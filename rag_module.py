"""ETF FAQ RAG와 금융위원회 ETF 시세 OpenAPI 모듈."""
from __future__ import annotations

import hashlib, os, re, shutil, tempfile
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote

import fitz
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
ETF_API_URL = "https://apis.data.go.kr/1160100/service/GetSecuritiesProductInfoService/getETFPriceInfo"
ETF_API_SOURCE = "금융위원회 증권상품시세정보 OpenAPI"
BREVITY_WORDS = ("간단", "쉽게", "한 줄", "한줄", "요약", "뭐라는")
CHAT_MODEL, EMBEDDING_MODEL = "gemini-3.5-flash", "models/gemini-embedding-001"
RAG_MIN_RELEVANCE_SCORE, MAX_CACHED_INDEXES = 0.5, 2
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FAQ_PATH = BASE_DIR / "data" / "ETF_FAQ.pdf"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

class ConfigurationError(Exception): pass
class ETFAPIError(Exception): pass
class ETFAPIAuthenticationError(ETFAPIError): pass
class ETFAPIRateLimitError(ETFAPIError): pass
class ETFAPINetworkError(ETFAPIError): pass
class ETFAPINoDataError(ETFAPIError): pass
class ETFMultipleMatchesError(ETFAPIError): pass
class RAGError(Exception): pass
class GeminiError(RAGError): pass

NUMERIC_FIELDS = {"clpr", "vs", "fltRt", "nav", "mkp", "hipr", "lopr", "trqu", "trPrc", "mrktTotAmt", "nPptTotAmt", "stLstgCnt", "bssIdxClpr"}
ALL_FIELDS = ["basDt", "srtnCd", "isinCd", "itmsNm", "clpr", "vs", "fltRt", "nav", "mkp", "hipr", "lopr", "trqu", "trPrc", "mrktTotAmt", "nPptTotAmt", "stLstgCnt", "bssIdxIdxNm", "bssIdxClpr"]
ETF_BRANDS = ("KODEX", "TIGER", "ACE", "RISE", "SOL", "PLUS", "HANARO")
ETF_BRAND_PATTERN = "|".join(ETF_BRANDS)
ETF_CODE_PATTERN = r"(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{5})(?![A-Za-z0-9])"
QUERY_FILLER_WORDS = ("순자산가치", "NAV", "종가", "가격", "괴리율", "분석")
RAG_CONCEPT_WORDS = ("NAV", "순자산가치", "괴리율", "추적오차", "시장가격", "기준가격", "유동성공급자", "LP", "할증", "할인", "프리미엄")

def _key(name: str, fallback: str | None = None) -> str | None:
    return os.getenv(name) or (os.getenv(fallback) if fallback else None)

def environment_status() -> dict[str, bool]:
    return {"GEMINI_API_KEY": bool(_key("GEMINI_API_KEY")), "DATA_GO_KR_API_KEY": bool(_key("DATA_GO_KR_API_KEY", "DATA_GO_KR_SERVICE_KEY"))}

def _number(value: Any) -> int | float | None:
    if value is None or str(value).strip().lower() in {"", "null", "none"}: return None
    try:
        number = float(str(value).replace(",", "").strip())
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError): return None

def _date(value: str) -> str:
    try: datetime.strptime(value, "%Y%m%d")
    except ValueError as exc: raise ETFAPIError("날짜는 YYYYMMDD 형식의 유효한 날짜여야 합니다.") from exc
    return value

def _friendly_gemini_error(exc: Exception) -> GeminiError:
    text = str(exc).lower()
    if "quota" in text or "429" in text: return GeminiError("Gemini API 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
    if "api key" in text or "401" in text or "403" in text: return GeminiError("Gemini API 인증에 실패했습니다. 환경변수를 확인해 주세요.")
    return GeminiError("Gemini 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")

def normalize_etf_query(text: str) -> str:
    for old, new in {"코덱스":"KODEX", "타이거":"TIGER", "에이스":"ACE", "케이비스타":"RISE", "라이즈":"RISE", "쏠":"SOL", "아리랑":"PLUS", "플러스":"PLUS", "하나로":"HANARO"}.items():
        text = re.sub(old, new, text, flags=re.I)
    text = re.sub(r"\bKBSTAR\b", "RISE", text, flags=re.I)
    text = re.sub(r"\bARIRANG\b", "PLUS", text, flags=re.I)
    return text.strip()

def _canonical_etf_name(text: Any) -> str:
    """브랜드 표기와 공백 차이를 제거한 종목명 비교값."""
    return re.sub(r"\s+", "", normalize_etf_query(str(text or ""))).upper()

def resolve_faq_pdf_path() -> Path:
    """운영자가 등록한 FAQ PDF를 정해진 우선순위로 찾는다."""
    configured = os.getenv("FAQ_PDF_PATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute(): candidate = BASE_DIR / candidate
        candidate = candidate.resolve()
        if not candidate.is_file() or candidate.suffix.lower() != ".pdf":
            raise ConfigurationError("FAQ_PDF_PATH가 유효한 PDF 파일을 가리키지 않습니다.")
        return candidate
    if DEFAULT_FAQ_PATH.is_file(): return DEFAULT_FAQ_PATH.resolve()
    data_dir = BASE_DIR / "data"
    pdfs = sorted((p.resolve() for p in data_dir.glob("*.pdf") if p.is_file()), key=lambda p:p.name.lower()) if data_dir.is_dir() else []
    faq_pdfs = [p for p in pdfs if "faq" in p.name.lower()]
    if len(faq_pdfs) == 1: return faq_pdfs[0]
    if len(faq_pdfs) > 1: raise ConfigurationError("data 폴더에 FAQ PDF가 여러 개 있어 문서를 결정할 수 없습니다. FAQ_PDF_PATH를 설정해 주세요.")
    if len(pdfs) == 1: return pdfs[0]
    if not pdfs: raise ConfigurationError("등록된 ETF FAQ PDF를 찾지 못했습니다. data 폴더 또는 FAQ_PDF_PATH를 확인해 주세요.")
    raise ConfigurationError("data 폴더에 PDF가 여러 개 있어 FAQ 문서를 결정할 수 없습니다. FAQ_PDF_PATH를 설정해 주세요.")

def cleanup_old_indexes(current_digest: str, max_indexes: int = MAX_CACHED_INDEXES, root: Path | None = None) -> list[Path]:
    """현재 인덱스와 최근 인덱스만 남기고 이전 해시 인덱스를 정리한다."""
    if max_indexes < 1: raise ValueError("max_indexes는 1 이상이어야 합니다.")
    store = root or VECTORSTORE_DIR
    if not store.is_dir(): return []
    index_dirs = [path for path in store.iterdir() if path.is_dir() and re.fullmatch(r"[0-9a-f]{64}", path.name)]
    current = store / current_digest
    others = sorted((path for path in index_dirs if path != current), key=lambda path:path.stat().st_mtime, reverse=True)
    ordered = ([current] if current in index_dirs else []) + others
    deleted = ordered[max_indexes:]
    for path in deleted: shutil.rmtree(path)
    return deleted

class ETFAPIClient:
    allowed_params = {"basDt", "beginBasDt", "endBasDt", "likeSrtnCd", "isinCd", "itmsNm", "likeItmsNm", "likeBssIdxIdxNm", "beginTrqu", "endTrqu", "beginMrktTotAmt", "endMrktTotAmt", "beginNav", "endNav"}
    def __init__(self, api_key: str | None = None, timeout: int = 15):
        self.api_key = api_key or _key("DATA_GO_KR_API_KEY", "DATA_GO_KR_SERVICE_KEY")
        self.timeout, self.session = timeout, requests.Session()

    def _raise_api_error(self, code: str, message: str = "") -> None:
        messages = {"10":"잘못된 요청 파라미터입니다.", "20":"서비스 접근이 거부되었습니다.", "22":"공공데이터 API 요청 횟수를 초과했습니다.", "30":"등록되지 않은 서비스 키입니다.", "31":"만료된 서비스 키입니다.", "32":"등록되지 않은 IP입니다.", "99":"공공데이터 API에서 알 수 없는 오류가 발생했습니다."}
        msg = messages.get(str(code), message or "공공데이터 API 오류가 발생했습니다.")
        if str(code) in {"20", "30", "31", "32"}: raise ETFAPIAuthenticationError(msg)
        if str(code) == "22": raise ETFAPIRateLimitError(msg)
        raise ETFAPIError(msg)

    def _request(self, num_rows: int = 100, page_no: int = 1, **filters: Any) -> list[dict[str, Any]]:
        if not self.api_key: raise ConfigurationError("DATA_GO_KR_API_KEY가 설정되지 않았습니다.")
        unknown = set(filters) - self.allowed_params
        if unknown: raise ETFAPIError(f"지원하지 않는 검색 파라미터: {', '.join(sorted(unknown))}")
        for name in ("basDt", "beginBasDt", "endBasDt"):
            if filters.get(name): filters[name] = _date(str(filters[name]))
        # 공공데이터포털은 인코딩/디코딩 키를 함께 제공한다. requests의
        # params가 자체 인코딩하므로 인코딩 키는 먼저 한 번 디코딩한다.
        service_key = unquote(self.api_key.strip())
        params = {"serviceKey":service_key, "numOfRows":num_rows, "pageNo":page_no, "resultType":"json", **{k:v for k,v in filters.items() if v not in (None, "")}}
        try:
            response = self.session.get(ETF_API_URL, params=params, timeout=self.timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise ETFAPINetworkError("공공데이터 API에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요.") from exc
        except requests.RequestException as exc:
            raise ETFAPINetworkError("공공데이터 API 요청 중 네트워크 오류가 발생했습니다.") from exc
        if response.status_code in {401, 403}:
            raise ETFAPIAuthenticationError("공공데이터 API 인증 또는 접근이 거부되었습니다.")
        if response.status_code == 429:
            raise ETFAPIRateLimitError("공공데이터 API 요청 횟수를 초과했습니다.")
        if not response.ok:
            raise ETFAPIError(f"공공데이터 API가 HTTP {response.status_code} 오류를 반환했습니다.")
        try: payload = response.json()
        except ValueError:
            try:
                root = ET.fromstring(response.text)
                code = root.findtext(".//returnReasonCode") or root.findtext(".//resultCode") or "99"
                message = root.findtext(".//returnAuthMsg") or root.findtext(".//resultMsg") or ""
                self._raise_api_error(code, message)
            except ET.ParseError as exc: raise ETFAPIError("공공데이터 API 응답을 해석할 수 없습니다.") from exc
            return []
        header = payload.get("response", {}).get("header", {}); code = str(header.get("resultCode", "00"))
        if code not in {"00", "0"}: self._raise_api_error(code, str(header.get("resultMsg", "")))
        items = payload.get("response", {}).get("body", {}).get("items") or {}
        raw = items.get("item", []) if isinstance(items, dict) else []
        if isinstance(raw, dict): raw = [raw]
        return [self._normalize(x) for x in raw if isinstance(x, dict)]

    @staticmethod
    def _normalize(item: dict[str, Any]) -> dict[str, Any]:
        return {f: (_number(item.get(f)) if f in NUMERIC_FIELDS else item.get(f) or None) for f in ALL_FIELDS}

    def search_etfs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        query = normalize_etf_query(query); code = re.search(ETF_CODE_PATTERN, query, re.I)
        if code: rows = self._request(likeSrtnCd=code.group(1).upper())
        else:
            cleaned = re.sub(r"(?:최근|알려줘|찾아줘|가격|종가|거래량|거래대금|시가총액|NAV|ETF|추종하는)", " ", query, flags=re.I).strip()
            rows = self._request(likeItmsNm=cleaned)
            if not rows: rows = self._request(likeBssIdxIdxNm=cleaned)
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("srtnCd") or row.get("isinCd"))
            if key not in latest or str(row.get("basDt") or "") > str(latest[key].get("basDt") or ""): latest[key] = row
        candidates = sorted(latest.values(), key=lambda x: (str(x.get("itmsNm")), str(x.get("srtnCd"))))
        if not code:
            exact = [row for row in candidates if _canonical_etf_name(row.get("itmsNm")) == _canonical_etf_name(cleaned)]
            if exact: candidates = exact
        return candidates[:limit]

    def get_etf_by_code(self, code: str) -> list[dict[str, Any]]: return self._request(likeSrtnCd=code)
    def get_latest_etf(self, query: str) -> dict[str, Any]:
        candidates = self.search_etfs(query)
        if not candidates: raise ETFAPINoDataError("조건에 맞는 ETF 데이터를 찾지 못했습니다.")
        if len(candidates) > 1: raise ETFMultipleMatchesError("여러 ETF가 검색되었습니다.")
        rows = self.get_etf_by_code(str(candidates[0].get("srtnCd")))
        return max(rows or candidates, key=lambda x: str(x.get("basDt") or ""))
    def get_period_prices(self, code: str, begin: str, end: str) -> list[dict[str, Any]]:
        return self._request(num_rows=1000, likeSrtnCd=code, beginBasDt=_date(begin), endBasDt=_date(end))

class FAQRAGService:
    def __init__(self, api_key: str | None = None, chat_model: str = CHAT_MODEL, min_relevance_score: float = RAG_MIN_RELEVANCE_SCORE):
        self.api_key, self.chat_model = api_key or _key("GEMINI_API_KEY"), chat_model
        self.min_relevance_score = min_relevance_score
        self.vectorstore: FAISS | None = None
        self.document_name: str | None = None
        self.document_hash: str | None = None

    def build_vectorstore(self, uploaded_file: BinaryIO) -> str:
        if not self.api_key: raise ConfigurationError("GEMINI_API_KEY가 설정되지 않았습니다.")
        data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        digest = hashlib.sha256(data).hexdigest()
        if self.vectorstore is not None and digest == self.document_hash: return digest
        name, temp_path = Path(getattr(uploaded_file, "name", "ETF_FAQ.pdf")).name, None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                temp.write(data); temp_path = temp.name
            with fitz.open(temp_path) as pdf:
                docs = [Document(page_content=text, metadata={"source":name, "file_name":name, "page":i}) for i, page in enumerate(pdf) if (text := page.get_text("text").strip())]
            if not docs: raise RAGError("PDF에서 읽을 수 있는 텍스트를 찾지 못했습니다.")
            chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100, separators=["\n\n", "\n", ". ", "다. ", "요. ", " ", ""]).split_documents(docs)
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=self.api_key)
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            self.document_name, self.document_hash = name, digest
            return digest
        except (RAGError, ConfigurationError): raise
        except Exception as exc: raise _friendly_gemini_error(exc) from exc
        finally:
            if temp_path:
                try: os.unlink(temp_path)
                except OSError: pass

    def initialize_from_pdf(self, pdf_path: str | Path | None = None) -> str:
        """로컬 FAQ PDF의 기존 인덱스를 로드하거나 새로 생성한다."""
        if not self.api_key: raise ConfigurationError("GEMINI_API_KEY가 설정되지 않았습니다.")
        path = Path(pdf_path).resolve() if pdf_path else resolve_faq_pdf_path()
        if not path.is_file() or path.suffix.lower() != ".pdf": raise ConfigurationError("ETF FAQ PDF 경로가 올바르지 않습니다.")
        data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
        index_dir = VECTORSTORE_DIR / digest
        index_bundle = index_dir / "index.bin"
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=self.api_key)
        try:
            if index_bundle.is_file():
                # FAISS의 Windows 파일 writer는 한글 경로를 처리하지 못하므로
                # 메모리 직렬화 후 Python의 유니코드 경로 지원으로 저장/로드한다.
                self.vectorstore = FAISS.deserialize_from_bytes(
                    index_bundle.read_bytes(), embeddings, allow_dangerous_deserialization=True
                )
            else:
                with fitz.open(path) as pdf:
                    docs = [Document(page_content=text, metadata={"source":path.name, "file_name":path.name, "page":i}) for i,page in enumerate(pdf) if (text:=page.get_text("text").strip())]
                if not docs: raise RAGError("PDF에서 읽을 수 있는 텍스트를 찾지 못했습니다.")
                chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100, separators=["\n\n", "\n", ". ", "다. ", "요. ", " ", ""]).split_documents(docs)
                self.vectorstore = FAISS.from_documents(chunks, embeddings)
                index_dir.mkdir(parents=True, exist_ok=True)
                index_bundle.write_bytes(self.vectorstore.serialize_to_bytes())
            self.document_name, self.document_hash = path.name, digest
            try: cleanup_old_indexes(digest)
            except OSError: pass
            return digest
        except (ConfigurationError, RAGError): raise
        except Exception as exc: raise _friendly_gemini_error(exc) from exc

    def retrieve(self, question: str, k: int = 4) -> list[Document]:
        if self.vectorstore is None: raise RAGError("서비스의 ETF FAQ 지식베이스를 사용할 수 없습니다. 운영자에게 문의해 주세요.")
        try:
            concepts = [word for word in RAG_CONCEPT_WORDS if word.upper() in question.upper()]
            matches = self.vectorstore.similarity_search_with_relevance_scores(question, k=k)
            if len(concepts) > 1 and any(marker in question.lower() for marker in ("차이", "다르", "달라", "비교")):
                for concept in concepts: matches += self.vectorstore.similarity_search_with_relevance_scores(concept, k=1)
            unique: dict[tuple[Any, ...], tuple[Document, float]] = {}
            for doc, score in matches:
                if score < self.min_relevance_score: continue
                key = (doc.metadata.get("file_name"), doc.metadata.get("page"), doc.page_content)
                if key not in unique or score > unique[key][1]: unique[key] = (doc, score)
            return [doc for doc, score in sorted(unique.values(), key=lambda item:item[1], reverse=True)]
        except Exception as exc: raise _friendly_gemini_error(exc) from exc

    def answer(self, question: str, conversation_context: dict[str, Any] | None = None) -> dict[str, Any]:
        search_question = question + (" ETF NAV 괴리율 유동성공급자 LP" if conversation_context else "")
        docs = self.retrieve(search_question)
        if not docs: return {"answer":"업로드된 한국거래소 ETF FAQ 문서에서는 해당 내용을 확인할 수 없습니다.", "sources":[]}
        context = "\n\n".join(f"[{d.metadata['file_name']} {d.metadata['page']+1}페이지]\n{d.page_content}" for d in docs)
        previous = ""
        if conversation_context:
            previous = "\n\n직전 분석 정보:\n" + "\n".join(f"- {key}: {conversation_context.get(key)}" for key in ("itmsNm", "srtnCd", "basDt", "difference", "rate", "status") if conversation_context.get(key) is not None)
        denial = "업로드된 한국거래소 ETF FAQ 문서에서는 해당 내용을 확인할 수 없습니다."
        system = f"차분한 금융 데이터 분석가처럼 한국어 존댓말로 답하세요. 핵심을 먼저 말하고 전문용어는 짧게 풀어 2~4문장으로 설명하세요. 인사, 칭찬, 감탄사, 이모지, 과장된 친근함은 쓰지 마세요. 제공된 FAQ 문맥과 직전 분석 정보만 근거로 사용하고 수치, 날짜, 제도를 바꾸거나 새로운 수치를 만들지 마세요. 비교 질문은 서로 다른 문맥 조각의 근거를 종합해 차이를 설명하세요. 단일 기준일의 괴리율이 0에 가깝다는 이유만으로 가장 안정적인 ETF라고 단정하지 말고 기간별 괴리율, 유동성, 호가 스프레드와 추적오차를 함께 봐야 한다고 설명하세요. 근거가 부족하면 '{denial}'라고 답하세요. 투자 권유나 수익 보장 표현 없이 본문만 작성하세요."
        try:
            llm = ChatGoogleGenerativeAI(model=self.chat_model, temperature=0, google_api_key=self.api_key)
            response = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"FAQ 문맥:\n{context}{previous}\n\n질문: {question}")])
            body = getattr(response, "text", None) or str(response.content)
        except Exception as exc: raise _friendly_gemini_error(exc) from exc
        if denial in body: return {"answer":denial, "sources":[]}
        sources = sorted({f"{d.metadata['file_name']}, {d.metadata['page']+1}페이지" for d in docs})
        return {"answer":body.strip()+"\n\n출처:\n"+"\n".join(f"- {x}" for x in sources), "sources":sources}

class QueryRouter:
    rag_words = RAG_CONCEPT_WORDS
    unsupported_words = ("추천", "전망", "예측", "실시간", "뉴스", "세금", "과세", "수수료", "보수", "분배금", "배당", "거래량", "거래대금", "시가총액", "고가", "저가", "포트폴리오")
    explanation_words = ("왜", "무엇", "뭐", "개념", "설명", "차이", "다르", "달라", "어떻게", "의미")
    data_request_words = ("최신", "일별", "데이터", "조회", "보여줘", "계산", "분석")
    follow_up_words = ("왜", "그럼", "그런", "이런", "이건", "무슨 뜻", "반대", "여기서", "이 수치", "언제 기준", "다시 설명", "거의", "안정", "볼 수") + BREVITY_WORDS
    def __init__(self, api_key: str | None = None): self.api_key = api_key or _key("GEMINI_API_KEY")
    def classify(self, question: str, has_context: bool = False) -> str:
        upper = question.upper()
        if any(x.upper() in upper for x in self.unsupported_words): return "UNSUPPORTED"
        normalized = normalize_etf_query(question)
        api = bool(re.search(ETF_CODE_PATTERN, question, re.I) or re.search(rf"\b(?:{ETF_BRAND_PATTERN})\s+\S+", normalized, re.I))
        rag = any(x.upper() in upper for x in self.rag_words)
        explanation = any(word in question for word in self.explanation_words)
        if api: return "HYBRID" if rag and explanation else "API"
        if has_context and any(x.upper() in upper for x in self.follow_up_words): return "FOLLOW_UP"
        if rag and not explanation and any(word in question for word in self.data_request_words): return "NEEDS_ETF"
        if rag: return "RAG"
        return "UNSUPPORTED"
    def extract_entities(self, question: str) -> dict[str, Any]:
        normalized = normalize_etf_query(question); match = re.search(ETF_CODE_PATTERN, normalized, re.I)
        name_match = re.search(rf"\b({ETF_BRAND_PATTERN})\s+([A-Za-z0-9가-힣&+._-]+)", normalized, re.I)
        etf_name = f"{name_match.group(1).upper()} {name_match.group(2)}" if name_match else None
        rag_query = normalized
        if etf_name: rag_query = re.sub(re.escape(name_match.group(0)), " ", rag_query, count=1, flags=re.I)
        for word in QUERY_FILLER_WORDS: rag_query = re.sub(re.escape(word), " ", rag_query, flags=re.I)
        rag_query = re.sub(r"\b(?:최근|알려줘|보여줘)\b|(?:와|과|랑|이랑)\s*$", " ", rag_query).strip(" ,·과와")
        return {"query":normalized, "etf_name":etf_name, "code":match.group(1).upper() if match else None, "rag_query":rag_query or normalized}

def _format_date(value: Any) -> str:
    try: return datetime.strptime(str(value), "%Y%m%d").strftime("%Y년 %m월 %d일")
    except ValueError: return str(value or "기준일 미상")

def calculate_divergence(row: dict[str, Any]) -> dict[str, Any]:
    """공식 종가와 NAV로 괴리율을 결정론적으로 계산한다."""
    close, nav = _number(row.get("clpr")), _number(row.get("nav"))
    result = {"calculable":False, "close":close, "nav":nav, "difference":None, "rate":None, "status":None, "reason":None}
    if close is None:
        result["reason"] = "종가가 제공되지 않아 괴리율을 계산할 수 없습니다."
        return result
    if nav is None:
        result["reason"] = "NAV가 제공되지 않아 괴리율을 계산할 수 없습니다."
        return result
    if nav <= 0:
        result["reason"] = "NAV가 0 이하이므로 괴리율을 계산할 수 없습니다."
        return result
    difference, rate = close - nav, (close - nav) / nav * 100
    status = "할증" if difference > 0 else "할인" if difference < 0 else "일치"
    result.update(calculable=True, difference=difference, rate=rate, status=status)
    return result

def _format_number(value: int | float | None, signed: bool = False) -> str:
    if value is None: return "제공되지 않음"
    if isinstance(value, float) and not value.is_integer(): return f"{value:+,.2f}" if signed else f"{value:,.2f}"
    return f"{int(value):+,}" if signed else f"{int(value):,}"

def format_etf_analysis(row: dict[str, Any], analysis: dict[str, Any], today: date | None = None) -> str:
    lines = [f"{row.get('itmsNm')}({row.get('srtnCd')})을 최근 공식 데이터 기준으로 확인했습니다.", "", "[분석 결과]",
             f"- 기준일: {_format_date(row.get('basDt'))}", f"- 종가: {_format_number(analysis['close'])}", f"- NAV: {_format_number(analysis['nav'])}"]
    if analysis["calculable"]:
        status_text = {"할증":"종가가 NAV보다 높은 할증 상태입니다.", "할인":"종가가 NAV보다 낮은 할인 상태입니다.", "일치":"종가와 NAV가 같습니다."}[analysis["status"]]
        lines += [f"- 가격 차이(종가 - NAV): {_format_number(analysis['difference'], signed=True)}",
                  f"- 괴리율: {analysis['rate']:+.2f}%",
                  f"- 상태: {status_text}"]
    else:
        lines.append(f"- 괴리율: 계산 불가 ({analysis['reason']})")
    notices = ["※ 실시간 시세가 아닌 일 단위 데이터이며, 할증·할인은 투자 판단 신호가 아닙니다."]
    try: age = ((today or date.today()) - datetime.strptime(str(row.get("basDt")), "%Y%m%d").date()).days
    except ValueError: age = None
    if age is not None and age > 3: notices.append(f"※ API가 제공한 최신 기준일이 오늘보다 {age}일 전입니다. 최근 거래일 데이터가 아직 반영되지 않았을 수 있습니다.")
    return "\n".join(lines+["", "계산식: (종가 - NAV) / NAV × 100", f"출처: {ETF_API_SOURCE}", *notices])

def _analysis_context(row: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    return {"itmsNm":row.get("itmsNm"), "srtnCd":row.get("srtnCd"), "basDt":row.get("basDt"), "difference":analysis.get("difference"), "rate":analysis.get("rate"), "status":analysis.get("status")}

def _fallback_interpretation(analysis: dict[str, Any]) -> str:
    if not analysis["calculable"]: return analysis["reason"]
    meanings = {"할증":"종가가 NAV보다 높은 가격에 형성된 상태입니다.", "할인":"종가가 NAV보다 낮은 가격에 형성된 상태입니다.", "일치":"종가와 NAV가 같은 수준입니다."}
    return meanings[analysis["status"]] + " 시장 수요와 LP의 호가 공급 등에 따라 차이가 생길 수 있으며, 이 결과만으로 매수 여부를 판단할 수는 없습니다."

def _unsupported_answer(question: str) -> str:
    if "실시간" in question: return "실시간 시세는 확인할 수 없습니다. ETF 이름이나 종목코드를 입력하면 현재 제공되는 최신 일별 데이터로 괴리율을 분석해드릴 수 있습니다."
    if any(word in question for word in ("추천", "전망", "예측", "포트폴리오")): return "종목 추천이나 가격 예측은 제공하지 않습니다. 대신 확인하려는 ETF의 이름이나 종목코드를 입력하면 종가와 NAV의 차이를 공식 데이터로 살펴볼 수 있습니다."
    if any(word in question for word in ("세금", "과세", "수수료", "보수", "분배금", "배당")): return "세금·비용·분배금은 현재 분석 범위에 포함되지 않습니다. 이 서비스에서는 국내 ETF의 괴리율·NAV와 관련 개념을 확인할 수 있습니다."
    return "이 서비스는 공식 일별 데이터로 국내 ETF의 괴리율·NAV를 분석하고 관련 개념을 설명합니다. 분석할 ETF 이름이나 6자리 종목코드를 입력해 주세요."

def _follow_up_fallback(question: str, context: dict[str, Any]) -> str:
    if any(word in question for word in BREVITY_WORDS):
        rate = context.get("rate")
        rate_text = f"{rate:+.2f}%" if isinstance(rate, (int, float)) else "0에 가까운 괴리율"
        return f"쉽게 말하면, 괴리율 {rate_text}는 그날 ETF 가격과 NAV가 거의 같았다는 뜻입니다. 다만 이것만으로 지금 사도 안전하다고 볼 수는 없습니다."
    if "언제 기준" in question:
        return f"직전 분석은 {_format_date(context.get('basDt'))} 기준 {context.get('itmsNm')} 데이터입니다. API의 최신 반영일이 실제 최근 거래일보다 늦을 수 있으므로 기준일을 함께 확인해야 합니다."
    if any(word in question for word in ("거의", "안정", "볼 수", "0")):
        rate = context.get("rate")
        rate_text = f"{rate:+.2f}%" if isinstance(rate, (int, float)) else "0에 가까운 괴리율"
        return f"괴리율 {rate_text}은 해당 기준일에 시장가격과 NAV가 가까웠다는 뜻입니다. 다만 하루 수치만으로 가장 안정적인 상태라고 판단할 수는 없습니다. 기간별 괴리율 흐름과 유동성, 호가 스프레드, 추적오차를 함께 살펴봐야 합니다."
    status = context.get("status")
    return f"직전 결과는 {context.get('itmsNm')}의 {status or '괴리율'} 상태를 보여줍니다. 시장 수요와 LP의 호가 공급 등에 따라 시장가격과 NAV의 차이가 생길 수 있으며, 한 시점의 결과만으로 상품의 안정성을 단정할 수는 없습니다."

def answer_user_query(question: str, api_client: ETFAPIClient, rag_service: FAQRAGService | None = None, selected_code: str | None = None, previous_context: dict[str, Any] | None = None) -> dict[str, Any]:
    router = QueryRouter(); intent = router.classify(question, has_context=bool(previous_context)); entities = router.extract_entities(question)
    result = {"intent":intent, "answer":"", "sources":[], "candidates":[], "data":None, "analysis":None, "conversation_context":previous_context if intent == "FOLLOW_UP" else None}
    if intent in {"API", "HYBRID"}:
        if selected_code:
            row = api_client.get_latest_etf(selected_code)
        else:
            candidates = api_client.search_etfs(entities["code"] or entities["etf_name"] or entities["query"])
            if not candidates: raise ETFAPINoDataError("조건에 맞는 ETF 데이터를 찾지 못했습니다.")
            if len(candidates)>1: result["candidates"] = candidates; return result
            row = api_client.get_latest_etf(str(candidates[0]["srtnCd"]))
        result["data"] = row
        result["analysis"] = calculate_divergence(row)
        result["conversation_context"] = _analysis_context(row, result["analysis"])
        result["answer"], result["sources"] = format_etf_analysis(row, result["analysis"]), [ETF_API_SOURCE]
        if intent == "API":
            interpretation = _fallback_interpretation(result["analysis"])
            if rag_service is not None and rag_service.vectorstore is not None:
                try:
                    rag = rag_service.answer("직전 ETF 분석 결과의 의미와 괴리율이 생길 수 있는 이유를 설명해 주세요.", result["conversation_context"])
                    if rag["sources"]: interpretation = rag["answer"]; result["sources"] += rag["sources"]
                except RAGError: pass
            result["answer"] += "\n\n[해석]\n" + interpretation
    if intent in {"RAG", "HYBRID", "FOLLOW_UP"}:
        if intent == "FOLLOW_UP":
            fallback = _follow_up_fallback(question, previous_context or {})
            if any(word in question for word in BREVITY_WORDS) or rag_service is None or rag_service.vectorstore is None:
                result["answer"] = fallback
            else:
                try:
                    rag = rag_service.answer(question, result["conversation_context"])
                    result["answer"] = rag["answer"] if rag["sources"] else fallback
                    result["sources"] += rag["sources"]
                except RAGError: result["answer"] = fallback
        elif rag_service is None or rag_service.vectorstore is None:
            notice = "FAQ 설명 기능을 현재 사용할 수 없습니다. 공식 데이터 분석 결과는 계속 확인할 수 있습니다."
            result["answer"] = (result["answer"]+"\n\n[관련 개념]\n"+notice) if intent == "HYBRID" else notice
        else:
            rag = rag_service.answer(entities["rag_query"] if intent == "HYBRID" else question, result["conversation_context"] if intent == "FOLLOW_UP" else None)
            result["answer"] = (result["answer"]+"\n\n[관련 개념]\n"+rag["answer"]) if intent == "HYBRID" else rag["answer"]
            result["sources"] += rag["sources"]
    if intent == "NEEDS_ETF": result["answer"] = "어떤 ETF의 괴리율을 확인할까요? ETF 이름이나 6자리 종목코드를 입력해 주세요. 예: KODEX 200 또는 069500"
    if intent == "UNSUPPORTED": result["answer"] = _unsupported_answer(question)
    return result
