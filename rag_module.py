"""ETF FAQ RAG와 금융위원회 ETF 시세 OpenAPI 모듈."""
from __future__ import annotations

import hashlib, json, os, re, tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
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
CHAT_MODEL, EMBEDDING_MODEL = "gemini-3.5-flash", "models/gemini-embedding-001"
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
FIELD_LABELS = {"clpr":"종가", "nav":"NAV", "mkp":"시가", "hipr":"고가", "lopr":"저가", "trqu":"거래량", "trPrc":"거래대금", "mrktTotAmt":"시가총액", "nPptTotAmt":"순자산총액", "fltRt":"등락률", "bssIdxIdxNm":"기초지수명", "bssIdxClpr":"기초지수 종가"}
METRIC_WORDS = {"종가":"clpr", "가격":"clpr", "NAV":"nav", "순자산가치":"nav", "시가":"mkp", "고가":"hipr", "저가":"lopr", "거래량":"trqu", "거래대금":"trPrc", "시가총액":"mrktTotAmt", "순자산총액":"nPptTotAmt", "등락률":"fltRt", "기초지수":"bssIdxIdxNm"}

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
    for old, new in {"코덱스":"KODEX", "타이거":"TIGER", "에이스":"ACE", "케이비스타":"KBSTAR", "쏠":"SOL"}.items():
        text = re.sub(old, new, text, flags=re.I)
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
        query = normalize_etf_query(query); code = re.search(r"(?<!\d)(\d{6})(?!\d)", query)
        if code: rows = self._request(likeSrtnCd=code.group(1))
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
    def __init__(self, api_key: str | None = None, chat_model: str = CHAT_MODEL):
        self.api_key, self.chat_model = api_key or _key("GEMINI_API_KEY"), chat_model
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
            return digest
        except (ConfigurationError, RAGError): raise
        except Exception as exc: raise _friendly_gemini_error(exc) from exc

    def retrieve(self, question: str, k: int = 4) -> list[Document]:
        if self.vectorstore is None: raise RAGError("서비스의 ETF FAQ 지식베이스를 사용할 수 없습니다. 운영자에게 문의해 주세요.")
        try: return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":k}).invoke(question)
        except Exception as exc: raise _friendly_gemini_error(exc) from exc

    def answer(self, question: str) -> dict[str, Any]:
        docs = self.retrieve(question)
        if not docs: return {"answer":"업로드된 한국거래소 ETF FAQ 문서에서는 해당 내용을 확인할 수 없습니다.", "sources":[]}
        context = "\n\n".join(f"[{d.metadata['file_name']} {d.metadata['page']+1}페이지]\n{d.page_content}" for d in docs)
        system = "제공된 FAQ 문서 문맥만 근거로 답하세요. 문서에 없는 내용은 추측하지 말고 질문과 무관한 문맥으로 답하지 마세요. 수치, 날짜, 제도를 왜곡하지 마세요. 근거가 부족하면 '업로드된 한국거래소 ETF FAQ 문서에서는 해당 내용을 확인할 수 없습니다.'라고 답하세요. 한국어로 명확하고 간결하게 쓰고 투자 권유나 수익 보장 표현을 하지 마세요. 본문만 작성하세요."
        try:
            llm = ChatGoogleGenerativeAI(model=self.chat_model, temperature=0, google_api_key=self.api_key)
            response = llm.invoke([SystemMessage(content=system), HumanMessage(content=f"문맥:\n{context}\n\n질문: {question}")])
            body = getattr(response, "text", None) or str(response.content)
        except Exception as exc: raise _friendly_gemini_error(exc) from exc
        sources = sorted({f"{d.metadata['file_name']}, {d.metadata['page']+1}페이지" for d in docs})
        return {"answer":body.strip()+"\n\n출처:\n"+"\n".join(f"- {x}" for x in sources), "sources":sources}

class QueryRouter:
    api_words, rag_words = tuple(METRIC_WORDS), ("ETF란", "ETF가 무엇", "무엇인지", "장점", "차이", "거래 방법", "매매", "상장", "설정", "환매", "유동성공급자", "LP", "괴리율", "추적오차", "세금", "위험", "FAQ", "개념")
    def __init__(self, api_key: str | None = None): self.api_key = api_key or _key("GEMINI_API_KEY")
    def classify(self, question: str) -> str:
        upper = question.upper(); api = any(x.upper() in upper for x in self.api_words); rag = any(x.upper() in upper for x in self.rag_words)
        if api and rag: return "HYBRID"
        if api or re.search(r"\b\d{6}\b", question): return "API"
        if rag: return "RAG"
        if re.search(r"\b(?:KODEX|TIGER|ACE|KBSTAR|SOL)\s+\S+", normalize_etf_query(question), re.I): return "API"
        if "ETF" in upper: return "GENERAL"
        return "UNSUPPORTED"
    def extract_entities(self, question: str) -> dict[str, Any]:
        normalized = normalize_etf_query(question); match = re.search(r"(?<!\d)(\d{6})(?!\d)", normalized)
        metric_text = normalized.upper()
        metrics = []
        # 긴 표현부터 소비해 '시가총액'을 '시가'로도 중복 인식하지 않는다.
        for word, field in sorted(METRIC_WORDS.items(), key=lambda item: len(item[0]), reverse=True):
            if word.upper() in metric_text:
                metrics.append(field); metric_text = metric_text.replace(word.upper(), " ")
        metrics = list(dict.fromkeys(metrics))
        name_match = re.search(r"\b(KODEX|TIGER|ACE|KBSTAR|SOL)\s+([A-Za-z0-9가-힣&+._-]+)", normalized, re.I)
        etf_name = f"{name_match.group(1).upper()} {name_match.group(2)}" if name_match else None
        rag_query = normalized
        if etf_name: rag_query = re.sub(re.escape(name_match.group(0)), " ", rag_query, count=1, flags=re.I)
        for word in sorted(METRIC_WORDS, key=len, reverse=True): rag_query = re.sub(re.escape(word), " ", rag_query, flags=re.I)
        rag_query = re.sub(r"\b(?:최근|알려줘|보여줘)\b|(?:와|과|랑|이랑)\s*$", " ", rag_query).strip(" ,·과와")
        api_metric_names = {"clpr":"close_price", "nav":"nav", "mkp":"open_price", "hipr":"high_price", "lopr":"low_price", "trqu":"trading_volume", "trPrc":"trading_value", "mrktTotAmt":"market_cap", "nPptTotAmt":"net_asset_total", "fltRt":"change_rate", "bssIdxIdxNm":"base_index"}
        final_metrics = metrics or ["clpr"]
        return {"query":normalized, "etf_name":etf_name, "code":match.group(1) if match else None,
                "metrics":final_metrics, "api_metrics":[api_metric_names.get(x, x) for x in final_metrics],
                "rag_query":rag_query or normalized}

def _format_date(value: Any) -> str:
    try: return datetime.strptime(str(value), "%Y%m%d").strftime("%Y년 %m월 %d일")
    except ValueError: return str(value or "기준일 미상")

def format_api_answer(row: dict[str, Any], metrics: list[str]) -> str:
    lines = [f"{_format_date(row.get('basDt'))} 기준 {row.get('itmsNm')}({row.get('srtnCd')}) 데이터입니다."]
    for field in metrics:
        value, label = row.get(field), FIELD_LABELS.get(field, field)
        if value is None: lines.append(f"- {label}: 제공되지 않음")
        elif field == "fltRt": lines.append(f"- {label}: {value:,.2f}%")
        elif isinstance(value, (int,float)): lines.append(f"- {label}: {value:,}")
        else: lines.append(f"- {label}: {value}")
    return "\n".join(lines+["", f"출처: {ETF_API_SOURCE}", "※ 실시간 시세가 아닌 일 단위 데이터입니다."])

def answer_user_query(question: str, api_client: ETFAPIClient, rag_service: FAQRAGService | None = None) -> dict[str, Any]:
    router = QueryRouter(); intent = router.classify(question); entities = router.extract_entities(question)
    result = {"intent":intent, "answer":"", "sources":[], "candidates":[], "data":None}
    if intent in {"API", "HYBRID"}:
        candidates = api_client.search_etfs(entities["code"] or entities["etf_name"] or entities["query"])
        if not candidates: raise ETFAPINoDataError("조건에 맞는 ETF 데이터를 찾지 못했습니다.")
        if len(candidates)>1: result["candidates"] = candidates; return result
        row = api_client.get_latest_etf(str(candidates[0]["srtnCd"])); result["data"] = row
        result["answer"], result["sources"] = format_api_answer(row, entities["metrics"]), [ETF_API_SOURCE]
    if intent in {"RAG", "GENERAL", "HYBRID"}:
        if rag_service is None or rag_service.vectorstore is None:
            notice = "서비스의 ETF FAQ 지식베이스를 사용할 수 없습니다. 운영자에게 문의해 주세요."
            result["answer"] = (result["answer"]+"\n\n[ETF 개념]\n"+notice).strip() if intent == "HYBRID" else notice
        else:
            rag = rag_service.answer(entities["rag_query"] if intent == "HYBRID" else question)
            result["answer"] = ("[시세 정보]\n"+result["answer"]+"\n\n[ETF 개념]\n"+rag["answer"]) if intent == "HYBRID" else rag["answer"]
            result["sources"] += rag["sources"]
    if intent == "UNSUPPORTED": result["answer"] = "현재 챗봇은 한국거래소 ETF FAQ 문서와 금융위원회 ETF 시세 데이터 범위에서 답변합니다."
    return result
