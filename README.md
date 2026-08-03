# ETF 질의응답 챗봇 MVP

한국거래소 ETF FAQ PDF 기반 RAG와 금융위원회 증권상품시세정보 OpenAPI를 결합한 Streamlit 챗봇입니다. Gemini는 FAQ 문맥의 답변 생성에 사용하며 API에 없는 시세를 만들지 않습니다.

## 데이터 소스와 지원 범위

- 운영자가 `data` 폴더에 등록한 한국거래소 ETF FAQ PDF: ETF 개념, 제도, 거래 방식, 괴리율, 추적오차 등
- 금융위원회 증권상품시세정보 OpenAPI: 종가, NAV, 시가·고가·저가, 거래량·거래대금, 시가총액, 기초지수 등 일 단위 데이터
- Google Gemini API: 문서 임베딩과 FAQ 문맥 기반 답변

뉴스, 실시간 시세, 미래 가격 예측, 종목 추천, Open DART 및 다른 데이터베이스 질문은 지원하지 않습니다. 검색어가 모호하면 최대 10개의 ETF 후보를 제시합니다.

## 파일

- `app.py`: Streamlit UI, 세션 및 후보 선택 관리
- `rag_module.py`: PDF RAG, Gemini, ETF API, 라우팅과 응답 정규화
- `test_gemini.py`: Gemini 연결 확인
- `.env.example`: 환경변수 예시

FAQ PDF는 기본적으로 `data/ETF_FAQ.pdf`에 등록합니다. 다른 위치나 이름을 사용하면 `FAQ_PDF_PATH` 환경변수로 지정할 수 있습니다. 앱 시작 시 PDF 해시와 일치하는 FAISS 인덱스를 불러오며, 없으면 자동 생성합니다.

## 요구 환경과 라이브러리

- Python 3.10 이상
- `streamlit`: 웹 채팅 UI
- `python-dotenv`: `.env` 환경변수 로드
- `requests`: 금융위원회 OpenAPI 호출
- `pymupdf`: FAQ PDF 텍스트 추출
- `faiss-cpu`: FAQ 벡터 검색 인덱스
- `langchain-core`, `langchain-community`, `langchain-text-splitters`: 문서·Retriever·텍스트 분할
- `langchain-google-genai`: Gemini 채팅 및 임베딩 연동

정확한 설치 목록은 `requirements.txt`에서 관리합니다. 개별 설치보다 다음의 일괄 설치 명령을 사용하세요.

## API 키 발급

### Gemini API 키

1. [Google AI Studio API Keys](https://aistudio.google.com/app/apikey)에 로그인합니다.
2. 프로젝트를 선택하거나 생성한 뒤 Gemini API 키를 발급합니다.
3. 발급한 값을 `.env`의 `GEMINI_API_KEY`에 입력합니다.

Gemini 키는 FAQ 문서 임베딩과 문서 기반 답변 생성에 사용됩니다. 계정과 지역, 요금제에 따라 사용량 제한이 적용될 수 있습니다.

### 금융위원회 공공데이터 API 키

1. 공공데이터포털의 [금융위원회 증권상품시세정보 OpenAPI](https://www.data.go.kr/data/15094806/openapi.do)에 로그인합니다.
2. **활용신청**을 완료하고 마이페이지에서 서비스 키를 확인합니다.
3. 인코딩 키 또는 디코딩 키 중 하나를 `.env`의 `DATA_GO_KR_API_KEY`에 입력합니다. 이 프로젝트는 두 형식을 모두 처리합니다.

활용신청 승인 상태, 일일 호출 한도 및 서비스 키 활성화 여부에 따라 호출이 실패할 수 있습니다.

## 설치와 환경변수 설정 (Windows PowerShell)

프로젝트 루트에서 다음 명령을 순서대로 실행합니다.

```powershell
python -m venv etf
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\etf\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

생성된 `.env`를 열어 다음 형식으로 설정합니다.

```dotenv
GEMINI_API_KEY=발급받은_Gemini_API_키
DATA_GO_KR_API_KEY=발급받은_공공데이터_서비스_키
FAQ_PDF_PATH=data/ETF_FAQ.pdf
```

`FAQ_PDF_PATH`는 프로젝트 루트 기준 상대경로나 절대경로를 사용할 수 있습니다. 기본 파일이 `data/ETF_FAQ.pdf`에 있다면 예시 값을 그대로 사용합니다.

API 키 양옆에 불필요한 공백을 넣지 마세요. `.env`를 Git에 커밋하거나 API 키를 코드, 로그, 화면에 출력해서는 안 됩니다.

## 연결 확인

Gemini 연결만 먼저 확인하려면 다음 명령을 실행합니다. 성공하면 짧은 Gemini 응답이 출력되지만 API 키 자체는 출력되지 않습니다.

```powershell
python test_gemini.py
```

공공데이터 API는 앱에서 `069500 NAV 알려줘` 또는 `KODEX 200 종가 알려줘`와 같은 질문으로 확인할 수 있습니다.

## 앱 실행

```powershell
streamlit run app.py
```

터미널에 표시되는 로컬 주소(일반적으로 `http://localhost:8501`)를 브라우저에서 엽니다. 운영자가 FAQ PDF를 등록한 상태에서 앱을 실행하면 지식베이스가 자동 준비되며, 고객은 파일 업로드 없이 바로 질문합니다.

종료할 때는 실행 중인 터미널에서 `Ctrl+C`를 누릅니다. 패키지나 코드를 변경한 뒤 동작이 반영되지 않으면 앱을 종료하고 같은 명령으로 다시 실행하세요.

## 질문 예시

- `KODEX 200 최근 종가 알려줘`
- `069500 NAV 알려줘`
- `반도체 ETF 찾아줘`
- `ETF의 괴리율이란 무엇인가요?`
- `KODEX 200 종가와 ETF의 괴리율 개념을 알려줘`

## 한계와 보안

시세는 실시간이 아니며 공공데이터의 최신 제공 기준일을 표시합니다. FAQ 답변 품질은 PDF의 텍스트 추출 가능 여부에 좌우됩니다. 기간 수익률과 ETF 간 비교는 초기 MVP 범위 밖입니다. `.env`는 Git에서 제외되며 본 서비스는 투자 추천이나 수익 보장 서비스가 아닙니다.
