import streamlit as st
from rag_module import ETFAPIClient, ETFAPIError, FAQRAGService, GeminiError, RAGError, ConfigurationError, answer_user_query, environment_status, resolve_faq_pdf_path

st.set_page_config(page_title="ETF 괴리율·NAV 해석 도우미", page_icon="📊")
st.title("ETF 괴리율·NAV 해석 도우미")
st.caption("공식 일별 데이터로 국내 ETF의 종가와 NAV 차이를 계산하고 관련 개념을 설명합니다.")
st.info("ETF 이름 또는 6자리 종목코드를 입력하세요. 예: `KODEX 200` 또는 `069500`")
st.caption("분석 후에는 ‘왜 그런 거야?’, ‘할인은 무슨 뜻이야?’처럼 이어서 물어볼 수 있습니다.")

defaults = {"messages":[], "rag_service":None, "document_name":None, "document_processed":False, "document_hash":None, "pending_candidates":[], "pending_query":None, "last_analysis_context":None}
for key,value in defaults.items():
    if key not in st.session_state: st.session_state[key]=value

def render_analysis_details(data, analysis):
    if data:
        with st.expander("공식 API 데이터와 계산 결과"):
            st.json({"공식 API 데이터":data, "결정론적 계산":analysis})

@st.cache_resource(show_spinner=False)
def load_faq_service(pdf_path: str, modified_ns: int, file_size: int) -> FAQRAGService:
    service = FAQRAGService()
    service.initialize_from_pdf(pdf_path)
    return service

knowledge_error = None
if st.session_state.rag_service is None:
    try:
        faq_path = resolve_faq_pdf_path()
        with st.spinner("ETF FAQ 지식베이스를 준비하는 중입니다..."):
            stat = faq_path.stat()
            service = load_faq_service(str(faq_path), stat.st_mtime_ns, stat.st_size)
            st.session_state.update(rag_service=service, document_name=service.document_name, document_processed=True, document_hash=service.document_hash)
    except (ConfigurationError, RAGError, GeminiError) as exc:
        knowledge_error = str(exc)

with st.sidebar:
    st.header("서비스 상태")
    for name,ready in environment_status().items(): st.write(("✅" if ready else "⚠️")+f" {name}: "+("설정됨" if ready else "미설정"))
    st.write("FAQ 지식베이스:", "사용 가능" if st.session_state.document_processed else "사용 불가")
    if knowledge_error: st.error(knowledge_error)
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages=[]; st.session_state.pending_candidates=[]; st.session_state.pending_query=None; st.session_state.last_analysis_context=None; st.rerun()
    st.divider(); st.info("지원: 국내 ETF 종가·NAV·괴리율 분석, 괴리율·LP·추적오차 개념\n\n데이터: 금융위원회 OpenAPI가 제공하는 최신 일 단위 기준값(최근 거래일보다 반영이 늦을 수 있음)\n\n미지원: 추천, 예측, 실시간 시세, 뉴스, 세금, 포트폴리오")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_analysis_details(message.get("data"), message.get("analysis"))

if st.session_state.pending_candidates:
    st.subheader("ETF 선택"); options=st.session_state.pending_candidates
    labels=[f"{x.get('itmsNm')} | {x.get('srtnCd')} | {x.get('bssIdxIdxNm') or '-'} | {x.get('basDt') or '-'}" for x in options]
    selected=st.selectbox("검색 결과가 여러 개입니다. 종목을 선택하세요.", range(len(labels)), format_func=lambda i:labels[i])
    if st.button("선택한 ETF로 조회"):
        try:
            result=answer_user_query(st.session_state.pending_query or "",ETFAPIClient(),st.session_state.rag_service,selected_code=str(options[selected]["srtnCd"]),previous_context=st.session_state.last_analysis_context)
            if result.get("conversation_context"): st.session_state.last_analysis_context=result["conversation_context"]
            answer=result["answer"]; st.session_state.messages.append({"role":"assistant","content":answer,"data":result.get("data"),"analysis":result.get("analysis")})
            st.session_state.pending_candidates=[]; st.session_state.pending_query=None; st.rerun()
        except (ConfigurationError,ETFAPIError,RAGError,GeminiError) as exc: st.error(str(exc))

if question := st.chat_input("ETF 이름·종목코드 또는 괴리율 관련 질문을 입력하세요"):
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("답변을 준비하는 중입니다..."): result=answer_user_query(question,ETFAPIClient(),st.session_state.rag_service,previous_context=st.session_state.last_analysis_context)
            if result["candidates"]:
                st.session_state.pending_candidates=result["candidates"]; st.session_state.pending_query=question; answer="검색 결과가 여러 개입니다. 아래에서 ETF를 선택해 주세요."
            else:
                answer=result["answer"]
                if result.get("conversation_context"): st.session_state.last_analysis_context=result["conversation_context"]
            st.markdown(answer); st.session_state.messages.append({"role":"assistant","content":answer,"data":result.get("data"),"analysis":result.get("analysis")})
            render_analysis_details(result.get("data"), result.get("analysis"))
            if result["candidates"]: st.rerun()
        except (ConfigurationError,ETFAPIError,RAGError,GeminiError) as exc:
            answer=f"오류: {exc}"; st.error(answer); st.session_state.messages.append({"role":"assistant","content":answer})
