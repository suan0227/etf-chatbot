import streamlit as st
from rag_module import ETFAPIClient, ETFAPIError, FAQRAGService, GeminiError, RAGError, ConfigurationError, answer_user_query, environment_status, format_api_answer, QueryRouter, resolve_faq_pdf_path

st.set_page_config(page_title="ETF 질의응답 챗봇", page_icon="📊")
st.title("ETF 질의응답 챗봇")
st.caption("한국거래소 ETF FAQ와 금융위원회 일 단위 시세 데이터를 이용합니다.")

defaults = {"messages":[], "rag_service":None, "document_name":None, "document_processed":False, "document_hash":None, "pending_candidates":[], "pending_query":None}
for key,value in defaults.items():
    if key not in st.session_state: st.session_state[key]=value

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
        st.session_state.messages=[]; st.session_state.pending_candidates=[]; st.session_state.pending_query=None; st.rerun()
    st.divider(); st.info("FAQ 질문은 서비스에 등록된 한국거래소 ETF FAQ에 근거합니다.\n\n시세 질문은 금융위원회 OpenAPI에 근거합니다. 시세는 실시간이 아닌 일 단위 데이터입니다.\n\n투자 추천 서비스가 아닙니다.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if st.session_state.pending_candidates:
    st.subheader("ETF 선택"); options=st.session_state.pending_candidates
    labels=[f"{x.get('itmsNm')} | {x.get('srtnCd')} | {x.get('bssIdxIdxNm') or '-'} | {x.get('basDt') or '-'}" for x in options]
    selected=st.selectbox("검색 결과가 여러 개입니다. 종목을 선택하세요.", range(len(labels)), format_func=lambda i:labels[i])
    if st.button("선택한 ETF로 조회"):
        try:
            row=ETFAPIClient().get_latest_etf(str(options[selected]["srtnCd"])); metrics=QueryRouter().extract_entities(st.session_state.pending_query or "")["metrics"]
            answer=format_api_answer(row,metrics); st.session_state.messages.append({"role":"assistant","content":answer})
            st.session_state.pending_candidates=[]; st.session_state.pending_query=None; st.rerun()
        except (ConfigurationError,ETFAPIError) as exc: st.error(str(exc))

if question := st.chat_input("ETF 질문을 입력하세요"):
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("답변을 준비하는 중입니다..."): result=answer_user_query(question,ETFAPIClient(),st.session_state.rag_service)
            if result["candidates"]:
                st.session_state.pending_candidates=result["candidates"]; st.session_state.pending_query=question; answer="검색 결과가 여러 개입니다. 아래에서 ETF를 선택해 주세요."
            else: answer=result["answer"]
            st.markdown(answer); st.session_state.messages.append({"role":"assistant","content":answer})
            if result.get("data"):
                with st.expander("정규화된 API 데이터"): st.dataframe([result["data"]],use_container_width=True)
            if result["candidates"]: st.rerun()
        except (ConfigurationError,ETFAPIError,RAGError,GeminiError) as exc:
            answer=f"오류: {exc}"; st.error(answer); st.session_state.messages.append({"role":"assistant","content":answer})
