import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from rag_module import FAQRAGService, QueryRouter, answer_user_query, calculate_divergence, cleanup_old_indexes, format_etf_analysis


class FakeAPIClient:
    def __init__(self):
        self.rows = [
            {"srtnCd": "069500", "itmsNm": "KODEX 200", "basDt": "20260818", "clpr": 42_000, "nav": 41_500},
            {"srtnCd": "999999", "itmsNm": "KODEX 200 TR", "basDt": "20260818", "clpr": 13_000, "nav": 13_100},
        ]

    def search_etfs(self, query):
        return self.rows

    def get_latest_etf(self, code):
        return next(row for row in self.rows if row["srtnCd"] == code)


class FakeRAGService:
    vectorstore = object()

    def __init__(self):
        self.questions = []
        self.contexts = []

    def answer(self, question, conversation_context=None):
        self.questions.append(question)
        self.contexts.append(conversation_context)
        return {"answer": "괴리율 개념 답변", "sources": ["ETF_FAQ.pdf, 1페이지"]}


class FakeVectorStore:
    def similarity_search_with_relevance_scores(self, question, k):
        self.last_question = question
        self.questions = getattr(self, "questions", []) + [question]
        return [
            (Document(page_content="관련 문서", metadata={"file_name":"ETF_FAQ.pdf", "page":0}), 0.8),
            (Document(page_content="무관한 문서", metadata={"file_name":"ETF_FAQ.pdf", "page":1}), 0.4),
        ]


class ComparisonVectorStore:
    def similarity_search_with_relevance_scores(self, question, k):
        tracking = Document(page_content="추적오차는 기초지수와 ETF 기준가격의 차이입니다.", metadata={"file_name":"ETF_FAQ.pdf", "page":2})
        divergence = Document(page_content="괴리율은 시장가격과 순자산가치의 차이입니다.", metadata={"file_name":"ETF_FAQ.pdf", "page":3})
        if question == "괴리율": return [(divergence, 0.9)]
        if question == "추적오차": return [(tracking, 0.9)]
        return [(tracking, 0.8)]


class RAGModuleTests(unittest.TestCase):
    def test_selected_candidate_keeps_hybrid_rag_answer(self):
        client, rag = FakeAPIClient(), FakeRAGService()
        question = "KODEX 200 종가와 ETF의 괴리율 개념을 알려줘"

        pending = answer_user_query(question, client, rag)
        resolved = answer_user_query(question, client, rag, selected_code="069500")

        self.assertEqual(len(pending["candidates"]), 2)
        self.assertIn("[분석 결과]", resolved["answer"])
        self.assertIn("[관련 개념]", resolved["answer"])
        self.assertIn("괴리율 개념 답변", resolved["answer"])
        self.assertEqual(resolved["data"]["srtnCd"], "069500")
        self.assertAlmostEqual(resolved["analysis"]["rate"], 500 / 41_500 * 100)
        self.assertEqual(len(rag.questions), 1)

    def test_divergence_is_calculated_without_llm(self):
        analysis = calculate_divergence({"clpr": 10_100, "nav": 10_000})

        self.assertTrue(analysis["calculable"])
        self.assertEqual(analysis["difference"], 100)
        self.assertEqual(analysis["rate"], 1)
        self.assertEqual(analysis["status"], "할증")

    def test_divergence_rejects_missing_or_invalid_nav(self):
        missing = calculate_divergence({"clpr": 10_100, "nav": None})
        zero = calculate_divergence({"clpr": 10_100, "nav": 0})

        self.assertFalse(missing["calculable"])
        self.assertFalse(zero["calculable"])
        self.assertIn("NAV", missing["reason"])

    def test_code_only_query_returns_focused_analysis(self):
        result = answer_user_query("069500", FakeAPIClient(), selected_code="069500")

        self.assertEqual(result["intent"], "API")
        self.assertIn("괴리율", result["answer"])
        self.assertIn("투자 판단 신호가 아닙니다", result["answer"])
        self.assertIn("[해석]", result["answer"])
        self.assertTrue(result["analysis"]["calculable"])

    def test_api_analysis_uses_rag_for_natural_interpretation(self):
        rag = FakeRAGService()

        result = answer_user_query("069500", FakeAPIClient(), rag, selected_code="069500")

        self.assertIn("[해석]", result["answer"])
        self.assertIn("괴리율 개념 답변", result["answer"])
        self.assertEqual(rag.contexts[0]["itmsNm"], "KODEX 200")

    def test_follow_up_uses_previous_analysis_context(self):
        rag = FakeRAGService()
        initial = answer_user_query("069500", FakeAPIClient(), selected_code="069500")

        follow_up = answer_user_query("왜 그런 거야?", FakeAPIClient(), rag, previous_context=initial["conversation_context"])

        self.assertEqual(follow_up["intent"], "FOLLOW_UP")
        self.assertIn("괴리율 개념 답변", follow_up["answer"])
        self.assertEqual(rag.contexts[0]["status"], "할증")

    def test_retrieve_filters_low_relevance_documents(self):
        service = FAQRAGService(api_key="test", min_relevance_score=0.5)
        service.vectorstore = FakeVectorStore()

        docs = service.retrieve("ETF 괴리율")

        self.assertEqual([doc.page_content for doc in docs], ["관련 문서"])

    def test_follow_up_context_enriches_rag_retrieval(self):
        service = FAQRAGService(api_key="test", min_relevance_score=0.5)
        service.vectorstore = FakeVectorStore()
        llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(text="차분한 설명"))

        with patch("rag_module.ChatGoogleGenerativeAI", return_value=llm):
            service.answer("왜 그런 거야?", {"itmsNm":"KODEX 200", "status":"할증"})

        self.assertTrue(any("NAV 괴리율" in question for question in service.vectorstore.questions))

    def test_comparison_retrieval_collects_each_concept(self):
        service = FAQRAGService(api_key="test", min_relevance_score=0.5)
        service.vectorstore = ComparisonVectorStore()

        docs = service.retrieve("괴리율과 추적오차는 어떻게 달라?")

        self.assertEqual({doc.metadata["page"] for doc in docs}, {2, 3})

    def test_refusal_does_not_show_misleading_sources(self):
        service = FAQRAGService(api_key="test", min_relevance_score=0.5)
        service.vectorstore = FakeVectorStore()
        refusal = "업로드된 한국거래소 ETF FAQ 문서에서는 해당 내용을 확인할 수 없습니다."
        llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(text=refusal))

        with patch("rag_module.ChatGoogleGenerativeAI", return_value=llm):
            result = service.answer("관련 없는 질문")

        self.assertEqual(result["sources"], [])
        self.assertNotIn("출처:", result["answer"])

    def test_router_limits_questions_to_price_nav_concepts(self):
        router = QueryRouter()

        self.assertEqual(router.classify("ETF 괴리율은 왜 생기나요?"), "RAG")
        self.assertEqual(router.classify("KODEX 200 괴리율 분석"), "API")
        self.assertEqual(router.classify("KODEX 200 괴리율이 왜 생겨?"), "HYBRID")
        self.assertEqual(router.classify("RISE 200"), "API")
        self.assertEqual(router.classify("ETF 수수료는 얼마나 떼요?"), "UNSUPPORTED")
        self.assertEqual(router.classify("왜 그런 거야?", has_context=True), "FOLLOW_UP")
        self.assertEqual(router.classify("그럼 종목 추천해줘", has_context=True), "UNSUPPORTED")

    def test_router_requests_etf_for_data_question_without_instrument(self):
        router = QueryRouter()

        self.assertEqual(router.classify("가장 최신 일별 데이터 괴리율 보여줘"), "NEEDS_ETF")
        self.assertEqual(router.classify("TIGER 미국 괴리율"), "API")

    def test_missing_etf_does_not_call_rag(self):
        rag = FakeRAGService()

        result = answer_user_query("가장 최신 일별 데이터 괴리율 보여줘", FakeAPIClient(), rag)

        self.assertEqual(result["intent"], "NEEDS_ETF")
        self.assertIn("ETF 이름이나 6자리 종목코드", result["answer"])
        self.assertEqual(rag.questions, [])

    def test_stability_follow_up_uses_numeric_context_without_rag(self):
        initial = answer_user_query("069500", FakeAPIClient(), selected_code="069500")

        result = answer_user_query("이건 거의 괴리율이 0이네? 가장 안정적인 상태로 볼 수 있나?", FakeAPIClient(), previous_context=initial["conversation_context"])

        self.assertEqual(result["intent"], "FOLLOW_UP")
        self.assertIn("하루", result["answer"])
        self.assertIn("안정적", result["answer"])
        self.assertAlmostEqual(initial["conversation_context"]["rate"], 500 / 41_500 * 100)

    def test_brevity_follow_up_summarizes_context_without_rag(self):
        rag = FakeRAGService()
        context = {"itmsNm":"KODEX 200", "srtnCd":"069500", "basDt":"20260814", "rate":-0.28, "status":"할인"}

        result = answer_user_query("좀 간단하게 말해봐 뭐라는 거야", FakeAPIClient(), rag, previous_context=context)

        self.assertEqual(result["intent"], "FOLLOW_UP")
        self.assertIn("-0.28%", result["answer"])
        self.assertIn("지금 사도 안전", result["answer"])
        self.assertLessEqual(len(result["answer"]), 100)
        self.assertEqual(rag.questions, [])

    def test_old_api_date_shows_source_delay_warning(self):
        row = {"srtnCd":"069500", "itmsNm":"KODEX 200", "basDt":"20260814", "clpr":42_000, "nav":41_500}

        answer = format_etf_analysis(row, calculate_divergence(row), today=date(2026, 8, 19))

        self.assertIn("5일 전", answer)
        self.assertIn("아직 반영되지 않았을 수 있습니다", answer)

    def test_router_supports_new_codes_and_legacy_brand_names(self):
        router = QueryRouter()

        self.assertEqual(router.extract_entities("0093A0")["code"], "0093A0")
        self.assertEqual(router.classify("0093A0"), "API")
        self.assertEqual(router.extract_entities("KBSTAR 200")["etf_name"], "RISE 200")

    def test_unsupported_answer_suggests_supported_questions(self):
        result = answer_user_query("오늘 환율은?", FakeAPIClient())

        self.assertIn("괴리율·NAV", result["answer"])
        self.assertIn("ETF 이름이나 6자리 종목코드", result["answer"])

    def test_unsupported_answer_is_specific_to_the_question(self):
        realtime = answer_user_query("실시간 시세 알려줘", FakeAPIClient())
        recommendation = answer_user_query("ETF 추천해줘", FakeAPIClient())

        self.assertIn("일별", realtime["answer"])
        self.assertIn("추천", recommendation["answer"])

    def test_cleanup_keeps_current_and_one_previous_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old, previous, current = (root / (char * 64) for char in "abc")
            for timestamp, path in enumerate((old, previous, current), start=1):
                path.mkdir()
                (path / "index.bin").write_bytes(b"index")
                os.utime(path, (timestamp, timestamp))
            unrelated = root / "notes"
            unrelated.mkdir()

            deleted = cleanup_old_indexes(current.name, max_indexes=2, root=root)

            self.assertEqual(deleted, [old])
            self.assertFalse(old.exists())
            self.assertTrue(previous.exists())
            self.assertTrue(current.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
