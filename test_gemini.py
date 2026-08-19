"""배포 전 Gemini API 키와 채팅 모델 연결을 수동 점검하는 스크립트."""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_module import CHAT_MODEL


def main() -> None:
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    response = llm.invoke("안녕하세요. 한 문장으로 자기소개해 주세요.")
    print(response.text)


if __name__ == "__main__":
    main()
