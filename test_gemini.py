from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
)

response = llm.invoke("안녕하세요. 한 문장으로 자기소개해 주세요.")

print(response.text)