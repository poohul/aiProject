import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.runnables import RunnablePassthrough

# --- OpenAI API 키 설정 ---
# 1. 환경 변수로 설정하는 것을 강력히 권장합니다.
#    예: export OPENAI_API_KEY="YOUR_API_KEY_HERE"
# 2. 코드 내에서 직접 설정할 수도 있지만, 보안상 좋지 않습니다.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# API 키가 설정되었는지 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 환경 변수를 설정해주세요.")

# --- LangChain LLM 초기화 (OpenAI 사용) ---
# temperature 값을 조정하여 생성되는 텍스트의 창의성을 조절할 수 있습니다.
llm = OpenAI(temperature=0.7)

# --- 프롬프트 템플릿 정의 ---
# 'topic' 변수를 사용하는 간단한 프롬프트입니다.
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short, concise, and informative summary about {topic}.",
)

# --- LangChain 체인 구성 ---
# RunnablePassthrough를 사용하여 입력 변수를 다음 구성 요소로 전달합니다.
chain = prompt | llm

# --- 체인 실행 및 결과 출력 ---
topic = "Generative AI"
print(f"Generating summary for: {topic}\n")

try:
    response = chain.invoke({"topic": topic})
    print("--- Generated Summary ---")
    print(response.strip()) # 불필요한 공백 제거
except Exception as e:
    print(f"An error occurred: {e}")