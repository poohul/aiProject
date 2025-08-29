import os
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Hugging Face API 토큰 로드 확인 (Windows 환경 변수에서 자동 로드)
# 디버깅을 위해 토큰이 제대로 로드되었는지 확인하는 코드입니다.
loaded_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if loaded_token:
    print(f"DEBUG: Hugging Face API Token successfully loaded from environment.")
    print(f"DEBUG: Token starts with: {loaded_token[:5]}... ends with: {loaded_token[-5:]}")
else:
    print("DEBUG: Hugging Face API Token NOT found in environment variables. Please check your system environment variables and restart terminal/IDE.")
    # 토큰이 없으면 더 이상 진행하지 않도록 예외를 발생시킬 수도 있습니다.
    # raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set.")

# 2. LLM 초기화 (openai-community/gpt2 사용)
# 이 부분에서 Hugging Face Inference Endpoint에 연결을 시도합니다.
try:
    print("\nAttempting to initialize HuggingFaceEndpoint with openai-community/gpt2...")
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-70B-Instruct",  # GPT2 모델 사용
        temperature=0.7,                  # 창의성 조절 (0.0은 보수적, 1.0은 창의적)
        max_new_tokens=100                # 생성할 최대 토큰 수
    )
    print("HuggingFaceEndpoint initialized successfully!")

    # 3. 프롬프트 템플릿 정의
    prompt_template = PromptTemplate.from_template(
        "Please write a short story about {topic}. Be concise."
    )

    # 4. LLM 체인 구성 (Prompt + LLM + Output Parser)
    # LangChain Expression Language (LCEL)을 사용하여 체인 구성
    chain = prompt_template | llm | StrOutputParser()

    # 5. LLM 호출 및 결과 출력
    print("\nGenerating response...")
    topic = "a friendly robot"
    response = chain.invoke({"topic": topic})

    print("\n--- Generated Story ---")
    print(response)
    print("-----------------------")

except Exception as e:
    import traceback
    traceback.print_exc() # 이 라인을 추가하세요!

    print(f"\nAn error occurred during LLM initialization or invocation: {e}")
    print("\nPossible reasons for this error (especially StopIteration):")
    print("  - Your HUGGINGFACEHUB_API_TOKEN is not correctly set in environment variables and applied.")
    print("  - The Hugging Face Inference API for 'openai-community/gpt2' is currently unavailable or experiencing issues.")
    print("  - Network connectivity problems (e.g., firewall blocking access to api-inference.huggingface.co).")
    print("  - If using a Gated Model (GPT2 is not, but others like Flan-T5 might be), you must accept its terms on Hugging Face Hub.")