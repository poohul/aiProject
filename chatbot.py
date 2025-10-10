# chatbot.py
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from commonUtil.timeCheck import logging_time

# --------------------------------------------------------
# 💬 프롬프트: 최신 날짜 문서 기준으로 답변하도록 지시
# --------------------------------------------------------
PROMPT_TEMPLATE = """
당신은 회사 게시판의 문서들을 분석하는 AI 비서입니다.

아래는 사용자의 질문과 관련된 문서들의 내용 요약입니다:
{context}

[지시사항]
1. 여러 문서가 관련된 경우, 반드시 metadata의 'date' 값이 가장 최신인 문서를 기준으로 답변하세요.
2. 날짜가 동일한 경우 metadata의 '게시일시'를 비교하여 최신 게시글을 선택하세요.
3. 반드시 실제 문서에 존재하는 내용만 사용하세요. 추측하지 마세요.
4. 가능하다면 다음 정보를 함께 포함하세요:
   - 게시일자
   - 게시자
   - 제목
5.사용자 요청이 '제목에 포함된 단어' 를 찾는 경우 반드시 문서의 '제목' 필드에 포함되어 있어야함.  
6.다음 문서들 중에서 사용자 질문에 해당하는 날짜(date)가 메타데이터로 제공되어 있다면,
반드시 해당 날짜(date)가 정확히 일치하는 문서만 골라서 답변하세요.
그 외의 날짜는 무시하세요.
[사용자 질문]
{question}

최종 요약된 답변:
"""

@logging_time
def get_answer(qa, query):
    """질문에 대한 답변 생성"""
    return qa.invoke({"query": query})["result"]

def load_vector_db(persist_dir="./chroma_db2"):
    """벡터 DB 로드"""
    embeddings = HuggingFaceEmbeddings(
        model_name="upskyy/gte-base-korean",
        model_kwargs={'trust_remote_code': True}
    )
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db

def main():
    print("🤖 게시판 기반 챗봇 (Ctrl+C 로 종료)\n")

    # -------------------------
    # ① 벡터 DB 로드
    # -------------------------
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 30})  # k 조절 가능

    # -------------------------
    # ② LLM + 프롬프트 정의
    # -------------------------
    llm = Ollama(model="llama3.1:8b")  # PC 버전 (빠름, 정확함)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # -------------------------
    # ③ 대화 루프
    # -------------------------
    while True:
        try:
            query = input("\n🗨️ 질문: ").strip()
            if not query:
                continue

            print("\n⏳ 검색 중...")
            answer = get_answer(qa, query)
            print(f"\n💡 답변:\n{answer}")

            docs = retriever.get_relevant_documents(query)
            for i, doc in enumerate(docs, 1):
                # print(f"  [{i}] {doc.metadata.get('source', '알 수 없음'),}")
                print(f"  [{i}] 출처: {doc.metadata.get('source', '알 수 없음')} / 날짜: {doc.metadata.get('date', '날짜 없음')}")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break


if __name__ == "__main__":
    main()
