# chatbot_fixed_v3.py
import re
from typing import List
from commonUtil.timeCheck import logging_time
# ---------- 전역 설정 (토큰 기반 분할 기준) ----------
DB_FOLDER = "./chroma_db3" #-- 기본은 ./chroma_db2
# ----------------------------------------------------

# --- Imports: try newest, fallback to older packages if necessary ---
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma  # older langchain versions

try:
    from langchain_ollama import OllamaLLM
except Exception:
    from langchain_community.llms import Ollama as OllamaLLM  # fallback (older)

# Embeddings import fallback handling
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        raise ImportError("HuggingFaceEmbeddings import failed. Install langchain_huggingface or langchain-community.")

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

# ---------- PROMPT ----------
PROMPT_TEMPLATE = """당신은 회사 게시판의 문서들을 분석하는 AI 비서입니다.

아래는 검색된 문서들의 실제 내용입니다:
{context}

[지시사항]
1. 반드시 문서 내용에만 근거하여 답변하세요.
2. 문서에 없는 내용은 추측하지 말고, "관련 문서에서 해당 내용을 찾을 수 없습니다."라고 답하세요.
3. 여러 문서가 검색된 경우, metadata의 'date' 값이 가장 최신인 문서를 기준으로 답변하세요.
4. 날짜가 동일한 경우 metadata의 '게시일시'를 비교해 최신 게시글을 선택하세요.
5. 가능한 경우 다음 정보를 포함하세요: 제목, 게시자, 게시일자.

[사용자 질문]
{question}
"""

# ---------- 벡터 DB 로드 ----------
def load_vector_db(persist_dir: str = DB_FOLDER):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db

# ---------- LLM 로드 ----------
def load_llm():
    # temperature 낮게 해서 추측 줄임
    try:
        return OllamaLLM(model="llama3.1:8b", temperature=0.0)
    except TypeError:
        # 일부 래퍼는 키워드명이 다를 수 있어 positional fallback
        return OllamaLLM("llama3.1:8b")

# ---------- QA 체인 생성 ----------
def create_qa_chain():
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = load_llm()

    # map_prompt / combine_prompt 을 PromptTemplate 으로 명시
    map_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )
    combine_prompt = PromptTemplate(
        input_variables=["summaries", "question"],
        template=(
            "아래는 여러 문서에서 추출한 요약입니다:\n\n{summaries}\n\n"
            "위 요약들을 바탕으로 사용자의 질문에 대해 문서 근거만 사용하여 최종 답변을 작성하세요. "
            "문서에 없으면 '관련 문서에서 해당 내용을 찾을 수 없습니다.'라고 답하세요.\n\n"
            "사용자 질문: {question}\n"
        )
    )

    # RetrievalQA.from_chain_type 에서 map_prompt / combine_prompt 전달
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",  # map_reduce 대신 stuff 사용
        )
    except TypeError:
        # 일부 LangChain 버전에서 key 이름이 다를 수 있으므로 기본 stuff 로 fallback
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

    return qa, retriever

# ---------- 제목 기반 필터 ----------
def filter_by_title(query: str, docs: List[Document]) -> List[Document]:
    title_pattern = re.search(r"제목.*?(?:이|에)?\s*([^\s]+)", query)
    if not title_pattern:
        return docs
    keyword = title_pattern.group(1).strip()
    filtered = [d for d in docs if keyword in (d.metadata.get("title") or "")]
    return filtered if filtered else docs

# ---------- 답변 생성 ----------
@logging_time
def get_answer(qa, query: str):
    # RetrievalQA 내부에서 retriever를 사용하므로 그냥 invoke
    try:
        result = qa.invoke({"query": query})
        # result may be a dict or string depending on version
        if isinstance(result, dict):
            return result.get("result") or result.get("answer") or str(result)
        return str(result)
    except Exception:
        # some versions expect run or call
        try:
            return qa.run(query)
        except Exception as e:
            return f"LLM 체인 호출 중 오류: {e}"

# ---------- 메인 ----------
def main():
    print("🤖 게시판 기반 챗봇 (Ctrl+C 로 종료)\n")
    qa, retriever = create_qa_chain()

    while True:
        try:
            query = input("🗨️ 질문: ").strip()
            if not query:
                continue

            docs = retriever.get_relevant_documents(query)
            docs = filter_by_title(query, docs)

            # 디버깅용: 검색된 context 간단 출력
            print("\n🔎 검색된 문서(요약):")
            for i, d in enumerate(docs, 1):
                title = d.metadata.get("title", "제목 없음")
                date = d.metadata.get("date", "날짜 없음")
                source = d.metadata.get("source", "출처 없음")
                snippet = d.page_content[:200].replace("\n", " ")
                print(f"  [{i}] {title} / {date} / {source}\n       {snippet}...\n")

            response = get_answer(qa, query)
            print("\n💬 답변:\n", response)

            print("\n📚 참고 문서 목록:")
            for i, d in enumerate(docs, 1):
                print(f"  [{i}] 출처: {d.metadata.get('source','알 수 없음')} / 날짜: {d.metadata.get('date','알 수 없음')}")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break

if __name__ == "__main__":
    main()
