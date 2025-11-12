# chatbot_fixed_v5.py (제목 필터를 title 메타데이터 필드로 직접 타겟팅)
import re
from typing import Dict, Any, List, Optional, Union
from commonUtil.timeCheck import logging_time
from datetime import datetime
import copy

# ---------- 전역 설정 (토큰 기반 분할 기준) ----------
DB_FOLDER = "./chroma_db3"  # -- 기본은 ./chroma_db2
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

    # retriever는 기본 k=10 설정만 가진 상태로 생성
    retriever = db.as_retriever(search_kwargs={"k": 10})
    llm = load_llm()

    # (map_prompt, combine_prompt 생략 - 기존과 동일)
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

    # RetrievalQA.from_chain_type
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
        )
    except TypeError:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

    # Note: retriever를 main 함수로 보낼 때 db 객체를 참조하게 되므로,
    # 여기서는 db 객체와 qa, retriever를 모두 반환합니다.
    return qa, db, retriever


# ---------- 질문에서 필터 조건 추출 (핵심 로직 수정) ----------

def extract_chroma_filter(query: str) -> Union[Dict[str, Any]]:
    """
    사용자 쿼리에서 ChromaDB 검색을 위한 필터링 인자(kwargs)를 추출합니다.
    제목 검색 요청 시 'title' 메타데이터 필드를 직접 타겟팅하도록 수정되었습니다.
    """

    # 1. 필터 조건들을 저장할 리스트 초기화
    where_conditions: List[Dict[str, Any]] = []
    search_kwargs: Dict[str, Any] = {}

    # --- A. 제목 필터링 로직: '제목 xx 포함' 패턴 (수정됨) ---
    title_pattern = re.search(r"(제목|타이틀)[^\s]*\s*(?:(?:에|이)?\s*(?:포함된|있는)?\s*|.*?\s*)\s*([^\s]+)", query)
    if title_pattern:
        keyword = title_pattern.group(2).strip()
        if keyword:
            # 💥 수정: where_document 대신, title 메타데이터 필드를 $contains로 직접 필터링 시도
            # 이렇게 하면 제목 필드에만 해당 키워드가 포함된 문서를 찾도록 요청합니다.
            where_conditions.append({"title": {"$contains": keyword}})

            # 참고: ChromaDB는 string metadata의 부분 문자열 $contains를 완벽하게 지원하지 않을 수 있지만,
            # 사용자 요구사항을 충족하기 위한 최선의 구현입니다.

    # --- B. 날짜 필터링 로직: 'YYYY-MM-DD' 형식의 메타데이터 'date' 필드에 적용 ---

    # 패턴 1: 'YYYY년 이후' / 'YYYY년도 이후'
    after_year_pattern = re.search(r"(\d{4})년(?:도)?\s*이후", query)
    if after_year_pattern:
        year = after_year_pattern.group(1)
        # 해당 년도의 시작일(YYYY-01-01) $gte (크거나 같다) 조건
        where_conditions.append({"date": {"$gte": f"{year}-01-01"}})

    # 패턴 2: 'YYYY년 MM월 내' / 'YYYY년 MM월까지' (해당 월의 마지막 날짜 $lt)
    within_month_pattern = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(?:이내|내|까지)", query)
    if within_month_pattern:
        year = int(within_month_pattern.group(1))
        month = int(within_month_pattern.group(2))

        # 다음 달의 시작 날짜를 구해서 $lt (작다) 조건을 사용 (해당 월 포함)
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"

        # 다음 달 1일 미만 $lt 조건
        where_conditions.append({"date": {"$lt": end_date}})

    # --- C. 최종 필터 구조 조립 ---

    if where_conditions:
        if len(where_conditions) == 1:
            # 조건이 하나일 경우: $and 없이 단일 필터만 사용 (ChromaDB 오류 방지)
            search_kwargs["where"] = where_conditions[0]
        else:
            # 조건이 두 개 이상일 경우: $and로 묶어서 사용
            search_kwargs["where"] = {"$and": where_conditions}

    # 최종 필터 반환 (where만 포함될 수 있음)
    return search_kwargs if search_kwargs else None


# ---------- 답변 생성 (invoke 사용) ----------
@logging_time
def get_answer(qa, db, query: str):
    # 1. 질문에서 메타데이터 필터를 추출
    metadata_filter = extract_chroma_filter(query)

    # 2. 검색 인자 설정 (기본 k=10)
    current_search_kwargs = {"k": 10}

    # invoke 메서드 사용을 위해 필터를 config 딕셔너리로 래핑
    config_for_invoke = {"configurable": metadata_filter} if metadata_filter else {}

    if metadata_filter:
        # 이 로그 메시지는 실제로 실행될 때만 나타납니다.
        print(f"✅ 메타데이터 필터 적용 (invoke config): {metadata_filter}")

    # 3. 새로운 search_kwargs를 가진 동적 retriever 생성 (k=10만 포함)
    dynamic_retriever = db.as_retriever(search_kwargs=current_search_kwargs)

    # 4. 문서 검색: invoke(query, config={...}) 패턴을 사용
    docs = dynamic_retriever.invoke(query, config=config_for_invoke)

    # 5. 디버깅용: 검색된 context 간단 출력
    print("\n🔎 검색된 문서(요약):")
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "제목 없음")
        date = d.metadata.get("date", "날짜 없음")
        source = d.metadata.get("source", "출처 없음")
        snippet = d.page_content[:200].replace("\n", " ")
        print(f"  [{i}] {title} / {date} / {source}\n       {snippet}...\n")

    # 6. LLM 답변 생성
    # 6-1. 검색된 문서들을 하나의 컨텍스트 문자열로 합치기
    context = "\n\n---\n\n".join(
        [d.page_content + f" (제목: {d.metadata.get('title', 'N/A')}, 게시일: {d.metadata.get('date', 'N/A')})" for d in
         docs])

    # 6-2. 프롬프트 템플릿에 컨텍스트와 질문을 채우기
    final_prompt = PROMPT_TEMPLATE.format(context=context, question=query)

    # 6-3. LLM에 직접 질문 (QA 체인 대신 LLM만 호출)
    llm = load_llm()
    try:
        response = llm.invoke(final_prompt)
        return response, docs
    except Exception as e:
        return f"LLM 호출 중 오류: {e}", docs


# ---------- 메인 (기존과 동일) ----------
def main():
    print("🤖 게시판 기반 챗봇 (Ctrl+C 로 종료)\n")
    # qa, db, retriever를 모두 받도록 수정
    qa, db, retriever = create_qa_chain()

    while True:
        try:
            query = input("🗨️ 질문: ").strip()
            if not query:
                continue

            # get_answer 함수가 동적으로 retriever를 사용하도록 수정했으므로,
            # 여기서는 docs 검색 및 filter_by_title 호출 로직을 제거하고
            # get_answer에 db 객체를 전달합니다.
            response, docs = get_answer(qa, db, query)

            # response는 LLM의 최종 답변, docs는 검색된 문서 목록
            print("\n💬 답변:\n", response)

            print("\n📚 참고 문서 목록:")
            for i, d in enumerate(docs, 1):
                print(
                    f"  [{i}] 제목: {d.metadata.get('title', '알 수 없음')} / 날짜: {d.metadata.get('date', '알 수 없음')} / 출처: {d.metadata.get('source', '알 수 없음')}")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break


if __name__ == "__main__":
    main()