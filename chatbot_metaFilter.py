# chatbot_fixed_v6.py (제목 필터를 Python에서 후처리)
import re
from typing import Dict, Any, List, Optional, Union
from commonUtil.timeCheck import logging_time
from datetime import datetime, timezone
import time  # time 모듈은 datetime 객체를 타임스탬프로 변환하는 데 사용됩니다.

# ---------- 전역 설정 (토큰 기반 분할 기준) ----------
DB_FOLDER = "./chroma_db3"  # -- 기본은 ./chroma_db2
V_Kwargs = 10
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
def load_llm(gpu_acceleration: bool = False):
    """
    LLM을 로드합니다. GPU 가속 옵션에 따라 최적의 파라미터를 사용합니다.
    """
    model_name = "llama3.1:8b"

    config_params = {
        "temperature": 0.0,
        "model": model_name
    }

    if gpu_acceleration:
        # 💡 GPU 사용 시 성능 최적화를 위한 파라미터 추가
        # num_gpu: 사용할 GPU 개수 (1개 사용을 명시)
        # mirostat: 샘플링 전략을 켜서 성능과 품질을 개선 (선택 사항)
        config_params.update({
            "num_gpu": 1,
            "mirostat": 2  # Mirostat 샘플링 v2 적용
        })
        print(f"🚀 GPU 가속 옵션 활성화: {model_name}")
    else:
        # CPU 사용 시: 메모리 사용량 및 추론 속도를 고려한 기본 설정 유지
        print(f"💻 CPU 모드 활성화: {model_name}")

    try:
        return OllamaLLM(**config_params)
    except TypeError:
        # 래퍼 버전 차이로 인한 키워드 에러 방지 (fallback)
        return OllamaLLM(model=model_name, temperature=0.0)


# ---------- QA 체인 생성 ----------
def create_qa_chain(gpu_acceleration: bool = False):
    db = load_vector_db()

    # retriever는 기본 k=10 설정만 가진 상태로 생성
    retriever = db.as_retriever(search_kwargs={"k": V_Kwargs})
    # llm = load_llm()
    llm = load_llm(gpu_acceleration=gpu_acceleration)

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


# ---------- 질문에서 필터 조건 추출 (수정: 제목 키워드만 별도 반환) ----------

def extract_chroma_filter(query: str) -> tuple[Union[Dict[str, Any], None], Union[str, None]]:
    """
    사용자 쿼리에서 ChromaDB 검색을 위한 필터링 인자와 제목 키워드를 추출합니다.

    Returns:
        tuple: (search_kwargs, title_keyword)
            - search_kwargs: ChromaDB에서 사용할 필터 (날짜 필터만 포함)
            - title_keyword: 제목에서 검색할 키워드 (Python 후처리용)
    """

    # 1. 필터 조건들을 저장할 리스트 초기화
    where_conditions: List[Dict[str, Any]] = []
    search_kwargs: Dict[str, Any] = {}
    title_keyword = None

    # --- A. 제목 필터링 로직: '제목 xx 포함' 패턴 (수정: 키워드만 추출) ---
    title_pattern = re.search(r"(제목|타이틀)[^\s]*\s*(?:(?:에|이)?\s*(?:포함된|있는)?\s*|.*?\s*)\s*([^\s]+)", query)
    if title_pattern:
        keyword = title_pattern.group(2).strip()
        if keyword:
            # ✅ 수정: ChromaDB 필터에 추가하지 않고, 반환용 변수에만 저장
            title_keyword = keyword
            print(f"🔍 제목 키워드 감지: '{keyword}' (Python 후처리 예정)")

    # --- B. 날짜 필터링 로직: 'YYYY-MM-DD' 형식의 메타데이터 'date' 필드에 적용 ---

    # 패턴 1: 'YYYY년 이후' / 'YYYY년도 이후'
    after_year_pattern = re.search(r"(\d{4})년(?:도)?\s*이후", query)
    if after_year_pattern:
        year = int(after_year_pattern.group(1))
        start_date_utc = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        start_timestamp = start_date_utc.timestamp()
        where_conditions.append({"date": {"$gte": start_timestamp}})

    # 패턴 2: 'YYYY년 MM월 내' / 'YYYY년 MM월까지' (해당 월의 마지막 날짜 $lt)
    within_month_pattern = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(?:이내|내|까지)", query)
    if within_month_pattern:
        year = int(within_month_pattern.group(1))
        month = int(within_month_pattern.group(2))

        # 1. 다음 달의 시작 날짜를 구합니다.
        if month == 12:
            next_month_start = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        else:
            next_month_start = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # 2. 해당 날짜를 유닉스 타임스탬프(초)로 변환합니다.
        end_timestamp_exclusive = next_month_start.timestamp()

        # 3. 숫자 값으로 $lt (작다, 미만) 조건을 적용합니다.
        where_conditions.append({"date": {"$lt": end_timestamp_exclusive}})

    # --- C. 최종 필터 구조 조립 (날짜 필터만) ---

    if where_conditions:
        if len(where_conditions) == 1:
            # 조건이 하나일 경우: $and 없이 단일 필터만 사용
            search_kwargs["where"] = where_conditions[0]
        else:
            # 조건이 두 개 이상일 경우: $and로 묶어서 사용
            search_kwargs["where"] = {"$and": where_conditions}

    # 최종 필터와 제목 키워드 반환
    return (search_kwargs if search_kwargs else None, title_keyword)


# ---------- 답변 생성 (invoke 사용) ----------
@logging_time
def get_answer(qa, db, query: str):
    # 1. 질문에서 메타데이터 필터와 제목 키워드를 추출
    metadata_filter, title_keyword = extract_chroma_filter(query)

    # 2. 검색 인자 설정 (기본 k=10)
    current_k = V_Kwargs

    # 3. 문서 검색: ChromaDB의 raw API를 사용하여 날짜 필터링만 적용
    try:
        if metadata_filter and 'where' in metadata_filter:
            # 날짜 필터가 있는 경우
            where_condition = metadata_filter['where']
            print(f"✅ ChromaDB 날짜 필터링: K={current_k}, Where={where_condition}")
            docs = db.similarity_search(
                query=query,
                k=current_k,
                filter=where_condition
            )
        else:
            # 필터가 없는 경우: 유사도 검색만 수행
            docs = db.similarity_search(
                query=query,
                k=current_k
            )

    except Exception as e:
        # ChromaDB 검색 오류 발생 시
        print(f"⚠️ ChromaDB 검색 오류 발생. 필터 없이 재시도: {e}")
        docs = db.similarity_search(query=query, k=current_k)

    # 4. ✅ 제목 키워드로 Python에서 후처리 필터링 (대소문자 구분 없이)
    if title_keyword:
        original_count = len(docs)
        # 대소문자 구분 없이 검색하고, 공백 제거 후 비교
        keyword_lower = title_keyword.lower().strip()
        filtered_docs = []
        for d in docs:
            title = d.metadata.get('title', '').lower().strip()
            if keyword_lower in title:
                filtered_docs.append(d)
            else:
                # 디버깅: 필터링된 제목 출력
                print(f"  ❌ 제외된 제목: '{d.metadata.get('title', '')}' (키워드 '{title_keyword}' 없음)")
        docs = filtered_docs
        print(f"🔍 제목 '{title_keyword}' 필터 적용: {original_count}개 → {len(docs)}개 문서")

    # 5. 검색된 문서가 없는 경우 조기 반환
    if not docs:
        return "검색 조건에 맞는 문서를 찾을 수 없습니다.", []

    context_parts = []
    # 6. 디버깅용: 검색된 context 간단 출력
    print("\n🔎 검색된 문서(요약):")
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "제목 없음")
        date_str = conv_timestamp(d.metadata.get("date", "날짜 없음"))
        source = d.metadata.get("source", "출처 없음")
        snippet = d.page_content[:200].replace("\n", " ")
        print(f"  [{i}] {title} / {date_str} / {source}\n       {snippet}...\n")
        context_part = (
                d.page_content +
                f" (제목: {d.metadata.get('title', 'N/A')}, 게시일: {date_str})"
        )
        context_parts.append(context_part)

    # 7. LLM 답변 생성
    context = "\n\n---\n\n".join(context_parts)
    final_prompt = PROMPT_TEMPLATE.format(context=context, question=query)

    # LLM에 직접 질문
    # llm = load_llm()
    llm = load_llm(gpu_acceleration=use_gpu)
    try:
        response = llm.invoke(final_prompt)
        return response, docs
    except Exception as e:
        return f"LLM 호출 중 오류: {e}", docs


# ---------- 메인 (기존과 동일) ----------
def main():
    print("🤖 게시판 기반 챗봇 (Ctrl+C 로 종료)\n")
    global use_gpu
    # 사용자로부터 GPU 가속 여부를 입력받는 로직 추가
    use_gpu_input = input("💡 GPU 가속을 사용하시겠습니까? (y/n): ").strip().lower()
    use_gpu = use_gpu_input == 'y'
    qa, db, retriever = create_qa_chain(gpu_acceleration=use_gpu)
    # qa, db, retriever = create_qa_chain()

    while True:
        try:
            query = input("🗨️ 질문: ").strip()
            if not query:
                continue

            response, docs = get_answer(qa, db, query)

            # response는 LLM의 최종 답변, docs는 검색된 문서 목록
            print("\n💬 답변:\n", response)

            print("\n📚 참고 문서 목록:")
            for i, d in enumerate(docs, 1):
                date_str = conv_timestamp(d.metadata.get('date', None))
                print(
                    f"  [{i}] 제목: {d.metadata.get('title', '알 수 없음')} / 날짜: {date_str} / 출처: {d.metadata.get('source', '알 수 없음')}")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break


def conv_timestamp(timestamp):
    date_str = '알 수 없음'
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        try:
            # 타임스탬프를 datetime 객체로 변환하고 원하는 형식으로 포맷팅
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        except Exception:
            date_str = '변환 오류'
    return date_str


if __name__ == "__main__":
    main()