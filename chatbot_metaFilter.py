# chatbot_fixed_v6_reranked.py (제목 필터를 Python에서 후처리 + K 확대/Rerank)
import re
from typing import Dict, Any, List, Optional, Union
from commonUtil.timeCheck import logging_time
from datetime import datetime, timezone
import time  # time 모듈은 datetime 객체를 타임스탬프로 변환하는 데 사용됩니다.
from dateutil.relativedelta import relativedelta
from pathlib import Path
# Reranker 함수 임포트
from reranker import rerank_documents

# ---------- 전역 설정 (토큰 기반 분할 기준) ----------
DB_FOLDER = "./chroma_db3"  # -- 기본은 ./chroma_db2
# V_Kwargs = 10 # -- 기존 설정 대신 K 확장 설정 사용
# V_MODEL_NAME = "llama3.2:3b"
V_MODEL_NAME = "llama3.1:8b"

# ----------------------------------------------------

# ---------- ✅ Reranking 관련 전역 설정 추가 ----------
USE_RERANKER = True  # 재순위화 사용 여부 (True/False)
INITIAL_K = 20  # 초기 ChromaDB 검색 K 값
FINAL_K = 10  # 재순위화 후 최종 반환할 문서 개수
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

# ---------- PROMPT (생략) ----------
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


# ---------- LLM 로드 (생략) ----------
def load_llm(gpu_acceleration: bool = False):
    """
    LLM을 로드합니다. GPU 가속 옵션에 따라 최적의 파라미터를 사용합니다.
    """
    model_name = V_MODEL_NAME

    config_params = {
        "temperature": 0.0,
        "model": model_name
    }

    if gpu_acceleration:
        # 💡 GPU 사용 시 성능 최적화를 위한 파라미터 추가
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


# ---------- QA 체인 생성 (Retriever k 값 수정) ----------
def create_qa_chain(gpu_acceleration: bool = False):
    db = load_vector_db()

    # Note: RetrievalQA 체인을 사용하지 않고, get_answer에서 raw 검색을 할 것이므로
    # 이 함수의 retriever는 기본 K를 가지도록 유지하거나, 단순 반환만 합니다.
    # get_answer에서 raw API를 사용해 K=INITIAL_K를 명시할 것입니다.
    retriever = db.as_retriever(search_kwargs={"k": INITIAL_K})  # K값 전역 변수 반영
    llm = load_llm(gpu_acceleration=gpu_acceleration)

    # PromptTemplate은 동일하게 사용
    map_prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)

    # RetrievalQA 체인 생성 로직은 유지 (사용하지 않더라도 구조 유지)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    return qa, db, retriever


# ---------- 질문에서 필터 조건 추출 (생략 - 기존과 동일) ----------
def extract_chroma_filter(query: str) -> tuple[Union[Dict[str, Any], None], Union[str, None]]:
    # ... (기존 extract_chroma_filter 함수 내용 그대로 사용) ...
    """
    사용자 쿼리에서 ChromaDB 검색을 위한 필터링 인자와 제목 키워드를 추출합니다.
    (기존 extract_chroma_filter 함수 내용 그대로 포함)
    """

    # 💡 dateutil.relativedelta 사용을 위해 설치가 필요합니다: pip install python-dateutil
    # 현재 시점 (UTC 기준)
    now_utc = datetime.now(timezone.utc)

    # 1. 필터 조건들을 저장할 리스트 초기화
    where_conditions: List[Dict[str, Any]] = []
    search_kwargs: Dict[str, Any] = {}
    title_keyword = None

    # --- A. 제목 필터링 로직 (기존과 동일) ---
    title_pattern = re.search(r"(제목|타이틀)[^\s]*\s*(?:(?:에|이)?\s*(?:포함된|있는)?\s*|.*?\s*)\s*([^\s]+)", query)
    if title_pattern:
        keyword = title_pattern.group(2).strip()
        if keyword:
            title_keyword = keyword
            print(f"🔍 제목 키워드 감지: '{keyword}' (Python 후처리 예정)")

    # --- B. 날짜 필터링 로직: 'YYYY-MM-DD' 형식의 메타데이터 'date' 필드에 적용 ---

    # 패턴 1: 'YYYY년 이후' / 'YYYY년도 이후' (기존)
    after_year_pattern = re.search(r"(\d{4})년(?:도)?\s*이후", query)
    if after_year_pattern:
        year = int(after_year_pattern.group(1))
        start_date_utc = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        start_timestamp = start_date_utc.timestamp()
        where_conditions.append({"date": {"$gte": start_timestamp}})

    # 패턴 2: 'YYYY년 MM월 내' / 'YYYY년 MM월까지' (기존)
    within_month_pattern = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(?:이내|내|까지)", query)
    if within_month_pattern:
        year = int(within_month_pattern.group(1))
        month = int(within_month_pattern.group(2))

        if month == 12:
            next_month_start = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        else:
            next_month_start = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        end_timestamp_exclusive = next_month_start.timestamp()
        where_conditions.append({"date": {"$lt": end_timestamp_exclusive}})

    # --- B-1. ✅ 추가된 로직: '지난 X개월' / '지난 X년' ---

    # 패턴 3: '지난 X개월'
    last_months_pattern = re.search(r"지난\s*(\d+)\s*개월", query)
    if last_months_pattern:
        months = int(last_months_pattern.group(1))
        # relativedelta를 사용하여 X개월 전 시점을 정확히 계산
        start_date_limit = now_utc - relativedelta(months=months)
        start_timestamp = start_date_limit.timestamp()
        where_conditions.append({"date": {"$gte": start_timestamp}})
        print(f"✅ 날짜 필터 감지: 지난 {months}개월 (UTC 기준: {start_date_limit.strftime('%Y-%m-%d %H:%M')})")

    # 패턴 4: '지난 X년'
    last_years_pattern = re.search(r"지난\s*(\d+)\s*년", query)
    if last_years_pattern:
        years = int(last_years_pattern.group(1))
        # relativedelta를 사용하여 X년 전 시점을 정확히 계산
        start_date_limit = now_utc - relativedelta(years=years)
        start_timestamp = start_date_limit.timestamp()
        where_conditions.append({"date": {"$gte": start_timestamp}})
        print(f"✅ 날짜 필터 감지: 지난 {years}년 (UTC 기준: {start_date_limit.strftime('%Y-%m-%d %H:%M')})")

    # --- C. 최종 필터 구조 조립 (기존과 동일) ---

    if where_conditions:
        if len(where_conditions) == 1:
            search_kwargs["where"] = where_conditions[0]
        else:
            search_kwargs["where"] = {"$and": where_conditions}

    # 최종 필터와 제목 키워드 반환
    return (search_kwargs if search_kwargs else None, title_keyword)


# ---------- 답변 생성 (invoke 사용) ----------
@logging_time
def get_answer(qa, db, query: str):
    global USE_RERANKER, INITIAL_K, FINAL_K  # 전역 변수 사용 선언

    # 1. 질문에서 메타데이터 필터와 제목 키워드를 추출
    metadata_filter, title_keyword = extract_chroma_filter(query)

    # 2. 검색 인자 설정 (K 값은 INITIAL_K 사용)
    current_k = INITIAL_K

    # 3. 문서 검색: ChromaDB의 raw API를 사용하여 날짜 필터링만 적용
    try:
        if metadata_filter and 'where' in metadata_filter:
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
        print(f"⚠️ ChromaDB 검색 오류 발생. 필터 없이 재시도: {e}")
        docs = db.similarity_search(query=query, k=current_k)

    # 4. 제목 키워드로 Python에서 후처리 필터링 (기존 로직 유지)
    if title_keyword:
        original_count = len(docs)
        keyword_lower = title_keyword.lower().strip()
        filtered_docs = []
        for d in docs:
            title = d.metadata.get('title', '').lower().strip()
            if keyword_lower in title:
                filtered_docs.append(d)
            else:
                print(f"  ❌ 제외된 제목: '{d.metadata.get('title', '')}' (키워드 '{title_keyword}' 없음)")
        docs = filtered_docs
        print(f"🔍 제목 '{title_keyword}' 필터 적용: {original_count}개 → {len(docs)}개 문서")

    # 5. 검색된 문서가 없는 경우 조기 반환
    # 5. 검색된 문서가 없는 경우 조기 반환
    if not docs:
        return "검색 조건에 맞는 문서를 찾을 수 없습니다.", []

    # ✅ 5-1. 파일 저장 로직 수정: 사용자 지정 경로 및 파일명에 질문 내용 반영
    try:
        # 1. 파일 경로 설정: Pathlib 사용 (import는 파일 상단에 있다고 가정)
        import os
        from pathlib import Path

        # 스크립트가 실행되는 디렉토리 기준으로 'data' 폴더 설정
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"

        # 2. 질문에서 파일명으로 사용할 수 없는 문자 제거 및 길이 제한
        # 파일명으로 사용할 수 없는 문자 제거
        safe_query = re.sub(r'[\\/:*?"<>|]', '', query).strip()

        # 파일명 길이 제한 (예: 50자)
        file_name_base = safe_query[:50] if len(safe_query) > 50 else safe_query

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 3. 최종 파일 경로 설정 및 폴더 생성
        # 'data' 폴더가 없으면 생성 (exist_ok=True로 권한 오류 일부 방지)
        data_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{file_name_base}_{timestamp}.txt"
        file_path = data_dir / file_name  # 최종 파일 경로: [스크립트경로]/data/[질문_타임스탬프].txt

        # 4. 파일 내용 생성
        file_content = f"--- RAG Initial Retrieval Log ---\n"
        file_content += f"Query: {query}\n"
        file_content += f"Initial K: {INITIAL_K}\n"
        file_content += f"Documents Retrieved: {len(docs)}\n"
        file_content += "---------------------------------------\n\n"

        # 5. 문서 내용 목록 추가
        for i, d in enumerate(docs, 1):
            title = d.metadata.get("title", "제목 없음")
            date_str = conv_timestamp(d.metadata.get("date", "날짜 없음"))
            source = d.metadata.get("source", "출처 없음")

            file_content += f"[{i}] Title: {title}\n"
            file_content += f"    Date: {date_str}, Source: {source}\n"
            file_content += f"    Content Snippet (First 500 chars):\n"

            # f-string 오류 해결: replace('\n', ' ')를 먼저 변수에 저장
            replaced_content = d.page_content[:500].replace('\n', ' ')
            file_content += f"    {replaced_content}...\n\n"

            file_content += "---\n"

        # 6. 파일 쓰기
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        print(f"\n💾 검색된 {len(docs)}개 문서 목록을 파일로 저장했습니다: {file_path}")

    except Exception as e:
        print(f"\n⚠️ 문서 목록 파일 저장 중 오류 발생: {e}")
    # 6. ✅ 재순위화 로직 적용 (선택 사항)
    if USE_RERANKER:
        print(f"--- 🔄 재순위화 시작 (Total {len(docs)}개 문서) ---")
        reranked_results = rerank_documents(query, docs, top_k=FINAL_K)

        # 재순위화 결과에서 다시 Document 객체 리스트로 변환 (LLM 입력 형식 맞춤)
        docs = [
            Document(
                page_content=d['content'],
                metadata={**d['metadata'], 'rerank_score': d['rerank_score']}
            )
            for d in reranked_results
        ]
        print(f"--- ✅ 최종 {len(docs)}개 문서 반환 (Reranked) ---")
    else:
        # 재순위화 건너뛰기: 초기 검색 결과에서 FINAL_K개만 사용
        docs = docs[:FINAL_K]
        print(f"--- ✅ 최종 {len(docs)}개 문서 반환 (Initial Top-K) ---")

    # 7. 디버깅용: 검색된 context 간단 출력
    context_parts = []
    print("\n🔎 검색된 문서(요약):")
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "제목 없음")
        date_str = conv_timestamp(d.metadata.get("date", "날짜 없음"))
        source = d.metadata.get("source", "출처 없음")
        snippet = d.page_content[:200].replace("\n", " ")
        score_info = f"[Score: {d.metadata.get('rerank_score', 'N/A'):.4f}]" if 'rerank_score' in d.metadata else ""

        print(f"  {score_info} [{i}] {title} / {date_str} / {source}\n       {snippet}...\n")
        context_part = (
                d.page_content +
                f" (제목: {d.metadata.get('title', 'N/A')}, 게시일: {date_str}, Score: {score_info})"
        )
        context_parts.append(context_part)

    # 8. LLM 답변 생성
    context = "\n\n---\n\n".join(context_parts)
    final_prompt = PROMPT_TEMPLATE.format(context=context, question=query)

    llm = load_llm(gpu_acceleration=use_gpu)
    try:
        response = llm.invoke(final_prompt)
        return response, docs
    except Exception as e:
        return f"LLM 호출 중 오류: {e}", docs


# ---------- 메인 (기존과 동일) ----------
def main():
    print("🤖 게시판 기반 챗봇 (Ctrl+C 로 종료)\n")
    global use_gpu, USE_RERANKER, INITIAL_K, FINAL_K  # 전역 변수 사용 선언

    # --- 전역 옵션 설정 ---
    use_gpu_input = input("💡 GPU 가속을 사용하시겠습니까? (y/n): ").strip().lower()
    use_gpu = use_gpu_input == 'y'

    # 재순위화 옵션 입력 받기
    rerank_input = input(f"⭐ 재순위화(Reranking)를 사용하시겠습니까? (y/n, 기본={USE_RERANKER}): ").strip().lower()
    USE_RERANKER = rerank_input == 'y'

    # K 값 조정 옵션
    if USE_RERANKER:
        try:
            initial_k_input = input(f"🔍 초기 검색 K값 (기본={INITIAL_K}): ").strip()
            if initial_k_input:
                INITIAL_K = int(initial_k_input)

            final_k_input = input(f"✅ 최종 반환 K값 (기본={FINAL_K}): ").strip()
            if final_k_input:
                FINAL_K = int(final_k_input)

        except ValueError:
            print("❗ K값 설정이 잘못되었습니다. 기본값으로 재설정합니다.")
            INITIAL_K = 100
            FINAL_K = 10

    else:
        # Rerank를 사용하지 않으면 K는 FINAL_K 값만 가짐
        INITIAL_K = FINAL_K

    print(f"\n** RAG 설정: Reranker={USE_RERANKER}, K_init={INITIAL_K}, K_final={FINAL_K} **")
    # --- 옵션 설정 완료 ---

    qa, db, retriever = create_qa_chain(gpu_acceleration=use_gpu)

    while True:
        try:
            query = input("🗨️ 질문: ").strip()
            if not query:
                continue

            response, docs = get_answer(qa, db, query)

            print("\n💬 답변:\n", response)

            print("\n📚 참고 문서 목록:")
            for i, d in enumerate(docs, 1):
                date_str = conv_timestamp(d.metadata.get('date', None))
                score_info = f" | Rerank Score: {d.metadata.get('rerank_score', 'N/A'):.4f}" if 'rerank_score' in d.metadata else ""
                print(
                    f"  [{i}] 제목: {d.metadata.get('title', '알 수 없음')} / 날짜: {date_str} / 출처: {d.metadata.get('source', '알 수 없음')}{score_info}")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break


def conv_timestamp(timestamp):
    date_str = '알 수 없음'
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        try:
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        except Exception:
            date_str = '변환 오류'
    return date_str


if __name__ == "__main__":
    main()