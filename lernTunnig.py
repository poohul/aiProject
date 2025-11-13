# hil_data_collector.py
import re
import json
import time
from typing import Dict, Any, List, Union
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from pathlib import Path

# --- RAG 설정 상수 ---
DB_FOLDER = "./chroma_db3"
INITIAL_K = 100  # HIL 모드에서는 넓은 범위 검색을 위해 K를 크게 설정합니다.
# ----------------------

# --- 학습 데이터 저장 경로 ---
HIL_DATA_DIR = Path("./hil_training_data")
# ------------------------------

# --- Langchain Imports ---
try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        raise ImportError("HuggingFaceEmbeddings import failed. Check dependencies.")

# --- 전역 변수 (HIL 모드에서는 사용하지 않지만 로직 유지를 위해 선언)
use_gpu = False


# ---------- 벡터 DB 로드 ----------
def load_vector_db(persist_dir: str = DB_FOLDER):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db


# ---------- 타임스탬프 변환 함수 ----------
def conv_timestamp(timestamp):
    date_str = '알 수 없음'
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        try:
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        except Exception:
            date_str = '변환 오류'
    return date_str


# ---------- 필터 추출 함수 (기존 로직 재사용) ----------
# LLM 호출이 없으므로, LLM/QA 체인 관련 함수는 제거하고 핵심 함수만 남깁니다.
def extract_chroma_filter(query: str) -> tuple[Union[Dict[str, Any], None], Union[str, None]]:
    """
    사용자 쿼리에서 ChromaDB 검색을 위한 필터링 인자와 제목 키워드를 추출합니다.
    (기존 extract_chroma_filter 함수 내용 그대로 포함)
    """
    now_utc = datetime.now(timezone.utc)
    where_conditions: List[Dict[str, Any]] = []
    search_kwargs: Dict[str, Any] = {}
    title_keyword = None

    # A. 제목 필터링 로직 (생략)
    title_pattern = re.search(r"(제목|타이틀)[^\s]*\s*(?:(?:에|이)?\s*(?:포함된|있는)?\s*|.*?\s*)\s*([^\s]+)", query)
    if title_pattern:
        keyword = title_pattern.group(2).strip()
        if keyword:
            title_keyword = keyword
            print(f"🔍 제목 키워드 감지: '{keyword}' (Python 후처리 예정)")

    # B. 날짜 필터링 로직 (생략)
    after_year_pattern = re.search(r"(\d{4})년(?:도)?\s*이후", query)
    if after_year_pattern:
        year = int(after_year_pattern.group(1))
        start_date_utc = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        start_timestamp = start_date_utc.timestamp()
        where_conditions.append({"date": {"$gte": start_timestamp}})

    within_month_pattern = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(?:이내|내|까지)", query)
    if within_month_pattern:
        year = int(within_month_pattern.group(1))
        month = int(within_month_pattern.group(2))
        next_month_start = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=timezone.utc) if month < 12 else datetime(
            year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_timestamp_exclusive = next_month_start.timestamp()
        where_conditions.append({"date": {"$lt": end_timestamp_exclusive}})

    last_months_pattern = re.search(r"지난\s*(\d+)\s*개월", query)
    if last_months_pattern:
        months = int(last_months_pattern.group(1))
        start_date_limit = now_utc - relativedelta(months=months)
        start_timestamp = start_date_limit.timestamp()
        where_conditions.append({"date": {"$gte": start_timestamp}})

    last_years_pattern = re.search(r"지난\s*(\d+)\s*년", query)
    if last_years_pattern:
        years = int(last_years_pattern.group(1))
        start_date_limit = now_utc - relativedelta(years=years)
        start_timestamp = start_date_limit.timestamp()
        where_conditions.append({"date": {"$gte": start_timestamp}})

    # C. 최종 필터 구조 조립 (생략)
    if where_conditions:
        search_kwargs["where"] = where_conditions[0] if len(where_conditions) == 1 else {"$and": where_conditions}

    return (search_kwargs if search_kwargs else None, title_keyword)


# ---------- HIL 데이터 수집 메인 로직 ----------
def interactive_labeling_mode(db, query: str):
    """
    사용자에게 K=100 문서를 보여주고 정답을 선택하도록 유도하며 데이터를 저장합니다.
    """

    # 1. 질문에서 메타데이터 필터와 제목 키워드를 추출
    metadata_filter, title_keyword = extract_chroma_filter(query)
    current_k = INITIAL_K

    # 2. 문서 검색 (Initial Retrieval)
    try:
        if metadata_filter and 'where' in metadata_filter:
            where_condition = metadata_filter['where']
            docs = db.similarity_search(query=query, k=current_k, filter=where_condition)
        else:
            docs = db.similarity_search(query=query, k=current_k)

    except Exception as e:
        print(f"⚠️ ChromaDB 검색 오류 발생. 필터 없이 재시도: {e}")
        docs = db.similarity_search(query=query, k=current_k)

    # 3. 제목 키워드로 Python에서 후처리 필터링 (기존 로직 유지)
    if title_keyword:
        original_count = len(docs)
        keyword_lower = title_keyword.lower().strip()
        filtered_docs = []
        for d in docs:
            title = d.metadata.get('title', '').lower().strip()
            if keyword_lower in title:
                filtered_docs.append(d)
        docs = filtered_docs
        print(f"🔍 제목 '{title_keyword}' 필터 적용: {original_count}개 → {len(docs)}개 문서")

    if not docs:
        print("검색된 문서가 없어 라벨링을 진행할 수 없습니다.")
        return

    print(f"\n--- 🧠 Active Learning: 정답 문서 선택 ({len(docs)}개 중) ---")

    # 4. 사용자에게 문서 목록 보여주기
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "제목 없음")
        date_str = conv_timestamp(d.metadata.get("date", "날짜 없음"))
        source = d.metadata.get("source", "출처 없음")
        snippet = d.page_content[:100].replace("\n", " ")
        print(f"[{i}] 제목: {title} | 날짜: {date_str} | 출처: {source} | 내용: {snippet}...")

    # 5. 사용자 입력 받기
    try:
        selection = input("\n💡 가장 정확한 답변 문서의 번호를 입력하세요 (취소: 0): ").strip()
        doc_index = int(selection) - 1

        if doc_index < 0 or doc_index >= len(docs):
            print("라벨링 취소 또는 잘못된 번호 입력.")
            return

        # 6. 긍정 쌍 (Positive) 및 부정 쌍 (Negative) 자동 구축
        positive_doc = docs[doc_index]
        print(f"\n✅ 정답 문서 선택됨: {positive_doc.metadata.get('title')}")

        # 나머지 문서를 부정 쌍으로 간주 (Top-20 오답만 추출하는 것이 일반적)
        # 100개 중 정답을 뺀 나머지 19개 문서 (Top-20에서)를 Hard Negatives로 사용
        negative_docs = [d for i, d in enumerate(docs) if i != doc_index and i < 20]

        # 7. 트립렛 데이터 저장
        HIL_DATA_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파일명에 질문 내용을 포함 (파일명 안전 처리)
        safe_query = re.sub(r'[\\/:*?"<>|]', '', query).strip()[:30]  # 파일명 길이를 30자로 제한
        data_file = HIL_DATA_DIR / f"triplet_{safe_query}_{timestamp}.json"

        # 저장할 데이터 구조
        triplet_data = {
            "query": query,
            "positive": {
                "content": positive_doc.page_content,
                "title": positive_doc.metadata.get("title", "N/A"),
                "source": positive_doc.metadata.get("source", "N/A"),
            },
            "negatives": [
                {"content": d.page_content, "title": d.metadata.get("title", "N/A")}
                for d in negative_docs
            ]
        }

        with open(data_file, 'w', encoding='utf-8') as f:
            # ensure_ascii=False: 한글 깨짐 방지
            json.dump(triplet_data, f, ensure_ascii=False, indent=4)

        print(f"💾 학습 데이터 트립렛이 성공적으로 저장되었습니다: {data_file}")
        print(f"    (부정 문서 {len(negative_docs)}개 포함)")

    except ValueError:
        print("유효하지 않은 입력입니다. 숫자를 입력해 주세요.")
    except Exception as e:
        print(f"⚠️ 데이터 저장 중 오류 발생: {e}")


# ---------- 메인 실행 함수 (HIL 모드 전용) ----------
def main():
    print("🧠 능동 학습(HIL) 데이터 수집 모드 시작\n")

    db = load_vector_db()

    while True:
        try:
            query = input("🗨️ 학습 데이터로 만들 질문을 입력하세요 (Ctrl+C 로 종료): ").strip()
            if not query:
                continue

            interactive_labeling_mode(db, query)

        except KeyboardInterrupt:
            print("\n👋 데이터 수집 모드를 종료합니다.")
            break


if __name__ == "__main__":
    main()