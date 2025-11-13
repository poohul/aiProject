# hil_data_collector_multi_pair.py (Positive/Negative 복수 선택 모드)
import re
import json
import time
from typing import Dict, Any, List, Union, Set  # Set 타입 힌트 추가
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from pathlib import Path

# --- RAG 설정 상수 ---
DB_FOLDER = "./chroma_db3"
INITIAL_K = 200
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

# --- 전역 변수
use_gpu = False


# ---------- 벡터 DB 로드 ----------
def load_vector_db(persist_dir: str = DB_FOLDER):
    """벡터 DB(Chroma)와 임베딩 모델을 로드합니다."""
    # 실제 임베딩 모델 이름을 사용해야 합니다. (이전 스크립트에서 사용된 이름)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db


# ---------- 타임스탬프 변환 함수 ----------
def conv_timestamp(timestamp: Union[int, float, str]) -> str:
    """타임스탬프를 YYYY-MM-DD 형식의 문자열로 변환합니다."""
    date_str = '알 수 없음'
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        try:
            # UTC 타임스탬프를 로컬 시간으로 변환
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        except Exception:
            date_str = '변환 오류'
    return date_str


# ---------- 필터 추출 함수 ----------
def extract_chroma_filter(query: str) -> tuple[Union[Dict[str, Any], None], Union[str, None]]:
    """사용자 쿼리에서 ChromaDB 검색을 위한 필터링 인자를 추출합니다."""
    now_utc = datetime.now(timezone.utc)
    where_conditions: List[Dict[str, Any]] = []
    search_kwargs: Dict[str, Any] = {}
    title_keyword = None

    # A. 제목 필터링 로직
    title_pattern = re.search(r"(제목|타이틀)[^\s]*\s*(?:(?:에|이)?\s*(?:포함된|있는)?\s*|.*?\s*)\s*([^\s]+)", query)
    if title_pattern:
        keyword = title_pattern.group(2).strip()
        if keyword:
            title_keyword = keyword

    # B. 날짜 필터링 로직 (GTE, LT 기반으로 구현)
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

    # C. 최종 필터 구조 조립
    if where_conditions:
        search_kwargs["where"] = where_conditions[0] if len(where_conditions) == 1 else {"$and": where_conditions}

    return (search_kwargs if search_kwargs else None, title_keyword)


# ---------- HIL 데이터 수집 메인 로직 (복수 쌍 모드) ----------
def interactive_labeling_mode(db, query: str):
    """
    사용자에게 Positive 문서와 Negative 문서를 복수로 선택하도록 유도하여 트립렛을 저장합니다.
    """

    # 1-3. 문서 검색 및 필터링
    metadata_filter, title_keyword = extract_chroma_filter(query)
    current_k = INITIAL_K

    try:
        if metadata_filter and 'where' in metadata_filter:
            where_condition = metadata_filter['where']
            docs = db.similarity_search(query=query, k=current_k, filter=where_condition)
        else:
            docs = db.similarity_search(query=query, k=current_k)

    except Exception as e:
        print(f"⚠️ ChromaDB 검색 오류 발생. 필터 없이 재시도: {e}")
        docs = db.similarity_search(query=query, k=current_k)

    if title_keyword:
        original_count = len(docs)
        keyword_lower = title_keyword.lower().strip()
        filtered_docs = [d for d in docs if keyword_lower in d.metadata.get('title', '').lower().strip()]
        docs = filtered_docs
        print(f"🔍 제목 '{title_keyword}' 필터 적용: {original_count}개 → {len(docs)}개 문서")

    if not docs:
        print("검색된 문서가 없어 라벨링을 진행할 수 없습니다.")
        return

    print(f"\n--- 🧠 Active Learning: 정답/오답 문서 복수 선택 ({len(docs)}개 중) ---")

    # 4. 사용자에게 문서 목록 보여주기
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "제목 없음")
        date_str = conv_timestamp(d.metadata.get("date", "날짜 없음"))
        source = d.metadata.get("source", "출처 없음")
        snippet = d.page_content[:100].replace("\n", " ")
        print(f"[{i}] 제목: {title} | 날짜: {date_str} | 출처: {source} | 내용: {snippet}...")

    # 5. 사용자 입력 받기 - Positive/Negative 복수 선택
    try:
        def get_indices(prompt: str) -> List[int]:
            """콤마 구분자 입력을 받아 유효한 문서 인덱스 리스트로 반환합니다."""
            selection = input(prompt).strip()
            if not selection or selection == '0':
                return []

            indices: Set[int] = set()
            for s in selection.split(','):
                try:
                    idx = int(s.strip()) - 1
                    if 0 <= idx < len(docs):
                        indices.add(idx)
                    else:
                        print(f"경고: 유효하지 않은 번호 '{s.strip()}'는 무시되었습니다.")
                except ValueError:
                    pass
            return sorted(list(indices))

        # 5a. Positive 문서 선택
        pos_indices = get_indices("\n💡 1. **정답(Positive)** 문서 번호를 콤마(,)로 구분하여 입력하세요 (예: 1,2,5 / 취소: 0): ")
        if not pos_indices:
            print("Positive 문서 선택 취소.")
            return

        # 5b. Negative 문서 선택
        neg_indices = get_indices("💡 2. **오답(Negative)** 문서 번호를 콤마(,)로 구분하여 입력하세요 (예: 8,10,12 / 취소: 0): ")

        # 6. 유효성 검사 및 데이터 추출

        # Positive/Negative 중복 선택 확인
        pos_set = set(pos_indices)
        neg_set = set(neg_indices)

        if pos_set.intersection(neg_set):
            print("🚨 오류: Positive와 Negative로 중복 선택된 문서가 있습니다. 라벨링 취소.")
            return

        if not neg_indices:
            print("경고: Negative 문서가 선택되지 않아, Positive-Only 쌍으로 저장됩니다.")

        positive_docs = [docs[idx] for idx in pos_indices]
        negative_docs = [docs[idx] for idx in neg_indices]

        print(f"\n✅ Positive 문서 {len(positive_docs)}개 선택됨.")
        print(f"❌ Negative 문서 {len(negative_docs)}개 선택됨.")

        # 7. 트립렛 데이터 저장
        HIL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파일명 안전 처리
        safe_query = re.sub(r'[\\/:*?"<>|]', '', query).strip()[:30]
        data_file = HIL_DATA_DIR / f"triplet_multi_{safe_query}_{timestamp}.json"

        # 저장할 데이터 구조
        triplet_data: Dict[str, Any] = {
            "query": query,
            "positive": [
                {
                    "content": d.page_content,
                    "title": d.metadata.get("title", "N/A"),
                    "source": d.metadata.get("source", "N/A"),
                } for d in positive_docs
            ],
            "negatives": [
                {
                    "content": d.page_content,
                    "title": d.metadata.get("title", "N/A")
                } for d in negative_docs
            ]
        }

        # Positive 필드를 단일 딕셔너리가 아닌 리스트로 저장하도록 구조 변경
        if len(positive_docs) == 1:
            # 이전 포맷과의 호환성을 위해 1개일 때는 리스트 대신 단일 딕셔너리로 저장
            triplet_data['positive'] = triplet_data['positive'][0]

        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(triplet_data, f, ensure_ascii=False, indent=4)

        print(f"💾 학습 데이터 트립렛(N:M 쌍)이 성공적으로 저장되었습니다: {data_file}")

    except Exception as e:
        print(f"⚠️ 데이터 처리/저장 중 오류 발생: {e}")


# ---------- 메인 실행 함수 (HIL 모드 전용) ----------
def main():
    print("🧠 능동 학습(HIL) 데이터 수집 모드 시작 (Positive/Negative 복수 선택)")

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