import os
import json
from pathlib import Path
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from commonUtil.timeCheck import logging_time
from transformers import AutoTokenizer
# 날짜 처리를 위해 datetime 모듈 추가
from datetime import datetime

# ---------- 전역 설정 (토큰 기반 분할 기준) ----------
TOKEN_CHUNK_SIZE = 500
TOKEN_CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
DB_FOLDER = "./chroma_db3"
# 게시일시 포맷 정의 (현재 제공된 포맷: 2025-09-22 09:24:27)
DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# ----------------------------------------------------

# 토크나이저를 전역적으로 로드
try:
    TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Error loading tokenizer: {e}. Ensure 'transformers' is installed.")
    TOKENIZER = None


def extract_text_from_file(file_path: str) -> dict:
    """파일을 읽어 JSON이면 변환, 아니면 그대로 텍스트 리턴"""
    # (기존 코드는 변경 없음)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return {"text": "", "date": "", "title": "", "body": ""}

    if content.startswith("{") and content.endswith("}"):
        try:
            data = json.loads(content)
            title = data.get("제목", "")
            body = data.get("본문", "")
            author = data.get("게시자", "")
            date = data.get("게시일시", "") # <-- 여기서 날짜 문자열 획득

            text = f"제목: {title}\n내용: {body}\n게시자: {author}\n게시일시: {date}"

            return {
                "text": text.strip(),
                "date": date,
                "title": title,
                "body": body
            }

        except Exception:
            return {"text": content, "date": "", "title": "", "body": ""}
    else:
        return {"text": content, "date": "", "title": "", "body": ""}


def chunk_text_by_token(text, chunk_size=TOKEN_CHUNK_SIZE, chunk_overlap=TOKEN_CHUNK_OVERLAP):
    """본문을 토큰 기반으로 나눠주는 chunking 함수 (HuggingFace Tokenizer 사용)"""
    if not TOKENIZER:
        return [text]  # 토크나이저 로드 실패 시 분할하지 않고 원본 반환

    # 텍스트를 토큰 ID 리스트로 변환
    tokens = TOKENIZER.encode(text, add_special_tokens=False)

    chunks = []

    # 토큰 ID를 기준으로 슬라이싱
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        # 청크의 시작과 끝 인덱스
        start_idx = i
        end_idx = i + chunk_size

        # 토큰 ID 청크 추출
        token_chunk = tokens[start_idx:end_idx]

        # 토큰 ID 청크를 다시 텍스트로 디코딩
        chunk_text = TOKENIZER.decode(token_chunk)

        chunks.append(chunk_text)

        # 마지막 청크 처리 후 루프 종료
        if end_idx >= len(tokens):
            break

    return chunks


def load_documents_from_folder(folder_path: str):
    """폴더 내 모든 txt 파일을 Document 객체 리스트로 로드 (chunking 적용)"""
    documents = []

    print("🔍 Scanning for .txt files...")
    all_txt_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files if f.lower().endswith(".txt")
    ]

    if not all_txt_files:
        print("⚠️ No .txt files found.")
        return documents

    print(f"📊 Found {len(all_txt_files)} .txt files\n")

    for file_path in tqdm(all_txt_files, desc="📖 Loading & parsing files", unit="file"):
        try:
            result = extract_text_from_file(file_path)

            text = result["text"]
            date_str = result["date"] # 원본 날짜 문자열
            title = result["title"]
            body = result["body"]

            if not text:
                tqdm.write(f"  ⚠️ Empty file skipped: {file_path}")
                continue

            # ⭐⭐⭐ 핵심 수정 부분: 날짜 문자열을 타임스탬프로 변환 ⭐⭐⭐
            date_timestamp = 0.0
            if date_str:
                try:
                    # '2025-09-22 09:24:27' 포맷 파싱
                    dt_obj = datetime.strptime(date_str, DATE_TIME_FORMAT)
                    date_timestamp = dt_obj.timestamp() # float형 타임스탬프
                except ValueError:
                    tqdm.write(f"  ⚠️ Invalid date format in {file_path}. Storing 0.0.")
                    date_timestamp = 0.0 # 파싱 실패 시 0.0으로 저장

            # ✅ 제목만 따로 벡터화
            if title:
                doc_title = Document(
                    page_content=title,
                    metadata={
                        "source": file_path,
                        # ⭐ date 필드에 타임스탬프 저장
                        "date": date_timestamp,
                        "title": title,
                        "type": "title"
                    },
                )
                documents.append(doc_title)

            # ✅ 본문 chunking 적용 (토큰 기반 함수 사용)
            chunks = []
            if body:
                chunks = chunk_text_by_token(body, chunk_size=TOKEN_CHUNK_SIZE, chunk_overlap=TOKEN_CHUNK_OVERLAP)

                for i, chunk in enumerate(chunks):
                    doc_body = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            # ⭐ date 필드에 타임스탬프 저장
                            "date": date_timestamp,
                            "title": title,
                            "chunk_index": i,
                            "type": "body"
                        },
                    )
                    documents.append(doc_body)

            rel_path = os.path.relpath(file_path, folder_path)
            tqdm.write(f"  ✅ {rel_path} (chunks: {len(chunks) if body else 0}, title added: {'Y' if title else 'N'}, date stored as timestamp: {'Y' if date_timestamp else 'N'})")

        except Exception as e:
            tqdm.write(f"  ❌ Error reading {file_path}: {e}")

    print(f"\n✨ Successfully loaded: {len(documents)} entries (chunked body + title 포함)")
    return documents


@logging_time
def create_vector_db(folder_path, persist_dir=DB_FOLDER):
    print(f"\n{'=' * 60}")
    print(f"📂 Target folder: {folder_path}")
    print(f"💾 Vector DB will be saved to: {persist_dir}")
    print(f"{'=' * 60}\n")

    documents = load_documents_from_folder(folder_path)

    if not documents:
        print("⚠️ No valid .txt or JSON files found. Aborting.")
        return None

    # 토큰 기반으로 분할했기 때문에, 평균 길이는 토큰 수(500)에 근접할 것임.
    avg_length = sum(len(TOKENIZER.encode(doc.page_content, add_special_tokens=False)) for doc in documents) / len(
        documents)
    print(f"📊 Average chunk length: {avg_length:.1f} tokens")

    print("\n🔍 Creating embeddings (this may take a while)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("💾 Saving to Chroma vector DB...")
    # ChromaDB는 이제 메타데이터의 date 필드를 float으로 인식합니다.
    db = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)

    print(f"\n{'=' * 60}")
    print(f"✅ Vector DB successfully created!")
    print(f"📍 Location: {persist_dir}")
    print(f"📊 Total embedded entries: {len(documents)} (chunked body + title 포함)")
    print(f"📈 Average chunk length: {avg_length:.1f} tokens")
    print(f"{'=' * 60}")
    return db


if __name__ == "__main__":
    folder_path = input("📁 Enter folder path containing TXT files: ").strip()

    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
    elif not os.path.isdir(folder_path):
        print(f"❌ Error: '{folder_path}' is not a directory!")
    else:
        create_vector_db(folder_path)