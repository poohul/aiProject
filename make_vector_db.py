# make_vector_db.py
import os
import json
from pathlib import Path
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from commonUtil.timeCheck import logging_time


def extract_text_from_file(file_path: str) -> dict:
    """파일을 읽어 JSON이면 변환, 아니면 그대로 텍스트 리턴"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return ""

    if content.startswith("{") and content.endswith("}"):
        try:
            data = json.loads(content)
            title = data.get("제목", "")
            body = data.get("본문", "")
            author = data.get("게시자", "")
            date = data.get("게시일시", "")

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


def load_documents_from_folder(folder_path: str):
    """폴더 내 모든 txt 파일을 Document 객체 리스트로 로드"""
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
            date = result["date"]
            title = result["title"]
            body = result["body"]

            if not text:
                tqdm.write(f"  ⚠️ Empty file skipped: {file_path}")
                continue

            # ✅ 본문 문서
            doc_body = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "date": date,
                    "title": title,
                    "type": "body"
                },
            )

            # ✅ 제목 전용 문서 (NEW)
            if title:
                doc_title = Document(
                    page_content=title,
                    metadata={
                        "source": file_path,
                        "date": date,
                        "title": title,
                        "type": "title"
                    },
                )
                documents.extend([doc_body, doc_title])
            else:
                documents.append(doc_body)

            rel_path = os.path.relpath(file_path, folder_path)
            tqdm.write(f"  ✅ {rel_path} ({len(text)} chars, title added: {'Y' if title else 'N'})")

        except Exception as e:
            tqdm.write(f"  ❌ Error reading {file_path}: {e}")

    print(f"\n✨ Successfully loaded: {len(documents)} entries (body + title 포함)")
    return documents


@logging_time
def create_vector_db(folder_path, persist_dir="./chroma_db2"):
    print(f"\n{'=' * 60}")
    print(f"📂 Target folder: {folder_path}")
    print(f"💾 Vector DB will be saved to: {persist_dir}")
    print(f"{'=' * 60}\n")

    documents = load_documents_from_folder(folder_path)

    if not documents:
        print("⚠️ No valid .txt or JSON files found. Aborting.")
        return None

    avg_length = sum(len(doc.page_content) for doc in documents) / len(documents)
    print(f"📊 Average text length: {avg_length:.1f} chars")

    print("\n🔍 Creating embeddings (this may take a while)...")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    # 집 메인 pc 용
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="upskyy/gte-base-korean",
    #     model_kwargs={'trust_remote_code': True}
    # )

    print("💾 Saving to Chroma vector DB...")
    db = Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)

    print(f"\n{'=' * 60}")
    print(f"✅ Vector DB successfully created!")
    print(f"📍 Location: {persist_dir}")
    print(f"📊 Total embedded entries: {len(documents)} (body + title 포함)")
    print(f"📈 Average text length: {avg_length:.1f} chars")
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
