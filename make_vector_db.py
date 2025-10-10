# make_vector_db.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
# from timeCheck import logging_time
from commonUtil.timeCheck import logging_time

def load_texts_from_folder(folder_path):
    texts = []

    # 1단계: 모든 txt 파일 경로 수집
    print("🔍 Scanning for .txt files...")
    all_txt_files = []
    for root, dirs, files in os.walk(folder_path):
        # 특정 폴더 제외 (선택사항)
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv']]

        for filename in files:
            if filename.lower().endswith(".txt"):
                file_path = os.path.join(root, filename)
                all_txt_files.append(file_path)

    if not all_txt_files:
        return texts

    print(f"📊 Found {len(all_txt_files)} .txt files\n")

    # 2단계: 진행률 표시하며 파일 읽기
    for file_path in tqdm(all_txt_files, desc="📖 Loading files", unit="file"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                    # 상대 경로 표시
                    rel_path = os.path.relpath(file_path, folder_path)
                    tqdm.write(f"  ✅ {rel_path} ({len(content)} chars)")
        except Exception as e:
            rel_path = os.path.relpath(file_path, folder_path)
            tqdm.write(f"  ⚠️ Error reading {rel_path}: {e}")

    print(f"\n✨ Successfully loaded: {len(texts)}/{len(all_txt_files)} files")
    return texts


@logging_time
def create_vector_db(folder_path, persist_dir="./chroma_db2"):
    print(f"\n{'=' * 60}")
    print(f"📂 Target folder: {folder_path}")
    print(f"💾 Vector DB will be saved to: {persist_dir}")
    print(f"{'=' * 60}\n")

    texts = load_texts_from_folder(folder_path)

    if not texts:
        print("⚠️ No .txt files found or loaded. Aborting.")
        return

    print("\n🔍 Creating embeddings (this may take a while)...")
    # embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    embeddings = HuggingFaceEmbeddings(model_name="upskyy/gte-base-korean")
    print("💾 Saving to Chroma vector DB...")
    db = Chroma.from_texts(texts, embeddings, persist_directory=persist_dir)

    print(f"\n{'=' * 60}")
    print(f"✅ Vector DB successfully created!")
    print(f"📍 Location: {persist_dir}")
    print(f"📊 Total documents: {len(texts)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    folder_path = input("📁 Enter folder path containing TXT files: ").strip()

    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
    elif not os.path.isdir(folder_path):
        print(f"❌ Error: '{folder_path}' is not a directory!")
    else:
        create_vector_db(folder_path)