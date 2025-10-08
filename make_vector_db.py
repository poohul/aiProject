# make_vector_db.py
import os
from langchain_huggingface import HuggingFaceEmbeddings # 변경된 import 문
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                    print(f"✅ Loaded: {filename} ({len(content)} chars)")
    return texts

def create_vector_db(folder_path, persist_dir="./chroma_db2"):
    print(f"\n📂 Reading text files from: {folder_path}")
    texts = load_texts_from_folder(folder_path)

    if not texts:
        print("⚠️ No .txt files found. Aborting.")
        return

    print("\n🔍 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    print("💾 Saving Chroma vector DB...")
    db = Chroma.from_texts(texts, embeddings, persist_directory=persist_dir)
    # db.persist()
    print(f"✅ Vector DB saved to {persist_dir}")

if __name__ == "__main__":
    folder_path = input("📁 Enter folder path containing TXT files: ").strip()
    create_vector_db(folder_path)
