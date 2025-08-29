from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# print("▶️ 임베딩 모델 로드 시작")

def create_vector_db(docs, persist_directory="vectorstore"):
    # 1. 로컬 임베딩 모델 선택
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ 임베딩 모델 로드 완료")

    # 2. Chroma DB에 저장
    print("▶️ 벡터 DB 생성 시작")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print("✅ 벡터 DB 생성 완료")

    vectordb.persist()
    print(f"✅ 벡터 DB가 '{persist_directory}'에 저장되었습니다.")
    return vectordb