# chatbot.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def load_vector_db(persist_dir="./chroma_db2"):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return db

def main():
    print("🤖 게시판 기반 챗봇 (Ctrl+C 로 종료)\n")
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="llama3.1:8b")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    while True:
        try:
            query = input("\n🗨️ 질문: ").strip()
            if not query:
                continue
            answer = qa.run(query)
            print(f"\n💡 답변: {answer}")
        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break

if __name__ == "__main__":
    main()
