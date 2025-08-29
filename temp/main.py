# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from loader import load_and_split_pdf
from embedder import create_vector_db
from qa import qa

if __name__ == "__main__":
    # file_path = "data/sample.pdf"  # 테스트할 PDF 파일 경로
    # docs = load_and_split_pdf(file_path)

    # print(f"문서 개수: {len(docs)}")
    # if docs:
    #     print(docs[0].page_content[:100])

    #print(docs)
    # print(f"총 {len(docs)} 개의 청크가 생성되었습니다.")
    # print("예시:")
    # print(docs[0].page_content[:500])  # 첫 chunk의 앞부분 출력

    # 백터db 생성
    # vectordb = create_vector_db(docs)


    # 질문 실행
    question = "이 문서의 핵심 내용은 무엇인가요?"
    response = qa.invoke({"query": question})

    print("💬 답변:", response["result"])
    print("📚 출처 문서 수:", len(response["source_documents"]))