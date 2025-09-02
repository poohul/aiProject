from vector_db_creator import VectorDBCreator
from gpt2_llm_model import GPT2LLMSystem
import os


def setup_complete_system(text_file_path: str, collection_name: str = "my_documents"):
    """
    전체 시스템 설정 및 실행

    Args:
        text_file_path: 읽을 텍스트 파일 경로
        collection_name: 생성할 컬렉션 이름
    """
    print("=== 벡터 DB 및 LLM 시스템 설정 시작 ===\n")

    # 1단계: 벡터 DB 생성
    print("1. 벡터 DB 생성 중...")
    vector_creator = VectorDBCreator()

    # 텍스트 파일 읽기
    text_content = vector_creator.read_text_file(text_file_path)
    if not text_content:
        print("텍스트 파일을 읽을 수 없습니다. 프로그램을 종료합니다.")
        return

    # 텍스트 청크 분할
    chunks = vector_creator.split_text_into_chunks(text_content)

    # 벡터 DB 생성
    success = vector_creator.create_vector_db(collection_name, chunks)
    if not success:
        print("벡터 DB 생성에 실패했습니다.")
        return

    print("벡터 DB 생성 완료!\n")

    # 2단계: LLM 시스템 초기화
    print("2. GPT-2 LLM 시스템 초기화 중...")
    llm_system = GPT2LLMSystem(collection_name=collection_name)

    print("시스템 설정 완료!\n")

    # 3단계: 시스템 테스트
    # print("=== 시스템 테스트 ===")
    # test_queries = [
    #     "이 문서의 주요 내용을 요약해주세요",
    #     "가장 중요한 정보는 무엇인가요?",
    #     "이 내용에 대해 질문이 있습니다"
    # ]
    #
    # for i, query in enumerate(test_queries, 1):
    #     print(f"\n테스트 {i}: {query}")
    #     print("-" * 50)
    #
    #     # 관련 문서 검색
    #     search_results = vector_creator.search_similar_documents(query, n_results=2)
    #     if search_results.get('documents'):
    #         print("검색된 관련 문서:")
    #         for j, doc in enumerate(search_results['documents'][0]):
    #             print(f"  {j + 1}. {doc[:100]}...")
    #
    #     # LLM 응답 생성
    #     response = llm_system.generate_response(query, use_context=True)[0]
    #     print(f"\nAI 응답: {response}")
    #     print("=" * 80)

    # 4단계: 대화형 모드
    print("\n=== 대화형 모드 시작 ===")
    print("이제 자유롭게 질문하실 수 있습니다!")
    llm_system.chat()


def create_sample_text_file():
    """샘플 텍스트 파일 생성 (테스트용)"""
    sample_content = """
    인공지능의 발전과 미래

    인공지능(AI)은 현대 사회에서 가장 혁신적인 기술 중 하나입니다. 
    머신러닝과 딥러닝의 발전으로 AI는 다양한 분야에서 활용되고 있습니다.

    자연어처리 기술의 발전으로 GPT와 같은 대화형 AI가 등장했습니다.
    이러한 기술들은 텍스트 생성, 번역, 요약 등 다양한 작업을 수행할 수 있습니다.

    컴퓨터 비전 분야에서도 AI는 놀라운 성과를 보이고 있습니다.
    이미지 인식, 객체 탐지, 얼굴 인식 등의 기술이 급속도로 발전하고 있습니다.

    미래에는 AI가 의료, 교육, 교통, 금융 등 모든 분야에서 중요한 역할을 할 것으로 예상됩니다.
    하지만 AI의 발전과 함께 윤리적 고려사항과 안전성 문제도 중요하게 다뤄져야 합니다.

    AI 기술의 민주화와 접근성 향상을 통해 더 많은 사람들이 AI의 혜택을 받을 수 있도록 해야 합니다.
    동시에 AI가 인간의 일자리를 대체하는 문제에 대해서도 신중하게 접근해야 합니다.

    결론적으로, 인공지능은 인간의 삶을 개선하는 도구로 활용되어야 하며, 
    지속적인 연구와 개발을 통해 더욱 발전시켜 나가야 할 기술입니다.
    """

    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write(sample_content.strip())
    print("샘플 텍스트 파일 'sample.txt'가 생성되었습니다.")


if __name__ == "__main__":
    # 사용자로부터 텍스트 파일 경로 입력받기
    print("=== 텍스트 파일 경로 설정 ===")
    print("읽고 싶은 텍스트 파일의 전체 경로를 입력하세요.")
    print("예시: C:\\Users\\username\\Documents\\my_text.txt")
    print("또는 상대 경로: ./data/sample.txt\n")

    while True:
        text_file_path = input("파일 경로: ").strip()

        # 빈 입력시 샘플 파일 생성 옵션
        if not text_file_path:
            use_sample = input("샘플 파일을 생성하시겠습니까? (y/n): ").lower()
            if use_sample in ['y', 'yes', '예']:
                text_file_path = "sample.txt"
                create_sample_text_file()
                break
            else:
                continue

        # 따옴표 제거 (드래그 앤 드롭시 생기는 따옴표)
        text_file_path = text_file_path.strip('"').strip("'")

        # 파일 존재 확인
        if os.path.exists(text_file_path):
            print(f"파일 '{text_file_path}'을 찾았습니다!")
            break
        else:
            print(f"파일 '{text_file_path}'을 찾을 수 없습니다. 다시 입력해주세요.\n")

    # 컬렉션 이름 설정
    collection_name = input("\n컬렉션 이름을 입력하세요 (기본값: my_documents): ").strip()
    if not collection_name:
        collection_name = "my_documents"

    # 전체 시스템 실행
    setup_complete_system(text_file_path, collection_name)