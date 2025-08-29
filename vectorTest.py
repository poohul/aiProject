import numpy as np


def simple_text_embedding(text):
    """매우 간단한 텍스트 임베딩 (테스트용)"""
    # 단어 기반 간단한 벡터화
    words = text.lower().split()
    vocab = ["인공지능", "기계학습", "자연어처리", "딥러닝", "데이터", "기술", "컴퓨터", "학습", "패턴", "예측"]

    vector = np.zeros(len(vocab))
    for i, word in enumerate(vocab):
        if word in text:
            vector[i] = 1

    # 정규화
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def main():
    print("=== 기본 벡터 검색 테스트 ===")

    # 텍스트 데이터
    documents = [
        "인공지능이란 무엇인가? 인공지능은 컴퓨터가 인간처럼 생각하고 학습할 수 있도록 하는 기술입니다.",
        "기계학습의 기초. 기계학습은 데이터를 통해 패턴을 찾아 예측하는 기술입니다.",
        "자연어처리의 최신 동향. 자연어처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.",
        "딥러닝으로 하는 이미지 분류. 딥러닝은 인공신경망을 이용한 기계학습 방법입니다."
    ]

    print("문서 임베딩 생성 중...")
    try:
        # 문서 벡터화
        doc_vectors = []
        for i, doc in enumerate(documents):
            vector = simple_text_embedding(doc)
            doc_vectors.append(vector)
            print(f"문서 {i + 1} 벡터 생성 완료: {vector.shape}")

        doc_vectors = np.array(doc_vectors)
        print(f"전체 문서 벡터 형태: {doc_vectors.shape}")

    except Exception as e:
        print(f"문서 벡터화 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return

    # 질문
    query = "기계학습이란 무엇인가요?"
    print(f"\n질문: {query}")

    try:
        # 질문 벡터화
        query_vector = simple_text_embedding(query)
        print(f"질문 벡터 생성 완료: {query_vector.shape}")

        # 유사도 계산
        similarities = []
        for i, doc_vector in enumerate(doc_vectors):
            sim = cosine_similarity(query_vector, doc_vector)
            similarities.append((i, sim))
            print(f"문서 {i + 1} 유사도: {sim:.4f}")

        # 결과 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)

        print("\n=== 검색 결과 ===")
        for i, (doc_idx, sim) in enumerate(similarities[:2]):
            print(f"{i + 1}. 유사도: {sim:.4f}")
            print(f"   문서: {documents[doc_idx]}")
            print()

    except Exception as e:
        print(f"검색 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"전체 실행 중 오류: {e}")
        import traceback

        traceback.print_exc()