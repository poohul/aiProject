import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import faiss
import torch

# 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token

# 텍스트 데이터 예제
texts = [
    "인공지능이란 무엇인가?",
    "기계학습의 기초.",
    "자연어처리의 최신 동향.",
    "딥러닝으로 하는 이미지 분류."
]


# 개선된 벡터 생성 함수
def embed_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # 어텐션 마스크를 사용한 평균
    attention_mask = inputs['attention_mask']
    embeddings = outputs.last_hidden_state

    # 패딩 토큰 제외하고 평균 계산
    masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
    summed = masked_embeddings.sum(dim=1)
    lengths = attention_mask.sum(dim=1).unsqueeze(-1)
    mean_embeddings = summed / lengths

    return mean_embeddings.detach().numpy()


def analyze_similarities():
    print("=== 토큰화 분석 ===")
    query = "기계학습 기초"

    for i, text in enumerate(texts):
        tokens = tokenizer.tokenize(text)
        print(f"{i}. '{text}' -> 토큰: {tokens}")

    query_tokens = tokenizer.tokenize(query)
    print(f"질문 '{query}' -> 토큰: {query_tokens}")

    print("\n=== 벡터 생성 및 유사도 분석 ===")

    # 벡터 생성
    doc_vectors = []
    for text in texts:
        vec = embed_text(text)
        doc_vectors.append(vec)

    query_vec = embed_text(query)

    # 코사인 유사도 직접 계산
    def cosine_similarity(v1, v2):
        return np.dot(v1.flatten(), v2.flatten()) / (
                np.linalg.norm(v1.flatten()) * np.linalg.norm(v2.flatten())
        )

    print(f"질문: '{query}'")
    print("유사도 분석:")

    similarities = []
    for i, doc_vec in enumerate(doc_vectors):
        cos_sim = cosine_similarity(query_vec, doc_vec)
        l2_dist = np.linalg.norm(query_vec - doc_vec)
        similarities.append((i, cos_sim, l2_dist))
        print(f"{i + 1}. '{texts[i]}'")
        print(f"   코사인 유사도: {cos_sim:.4f}")
        print(f"   L2 거리: {l2_dist:.4f}")
        print()

    # FAISS 결과와 비교
    print("=== FAISS 검색 결과 ===")
    vector_data = np.vstack(doc_vectors).astype('float32')
    query_vec_f32 = query_vec.astype('float32')

    dimension = vector_data.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vector_data)

    distances, indices = index.search(query_vec_f32, k=len(texts))

    print("FAISS L2 거리 기준 순위:")
    for rank, idx in enumerate(indices[0]):
        print(f"{rank + 1}순위: '{texts[idx]}' (거리: {distances[0][rank]:.4f})")

    # 코사인 유사도 기준 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    print("\n코사인 유사도 기준 순위:")
    for rank, (idx, cos_sim, l2_dist) in enumerate(similarities):
        print(f"{rank + 1}순위: '{texts[idx]}' (유사도: {cos_sim:.4f})")


def test_english_comparison():
    print("\n=== 영어로 테스트 (비교용) ===")

    english_texts = [
        "What is artificial intelligence?",
        "Machine learning basics.",
        "Natural language processing trends.",
        "Deep learning image classification."
    ]

    query = "machine learning basics"

    print("영어 토큰화:")
    for i, text in enumerate(english_texts):
        tokens = tokenizer.tokenize(text)
        print(f"{i}. '{text}' -> {len(tokens)} 토큰")

    query_tokens = tokenizer.tokenize(query)
    print(f"질문 '{query}' -> {len(query_tokens)} 토큰")

    # 영어 벡터 검색
    doc_vectors = [embed_text(text) for text in english_texts]
    query_vec = embed_text(query)

    vector_data = np.vstack(doc_vectors).astype('float32')
    dimension = vector_data.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vector_data)

    distances, indices = index.search(query_vec.astype('float32'), k=2)

    print("\n영어 검색 결과:")
    for rank, idx in enumerate(indices[0]):
        print(f"{rank + 1}순위: '{english_texts[idx]}' (거리: {distances[0][rank]:.4f})")


if __name__ == "__main__":
    analyze_similarities()
    test_english_comparison()