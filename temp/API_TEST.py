import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import faiss
import torch

# 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 텍스트 데이터 예제
texts = [
    "인공지능이란 무엇인가?",
    "기계학습의 기초.",
    "자연어처리의 최신 동향.",
    "딥러닝으로 하는 이미지 분류.",
    "오늘 날씨는 21도입니다",
    "화창한 날씨가 계속 됩니다",
]


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

    attention_mask = inputs['attention_mask']
    embeddings = outputs.last_hidden_state

    masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
    summed = masked_embeddings.sum(dim=1)
    lengths = attention_mask.sum(dim=1).unsqueeze(-1)
    mean_embeddings = summed / lengths

    return mean_embeddings.detach().numpy()


def cosine_similarity(v1, v2):
    """코사인 유사도 계산"""
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    return np.dot(v1_flat, v2_flat) / (
            np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat)
    )


def smart_search(query, similarity_threshold=0.3):
    """임계값을 사용한 스마트 검색"""
    print(f"질문: '{query}'")

    # 쿼리 벡터 생성
    query_vec = embed_text(query)

    # 모든 문서와 유사도 계산
    similarities = []
    for i, text in enumerate(texts):
    #     for i, text in enumerate(1):
        doc_vec = embed_text(text)
        sim = cosine_similarity(query_vec, doc_vec)
        similarities.append((i, sim, text))

    # 유사도 기준으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== 모든 문서의 유사도 ===")
    for i, (idx, sim, text) in enumerate(similarities):
        print(f"{i + 1}. 유사도: {sim:.4f} - '{text}'")

    # 임계값 체크
    best_similarity = similarities[0][1]

    if best_similarity < similarity_threshold:
        print(f"\n⚠️  최고 유사도({best_similarity:.4f})가 임계값({similarity_threshold})보다 낮습니다.")
        print("❌ 관련된 정보를 찾을 수 없습니다.")
        print("💡 다른 질문을 해보시거나, 다음 주제들에 대해 물어보세요:")
        for text in texts:
            print(f"   - {text}")
        return None
    else:
        print(f"\n✅ 관련 정보를 찾았습니다! (유사도: {best_similarity:.4f})")
        print(f"📄 가장 관련된 문서: '{similarities[0][2]}'")
        return similarities[0]


def test_various_queries():
    """다양한 질문으로 테스트"""
    test_queries = [
        # "기계학습이란 무엇인가요?",  # 관련 있는 질문
        # "딥러닝 이미지 분류",  # 관련 있는 질문
        # "오늘 날씨가 어때요?",  # 관련 없는 질문
        # "점심 메뉴 추천해주세요",  # 관련 없는 질문
        # "자연어처리 동향",  # 관련 있는 질문
        "오늘 날씨가 어때요?",
    ]

    print("=== 다양한 질문 테스트 ===\n")

    for query in test_queries:
        result = smart_search(query, similarity_threshold=0.3)
        print("-" * 50)


def adjust_threshold_test():
    """임계값 조정 테스트"""
    query = "오늘 날씨가 어때요?"
    thresholds = [0.1, 0.3, 0.5, 0.7]

    print(f"=== 임계값 조정 테스트 (질문: '{query}') ===\n")

    for threshold in thresholds:
        print(f"임계값: {threshold}")
        result = smart_search(query, similarity_threshold=threshold)
        print("-" * 30)


if __name__ == "__main__":
    # 기본 테스트
    test_various_queries()

    print("\n" + "=" * 60 + "\n")

    # 임계값 조정 테스트
    adjust_threshold_test()