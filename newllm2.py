import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import faiss
import torch

# 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# 패딩 토큰 설정 (GPT-2는 기본적으로 패딩 토큰이 없음)
tokenizer.pad_token = tokenizer.eos_token

# 텍스트 데이터 예제
texts = [
    "인공지능이란 무엇인가?",
    "기계학습의 기초.",
    "자연어처리의 최신 동향.",
    "딥러닝으로 하는 이미지 분류.",
    "오늘 날씨는 21도 입니다",
]


# 벡터 생성 함수
def embed_text(text):
    # 패딩과 어텐션 마스크 설정
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():  # 그래디언트 계산 비활성화
        outputs = model(**inputs)

    # 마지막 히든 스테이트의 평균 (패딩된 토큰 제외)
    attention_mask = inputs['attention_mask']
    embeddings = outputs.last_hidden_state

    # 어텐션 마스크를 사용하여 패딩 토큰 제외하고 평균 계산
    masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
    summed = masked_embeddings.sum(dim=1)
    lengths = attention_mask.sum(dim=1).unsqueeze(-1)
    mean_embeddings = summed / lengths

    return mean_embeddings.detach().numpy()


# 벡터 데이터 생성
print("텍스트 임베딩 생성 중...")
vector_data = np.vstack([embed_text(text) for text in texts])
print(f"벡터 데이터 형태: {vector_data.shape}")

# FAISS 벡터 인덱스 생성
dimension = vector_data.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vector_data.astype('float32'))  # float32로 변환

print(f"FAISS 인덱스 생성 완료. 차원: {dimension}, 벡터 수: {index.ntotal}")

# 검색 테스트
query = "오늘 날씨는?"
print(f"\n검색 쿼리: '{query}'")
query_vec = embed_text(query).astype('float32')
distances, indices = index.search(query_vec, k=2)

print("\n=== 검색 결과 ===")
print("검색 결과 인덱스:", indices[0])
print("거리:", distances[0])

for i, idx in enumerate(indices[0]):
    print(f"{i + 1}. 관련 텍스트: '{texts[idx]}' (거리: {distances[0][i]:.4f})")