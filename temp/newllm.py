import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import faiss

# 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# 텍스트 데이터 예제
texts = [
    "인공지능이란 무엇인가?",
    "기계학습의 기초.",
    "자연어처리의 최신 동향.",
    "딥러닝으로 하는 이미지 분류."
]

# 벡터 생성 함수
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # 마지막 히든스테이트 평균
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# 벡터 데이터 생성
vector_data = np.vstack([embed_text(text) for text in texts])

# FAISS 벡터 인덱스 생성
dimension = vector_data.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vector_data)

# 검색 테스트
query = "기계학습 기초"
query_vec = embed_text(query)
distances, indices = index.search(query_vec, k=2)

print("검색 결과 인덱스:", indices)
print("거리:", distances)
for idx in indices[0]:
    print("관련 텍스트:", texts[idx])
