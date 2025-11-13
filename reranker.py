# reranker.py
from sentence_transformers import CrossEncoder

# 재순위화 모델을 전역 변수로 저장 (메모리에서 한 번만 로드)
RERANKER_MODEL = None
RERANKER_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2'
# RERANKER_NAME = './custom_kyoboDTS_bbs_reranker'
# RERANKER_NAME = 'C:/Users/yjb/PycharmProjects/aiProject/custom_kyoboDTS_bbs_reranker'



def load_reranker_model():
    """재순위화 모델을 로드하고 전역 변수에 저장합니다."""
    global RERANKER_MODEL
    if RERANKER_MODEL is None:
        print(f"Loading Reranker Model: {RERANKER_NAME}")
        RERANKER_MODEL = CrossEncoder(RERANKER_NAME)
    return RERANKER_MODEL


def rerank_documents(query: str, retrieved_docs: list, top_k: int = 10) -> list:
    """
    검색된 문서 목록을 재순위화하여 상위 K개를 반환합니다.

    :param query: 사용자 질문 (Anchor)
    :param retrieved_docs: ChromaDB 등에서 검색된 (content, metadata) 튜플 리스트
    :param top_k: 재순위화 후 최종적으로 반환할 문서의 개수
    :return: 재순위화 및 정렬된 문서 딕셔너리 리스트
    """
    if not retrieved_docs:
        return []

    model = load_reranker_model()

    # 1. Reranker 입력 형식 생성: [(Query, Document1), (Query, Document2), ...]
    # retrieved_docs는 (content, metadata) 튜플을 포함하는 리스트라고 가정
    rerank_input = [(query, doc.page_content) for doc in retrieved_docs]

    # 2. 관련성 점수 예측
    scores = model.predict(rerank_input)

    # 3. 점수와 원본 데이터를 결합
    reranked_results = []
    for score, doc in zip(scores, retrieved_docs):
        # 기존 Document 객체를 사용하여 결과를 재구성
        doc_dict = {
            'content': doc.page_content,
            'metadata': doc.metadata,
            'rerank_score': float(score)  # float으로 변환
        }
        reranked_results.append(doc_dict)

    # 4. 점수 기준으로 내림차순 정렬
    reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)

    # 5. 최종 Top-K 반환
    return reranked_results[:top_k]