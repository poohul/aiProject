# reranker_finetuner.py
import json
from pathlib import Path
import torch
from torch import nn  # nn 모듈 추가
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample, losses

# 1. 설정
RERANKER_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2'
HIL_DATA_DIR = Path("./hil_training_data")
OUTPUT_MODEL_PATH = './custom_finetuned_reranker'

# 하이퍼파라미터 설정
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# 2. 데이터 로드 및 준비
def load_and_prepare_data(data_dir: Path):
    """저장된 JSON 트립렛 데이터를 로드하고 InputExample 리스트로 변환합니다."""
    train_examples = []

    for json_file in data_dir.glob("triplet_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        query = data['query']
        pos_content = data['positive']['content']

        # 긍정 쌍 (Positive Pair) 추가: 레이블 1.0 (매우 관련 있음)
        train_examples.append(InputExample(texts=[query, pos_content], label=1.0))

        # 부정 쌍 (Negative Pairs) 추가: 레이블 0.0 (관련 없음)
        for neg in data['negatives']:
            neg_content = neg['content']
            train_examples.append(InputExample(texts=[query, neg_content], label=0.0))

    print(f"💾 총 {len(train_examples)}개의 학습 쌍을 준비했습니다.")
    return train_examples


# 3. 모델 파인튜닝 함수
def fine_tune_reranker():
    # 데이터 로드
    train_examples = load_and_prepare_data(HIL_DATA_DIR)
    if not train_examples:
        print("학습 데이터가 부족하여 파인튜닝을 시작할 수 없습니다.")
        return

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # 모델 로드 (GPU 사용 설정)
    model = CrossEncoder(RERANKER_NAME, device=device)

    # 3. 손실 함수 설정: CrossEncoder의 표준 손실 함수 (Binary Classification)
    # CrossEncoder는 0과 1 사이의 점수를 예측하므로 BCEWithLogitsLoss가 적합합니다.
    loss_fct = nn.BCEWithLogitsLoss()

    # 4. 모델 학습 실행 (TypeError 해결)
    print(f"🚀 파인튜닝 시작: {NUM_EPOCHS} 에포크")
    model.fit(
        # 💡 수정된 부분: train_objectives 대신 train_dataloader와 loss_fct를 사용
        train_dataloader=train_dataloader,
        loss_fct=loss_fct,
        epochs=NUM_EPOCHS,
        warmup_steps=100,
        output_path=OUTPUT_MODEL_PATH,
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True
    )

    print(f"\n✅ 파인튜닝 완료. 모델이 저장되었습니다: {OUTPUT_MODEL_PATH}")


if __name__ == '__main__':
    fine_tune_reranker()