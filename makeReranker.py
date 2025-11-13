# reranker_finetuner.py
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
import re
from typing import List, Dict, Any, Union # 👈 Union, List, Dict, Any를 import합니다.

# 1. 설정
RERANKER_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2'  # 기본 모델로 재시작 권장
HIL_DATA_DIR = Path("./hil_training_data")
OUTPUT_MODEL_PATH = './custom_kyoboDTS_bbs_reranker'  # 새로운 경로 권장
# 하이퍼파라미터 설정
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# GPU 사용 가능 여부 확인 및 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 📅 날짜 패턴: YYYY.MM.DD 또는 YYYY-MM-DD 형태를 감지하는 정규식
DATE_PATTERN = re.compile(r'\d{4}[.\s-]\d{2}[.\s-]\d{2}')


def clean_text(text: str) -> str:
    """텍스트에서 날짜 패턴을 제거하고 불필요한 공백을 정리합니다."""
    if not isinstance(text, str):
        return ""
    # 날짜 패턴을 찾아 공백으로 대체합니다.
    text = DATE_PATTERN.sub(' ', text)
    # 중복된 공백을 하나로 줄여서 텍스트를 깔끔하게 만듭니다.
    return ' '.join(text.split()).strip()


# 2. 데이터 로드 및 준비
def load_and_prepare_data(data_dir: Path) -> List[InputExample]:
    """
    저장된 JSON 트립렛 데이터를 로드하고 InputExample 리스트로 변환합니다.
    (복수 Positive/Negative 문서 지원)
    """
    train_examples = []

    for json_file in data_dir.glob("triplet_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ JSON Decode Error in file: {json_file}. Skipping.")
                continue

        query = data.get('query', '')
        if not query:
            continue

        # Positive 문서 처리 (리스트 또는 단일 딕셔너리 모두 처리)
        positives: Union[List[Dict[str, Any]], Dict[str, Any], None] = data.get('positive')
        if not positives:
            continue  # Positive 문서가 없는 트립렛은 학습 쌍을 만들 수 없으므로 건너뜁니다.

        if isinstance(positives, dict):
            # 단일 Positive 포맷인 경우, 리스트로 변환하여 처리 통일
            pos_list = [positives]
        elif isinstance(positives, list):
            pos_list = positives
        else:
            continue  # 이상한 타입은 건너뜁니다.

        # 긍정 쌍 (Positive Pair) 추가: 레이블 1.0 (매우 관련 있음)
        for pos in pos_list:
            pos_content = clean_text(pos.get('content', ''))
            if pos_content:
                train_examples.append(InputExample(texts=[query, pos_content], label=1.0))

        # 부정 쌍 (Negative Pairs) 추가: 레이블 0.0 (관련 없음)
        negatives: List[Dict[str, Any]] = data.get('negatives', [])
        for neg in negatives:
            neg_content = clean_text(neg.get('content', ''))
            if neg_content:
                train_examples.append(InputExample(texts=[query, neg_content], label=0.0))

    if not train_examples:
        print("학습 데이터가 부족하여 파인튜닝을 시작할 수 없습니다.")
    else:
        print(f"💾 총 {len(train_examples)}개의 학습 쌍을 준비했습니다.")

    return train_examples


# 3. 모델 파인튜닝 함수
def fine_tune_reranker():
    # 데이터 로드
    train_examples = load_and_prepare_data(HIL_DATA_DIR)
    if not train_examples:
        return

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # 모델 로드 (GPU 사용 설정)
    model = CrossEncoder(RERANKER_NAME, device=device)
    loss_fct = nn.BCEWithLogitsLoss()

    # ⭐⭐ 학습 전에 출력 폴더가 존재하는지 확인하고 생성 ⭐⭐
    output_path_obj = Path(OUTPUT_MODEL_PATH)
    try:
        output_path_obj.mkdir(parents=True, exist_ok=True)
        print(f"📁 모델 저장 디렉토리 생성 또는 확인 완료: {OUTPUT_MODEL_PATH}")
    except Exception as e:
        print(f"❌ 오류: 모델 저장 디렉토리 생성 실패. 권한 문제일 수 있습니다: {e}")
        return  # 디렉토리 생성 실패 시 함수 종료

    # 4. 모델 학습 실행
    print(f"🚀 파인튜닝 시작: {NUM_EPOCHS} 에포크")
    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=loss_fct,
        epochs=NUM_EPOCHS,
        warmup_steps=100,
        output_path=OUTPUT_MODEL_PATH,
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True
    )

    # ✅ 학습 완료 후 모델 수동 저장 (필수)
    model.save(OUTPUT_MODEL_PATH)

    # 저장된 모델의 절대 경로를 계산하여 출력
    absolute_path = output_path_obj.resolve()

    print(f"\n✅ 파인튜닝 완료. 모델이 저장되었습니다.")
    print(f"📁 모델 저장 경로 (절대 경로): {absolute_path}")


if __name__ == '__main__':
    fine_tune_reranker()