from pathlib import Path

"""
KcBERT 쇼핑 리뷰 파인튜닝용 설정 파일
- real_kcbert 구조에 맞게 경로 조정 버전
"""

# ============================================================
# 1. 경로 설정
# ============================================================

# 현재 파일: project/real_kcbert/kcbert_making/kcbert_finetune_config.py
BASE_DIR = Path(__file__).resolve().parent

# real_kcbert 폴더
PROJECT_ROOT = BASE_DIR.parent

# datasets 폴더: project/real_kcbert/datasets
DATA_DIR = PROJECT_ROOT / "datasets"

# 쇼핑 리뷰 데이터
DATA_PATH = DATA_DIR / "Fine_tuning_shopping.txt"

# 모델 저장 폴더: project/real_kcbert/kcbert_official_finetune
OUT_DIR = PROJECT_ROOT / "kcbert_official_finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace KcBERT base 모델 ID
MODEL_ID = "beomi/KcBERT-base"


# ============================================================
# 2. 하이퍼파라미터
# ============================================================

MAX_LEN = 128
BATCH = 32
EPOCHS = 3
LR = 3e-5
WARMUP = 0.06
SEED = 7
NUM_WORKERS = 2


