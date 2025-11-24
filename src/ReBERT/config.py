from pathlib import Path

# 이 파일(config.py)이 있는 폴더: rebert/rebert_making
BASE_DIR = Path(__file__).resolve().parent

# rebert 폴더
PROJECT_ROOT = BASE_DIR.parent

# datasets 폴더 (rebert/datasets)
DATA_DIR = PROJECT_ROOT / "datasets"

# ========= 공통 경로 설정 =========
CORPUS_PATH      = DATA_DIR / "Pre_training_dataset.txt"      # pre-training 코퍼스
DAPT_CORPUS_PATH = DATA_DIR / "DAPT_dataset.txt"              # DAPT용 도메인 코퍼스
FINE_TUNE_PATH   = DATA_DIR / "Fine_tuning_shopping.txt"      # (text \t label) 형식

# 모델/토크나이저 저장 경로 (rebert/kcbert_*)
TOKENIZER_DIR = PROJECT_ROOT / "kcbert_tokenizer"
MODEL_DIR     = PROJECT_ROOT / "kcbert_mlm"
DAPT_OUT_DIR  = MODEL_DIR / "dapt"

# 디렉터리 미리 생성 (존재하면 무시)
for d in [TOKENIZER_DIR, MODEL_DIR, DAPT_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ========= MLM Pre-training 하이퍼파라미터 =========
VOCAB_SIZE = 32000              # WordPiece vocab 크기
LOWER_CASE = False              # 한국어에선 보통 False가 많지만, 소문자 일관화가 유리하면 True 사용
BLOCK_SIZE = 128                # 최대 토큰 길이 (BERT 입력 시퀀스 길이)
MLM_PROB = 0.15                 # MLM에서 [MASK]로 가릴 비율
BATCH_SIZE = 32                 # 배치 크기 (GPU 메모리에 맞게 튜닝)
EPOCHS = 3                      # 사전학습 epoch 수
LEARNING_RATE = 1.8e-4          # 학습률 (MLM pretrain: 보통 1e-4 ~ 5e-4 사이)
WARMUP_RATIO = 0.12             # 전체 step 대비 warmup 비율
WEIGHT_DECAY = 0.01             # AdamW weight decay 계수
LOGGING_STEPS = 250             # 학습 중 로그 출력 주기
SAVE_STEPS = 2000               # 체크포인트 저장 주기
SEED = 7                        # 랜덤 시드 고정

# ========= BERT-small 구조 =========
HIDDEN_SIZE = 512               # 히든 차원 (기본 BERT-base는 768, 여기선 작게 512)
NUM_HIDDEN_LAYERS = 6           # Transformer 인코더 레이어 수
NUM_ATTENTION_HEADS = 8         # multi-head attention의 head 개수
INTERMEDIATE_SIZE = 2048        # FFN(중간층) 차원
MAX_POS_EMBED = 512             # 최대 position embedding 길이

# ========= DAPT 하이퍼파라미터 =========
DAPT_BLOCK_SIZE = 128           # DAPT용 입력 최대 길이
DAPT_MLM_PROB = 0.15            # DAPT에서도 동일하게 MLM 15%
DAPT_BATCH_SIZE = 16            # DAPT 배치 크기 (코퍼스/메모리에 따라 줄여놓음)
DAPT_EPOCHS = 1                 # DAPT epoch 수 (보통 1~2면 충분)
DAPT_LR = 5e-5                  # DAPT 학습률 (기존 모델 미세 조정이므로 작게)

# ========= 파인튜닝 하이퍼파라미터 =========
FT_BATCH_SIZE   = 32            # 파인튜닝 배치 크기
FT_EPOCHS       = 3             # 파인튜닝 epoch 수
FT_LR           = 3e-5          # 파인튜닝 학습률 (일반적인 분류 파인튜닝 범위)

