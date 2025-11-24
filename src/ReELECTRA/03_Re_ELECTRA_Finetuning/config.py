# config.py
CONFIG = {
    # paths
    "TOKENIZER_DIR": "./model/tokenizer",
    "FINETUNE_DATA_PATH": "./data/processed/model/finetuning_preprocessed.txt",

    # outputs
    "OUTPUT_DIR": "./model/ReELECTRA/finetuned/checkpoints",
    "FINAL_SAVE_DIR": "./model/ReELECTRA/finetuned",

    # tokenization
    "MAX_LENGTH": 128,

    # training defaults (device-aware override in train script)
    "LEARNING_RATE": 5e-5,
    "NUM_EPOCHS_CPU": 1,
    "NUM_EPOCHS_CUDA": 3,
    "BATCH_CPU": 2,
    "BATCH_CUDA": 16,
    "GRAD_ACCUM_CPU": 8,
    "GRAD_ACCUM_CUDA": 1,
    "DATALOADER_WORKERS_CPU": 0,
    "DATALOADER_WORKERS_CUDA": 2,
    "SAVE_STEPS": 1000,
    "LOGGING_STEPS": 50,
    "RANDOM_SEED": 1,
    "NUM_PROC_TOKENIZE": 2
}
