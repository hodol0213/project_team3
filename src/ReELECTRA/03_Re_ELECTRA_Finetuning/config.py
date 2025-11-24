# config.py
CONFIG = {
    # paths
    "TOKENIZER_DIR": "./tokenizer",
    "FINETUNE_DATA_PATH": "./datasets/Fine-tuning_shopping.txt",

    # outputs
    "OUTPUT_DIR": "./Re_ELECTRA_Finetuning_checkpoint",
    "FINAL_SAVE_DIR": "./Re_ELECTRA_Finetuning",

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
