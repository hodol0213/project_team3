CONFIG = {
    # paths
    "TOKENIZER_DIR": "./tokenizer",
    "DAPT_DATA_PATH": "./datasets/DAPT_dataset.txt",

    # checkpoints / outputs
    "OUTPUT_DIR": "./Re_ELECTRA_DAPT_checkpoint",
    "FINAL_SAVE_DIR": "./Re_ELECTRA_DAPT",

    # tokenization / dataloader
    "MAX_LENGTH": 128,
    "MLM_PROB": 0.15,
    "NUM_PROC": 2,

    # training default (will be adjusted by device)
    "DISCRIMINATOR_LR": 5e-5,
    "GENERATOR_LR_FACTOR": 0.5,

    "WARMUP_STEPS": 1000,
    "SAVE_STEPS": 1000,
    "LOGGING_STEPS": 50,

    "NUM_EPOCHS_CPU": 1,
    "NUM_EPOCHS_CUDA": 2,
    "BATCH_CPU": 2,
    "BATCH_CUDA": 16,
    "GRAD_ACCUM_CPU": 8,
    "GRAD_ACCUM_CUDA": 1,
    "DATALOADER_WORKERS_CPU": 0,
    "DATALOADER_WORKERS_CUDA": 2,
}
