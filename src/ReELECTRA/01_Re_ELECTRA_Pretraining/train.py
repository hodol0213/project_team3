from config import CONFIG
from data import get_tokenizer, build_datasets_and_collator
from model import init_models_and_optimizers, get_training_args, build_trainer, find_last_checkpoint, evaluate_models
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # tokenizer, datasets
    tokenizer = get_tokenizer(CONFIG["TOKENIZER_DIR"])
    tokenized_datasets, data_collator = build_datasets_and_collator(CONFIG["DATA_PATH"], tokenizer, max_length=CONFIG["MAX_LENGTH"])

    # models & optimizers
    generator, discriminator, gen_optimizer, disc_optimizer = init_models_and_optimizers(device=device, discriminator_lr=CONFIG["DISCRIMINATOR_LR"])

    # training args
    num_epochs = CONFIG["NUM_EPOCHS_CUDA"] if device=="cuda" else CONFIG["NUM_EPOCHS_CPU"]
    per_device_batch = CONFIG["BATCH_CUDA"] if device=="cuda" else CONFIG["BATCH_CPU"]
    grad_accum = CONFIG["GRAD_ACCUM_CUDA"] if device=="cuda" else CONFIG["GRAD_ACCUM_CPU"]
    workers = CONFIG["DATALOADER_WORKERS_CUDA"] if device=="cuda" else CONFIG["DATALOADER_WORKERS_CPU"]

    training_args = get_training_args(
        output_dir=CONFIG["OUTPUT_DIR"],
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        dataloader_num_workers=workers
    )

    # build trainer
    trainer = build_trainer(generator, discriminator, gen_optimizer, disc_optimizer, tokenizer, tokenized_datasets, data_collator, training_args, warmup_steps=CONFIG["WARMUP_STEPS"])

    # resume if checkpoint exists
    last_ckpt = find_last_checkpoint(training_args.output_dir)
    print("Resume from:", last_ckpt)

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(CONFIG["FINAL_SAVE_DIR"])

    # evaluate
    evaluate_models(generator, discriminator, tokenized_datasets, data_collator, device=device)

if __name__ == "__main__":
    main()
