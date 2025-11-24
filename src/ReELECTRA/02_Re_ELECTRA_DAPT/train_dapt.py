from config import CONFIG
from data import get_tokenizer, build_datasets_and_collator
from model import init_models_and_optimizers, get_training_args, build_trainer, find_last_checkpoint, evaluate_models
import torch
import os

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # tokenizer & data
    tokenizer = get_tokenizer(CONFIG["TOKENIZER_DIR"])
    tokenized_datasets, data_collator = build_datasets_and_collator(CONFIG["DAPT_DATA_PATH"], tokenizer, max_length=CONFIG["MAX_LENGTH"])

    # models & optimizers
    generator, discriminator, gen_optimizer, disc_optimizer = init_models_and_optimizers(device=device, discriminator_lr=CONFIG["DISCRIMINATOR_LR"])

    # training args (device-aware)
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

    # optional: if you prefer to resume from a specific step number, set STEP env var:
    # export DAPT_STEP=135000
    step_env = os.environ.get("DAPT_STEP")
    if step_env is not None:
        last_ckpt = os.path.join(CONFIG["OUTPUT_DIR"], f"checkpoint-{step_env}")
        print("Resume from specified STEP:", last_ckpt)
        if not os.path.exists(last_ckpt):
            print("Specified checkpoint does not exist, falling back to auto-find.")
            last_ckpt = find_last_checkpoint(training_args.output_dir)
    else:
        last_ckpt = find_last_checkpoint(training_args.output_dir)

    print("Resume from:", last_ckpt)

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(CONFIG["FINAL_SAVE_DIR"])

    # evaluation
    evaluate_models(generator, discriminator, tokenized_datasets, data_collator, device=device)

if __name__ == "__main__":
    main()
