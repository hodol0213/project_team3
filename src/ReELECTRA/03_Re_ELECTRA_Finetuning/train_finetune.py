# train_finetune.py
import os
import torch
from transformers import TrainingArguments, Trainer
from config import CONFIG
from data_finetune import get_tokenizer, prepare_datasets, load_raw_df
from model_finetune import load_model, evaluate_model_by_dataloader

def main():
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # tokenizer + datasets
    tokenizer = get_tokenizer(CONFIG["TOKENIZER_DIR"])
    raw_df = load_raw_df(CONFIG["FINETUNE_DATA_PATH"])
    train_ds, valid_ds = prepare_datasets(tokenizer=tokenizer, raw_df=raw_df,
                                          test_size=0.2, seed=CONFIG["RANDOM_SEED"],
                                          max_length=CONFIG["MAX_LENGTH"],
                                          num_proc=CONFIG["NUM_PROC_TOKENIZE"])
    # training args (device-aware)
    num_epochs = CONFIG["NUM_EPOCHS_CUDA"] if device == "cuda" else CONFIG["NUM_EPOCHS_CPU"]
    per_device_batch = CONFIG["BATCH_CUDA"] if device == "cuda" else CONFIG["BATCH_CPU"]
    grad_accum = CONFIG["GRAD_ACCUM_CUDA"] if device == "cuda" else CONFIG["GRAD_ACCUM_CPU"]
    workers = CONFIG["DATALOADER_WORKERS_CUDA"] if device == "cuda" else CONFIG["DATALOADER_WORKERS_CPU"]

    training_args = TrainingArguments(
        output_dir=CONFIG["OUTPUT_DIR"],
        overwrite_output_dir=True,
        learning_rate=CONFIG["LEARNING_RATE"],
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        logging_steps=CONFIG["LOGGING_STEPS"],
        save_strategy="steps",
        save_steps=CONFIG["SAVE_STEPS"],
        dataloader_num_workers=workers,
        dataloader_pin_memory=(device == "cuda"),
        weight_decay=0.01,
        load_best_model_at_end=False,
        report_to="none",
        push_to_hub=False,
    )

    # model init (from DAPT-finetuned directory)
    model = load_model(pretrained_dir="./Re_ELECTRA_DAPT", num_labels=2, device=device)

    # Trainer - note: we pass tokenizer via tokenizer argument for convenience (used for data collating)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
    )

    # train
    trainer.train()

    # save model + tokenizer
    os.makedirs(CONFIG["FINAL_SAVE_DIR"], exist_ok=True)
    trainer.save_model(CONFIG["FINAL_SAVE_DIR"])
    tokenizer.save_pretrained(CONFIG["FINAL_SAVE_DIR"])
    print("Saved finetuned model & tokenizer to:", CONFIG["FINAL_SAVE_DIR"])

    # evaluation (use model from memory)
    metrics = evaluate_model_by_dataloader(model, valid_ds, data_collator=None, batch_size=32, device=device)
    print("Eval metrics:", metrics)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # quick inference examples
    model.eval()
    texts = ["이 영화 정말 재미있다", "별로였다"]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=CONFIG["MAX_LENGTH"], return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        preds = logits.argmax(dim=-1)
    print("Sample preds:", preds.cpu().numpy())

if __name__ == "__main__":
    main()
