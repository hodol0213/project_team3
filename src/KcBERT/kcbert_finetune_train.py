import numpy as np
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from kcbert_finetune_config import (
    MODEL_ID,
    OUT_DIR,
    MAX_LEN,
    BATCH,
    EPOCHS,
    LR,
    WARMUP,
    SEED,
    NUM_WORKERS,
)
from kcbert_finetune_data import load_and_clean_data, train_valid_split


def build_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.model_max_length = MAX_LEN

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        problem_type="single_label_classification",
        id2label={0: "NEG", 1: "POS"},
        label2id={"NEG": 0, "POS": 1},
    )
    return tokenizer, model


def build_tokenized_datasets(tokenizer, train_df, valid_df):
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    valid_ds = Dataset.from_pandas(valid_df, preserve_index=False)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    valid_tok = valid_ds.map(tok_fn, batched=True, remove_columns=["text"])
    return train_tok, valid_tok


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def build_trainer(model, tokenizer, train_tok, valid_tok):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        seed=SEED,
        dataloader_num_workers=NUM_WORKERS,
        report_to=[],
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer


def train_and_evaluate():
    df = load_and_clean_data()
    train_df, valid_df = train_valid_split(df)

    tokenizer, model = build_tokenizer_and_model()

    train_tok, valid_tok = build_tokenized_datasets(tokenizer, train_df, valid_df)

    trainer = build_trainer(model, tokenizer, train_tok, valid_tok)

    print("=== Training ===")
    trainer.train()

    print("=== Evaluation ===")
    eval_out = trainer.evaluate()
    print(eval_out)

    pred = trainer.predict(valid_tok)
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["NEG", "POS"]))

    model.config.id2label = {0: "NEG", 1: "POS"}
    model.config.label2id = {"NEG": 0, "POS": 1}

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    print("Saved:", OUT_DIR.resolve())


if __name__ == "__main__":
    train_and_evaluate()
