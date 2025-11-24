from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from config import (
    FINE_TUNE_PATH,
    DAPT_OUT_DIR,
    MODEL_DIR,
    FT_BATCH_SIZE, FT_EPOCHS, FT_LR,
)


def load_finetune_dataframe(path: Path) -> pd.DataFrame:
    """Fine_tuning_shopping.txt 파일 로드 및 전처리."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label"],
        dtype={"text": str},
    )
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    print(df.head(3))
    print(df["label"].value_counts())
    return df


def train_valid_split(df: pd.DataFrame, test_size: float = 0.1, seed: int = 7):
    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    print(f"train: {len(train_df)} / valid: {len(valid_df)}")
    return train_df, valid_df


def build_tokenizer_and_model():
    """DAPT가 끝난 MLM 체크포인트에서 토크나이저와 분류 모델 불러오기."""
    dapt_dir = DAPT_OUT_DIR
    tokenizer = AutoTokenizer.from_pretrained(dapt_dir, use_fast=True)
    tokenizer.model_max_length = 128

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        dapt_dir,
        num_labels=num_labels,
        problem_type="single_label_classification",
        id2label={0: "NEG", 1: "POS"},
        label2id={"NEG": 0, "POS": 1},
    )
    print("Tokenizer:", tokenizer.name_or_path)
    print("Model:", model.name_or_path)
    return tokenizer, model


def tokenize_dataset(tokenizer, train_df, valid_df):
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    valid_ds = Dataset.from_pandas(valid_df, preserve_index=False)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    valid_tok = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return train_tok, valid_tok


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
    }


def build_trainer(model, tokenizer, train_tok, valid_tok):
    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "finetune"),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=FT_BATCH_SIZE,
        per_device_eval_batch_size=FT_BATCH_SIZE,
        learning_rate=FT_LR,
        num_train_epochs=FT_EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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


def run_finetune_and_eval():
    df = load_finetune_dataframe(FINE_TUNE_PATH)
    train_df, valid_df = train_valid_split(df)

    tokenizer, model = build_tokenizer_and_model()
    train_tok, valid_tok = tokenize_dataset(tokenizer, train_df, valid_df)

    trainer = build_trainer(model, tokenizer, train_tok, valid_tok)

    trainer.train()
    eval_res = trainer.evaluate()
    print("Eval:", eval_res)

    preds = np.argmax(trainer.predict(valid_tok).predictions, axis=-1)
    labels = np.array(valid_tok["label"])

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print("Confusion matrix [[TN FP],[FN TP]]")
    print(cm)
    print(classification_report(labels, preds, digits=4))


if __name__ == "__main__":
    run_finetune_and_eval()
