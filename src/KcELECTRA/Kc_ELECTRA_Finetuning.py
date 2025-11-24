import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    ElectraForSequenceClassification,
    Trainer
)
from torch.utils.data import DataLoader


def load_dataset(file_path: str, test_size: float = 0.2, random_state: int = 1):
    df = pd.read_csv(file_path, sep="\t", names=['document', 'label'])
    train, valid = train_test_split(df, test_size=test_size, shuffle=True, random_state=random_state)
    return train, valid


def tokenize_function(df, tokenizer, max_length: int = 128):
    encodings = tokenizer(
        list(df['document']),
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    dataset = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': df['label'].to_list()
    }
    return Dataset.from_dict(dataset)


def get_training_args(device: str):
    if device == "cpu":
        num_train_epochs = 1
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 8
        dataloader_num_workers = 0
        dataloader_pin_memory = False
    else:  # cuda
        num_train_epochs = 3
        per_device_train_batch_size = 16
        gradient_accumulation_steps = 2
        dataloader_num_workers = 2
        dataloader_pin_memory = True

    return TrainingArguments(
        output_dir="./Kc_ELECTRA_Finetuning_checkpoint",
        overwrite_output_dir=True,
        learning_rate=5e-5,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        weight_decay=0.01,
        load_best_model_at_end=False,
        report_to="none",
        push_to_hub=False,
    )


def init_model():
    return ElectraForSequenceClassification.from_pretrained(
        "beomi/KcELECTRA-small-v2022", num_labels=2
    )


def train_model(model, training_args, train_dataset, valid_dataset, tokenizer, save_dir="./Kc_ELECTRA_Finetuning"):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    return trainer


def evaluate_model(model, valid_dataset, device="cpu"):
    valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    for batch in eval_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")


def predict(model, tokenizer, texts, device="cpu", max_length=128):
    model.eval()
    tokenizer.model_max_length = max_length
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**encoding).logits
        preds = logits.argmax(dim=-1)
    return preds.cpu().numpy()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 데이터 로드
    train, valid = load_dataset("./datasets/Fine-tuning_shopping.txt")

    # 2. 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-small-v2022")

    # 3. 토큰화
    train_dataset = tokenize_function(train, tokenizer)
    valid_dataset = tokenize_function(valid, tokenizer)

    # 4. TrainingArguments
    training_args = get_training_args(device)

    # 5. 모델 초기화
    model = init_model()

    # 6. 학습 및 저장
    trainer = train_model(model, training_args, train_dataset, valid_dataset, tokenizer)

    # 7. 평가
    evaluate_model(model, valid_dataset, device)

    # 8. 예측
    texts = ["이 영화 정말 재미있다", "별로였다"]
    preds = predict(model, tokenizer, texts, device)
    print(preds)  # 예: [1, 0]
