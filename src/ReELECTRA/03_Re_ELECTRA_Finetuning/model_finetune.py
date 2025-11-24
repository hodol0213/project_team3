# model_finetune.py
import torch
from transformers import ElectraForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_model(pretrained_dir: str, num_labels: int = 2, device: str = "cpu"):
    model = ElectraForSequenceClassification.from_pretrained(pretrained_dir, num_labels=num_labels)
    model.to(device)
    return model

def compute_metrics_from_preds(all_preds, all_labels):
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return {"accuracy": acc, "f1": f1}

def evaluate_model_by_dataloader(model, dataset, data_collator=None, batch_size: int = 32, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    If dataset is a `datasets.Dataset`, ensure it has columns ['input_ids','attention_mask','labels'].
    data_collator is optional; if provided, use it to batch. Otherwise simple batch conversion.
    """
    model.eval()
    model.to(device)

    # Use DataLoader with collate_fn if provided
    if data_collator is not None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    else:
        # set dataset format to torch for direct batching
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_metrics_from_preds(all_preds, all_labels)
