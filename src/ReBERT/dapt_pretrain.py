import math
from itertools import chain
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline,
)

from config import (
    MODEL_DIR, DAPT_OUT_DIR, DAPT_CORPUS_PATH,
    DAPT_BLOCK_SIZE, DAPT_MLM_PROB, DAPT_BATCH_SIZE,
    DAPT_EPOCHS, DAPT_LR,
    WARMUP_RATIO, WEIGHT_DECAY, LOGGING_STEPS, SAVE_STEPS, SEED,
)


def load_base_mlm_for_dapt():
    """기존 MLM 학습 완료 모델 로드."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)
    return tokenizer, model


def build_dapt_dataset(tokenizer):
    """도메인 코퍼스를 로드해서 DAPT 학습용 dataset으로 변환."""
    raw = load_dataset("text", data_files=str(DAPT_CORPUS_PATH))["train"]

    def _valid(ex):
        t = ex["text"]
        return isinstance(t, str) and t.strip() != ""

    raw = raw.filter(_valid)

    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=DAPT_BLOCK_SIZE,
            return_special_tokens_mask=True,
        )

    tok = raw.map(tok_fn, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // DAPT_BLOCK_SIZE) * DAPT_BLOCK_SIZE
        return {
            k: [t[i:i + DAPT_BLOCK_SIZE] for i in range(0, total_length, DAPT_BLOCK_SIZE)]
            for k, t in concatenated.items()
        }

    train_dataset = tok.map(group_texts, batched=True)
    print(train_dataset)
    return train_dataset


def train_dapt():
    tokenizer, model = load_base_mlm_for_dapt()
    train_dataset = build_dapt_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=DAPT_MLM_PROB,
    )

    args = TrainingArguments(
        output_dir=str(DAPT_OUT_DIR),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=DAPT_BATCH_SIZE,
        learning_rate=DAPT_LR,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=DAPT_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        prediction_loss_only=True,
        fp16=True,
        gradient_checkpointing=True,
        seed=SEED,
        dataloader_num_workers=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    train_result = trainer.train()
    trainer.save_model(DAPT_OUT_DIR)
    tokenizer.save_pretrained(DAPT_OUT_DIR)

    train_loss = train_result.training_loss
    ppl = math.exp(train_loss)
    print(f"[DAPT] train loss: {train_loss:.4f} / ppl: {ppl:.2f}")

    mask_filler = pipeline(
        "fill-mask",
        model=str(DAPT_OUT_DIR),
        tokenizer=str(DAPT_OUT_DIR),
        device_map="auto",
    )
    test_sentence = "배송도 빠르고 [MASK]대비 상품도 괜찮았습니다"
    for pred in mask_filler(test_sentence):
        print(f"{pred['sequence']}  |  score={pred['score']:.4f}")


if __name__ == "__main__":
    train_dapt()
