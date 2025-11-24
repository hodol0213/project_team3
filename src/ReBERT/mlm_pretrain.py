import math
from itertools import chain

import torch
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from config import (
    CORPUS_PATH, TOKENIZER_DIR, MODEL_DIR,
    BLOCK_SIZE, MLM_PROB, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY,
    LOGGING_STEPS, SAVE_STEPS, SEED,
    HIDDEN_SIZE, NUM_HIDDEN_LAYERS, NUM_ATTENTION_HEADS,
    INTERMEDIATE_SIZE, MAX_POS_EMBED,
)


def get_tokenizer() -> BertTokenizerFast:
    """학습된 WordPiece 기반 HF 토크나이저 로드."""
    return BertTokenizerFast.from_pretrained(TOKENIZER_DIR)


def build_mlm_dataset(tokenizer: BertTokenizerFast):
    """텍스트 파일을 로드해서 MLM 학습용 dataset으로 변환."""
    raw_datasets = load_dataset("text", data_files=str(CORPUS_PATH))

    def _is_valid(example):
        t = example["text"]
        return isinstance(t, str) and t.strip() != ""

    raw_datasets = raw_datasets.filter(_is_valid)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=BLOCK_SIZE,
            return_special_tokens_mask=True,
        )

    remove_cols = raw_datasets["train"].column_names

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_cols,
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result = {
            k: [t[i:i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated.items()
        }
        return result

    lm_datasets = tokenized.map(
        group_texts,
        batched=True,
    )

    train_dataset = lm_datasets["train"]
    print(train_dataset)
    return train_dataset


def build_mlm_model(vocab_size: int) -> BertForMaskedLM:
    """작은 BERT 구조 정의 후 MLM 모델 생성."""
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_POS_EMBED,
        type_vocab_size=2,
        pad_token_id=0,
    )
    model = BertForMaskedLM(config)
    print("모델 파라미터 수:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    return model


def print_gpu_info():
    if torch.cuda.is_available():
        print("CUDA 사용 가능:", torch.cuda.is_available())
        print("GPU 이름:", torch.cuda.get_device_name(0))
        print("현재 GPU 메모리 사용량 (MB):", torch.cuda.memory_allocated() / 1024 ** 2)
    else:
        print("CUDA 사용 불가, CPU 사용")


def train_mlm():
    print_gpu_info()

    tokenizer = get_tokenizer()
    train_dataset = build_mlm_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    model = build_mlm_model(tokenizer.vocab_size)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=EPOCHS,
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
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    train_loss = train_result.training_loss
    ppl = math.exp(train_loss)
    print(f"최종 train loss: {train_loss:.4f} / ppl: {ppl:.2f}")


if __name__ == "__main__":
    train_mlm()
