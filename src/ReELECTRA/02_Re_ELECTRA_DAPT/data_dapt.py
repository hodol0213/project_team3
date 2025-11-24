from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from config import CONFIG

def get_tokenizer(tokenizer_dir: str = CONFIG["TOKENIZER_DIR"]):
    return AutoTokenizer.from_pretrained(tokenizer_dir)

def build_datasets_and_collator(
    file_path: str = CONFIG["DAPT_DATA_PATH"],
    tokenizer=None,
    max_length: int = CONFIG["MAX_LENGTH"],
    mlm_prob: float = CONFIG["MLM_PROB"],
    num_proc: int = CONFIG["NUM_PROC"]
):
    assert tokenizer is not None, "tokenizer must be provided"

    ds = load_dataset("text", data_files=file_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = ds.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
    )

    split = tokenized["train"].train_test_split(test_size=0.2)
    tokenized = DatasetDict({"train": split["train"], "validation": split["test"]})

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob
    )

    return tokenized, data_collator
