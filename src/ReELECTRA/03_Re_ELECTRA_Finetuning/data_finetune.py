# data_finetune.py
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from config import CONFIG

def load_raw_df(path: str = CONFIG["FINETUNE_DATA_PATH"]):
    # tab-separated, two columns: document \t label (no header)
    df = pd.read_csv(path, sep="\t", names=["document", "label"], header=None)
    return df

def get_tokenizer(tokenizer_dir: str = CONFIG["TOKENIZER_DIR"]):
    return AutoTokenizer.from_pretrained(tokenizer_dir)

def prepare_datasets(tokenizer,
                     raw_df=None,
                     test_size: float = 0.2,
                     seed: int = CONFIG["RANDOM_SEED"],
                     max_length: int = CONFIG["MAX_LENGTH"],
                     num_proc: int = CONFIG["NUM_PROC_TOKENIZE"]):
    """
    Returns (train_dataset, valid_dataset) as `datasets.Dataset` objects (tokenized)
    """
    if raw_df is None:
        raw_df = load_raw_df()

    train_df, valid_df = train_test_split(raw_df, test_size=test_size, shuffle=True, random_state=seed)

    # tokenize via huggingface datasets mapping (we'll convert pandas to datasets first)
    def tokenize_batch(examples):
        return tokenizer(examples["document"], padding="max_length", truncation=True, max_length=max_length)

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))

    train_ds = train_ds.map(tokenize_batch, batched=True, num_proc=num_proc, remove_columns=["document", "__index_level_0__"])
    valid_ds = valid_ds.map(tokenize_batch, batched=True, num_proc=num_proc, remove_columns=["document", "__index_level_0__"])

    # ensure labels column exists and is int
    train_ds = train_ds.rename_column("label", "labels")
    valid_ds = valid_ds.rename_column("label", "labels")

    # set format will be handled by Trainer automatically; for manual dataloader we'll set later
    return train_ds, valid_ds
