import csv
import pandas as pd
from sklearn.model_selection import train_test_split

from kcbert_finetune_config import DATA_PATH, SEED


def load_and_clean_data(path=DATA_PATH):
    """Fine_tuning_shopping.txt 로드 및 전처리"""
    assert path.exists(), f"File not found: {path}"

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label"],
        dtype={"text": str},
        engine="python",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        encoding_errors="ignore",
    )

    df = df.dropna(subset=["text"]).reset_index(drop=True)

    df["text"] = (
        df["text"]
        .str.replace(r"[\u200B-\u200D\uFEFF]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["label"] = (
        pd.to_numeric(df["label"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )

    df = df[df["text"].str.len() >= 3]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    print(df.head())
    print("Samples:", len(df))
    print("Label dist:\n", df["label"].value_counts())
    return df


def train_valid_split(df, test_size=0.1, seed=SEED):
    """Label 균형 고려하여 split"""
    label_counts = df["label"].value_counts()

    if label_counts.min() < 2:
        train_df, valid_df = train_test_split(df, test_size=test_size, random_state=seed)
    else:
        train_df, valid_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["label"]
        )

    print(f"Train: {len(train_df)} | Valid: {len(valid_df)}")
    return train_df, valid_df
