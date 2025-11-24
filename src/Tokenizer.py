import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer


def train_wordpiece_tokenizer(
    file_path: str,
    dir_path: str = "./model/tokenizer",
    vocab_size: int = 30000,
    limit_alphabet: int = 3000,
    min_frequency: int = 5,
    lowercase: bool = False
):

    # WordPiece 토크나이저 학습 및 vocab 저장
    tokenizer = BertWordPieceTokenizer(lowercase=lowercase)

    tokenizer.train(
        files=file_path,
        vocab_size=vocab_size,
        limit_alphabet=limit_alphabet,
        min_frequency=min_frequency,
    )

    os.makedirs(dir_path, exist_ok=True)
    tokenizer.save_model(dir_path)


def init_bert_tokenizer(vocab_file_path: str, dir_path: str = "./model/tokenizer"):
    # 학습된 vocab 기반으로 BertTokenizer 초기화 및 저장
    tokenizer = BertTokenizer(
        vocab_file=vocab_file_path,
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]"
    )

    tokenizer.save_pretrained(dir_path)
    return tokenizer


if __name__ == "__main__":
    # 1. WordPiece 학습
    file_path = "./data/processed/model/pretraining_preprocessed.txt"
    dir_path = "./model/tokenizer"
    train_wordpiece_tokenizer(file_path, dir_path)

    # 2. BertTokenizer 초기화
    vocab_file_path = os.path.join(dir_path, "vocab.txt")
    tokenizer = init_bert_tokenizer(vocab_file_path, dir_path)
