import io
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

from config import CORPUS_PATH, TOKENIZER_DIR, VOCAB_SIZE, LOWER_CASE


def preview_corpus(path: Path, n_lines: int = 5) -> None:
    """코퍼스 앞부분 몇 줄 출력."""
    assert path.exists(), f"코퍼스 파일을 찾을 수 없습니다: {path.resolve()}"
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= n_lines - 1:
                break
    print("\n... (앞부분 일부만 출력)\n")


def train_wordpiece_tokenizer() -> BertTokenizerFast:
    """WordPiece 토크나이저 학습 후 HF 토크나이저로 래핑."""
    preview_corpus(CORPUS_PATH)

    tokenizer_trainer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=LOWER_CASE,
        lowercase=LOWER_CASE,
    )

    tokenizer_trainer.train(
        files=[str(CORPUS_PATH)],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        limit_alphabet=1000,
        wordpieces_prefix="##",
        special_tokens=[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        ],
    )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_trainer.save_model(str(TOKENIZER_DIR))
    vocab_file = TOKENIZER_DIR / "vocab.txt"

    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_file),
        do_lower_case=LOWER_CASE,
        do_basic_tokenize=True,
    )
    tokenizer.save_pretrained(TOKENIZER_DIR)

    print("HF BertTokenizerFast 저장 완료:", TOKENIZER_DIR.resolve())
    return tokenizer


if __name__ == "__main__":
    tok = train_wordpiece_tokenizer()
    print(tok.tokenize("이 문장은 한국어 BERT 토크나이저 테스트입니다."))
