"""Deterministic tokenization compatible with the published vocabularies."""

import json
from importlib.resources import files
from pathlib import Path

SPECIAL_TOKENS = {"<NULL>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}


def tokenize(text: str, add_start_token: bool = True, add_end_token: bool = True) -> list[str]:
    # Retain the training pipeline's punctuation policy and case. Remove empty
    # fields without mutating a list while iterating, as the old tokenizer did.
    for punctuation in (";", ","):
        text = text.replace(punctuation, " " + punctuation)
    for punctuation in ("?", "."):
        text = text.replace(punctuation, "")
    tokens = text.split()
    return (["<START>"] if add_start_token else []) + tokens + (["<END>"] if add_end_token else [])


def validate_vocab(vocab: dict) -> dict[str, int]:
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Vocabulary must be a nonempty token-to-index mapping")
    if any(not isinstance(k, str) or type(v) is not int for k, v in vocab.items()):
        raise ValueError("Vocabulary tokens must be strings and indices must be integers")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary indices must be unique and contiguous from zero")
    if any(vocab.get(token) != index for token, index in SPECIAL_TOKENS.items()):
        raise ValueError(f"Vocabulary must use the special tokens {SPECIAL_TOKENS}")
    return vocab


def load_vocab(path: str | Path | None = None, size: int | None = None) -> dict[str, int]:
    if path is None:
        names = {468: "original.json", 714: "augmented.json"}
        if size not in names:
            raise ValueError(f"No bundled vocabulary with {size} tokens; supply --vocab")
        source = files("urban_change").joinpath("assets", "vocab", names[size])
        vocab = json.loads(source.read_text(encoding="utf-8"))
    else:
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_vocab(vocab)
    if size is not None and len(vocab) != size:
        raise ValueError(f"Vocabulary has {len(vocab)} tokens; checkpoint requires {size}")
    return vocab


def encode(tokens: list[str], vocab: dict[str, int], allow_unk: bool = True) -> list[int]:
    return [vocab.get(token, vocab["<UNK>"]) if allow_unk else vocab[token] for token in tokens]
