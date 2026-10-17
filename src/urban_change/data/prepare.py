"""Reproducible metadata generation without validation/test vocabulary leakage."""

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from urban_change.data.text import SPECIAL_TOKENS, load_vocab, tokenize


def prepare_metadata(
    captions_path: str | Path,
    output: str | Path,
    vocab_path: str | Path | None = None,
    min_count: int = 5,
    overwrite: bool = False,
) -> dict:
    if min_count < 0:
        raise ValueError("Minimum token count must be nonnegative")
    data = json.loads(Path(captions_path).read_text(encoding="utf-8"))
    entries = data.get("images", [])
    if not entries:
        raise ValueError("Caption JSON has no images")
    seen = set()
    prepared = []
    frequencies = Counter()
    for entry in entries:
        split, filename = entry["split"], entry["filename"]
        if (
            split not in {"train", "val", "test"}
            or Path(filename).name != filename
            or "\\" in filename
        ):
            raise ValueError(f"Invalid sample identity: {split}/{filename}")
        # Token files use the image stem, so stems must also be unique.
        stem = Path(filename).stem
        if stem in seen:
            raise ValueError(f"Duplicate image stem: {stem}")
        seen.add(stem)
        captions = [tokenize(sentence["raw"]) for sentence in entry["sentences"]]
        if not captions or any(len(tokens) <= 2 for tokens in captions):
            raise ValueError(f"Empty captions for {filename}")
        prepared.append((split, filename, captions))
        if split == "train":
            for tokens in captions:
                frequencies.update(tokens)
    if vocab_path is not None:
        vocab = load_vocab(vocab_path)
    else:
        vocab = dict(SPECIAL_TOKENS)
        # Preserve the original strict > threshold convention for compatibility.
        for token, count in sorted(frequencies.items()):
            if count > min_count and token not in vocab:
                vocab[token] = len(vocab)
    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Metadata output already exists: {destination}; use --overwrite")
    if destination.exists() and not (destination / "metadata.json").is_file():
        raise ValueError("Refusing to replace a directory not created by metadata preparation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".metadata-", dir=destination.parent))
    try:
        (staged / "tokens").mkdir()
        for split, filename, captions in prepared:
            (staged / "tokens" / f"{Path(filename).stem}.txt").write_text(
                json.dumps(captions) + "\n", encoding="utf-8"
            )
        for split in ("train", "val", "test"):
            names = sorted(
                filename for entry_split, filename, _ in prepared if entry_split == split
            )
            (staged / f"{split}.txt").write_text(
                "".join(name + "\n" for name in names), encoding="utf-8"
            )
        (staged / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")
        report = {
            "samples": len(prepared),
            "splits": {
                s: sum(split == s for split, _, _ in prepared) for s in ("train", "val", "test")
            },
            "vocab_size": len(vocab),
            "vocabulary_source": str(vocab_path) if vocab_path else "training split only",
            "maximum_caption_length": max(len(c) for _, _, captions in prepared for c in captions),
            "minimum_count_exclusive": min_count,
        }
        (staged / "metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # Existing metadata is replaced only after the new version is complete.
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=".metadata-backup-", dir=destination.parent))
            backup.rmdir()
            destination.replace(backup)
            try:
                staged.replace(destination)
            except BaseException:
                backup.replace(destination)
                raise
            else:
                shutil.rmtree(backup)
        else:
            staged.replace(destination)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return report
