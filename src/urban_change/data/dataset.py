"""A strict, portable LEVIR-MCI loader with explicit metadata and image roots."""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from urban_change.data.text import encode, load_vocab
from urban_change.inference import prepare_image


def read_labels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim == 3:
        if not np.array_equal(array[..., 0], array[..., 1]) or not np.array_equal(
            array[..., 1], array[..., 2]
        ):
            raise ValueError(
                f"Use indexed/grayscale label masks, not colored label_rgb masks: {path}"
            )
        array = array[..., 0]
    values = set(np.unique(array).tolist())
    if values.issubset({0, 1, 2}):
        return array.astype(np.int64)
    if not values.issubset({0, 128, 255}):
        raise ValueError(f"Unexpected mask values {sorted(values)} in {path}")
    return np.where(array == 255, 2, np.where(array == 128, 1, 0)).astype(np.int64)


def parse_record(record):
    filename, separator, index = record.rpartition("-")
    if separator and index.isdigit() and Path(filename).suffix:
        return filename, int(index)
    return record, None


class LevirMCIDataset(Dataset):
    def __init__(
        self,
        image_root: str | Path,
        metadata_root: str | Path,
        split: str,
        vocab: dict | None = None,
        max_length: int = 46,
        limit: int | None = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("Split must be train, val, or test")
        if max_length < 2:
            raise ValueError("Maximum caption length must be at least two")
        self.image_root, self.metadata_root = Path(image_root), Path(metadata_root)
        self.split, self.max_length = split, max_length
        self.vocab = vocab if vocab is not None else load_vocab(self.metadata_root / "vocab.json")
        names = (self.metadata_root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        self.records = [name.strip() for name in names if name.strip()]
        if not self.records:
            raise ValueError(f"No samples in {split}.txt")
        if len(set(self.records)) != len(self.records):
            raise ValueError(f"Duplicate records in {split}.txt; regenerate metadata")
        if limit is not None:
            if limit < 1:
                raise ValueError("Sample limit must be positive")
            self.records = self.records[:limit]
        for name in self.records:
            base, _ = parse_record(name)
            if Path(base).name != base or "\\" in base:
                raise ValueError(f"Split entries must be basenames: {name}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        filename, caption_index = parse_record(record)
        paths = {
            folder: self.image_root / self.split / folder / filename
            for folder in ("A", "B", "label")
        }
        images = []
        for folder in ("A", "B"):
            with Image.open(paths[folder]) as image:
                if image.size != (256, 256):
                    raise ValueError(f"Dataset images must be 256x256: {paths[folder]}")
                images.append(prepare_image(image)[0])
        labels = read_labels(paths["label"])
        if labels.shape != (256, 256):
            raise ValueError(f"Dataset masks must be 256x256: {paths['label']}")
        tokens_path = self.metadata_root / "tokens" / (Path(filename).stem + ".txt")
        captions = json.loads(tokens_path.read_text(encoding="utf-8"))
        if not captions:
            raise ValueError(f"No captions in {tokens_path}")
        all_tokens = np.zeros((len(captions), self.max_length), dtype=np.int64)
        lengths = np.zeros(len(captions), dtype=np.int64)
        for i, tokens in enumerate(captions):
            if not tokens or tokens[0] != "<START>" or tokens[-1] != "<END>":
                raise ValueError(f"Captions must contain START/END tokens: {tokens_path}")
            ids = encode(tokens, self.vocab)[: self.max_length]
            ids[-1] = self.vocab["<END>"]
            all_tokens[i, : len(ids)] = ids
            lengths[i] = len(ids)
        selected = (
            caption_index
            if caption_index is not None
            else random.randrange(len(captions))
            if self.split == "train"
            else 0
        )
        if not 0 <= selected < len(captions):
            raise ValueError(f"Caption index out of range: {record}")
        return {
            "image_a": images[0],
            "image_b": images[1],
            "labels": labels,
            "references": all_tokens,
            "reference_lengths": lengths,
            "tokens": all_tokens[selected],
            "length": lengths[selected],
            "name": filename,
        }
