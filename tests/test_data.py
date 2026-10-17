import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from urban_change.data.augment import augment_dataset
from urban_change.data.dataset import LevirMCIDataset, parse_record, read_labels
from urban_change.data.prepare import prepare_metadata
from urban_change.data.text import encode, load_vocab, tokenize, validate_vocab
from urban_change.data.validate import validate_dataset


@pytest.mark.parametrize("size", [468, 714])
def test_published_vocabularies(size):
    vocab = load_vocab(size=size)
    assert len(vocab) == size
    assert sorted(vocab.values()) == list(range(size))


def test_tokenization_and_unknown_tokens():
    assert tokenize("  a   building, appeared. ") == [
        "<START>",
        "a",
        "building",
        ",",
        "appeared",
        "<END>",
    ]
    assert tokenize("") == ["<START>", "<END>"]
    vocab = load_vocab(size=714)
    assert encode(["not_in_vocabulary"], vocab) == [1]
    with pytest.raises(KeyError):
        encode(["not_in_vocabulary"], vocab, allow_unk=False)
    with pytest.raises(ValueError):
        validate_vocab({"<NULL>": 0, "a": 0})


def test_metadata_uses_training_vocabulary_and_is_repeatable(dataset_fixture, tmp_path):
    fixture = dataset_fixture
    vocab = load_vocab(fixture["metadata"] / "vocab.json")
    assert "building" in vocab and "evaluationonly" not in vocab
    before = {
        p.relative_to(fixture["metadata"]): p.read_bytes()
        for p in fixture["metadata"].rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        prepare_metadata(fixture["captions"], fixture["metadata"])
    prepare_metadata(fixture["captions"], fixture["metadata"], min_count=0, overwrite=True)
    after = {
        p.relative_to(fixture["metadata"]): p.read_bytes()
        for p in fixture["metadata"].rglob("*")
        if p.is_file()
    }
    assert before == after
    other = tmp_path / "reused"
    prepare_metadata(fixture["captions"], other, vocab_path=fixture["metadata"] / "vocab.json")
    assert load_vocab(other / "vocab.json") == vocab


def test_duplicate_annotations_rejected(dataset_fixture, tmp_path):
    source = json.loads(dataset_fixture["captions"].read_text())
    source["images"].append(source["images"][0])
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_metadata(path, tmp_path / "bad-metadata")


def test_overwrite_protects_unrelated_directories(dataset_fixture, tmp_path):
    output = tmp_path / "user-files"
    output.mkdir()
    (output / "keep.txt").write_text("existing work")
    with pytest.raises(ValueError, match="Refusing"):
        prepare_metadata(dataset_fixture["captions"], output, overwrite=True)
    assert (output / "keep.txt").read_text() == "existing work"


def test_filename_hyphens_and_legacy_caption_indices():
    assert parse_record("train_city-block.png") == ("train_city-block.png", None)
    assert parse_record("train_city-block.png-3") == ("train_city-block.png", 3)


def test_dataset_paths_normalization_and_truncation(dataset_fixture):
    dataset = LevirMCIDataset(
        dataset_fixture["images"], dataset_fixture["metadata"], "train", max_length=4
    )
    sample = dataset[0]
    assert sample["image_a"].shape == (3, 256, 256)
    assert not np.array_equal(sample["image_a"], sample["image_b"])
    assert set(np.unique(sample["labels"])) == {0, 1, 2}
    assert sample["references"].shape == (5, 4)
    assert (sample["references"][:, -1] == 3).all()
    assert sample["length"] == 4
    report = validate_dataset(dataset_fixture["images"], dataset_fixture["metadata"])
    assert report["train"]["samples_checked"] == 2
    assert report["val"]["unknown_token_rate"] > 0


def test_missing_tokens_and_duplicate_split_fail(dataset_fixture):
    fixture = dataset_fixture
    filename = fixture["entries"][0]["filename"]
    (fixture["metadata"] / "tokens" / f"{Path(filename).stem}.txt").unlink()
    with pytest.raises(FileNotFoundError):
        LevirMCIDataset(fixture["images"], fixture["metadata"], "train")[0]
    split = fixture["metadata"] / "train.txt"
    split.write_text(split.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        LevirMCIDataset(fixture["images"], fixture["metadata"], "train")


def test_invalid_label_encoding_is_not_silently_remapped(tmp_path):
    path = tmp_path / "labels.png"
    Image.fromarray(np.full((4, 4), 17, dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="Unexpected mask"):
        read_labels(path)


def test_reversal_preserves_source_split_and_mask(dataset_fixture, tmp_path):
    fixture = dataset_fixture
    entry = fixture["entries"][0]
    mapping = {f"train/{entry['filename']}": ["a building was removed ."] * 5}
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture["images"].rglob("*.png")
    }
    output = tmp_path / "augmented"
    report = augment_dataset(fixture["captions"], fixture["images"], output, caption_map=mapping)
    assert report["original_samples"] == 4 and report["reversed_samples"] == 1
    augmented = json.loads((output / "captions.json").read_text())["images"]
    assert len(augmented) == 5
    reversed_entry = augmented[-1]
    assert reversed_entry["split"] == "train" and reversed_entry["imgid"] > 203
    assert all("removed" in s["raw"] for s in reversed_entry["sentences"])
    assert len({s["sentid"] for e in augmented for s in e["sentences"]}) == 25
    name = reversed_entry["filename"]
    assert (output / "images/train/A" / name).read_bytes() == (
        fixture["images"] / "train/B" / entry["filename"]
    ).read_bytes()
    assert (output / "images/train/label" / name).read_bytes() == (
        fixture["images"] / "train/label" / entry["filename"]
    ).read_bytes()
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    with pytest.raises(FileExistsError):
        augment_dataset(fixture["captions"], fixture["images"], output, caption_map=mapping)


def test_reversal_failure_has_no_silent_caption_fallback(dataset_fixture, tmp_path):
    def failing_rewriter(text):
        raise RuntimeError("Model service unavailable")

    output = tmp_path / "failed-augmentation"
    with pytest.raises(RuntimeError, match="unavailable"):
        augment_dataset(
            dataset_fixture["captions"], dataset_fixture["images"], output, rewrite=failing_rewriter
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".augmentation-*"))
