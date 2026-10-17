"""Validate file completeness, annotation encoding, and split identities."""

import numpy as np

from urban_change.data.dataset import LevirMCIDataset, parse_record


def validate_dataset(
    image_root, metadata_root, splits=("train", "val", "test"), limit=None
) -> dict:
    summary = {}
    all_names = set()
    for split in splits:
        dataset = LevirMCIDataset(image_root, metadata_root, split, limit=limit)
        labels = set()
        tokens = unknown = truncated = captions = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            # Reversed pairs stay with their source split. Detect filename reuse
            # across different splits, excluding explicit per-caption records.
            identity = sample["name"]
            if identity in all_names:
                raise ValueError(f"Image reused across splits: {identity}")
            labels.update(np.unique(sample["labels"]).tolist())
            references = sample["references"]
            tokens += int((references != 0).sum())
            unknown += int((references == 1).sum())
            truncated += int((sample["reference_lengths"] == dataset.max_length).sum())
            captions += len(references)
        all_names.update(parse_record(record)[0] for record in dataset.records)
        summary[split] = {
            "samples_checked": len(dataset),
            "label_classes": sorted(labels),
            "captions": captions,
            "unknown_token_rate": unknown / max(tokens, 1),
            "captions_at_maximum_length": truncated,
        }
    return summary
