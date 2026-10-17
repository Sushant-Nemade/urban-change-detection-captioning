---
license: mit
language:
  - en
task_categories:
  - image-segmentation
  - image-to-text
tags:
  - remote-sensing
  - urban-change
  - change-captioning
  - temporal-reversal
---

# LEVIR-MCI Augmented — Urban Change Detection & Change Captioning

This repository hosts LEVIR-MCI artifacts and an augmented archive published by **Sushant Nemade**. It supports research on pixel-level urban change detection and natural-language change captioning, including temporally reversed pairs for removal/deconstruction experiments.

The original imagery, masks, and captions originate from **LEVIR-MCI / Change-Agent** by Liu et al. This is an extension of that dataset, not a newly collected independent imagery source.

Project code: [urban-change-detection-captioning](https://github.com/Sushant-Nemade/urban-change-detection-captioning). Related checkpoint: [sushantnemade/MCI_augmented](https://huggingface.co/sushantnemade/MCI_augmented).

## Files and identity

Observed revision: `3844bd00748c4d79af90e7a002f919b0dd20bb13`.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `LEVIR-MCI-augmented_dataset.zip` | 4,212,576,803 | `eebea05866266496742fc31ad63327237e812cda8b3fdd1720123c839d1701c9` |
| `LEVIR-MCI-dataset.zip` | 2,771,943,190 | `8f7d52298fa3ca32aec983e84addef80f61ad7d981c0ff97a92c5880c4b83e9d` |

These identities come from Hub LFS metadata. Archive indices and annotation JSON were read through verified HTTPS ranges with ZIP-member CRC checks. Complete archive SHA-256 checks and per-image content audits were not performed.

| Archive | Train | Validation | Test | Total pairs |
|---|---:|---:|---:|---:|
| Original | 6,815 | 1,333 | 1,929 | 10,077 |
| Augmented | 10,222 | 2,000 | 2,893 | 15,115 |

The annotation counts agree with A-image entries in each ZIP. Every record has five captions, and image IDs are unique within each annotation file. The original JSON is `LevirCCcaptions.json`; the augmented JSON is `LEVIRCCcaptions_fully_cleaned.json`. No explicit per-record augmentation provenance was found. Generated-caption review status remains undocumented.

## Sample structure

The expected normalized interface uses `images/{train,val,test}/{A,B,label}/` and a caption JSON with an `images` list. Records include split, filename, image ID, change flag, and a `sentences` list. The preparation command accepts actual extracted paths; archive-internal directory names should be checked after extraction.

A is earlier and B later. Labels encode unchanged, road change, and building change. Original grayscale values are 0/128/255; the project maps them to class indices 0/1/2. `label_rgb` is a visualization and should not be used as an indexed training mask.

## Annotation provenance and reversal

Original LEVIR-MCI provides multiple human-written descriptions per pair. A reversed sample exchanges A/B, preserves the changed-object mask, and rewrites captions to reverse the temporal meaning while retaining counts, categories, and spatial relationships.

The historical augmentation code uses Azure OpenAI and includes a heuristic fallback. Therefore reversed captions must not all be labeled human-written or treated as independently validated. Human review and per-sample generation provenance remain to be documented for the existing hosted archive.

The new code supports reviewed local caption maps or explicitly marked Azure-generated rewrites. It records source/reversed identities and split membership, derives unique IDs from existing annotations, and aborts on rewrite failure instead of silently falling back. These safeguards apply to new runs and do not retroactively audit the archive.

## Splits and leakage

Keep each original pair and all its reversals in the same split. New augmentation defaults to training samples. Reversed validation/test pairs are correlated views of the same scenes; report them separately and do not count them as independent geographic observations. Build new vocabularies from training captions only; reuse the exact published vocabulary when evaluating an existing checkpoint.

## Use

```bash
urban-change download dataset-augmented --output data/raw/LEVIR-MCI-augmented_dataset.zip
# Extract and identify the actual images/ and caption JSON paths.
urban-change prepare --captions path/to/captions.json --output data/processed/levir_mci
urban-change validate-data --images path/to/images --metadata data/processed/levir_mci
```

Original dataset illustration:

![Original LEVIR-MCI examples](https://huggingface.co/datasets/sushantnemade/LEVIR-MCI-augmented/resolve/3844bd00748c4d79af90e7a002f919b0dd20bb13/dataset.png)

*The existing `dataset.png` originates from Change-Agent/LEVIR-MCI. It illustrates the original dataset; it is not the author’s removal-case result montage or evidence of augmented checkpoint performance.*

## Limitations and licensing

Temporal caption rewrites can invert the wrong event, lose an object/count, or introduce unsupported detail. Review originals and reversals together. Geographic and sensor coverage is inherited from LEVIR-MCI, not expanded by reversing observations.

The `license: mit` metadata above preserves the currently declared Hub value. It is not an independent confirmation of satellite imagery redistribution rights or all upstream asset terms. Confirm those terms and the scope of generated annotations before final release. Source-code licensing, including the SegFormer non-commercial restriction, is separate from the dataset’s declared metadata.

## Citation

Liu et al. *Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis.* IEEE TGRS, 2024. DOI: [10.1109/TGRS.2024.3425815](https://doi.org/10.1109/TGRS.2024.3425815). Original project: [Chen-Yang-Liu/Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent).
