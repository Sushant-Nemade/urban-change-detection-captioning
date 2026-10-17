# Data and temporal reversal

The source data is LEVIR-MCI, introduced with Change-Agent. The project’s augmented archive is hosted at [sushantnemade/LEVIR-MCI-augmented](https://huggingface.co/datasets/sushantnemade/LEVIR-MCI-augmented). ZIP indices and annotation JSON were inspected at revision `3844bd00748c4d79af90e7a002f919b0dd20bb13`:

| Archive | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Original | 6,815 | 1,333 | 1,929 | 10,077 |
| Augmented | 10,222 | 2,000 | 2,893 | 15,115 |

Both annotation files contain five captions per record and unique image IDs. No per-record `augmentation` provenance field was present. Counts agree with the ZIP indices for A images. This inspection used verified HTTPS ranges and ZIP-member CRC checks; it did not download or hash either entire archive, audit every image, or establish per-caption human review.

## Obtain and normalize the layout

```bash
urban-change download dataset-augmented --output data/raw/LEVIR-MCI-augmented_dataset.zip
```

After extraction, identify the actual image and caption JSON locations. The workflows accept explicit paths. A normalized layout is:

```text
data/raw/levir_mci/
  captions.json
  images/
    train/{A,B,label}/
    val/{A,B,label}/
    test/{A,B,label}/
```

Images and labels use the same filename within each split. `label_rgb` is for visualization, not training. Original grayscale labels are 0/128/255; indexed masks are 0/1/2.

## Generate metadata

```bash
urban-change prepare --captions data/raw/levir_mci/captions.json \
  --output data/processed/levir_mci
urban-change validate-data --images data/raw/levir_mci/images \
  --metadata data/processed/levir_mci
```

Preparation writes split lists, token JSON files, `vocab.json`, and `metadata.json`. It does not append to existing split lists. An existing output requires `--overwrite`; duplicate stems and malformed annotations fail. New vocabularies use **training captions only**, with tokens retained when their frequency is strictly greater than `--min-count` (default 5).

To evaluate a published checkpoint, preserve its vocabulary ordering rather than rebuilding it:

```bash
urban-change prepare --captions data/raw/levir_mci/captions.json \
  --vocab src/urban_change/assets/vocab/augmented.json \
  --output data/processed/levir_mci-published
```

The old 714-token vocabulary is retained for checkpoint compatibility. It is not presented as a newly leakage-free vocabulary; its historical construction used all provided caption entries. New training experiments should build their own vocabulary from training data and record it in the run.

## Create removal/deconstruction cases

Reversal exchanges `A` and `B`. Object-type change masks remain the same; captions must reverse the temporal meaning. A construction caption cannot simply be reused as a removal caption.

Use reviewed captions without an external service:

```json
{
  "train/train_000001.png": [
    "a building was removed .",
    "a building disappeared .",
    "one building is no longer present .",
    "a building was demolished .",
    "the building was removed from the area ."
  ]
}
```

This is a schema example, not an annotation for a specific real pair. Keys must match the actual changed samples and the number of captions must match each source record.

```bash
urban-change augment --captions data/raw/levir_mci/captions.json \
  --images data/raw/levir_mci/images \
  --caption-map reviewed-reversals.json \
  --output data/raw/levir_mci-reversed
```

The destination contains originals plus reversals, a new caption JSON, and `augmentation.json` with source identities and split mappings. The source dataset is untouched. Training samples are selected by default; `--splits` makes any extension to validation/test explicit. Original and reversed versions remain in the same split. Do not report a reversed test set as an independent collection of geographic scenes.

For Azure generation, install `.[augmentation]`, configure the names in `.env.example`, and replace `--caption-map` with `--azure --env-file .env`. Generated captions are marked as requiring review. API failures abort the output transaction; the pipeline never silently substitutes a word-replacement fallback.

Historical augmentation scripts used fallback rewrites and fixed ID offsets. New IDs derive from existing annotation IDs, outputs use a separate directory, and source/reversal mappings are explicit. Historical archives need their own provenance review; these safeguards do not retroactively validate existing generated captions.
