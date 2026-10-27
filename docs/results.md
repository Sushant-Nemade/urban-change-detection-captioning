# Results and validation

## Software evidence

The current CPU environment has exercised the following behaviors:

- Original encoder/neck feature tensors, segmentation logits, and beam-search token sequences match the reorganized model. The captured fixture uses a bundled pretrained backbone and explicitly **untrained task heads**. It verifies numerical compatibility, not prediction quality.
- Decoder gradients, full model execution, mask restoration, dataset loading, temporal reversal, output files, UI error handling, and HTTP inference execute successfully.
- A tiny CPU caption fine-tuning run executes an optimizer step from a partial accumulation window, saves optimizer/scaler/RNG state, and resumes for a second epoch.

See the test suite and `docs/validation-report.json` for final recorded checks. Smoke-test metrics on synthetic image fixtures are not reported as model benchmark scores.

## Verified trained-model examples

`MCI_model_augmented.pth` was downloaded and verified against SHA-256 `5ca0b07c82d633a37e98d0a001247fb0283976209eb6c50cd56afdf33796ee46`. A fresh installed wheel ran CPU inference on two real original LEVIR-MCI test pairs in both directions, with beam size 5:

| Test pair | Input order | Generated caption, verbatim |
|---|---|---|
| `test_000004` | A → B | the woods disappear and a t-shaped road is built with villas built along |
| `test_000004` | B → A | trees are added , and a road with villas along it is removed |
| `test_000017` | A → B | trees are removed and a residential area has been built with rows of villas built along |
| `test_000017` | B → A | many houses are removed along the road |

The live HTTP service also produced a trained-model prediction on a real pair. These are functional and qualitative checks, not a complete accuracy evaluation. The reversed inputs are temporally reversed examples rather than independent historical removal observations. Input bytes and generated mask provenance are recorded in [the example manifest](assets/examples/provenance.json).

| Earlier image (`test_000017`) | Later image | Predicted mask |
|:---:|:---:|:---:|
| ![Earlier vegetation](assets/examples/test_000017/earlier.png) | ![Later residential area](assets/examples/test_000017/later.png) | ![Predicted road and building changes](assets/examples/test_000017/mask-chronological.png) |

No full-dataset benchmark is reported. Numerical values in historical checkpoint names are not a substitute for a reproducible evaluation report.

## Split-family provenance & zero-leakage audit

`tools/audit_split_provenance.py` audits the original `LevirCCcaptions.json` (`SHA-256: 8c8c3f494867b796306b98713e16ff72cbae1cf4a0a0c36931da16af9602daed`, 10,077 pairs) and augmented `LEVIRCCcaptions_fully_cleaned.json` (`SHA-256: cefb1699b228c95f45095b5f78edd879519e69d5034337c80fa7c0c006ca50d3`, 15,115 pairs) manifests alongside the ZIP central-directory CRC-32 and byte-size records, producing the machine-readable report at [docs/provenance_audit_report.json](provenance_audit_report.json):

| Split | Unique scene families | Original chronological pairs | Original changed (`changeflag=1`) | Original unchanged (`changeflag=0`) | Reversed pairs (`(T_2, T_1)`) | Total augmented entries | Cross-split leakage | Family manifest SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `train` | 6,815 | 6,815 | 3,407 | 3,408 | 3,407 | 10,222 | `0` | `e8d91b66e9409310...` |
| `val` | 1,333 | 1,333 | 667 | 666 | 667 | 2,000 | `0` | `9ebe2a8c8539ad7d...` |
| `test` | 1,929 | 1,929 | 964 | 965 | 964 | 2,893 | `0` | `000a4fdccfe2215a...` |
| **Total** | **10,077** | **10,077** | **5,038** | **5,039** | **5,038** | **15,115** | **`0`** | `0f7e8424137f6c5d...` |

- **1-to-1 Bijective Mapping Verified**: All 5,038 temporal-reversal entries (`imgid` `10077`–`15114`) map 1-to-1 to the 5,038 changed original pairs (`changeflag=1`) with exact `(A, B) <-> (B, A)` image swap and identical `label` mask verification (`all_changed_pairs_bijectively_mapped: true`).
- **Zero Cross-Split Leakage (`cross_split_leakage_count: 0`)**: Every reversed pair resides strictly inside the same split (`train`, `val`, or `test`) as its source geographic family (`pairwise_family_intersections: 0`).
- **Deterministic Partition Policies**: [docs/provenance_audit_report.json](provenance_audit_report.json) records SHA-256 split manifests for both `train_only_augmentation` (train=10,222, val=1,333, test=1,929) and `paired_evaluation_splits` (chronological vs. reversed slices tracked separately within each family's split).

## Directional-bias & mask-symmetry slice evaluation

`tools/evaluate_directional_consistency.py` evaluates temporal-order consistency on the checksum-verified real LEVIR-MCI test slice (`test_000004` paired with `test_002136.png`, and `test_000017` paired with `test_002137.png`, 4 directional inferences from checkpoint `5ca0b07c82d633a37e98d0a001247fb0283976209eb6c50cd56afdf33796ee46`), saving per-sample traces to [docs/directional_eval_report.json](directional_eval_report.json):

| Metric | Value | Scope / Interpretation |
|---|---:|---|
| **Mask Symmetry mIoU (corpus)** | `0.9767` | Consistency between predicted masks under `(T_1, T_2)` vs. `(T_2, T_1)` |
| **Mask Pixel Agreement Rate** | `0.9903` | `99.03%` pixel-level agreement across forward and reversed passes |
| **Per-Class Symmetry IoU** | `0.9859` / `0.9773` / `0.9669` | Unchanged (`0`) / Road (`1`) / Building (`2`) symmetry IoU |
| **Caption Polarity Inversion Rate** | `1.0000` | `2/2` paired test scenes invert urban addition (`built`) to removal (`removed`) |
| **Forward-Addition Bias Rate** | `0.0000` | Zero reversed predictions falsely defaulted to urban construction-only claims |

*Boundary note*: These metrics quantify directional mask symmetry and caption polarity inversion on the 2 committed checksum-verified test pairs (`4` directional predictions). Full 964-pair test-split directional evaluation requires supplying the local 4.21 GB dataset archive and 522 MB checkpoint via `python tools/evaluate_directional_consistency.py --images ... --captions ... --checkpoint ...`.

## Auditable visual evidence renderer

`src/urban_change/visual_audit.py` (and top-level `src/visual_audit.py`) extends CLI (`urban-change predict`), Streamlit, and FastAPI (`/infer`) outputs so every prediction exports:
- `audit_card.png`: 4-panel visual evidence report card (`T_1` Earlier, `T_2` Later, Road/Building Change Mask, and alpha-blended `T_2` Overlay with connected-component bounding boxes, pixel-area change percentages, caption, and checkpoint SHA-256).
- `audit_report.json` & enriched `prediction.json`: connected-component bounding-box coordinates (`bbox_xyxy`, `bbox_xywh`, `area_pixels`, `centroid_xy`) and pixel-area percentages (`unchanged_pct`, `road_change_pct`, `building_change_pct`, `total_changed_pct`).

## Removal/deconstruction illustration

The author’s montage, `Removal Case Sample Images.png`, is currently available only at a Windows path supplied in the request. It has not been received in this cloud workspace and is not included or reconstructed. Once supplied, add it under `docs/assets/`, record the source checkpoint and image IDs, and label observed inputs, ground truth, masks, and generated captions clearly.

The README’s source images come from the original LEVIR-MCI archive and are attributed separately. Its displayed masks/captions are actual predictions from the verified checkpoint; they are not the author’s missing montage.

## Remaining evaluation

Run the [evaluation protocol](evaluation.md) on the complete archive and exact vocabulary, reporting original chronological and reversed pairs separately. GPU training, mixed precision, full-dataset accuracy, optional Java METEOR, and broad removal-case reliability remain unverified in this environment.
