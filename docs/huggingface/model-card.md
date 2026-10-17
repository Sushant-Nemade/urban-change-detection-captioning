---
language:
  - en
tags:
  - pytorch
  - remote-sensing
  - change-detection
  - change-captioning
  - temporal-reversal
datasets:
  - sushantnemade/LEVIR-MCI-augmented
---

# Urban Change Detection & Change Captioning — Augmented MCI

This repository contains the augmented MCI checkpoint published by **Sushant Nemade**, for joint urban change detection and captioning from an ordered pair of satellite images. The model architecture is adapted from [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent), rather than introduced as an original architecture here.

Code and documentation: [urban-change-detection-captioning](https://github.com/Sushant-Nemade/urban-change-detection-captioning).

## Inputs and outputs

- Input: aligned RGB images A (earlier) and B (later), normalized and resized to 256 × 256.
- Detection output: unchanged (0), road change (1), and building change (2).
- Caption output: an English description of the transition from A to B.

The visualization uses black, yellow, and red respectively. The mask identifies changed object categories, not construction versus demolition. Caption direction depends on image order.

## Architecture and vocabulary

Shared SegFormer MiT-B1 encoder, a temporal attentive neck with detection and caption branches, and a transformer caption decoder. The release code supports the published augmented 714-token vocabulary and preserves the original 468-token mapping for compatible original models. The exact token mapping is required; size alone is insufficient for custom checkpoints.

## Checkpoint identity

| File | Bytes | SHA-256 |
|---|---:|---|
| `MCI_model_augmented.pth` | 522,409,063 | `5ca0b07c82d633a37e98d0a001247fb0283976209eb6c50cd56afdf33796ee46` |

Observed repository revision: `330a2385435db812bff4c0a85ed3e4c1455fbb45`. At that revision, the epoch-9 filename and `MCI_model_augmented.pth` have the same LFS hash. Filenames containing training metrics are not independent evidence of a reproduced test benchmark.

## Use

Install the project with matching PyTorch/torchvision builds, then:

```bash
urban-change download model --output artifacts/MCI_model_augmented.pth
urban-change predict --checkpoint artifacts/MCI_model_augmented.pth \
  --image-a earlier.png --image-b later.png --output outputs/pair
```

A full checkpoint contains encoder weights; no separate SegFormer download is required for inference. This is custom paired-image inference, not a standard single-image Transformers pipeline.

## Augmentation and training context

The repository studies temporal reversal: swapping before/after imagery and rewriting captions to turn additions into removals and vice versa. This exposes the model to removal/deconstruction language. The historical augmentation implementation used Azure OpenAI rewrites with a heuristic fallback; per-sample review and generation records are not fully documented for the hosted archive. Generated reversals must not be described as uniformly human-authored.

The reorganized training workflow supports staged joint/caption/detection optimization, vocabulary transplantation, and resumable experiments. These engineering changes do not retroactively establish the exact settings used for this existing checkpoint. Release the corresponding run configuration and split manifest with any benchmark claim.

## Evaluation status

The trained checkpoint above was downloaded, SHA-256 verified, and loaded by the packaged implementation. Two original LEVIR-MCI test pairs were evaluated in both directions using beam size 5. For `test_000004`, A→B produced “the woods disappear and a t-shaped road is built with villas built along”; B→A produced “trees are added , and a road with villas along it is removed”. These are qualitative examples, not a full-dataset benchmark or independent historical demolition observations.

Code regression and training/resume checks additionally use untrained compatibility heads to validate software behavior. Report benchmark scores only with checkpoint hash, dataset revision, exact split, vocabulary, decoding settings, and metric protocol.

## Intended use and limitations

Research on aligned urban remote-sensing image pairs and joint change interpretation. Results may degrade with different sensors, resolutions, geography, seasons, misregistration, or object types. Captions may omit or invent details. Removal performance must be assessed on reviewed cases; reversing imagery does not itself prove temporal reasoning.

## Licensing requires review

The existing Hub metadata declares Apache-2.0. However, the inherited SegFormer implementation and pretrained backbone originate from NVIDIA’s source license with a non-commercial-use limitation. The compatibility of a blanket Apache-2.0 declaration with those upstream components has not been established. This draft intentionally makes no new blanket license declaration. Resolve the scope and update Hub metadata consistently before publishing this replacement card as a final release.

Preserve [upstream notices](https://github.com/Sushant-Nemade/urban-change-detection-captioning/blob/main/THIRD_PARTY_NOTICES.md) and the [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/LICENSE).

## Citation

Liu et al. *Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis.* IEEE TGRS, 2024. DOI: [10.1109/TGRS.2024.3425815](https://doi.org/10.1109/TGRS.2024.3425815).
