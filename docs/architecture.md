# Architecture

This project belongs to geospatial computer vision and vision-language learning. It jointly solves dense semantic change detection and autoregressive captioning from two aligned observations.

## Data contract

RGB images use means `[0.39073, 0.38623, 0.32989]` and standard deviations `[0.15329, 0.14628, 0.13648]`. Inference resizes to 256 × 256 using OpenCV bilinear interpolation. Training images must already be 256 × 256; implicit resampling of annotations is rejected.

Classes are unchanged (0), road change (1), and building change (2). Original grayscale masks encode these as 0/128/255. Indexed 0/1/2 masks are also accepted. Colored visualization masks are rejected as labels.

## Model graph

The shared SegFormer MiT-B1 backbone exposes 64-, 128-, 320-, and 512-channel feature maps at spatial sizes 64², 32², 16², and 8² for a 256² input. Both images pass through the same weights.

The attentive neck has separate captioning and detection paths. Temporal interactions operate on high-level features, while detection fuses multiscale differences into three class logits per pixel. The transformer decoder predicts caption tokens from the paired visual features. Beam search supports one image pair at a time.

Bundled vocabularies preserve the published mappings: 468 tokens for the original model and 714 for the augmented checkpoint. Equal vocabulary size does not establish equal token ordering. New checkpoints embed the mapping; custom legacy models can use `--vocab`.

## Compatibility boundary

`encoder_dict`, `encoder_trans_dict`, and `decoder_dict` remain the checkpoint sections. Parameter names and numerical model computation are retained. Obsolete commented versions and duplicate entry points are removed. All three sections load strictly, preventing missing weights from silently yielding random predictions.

The former MMCV/MMSegmentation imports supplied logging and checkpoint helpers to an unused initialization method. The live graph uses ordinary PyTorch operations. Removing that coupling eliminates a legacy build dependency without replacing the model.

Preprocessing, vocabulary selection, device handling, prediction, and rendering live in one inference module used by CLI, Streamlit, and FastAPI. CPU fallback is explicit; requesting unavailable CUDA fails clearly.
