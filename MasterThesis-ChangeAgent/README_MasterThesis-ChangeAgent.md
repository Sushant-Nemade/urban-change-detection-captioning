## LEVIR-MCI Change Detection & Captioning

Bi-temporal urban change detection and captioning on LEVIR-MCI with a dual-head model (SegFormer encoder + transformer decoder). Supports both the original LEVIR-MCI vocabulary (468 tokens) and an augmented version (714 tokens) that includes swapped A/B pairs and caption reversal.

## Requirements

- Python 3.10+ (recommended)
- GPU with CUDA for training (CPU works but is slow)
- Key libraries: PyTorch, mmcv/mmsegmentation, timm, einops, imageio, opencv-python, scikit-image, tqdm, nltk, pycocotools, streamlit

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r Multi_change/requirements.txt
```

SegFormer pretrained weights (at least `mit_b1.pth`) must be placed here:

```
Multi_change/model/pretrained/mit_b0.pth   # if used
Multi_change/model/pretrained/mit_b1.pth   # default backbone
Multi_change/model/pretrained/mit_b2.pth   # if used
```

## Data Preparation

Expected LEVIR-MCI layout (adapt `DATA_PATH_ROOT` accordingly):

```
<DATA_PATH_ROOT>/Levir-MCI-dataset/
	images/
		train/ A/ B/ label/ label_rgb/
		val/   A/ B/ label/ label_rgb/
		test/  A/ B/ label/ label_rgb/
```

Local augmented drop (extension of [LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main)):

- Multi_change/LEVIR_MCI-data/LEVIR-MCI-dataset

Set `DATA_PATH_ROOT` inside:

- Multi_change/augment_dataset_new.py
- Multi_change/preprocess_data.py

Captions:

- LevirCCcaptions.json (original) and LevirCCcaptions_augmented.json (augmented) should reside under Multi_change/data/LEVIR_MCI/.

## Dataset Augmentation (augment_dataset_new.py)

What it does:

- For change pairs (`changeflag=1`), swaps A/B images and rewrites captions (addition ↔ removal) via Azure OpenAI GPT.
- Copies A/B/label/label_rgb with new IDs/filenames.
- Produces LevirCCcaptions_augmented.json plus swapped_pairs.xlsx/CSV.

Environment variables (put in `.env`):

```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_RESOURCE=...
LLM_DEPLOYMENT=...
```

Run augmentation:

```bash
cd Multi_change
python augment_dataset_new.py .env
```

Behavior:

- If LevirCCcaptions_augmented.json already exists, the script will skip/overwrite based on its internal checks (adjust if you want forced regeneration).

## Preprocessing & Vocabulary (preprocess_data.py)

What it does:

- Reads LevirCCcaptions_augmented.json.
- Tokenizes captions with `<START>` / `<END>`.
- Writes token files under data/LEVIR_MCI/tokens/*.txt and splits (train.txt, val.txt, test.txt).
- Builds data/LEVIR_MCI/vocab.json (714-token augmented vocab).

Command:

```bash
cd Multi_change
python preprocess_data.py --dataset LEVIR_MCI --word_count_threshold 5
```

Also ensure the original vocab (468 tokens) is present:

```
data/LEVIR_MCI/vocab_original.json
```

## Training & Fine-Tuning (train_newoptimized.py)

Key arguments:

- `--train_stage`: s1 (pretrain) or s2 (finetune)
- `--train_goal`: 0=detection, 1=caption, 2=joint
- `--data_folder`, `--list_path`, `--token_folder`, `--vocab_file`, `--max_length`
- `--gpu_id`, `--batch_size`, `--num_epochs`, `--encoder_lr`, `--decoder_lr`
- Optional: `--use_loss_balance 1` for joint loss balance, `--checkpoint` to resume, `--cap_unfreeze_encoder_trans 1` to unfreeze encoder in caption finetuning.

Why stages (quick): s1 pretrains both heads jointly to build shared scene/change features; s2 caption-only sharpens language grounding; s2 detection-only (optional) sharpens masks; s2-fuse rejoins both heads with balanced losses to recover joint performance. Recommended schedule: s1 joint (goal=2) → s2 caption-only (goal=1) → s2 detection-only (goal=0, optional) → s2-fuse joint (goal=2 with loss balance); final reports use the best s2-fuse checkpoint.

### Train from scratch on augmented LEVIR-MCI (714 vocab)

Stage s1, joint training example:

```bash
cd Multi_change
python train_newoptimized.py \
	--train_stage s1 --train_goal 2 \
	--data_folder "<DATA_PATH_ROOT>/Levir-MCI-dataset/images" \
	--list_path "./data/LEVIR_MCI/" \
	--token_folder "./data/LEVIR_MCI/tokens/" \
	--vocab_file "vocab" \
	--max_length 46 \
	--num_epochs 30 \
	--encoder_lr 1e-4 --decoder_lr 2e-4 \
	--batch_size 8 \
	--gpu_id 0
```

Stage s2 examples (load an s1 checkpoint):

- Caption-only finetune:

```bash
python train_newoptimized.py \
	--train_stage s2 --train_goal 1 \
	--data_folder "<DATA_PATH_ROOT>/Levir-MCI-dataset/images" \
	--list_path "./data/LEVIR_MCI/" \
	--token_folder "./data/LEVIR_MCI/tokens/" \
	--vocab_file "vocab" \
	--max_length 46 \
	--checkpoint "checkpoints/s1_joint_best.pth" \
	--cap_unfreeze_encoder_trans 1 \
	--num_epochs 10 \
	--gpu_id 0
```

- Detection-only finetune:

```bash
python train_newoptimized.py \
	--train_stage s2 --train_goal 0 \
	--data_folder "<DATA_PATH_ROOT>/Levir-MCI-dataset/images" \
	--list_path "./data/LEVIR_MCI/" \
	--token_folder "./data/LEVIR_MCI/tokens/" \
	--vocab_file "vocab" \
	--max_length 46 \
	--checkpoint "checkpoints/s1_joint_best.pth" \
	--num_epochs 10 \
	--gpu_id 0
```

#### Stage s2 fuse (joint finetune)

Use an s1 joint checkpoint and keep both heads active with balanced losses:

```bash
python train_newoptimized.py \
	--train_stage s2 --train_goal 2 \
	--data_folder "<DATA_PATH_ROOT>/Levir-MCI-dataset/images" \
	--list_path "./data/LEVIR_MCI/" \
	--token_folder "./data/LEVIR_MCI/tokens/" \
	--vocab_file "vocab" \
	--max_length 46 \
	--checkpoint "checkpoints/s1_joint_best.pth" \
	--use_loss_balance 1 \
	--num_epochs 10 \
	--gpu_id 0
```

Hyperparameters:

- Batch size, learning rates, weight decay, and label smoothing are configurable via arguments; defaults are defined in the script. Adjust for your GPU memory (reduce batch size if OOM).

## Evaluation

Provided final checkpoint: `Multi_change/models_ckpt/MCI_model_augmented.pth` (renamed). Use it in the commands below or swap for your own.

### Single checkpoint (test_new.py)

Computes detection (mIoU/IoU/accuracy) and caption metrics (BLEU-1..4, METEOR, ROUGE-L, CIDEr). Options: `--beam_size`, `--det_tta`, `--save_mask`, `--save_caption`, `--result_path`.

Example:

```bash
cd Multi_change
python test_new.py \
	--data_folder "<DATA_PATH_ROOT>/Levir-MCI-dataset/images" \
	--list_path "./data/LEVIR_MCI/" \
	--token_folder "./data/LEVIR_MCI/tokens/" \
	--vocab_file "vocab" \
	--max_length 46 \
	--checkpoint "./models_ckpt/MCI_model_augmented.pth" \
	--beam_size 5 \
	--det_tta 0 \
	--save_mask --save_caption \
	--result_path "./predict_result" \
	--gpu_id 0
```

### Multiple checkpoints (eval_many.py)

Reads checkpoint paths from a text file and writes a CSV of metrics and inference time.

Create `ckpt_list.txt` (one path per line), then run:

```bash
cd Multi_change
python eval_many.py \
	--data_folder "<DATA_PATH_ROOT>/Levir-MCI-dataset/images" \
	--list_path "./data/LEVIR_MCI/" \
	--token_folder "./data/LEVIR_MCI/tokens/" \
	--ckpt_list ckpt_list.txt \
	--vocab_file vocab \
	--orig_vocab_file vocab_original \
	--beam_size 5 \
	--det_tta 0 \
	--split test \
	--out_csv results_eval_many.csv
```

## Inference CLI (predict_new.py)

What it does:

- Loads a single checkpoint; auto-detects vocab size (468 vs 714) and `max_length` from the checkpoint, picks the matching vocab JSON.
- Accepts arbitrary bi-temporal images (RGB/gray/RGBA), resizes to 256×256, normalizes, runs encoder/neck/decoder.
- Outputs a change mask (road=yellow, building=red) and a caption (if vocab is available).

Examples:

```bash
cd Multi_change
python predict_new.py \
	--imgA_path "samples/A.png" \
	--imgB_path "samples/B.png" \
	--checkpoint "./models_ckpt/MCI_model_augmented.pth" \
	--list_path "./data/LEVIR_MCI" \
	--mask_save_path "./CDmask.png" \
	--save_mask_size A \
	--gpu_id 0 \
	--beam_size 5
```

If the matching vocab JSON is missing, captioning is disabled but detection still runs.

## Streamlit App (streamlit_app.py)

Behavior:

- Sidebar: set checkpoint path, mask output size, and output directory.
- Main: upload Image A and B, click “Predict” to view A, B, mask, and caption; “Save Results” writes A/B/mask/caption to the chosen directory.
- Backbone fixed to segformer-mit_b1, input size 256, beam size 5; vocab auto-selected (714 vs 468) from checkpoint.

Run:

```bash
cd Multi_change
streamlit run streamlit_app.py
```

Typical usage: launch app → provide checkpoint → upload A/B → predict → save outputs.

## Reproducibility & Notes

- Use GPU for training; CPU inference is possible but slower.
- Adjust all placeholder paths (`<DATA_PATH_ROOT>`, `checkpoints/...`, etc.) to your environment.
- Keep image normalization consistent with training (the code handles this; do not pre-normalize differently).
- If you change vocab or `max_length`, regenerate tokens and vocab before training/eval.

## Acknowledgement

Thanks to the following repository: [Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent.git)
