# api_app.py

import os
import io
import time
import json
import base64

import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer

# --- Constants: MUST match training ---

MEAN = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
STD  = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
TARGET_SIZE = 256
BACKBONE = "segformer-mit_b1"
LIST_PATH_DEFAULT = "./data/LEVIR_MCI"
AUG_VOCAB_NAME = "vocab"
ORIG_VOCAB_NAME = "vocab_original"


def to_rgb3(pil_img: Image.Image) -> np.ndarray:
    # Convert to 3-channel RGB (handles grayscale/RGBA)
    return np.array(pil_img.convert("RGB"), dtype=np.float32)


def resize_hwc(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def norm_to_chw(img_hwc: np.ndarray) -> np.ndarray:
    chw = np.transpose(img_hwc, (2, 0, 1))
    for c in range(3):
        chw[c] = (chw[c] - MEAN[c]) / STD[c]
    return chw


def select_vocab_for_ckpt(dec_sd, list_path: str):
    """
    Decide which vocab JSON to use based on checkpoint decoder's vocab size.
    714 -> augmented vocab.json, 468 -> original vocab_original.json.
    If vocab is missing, captions are disabled.
    """
    ck_vocab_size = int(dec_sd["wdc.weight"].shape[0])
    ck_max_len = int(dec_sd["position_encoding.pe"].shape[0])

    if ck_vocab_size == 468:
        vocab_name = ORIG_VOCAB_NAME
    else:
        vocab_name = AUG_VOCAB_NAME

    vocab_path = os.path.join(list_path, vocab_name + ".json")

    caption_enabled = True
    if not os.path.exists(vocab_path):
        caption_enabled = False
        word_vocab = {"<NULL>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    else:
        with open(vocab_path, "r") as f:
            word_vocab = json.load(f)
        if len(word_vocab) != ck_vocab_size:
            raise RuntimeError(
                f"Vocab size mismatch: {vocab_name} has {len(word_vocab)} vs ckpt {ck_vocab_size}"
            )
    return word_vocab, vocab_name, ck_vocab_size, ck_max_len, caption_enabled


class InferenceModel:
    def __init__(self, ckpt_path: str, list_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if not os.path.isdir(list_path):
            raise FileNotFoundError(f"List/vocab folder not found: {list_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path
        self.list_path = list_path

        # Load checkpoint and decoder config
        ckpt = torch.load(ckpt_path, map_location=self.device)
        dec_sd = ckpt["decoder_dict"]

        word_vocab, vocab_used, ck_vocab_size, ck_max_len, caption_enabled = select_vocab_for_ckpt(
            dec_sd, list_path
        )
        self.word_vocab = word_vocab
        self.idx_to_word = {v: k for k, v in word_vocab.items()}
        self.specials = {
            word_vocab["<START>"],
            word_vocab["<END>"],
            word_vocab["<NULL>"],
        }
        self.vocab_used = vocab_used
        self.caption_enabled = caption_enabled
        self.ck_max_len = ck_max_len

        # Build model graph
        self.encoder = Encoder(BACKBONE).eval().to(self.device)
        self.encoder_trans = AttentiveEncoder(
            train_stage=None,
            n_layers=3,
            feature_size=[16, 16, 512],
            heads=8,
            dropout=0.1,
        ).eval().to(self.device)
        self.decoder = DecoderTransformer(
            encoder_dim=512,
            feature_dim=512,
            vocab_size=len(word_vocab),
            max_lengths=ck_max_len,
            word_vocab=word_vocab,
            n_head=8,
            n_layers=1,
            dropout=0.1,
        ).eval().to(self.device)

        # Load weights
        self.encoder.load_state_dict(ckpt["encoder_dict"], strict=True)
        self.encoder_trans.load_state_dict(ckpt["encoder_trans_dict"], strict=False)
        self.decoder.load_state_dict(dec_sd, strict=True)

    def prepare_images(self, pilA: Image.Image, pilB: Image.Image):
        npA = to_rgb3(pilA)
        npB = to_rgb3(pilB)
        origA = (npA.shape[0], npA.shape[1])
        origB = (npB.shape[0], npB.shape[1])

        npA = resize_hwc(npA, TARGET_SIZE)
        npB = resize_hwc(npB, TARGET_SIZE)
        chwA = norm_to_chw(npA)
        chwB = norm_to_chw(npB)

        tenA = (
            torch.from_numpy(chwA)
            .float()
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )
        tenB = (
            torch.from_numpy(chwB)
            .float()
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )
        return tenA, tenB, origA, origB

    @torch.no_grad()
    def forward_once(self, imgA, imgB):
        feat1_list, feat2_list = self.encoder(imgA, imgB)
        feat1_cap, feat2_cap, seg_pre = self.encoder_trans(feat1_list, feat2_list)
        return feat1_cap, feat2_cap, seg_pre

    @torch.no_grad()
    def infer_caption(self, feat1, feat2, beam_size: int = 5) -> str:
        if not self.caption_enabled:
            return ""
        if beam_size > 1:
            seq = self.decoder.sample_beam(feat1, feat2, k=beam_size)
        else:
            seq = self.decoder.sample(feat1, feat2, k=1)

        if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
            seq = seq[0]
        tokens = [
            self.idx_to_word.get(i, "<UNK>") for i in seq if i not in self.specials
        ]
        return " ".join(tokens).strip()

    @torch.no_grad()
    def infer_mask(self, seg_pre, save_size=None):
        pred = np.argmax(seg_pre.detach().cpu().numpy(), axis=1)[0].astype(np.uint8)
        if save_size is not None and (pred.shape[0], pred.shape[1]) != save_size:
            pred = cv2.resize(
                pred,
                (save_size[1], save_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        h, w = pred.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[pred == 1] = (255, 255, 0)  # road -> yellow (RGB)
        rgb[pred == 2] = (255, 0, 0)    # building -> red (RGB)
        return pred, rgb

    def predict(self, pilA: Image.Image, pilB: Image.Image, save_mask_size: str = "A"):
        """
        Full prediction: returns (caption, mask_rgb, elapsed_ms).
        save_mask_size: 'target', 'A', 'B'.
        """
        imgA, imgB, origA, origB = self.prepare_images(pilA, pilB)
        t0 = time.time()
        feat1, feat2, seg_pre = self.forward_once(imgA, imgB)
        caption = self.infer_caption(feat1, feat2, beam_size=5)

        if save_mask_size == "A":
            save_hw = origA
        elif save_mask_size == "B":
            save_hw = origB
        else:
            save_hw = None
        _, mask_rgb = self.infer_mask(seg_pre, save_size=save_hw)
        elapsed_ms = (time.time() - t0) * 1000.0
        return caption, mask_rgb, elapsed_ms


# --- FastAPI app ---

app = FastAPI(title="Change Agent REST API", version="1.0.0")
MODEL: InferenceModel | None = None


@app.on_event("startup")
def load_model_on_startup():
    """
    Load model once when the server starts.
    Uses env vars:
      CHECKPOINT_PATH (required)
      LIST_PATH (optional; default ./data/LEVIR_MCI)
    """
    global MODEL
    ckpt_path = os.getenv("CHECKPOINT_PATH")
    list_path = os.getenv("LIST_PATH", LIST_PATH_DEFAULT)

    if ckpt_path is None:
        raise RuntimeError(
            "Environment variable CHECKPOINT_PATH is not set. "
            "Set it to the .pth checkpoint path (inside container)."
        )

    MODEL = InferenceModel(ckpt_path=ckpt_path, list_path=list_path)
    print(
        f"Model loaded from {ckpt_path} on device {MODEL.device}. "
        f"Vocab used: {MODEL.vocab_used}. Caption: {'ON' if MODEL.caption_enabled else 'OFF'}."
    )


@app.get("/health")
def health():
    device_str = str(MODEL.device) if MODEL is not None else "unknown"
    return {"status": "ok", "device": device_str}


@app.post("/infer")
async def infer(
    imageA: UploadFile = File(...),
    imageB: UploadFile = File(...),
    save_mask_size: str = "A",  # "target", "A", or "B"
):
    """
    Run change detection + captioning on two uploaded images.

    Returns:
      - caption: string
      - mask_png_base64: PNG of predicted mask (road=yellow, building=red) base64-encoded
      - elapsed_ms: inference time in milliseconds

    """
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    if save_mask_size not in {"target", "A", "B"}:
        raise HTTPException(status_code=400, detail="Invalid save_mask_size.")

    try:
        contentA = await imageA.read()
        contentB = await imageB.read()
        pilA = Image.open(io.BytesIO(contentA))
        pilB = Image.open(io.BytesIO(contentB))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading images: {e}")

    caption, mask_rgb, elapsed_ms = MODEL.predict(
        pilA, pilB, save_mask_size=save_mask_size
    )

    # Encode mask to PNG (RGB->BGR for OpenCV, then encode)
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", mask_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode mask PNG.")
    mask_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return JSONResponse(
        {
            "caption": caption,
            "elapsed_ms": round(elapsed_ms, 2),
            "mask_png_base64": mask_b64,
        }
    )