"""One inference implementation for Python, CLI, Streamlit, and HTTP."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from urban_change.checkpoints import load_checkpoint
from urban_change.data.text import load_vocab
from urban_change.models.system import ChangeModel, config_from_checkpoint
from urban_change.runtime import resolve_device
from urban_change.visual_audit import render_visual_audit_card, summarize_mask_regions

MEAN = np.array([0.39073, 0.38623, 0.32989], dtype=np.float32) * 255
STD = np.array([0.15329, 0.14628, 0.13648], dtype=np.float32) * 255
PALETTE = np.array([[0, 0, 0], [255, 255, 0], [255, 0, 0]], dtype=np.uint8)
TARGET_SIZE = 256


def prepare_image(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.float32)
    if array.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
        array = cv2.resize(array, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = (array - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)


def decode_sequence(sequence, vocab: dict[str, int]) -> str:
    if sequence and isinstance(sequence[0], list):
        sequence = sequence[0]
    inverse = {index: token for token, index in vocab.items()}
    special = {vocab[key] for key in ("<NULL>", "<START>", "<END>")}
    return " ".join(inverse[index] for index in sequence if index not in special)


@dataclass
class Prediction:
    caption: str
    labels: np.ndarray
    mask_rgb: np.ndarray
    elapsed_ms: float
    checkpoint_sha256: str
    audit_card_rgb: np.ndarray | None = None
    audit_report: dict | None = None

    def metadata(self) -> dict:
        region_info = self.audit_report or summarize_mask_regions(self.labels)
        return {
            "caption": self.caption,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "checkpoint_sha256": self.checkpoint_sha256,
            "mask_shape": list(self.labels.shape),
            "classes": {"0": "unchanged", "1": "road change", "2": "building change"},
            "pixel_counts": region_info["pixel_counts"],
            "area_percentages": region_info["area_percentages"],
            "bounding_box_counts": region_info["bounding_box_counts"],
            "bounding_boxes": region_info["bounding_boxes"],
        }


class ChangePredictor:
    def __init__(
        self, checkpoint: str | Path, vocab: str | Path | None = None, device: str = "auto"
    ):
        self.device = resolve_device(device)
        checkpoint = Path(checkpoint).expanduser().resolve()
        state = load_checkpoint(checkpoint)
        size = int(state["decoder_dict"]["wdc.weight"].shape[0])
        if vocab is None and "vocab" in state:
            from urban_change.data.text import validate_vocab

            self.vocab = validate_vocab(state["vocab"])
            if len(self.vocab) != size:
                raise ValueError("Embedded vocabulary does not match decoder shape")
        else:
            self.vocab = load_vocab(vocab, size=size)
        self.model = ChangeModel(self.vocab, config_from_checkpoint(state))
        self.model.load_legacy_state(state)
        self.model.to(self.device).eval()
        with checkpoint.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()

    @torch.inference_mode()
    def predict(
        self, image_a: Image.Image, image_b: Image.Image, beam_size: int = 5, mask_size: str = "a"
    ) -> Prediction:
        if not 1 <= beam_size <= len(self.vocab):
            raise ValueError("Beam size must be between 1 and vocabulary size")
        if mask_size not in {"a", "b", "target"}:
            raise ValueError("Mask size must be a, b, or target")
        a = prepare_image(image_a).to(self.device)
        b = prepare_image(image_b).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        features_a, features_b, logits = self.model(a, b)
        if beam_size == 1:
            sequence = self.model.decoder.sample(features_a, features_b)
        else:
            sequence = self.model.decoder.sample_beam(features_a, features_b, k=beam_size)
        labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        size = {"a": image_a.size, "b": image_b.size, "target": (TARGET_SIZE, TARGET_SIZE)}[
            mask_size
        ]
        if labels.shape != (size[1], size[0]):
            labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - started) * 1000
        caption_text = decode_sequence(sequence, self.vocab)
        card_rgb, audit_meta = render_visual_audit_card(
            image_a=image_a,
            image_b=image_b,
            labels=labels,
            caption=caption_text,
            checkpoint_sha256=self.checkpoint_sha256,
        )
        return Prediction(
            caption_text,
            labels,
            PALETTE[labels],
            elapsed,
            self.checkpoint_sha256,
            audit_card_rgb=card_rgb,
            audit_report=audit_meta,
        )


def save_prediction(
    prediction: Prediction,
    output: str | Path,
    overwrite: bool = False,
    image_a: Image.Image | None = None,
    image_b: Image.Image | None = None,
) -> Path:
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {output}; choose another or use --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.labels).save(output / "labels.png")
    Image.fromarray(prediction.mask_rgb).save(output / "mask.png")

    card_rgb = prediction.audit_card_rgb
    audit_meta = prediction.audit_report
    if card_rgb is None or audit_meta is None:
        fallback_a = image_a if image_a is not None else Image.fromarray(prediction.mask_rgb)
        fallback_b = image_b if image_b is not None else Image.fromarray(prediction.mask_rgb)
        card_rgb, audit_meta = render_visual_audit_card(
            image_a=fallback_a,
            image_b=fallback_b,
            labels=prediction.labels,
            caption=prediction.caption,
            checkpoint_sha256=prediction.checkpoint_sha256,
        )
    Image.fromarray(card_rgb).save(output / "audit_card.png")
    (output / "audit_report.json").write_text(
        json.dumps(audit_meta, indent=2) + "\n", encoding="utf-8"
    )
    (output / "caption.txt").write_text(prediction.caption + "\n", encoding="utf-8")
    (output / "prediction.json").write_text(
        json.dumps(prediction.metadata(), indent=2) + "\n", encoding="utf-8"
    )
    return output
