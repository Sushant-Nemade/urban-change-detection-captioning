"""Evaluate exactly the requested split, checkpoint, and sample limit."""

import hashlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from urban_change.data.dataset import LevirMCIDataset
from urban_change.evaluation.metrics import SegmentationMetrics, caption_metrics
from urban_change.inference import ChangePredictor, decode_sequence


@torch.inference_mode()
def evaluate_model(
    model, vocab, dataset, device, beam_size=5, goal="joint", meteor_dir=None
) -> dict:
    if not 1 <= beam_size <= len(vocab):
        raise ValueError("Invalid beam size")
    if goal not in {"joint", "detection", "caption"}:
        raise ValueError("Unknown evaluation goal")
    model.eval()
    segmentation = SegmentationMetrics()
    references, hypotheses = [], []
    count = 0
    started = time.perf_counter()
    for sample in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        features_a, features_b, logits = model(
            sample["image_a"].to(device), sample["image_b"].to(device)
        )
        if goal != "caption":
            segmentation.update(sample["labels"].numpy(), logits.argmax(1).cpu().numpy())
        if goal != "detection":
            sequence = (
                model.decoder.sample(features_a, features_b)
                if beam_size == 1
                else model.decoder.sample_beam(features_a, features_b, k=beam_size)
            )
            references.append(
                [decode_sequence(tokens.tolist(), vocab) for tokens in sample["references"][0]]
            )
            hypotheses.append(decode_sequence(sequence, vocab))
        count += 1
    if not count:
        raise ValueError("Evaluation ran zero samples")
    report = {
        "samples": count,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "beam_size": beam_size,
        "goal": goal,
        "metrics_omitted": ["meteor"] if goal != "detection" and meteor_dir is None else [],
    }
    if goal != "caption":
        report["segmentation"] = segmentation.compute()
    if goal != "detection":
        report["captioning"] = caption_metrics(references, hypotheses, meteor_dir)
    return report


def evaluate_checkpoint(
    checkpoint,
    images,
    metadata,
    split="test",
    device="auto",
    vocab=None,
    beam_size=5,
    limit=None,
    meteor_dir=None,
):
    predictor = ChangePredictor(checkpoint, vocab, device)
    dataset = LevirMCIDataset(
        images, metadata, split, predictor.vocab, predictor.model.config.max_length, limit
    )
    report = evaluate_model(
        predictor.model,
        predictor.vocab,
        dataset,
        predictor.device,
        beam_size,
        meteor_dir=meteor_dir,
    )
    split_path = Path(metadata) / f"{split}.txt"
    report.update(
        checkpoint_sha256=predictor.checkpoint_sha256,
        split=split,
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        sample_limit=limit,
        vocabulary_size=len(predictor.vocab),
    )
    return report
