"""Directional-Bias & Mask-Symmetry Evaluation Harness (`tools/evaluate_directional_consistency.py`).

Evaluates temporal-reversal consistency on a checksum-verified test slice of paired (T1, T2)
and reversed (T2, T1) samples, computing:
  1. Mask Symmetry mIoU, Pixel Agreement Rate, and Per-Class (unchanged, road, building) Symmetry IoU
     between predicted segmentation masks under forward (T1 -> T2) vs reversed (T2 -> T1) order,
     plus ground-truth mIoU when reference labels are supplied.
  2. Caption Polarity Inversion Rate & Forward-Addition Bias Rate: analyzes whether construction/addition
     statements under (T1 -> T2) invert to removal/demolition/deconstruction statements under (T2 -> T1)
     rather than exhibiting forward-addition bias.
  3. Per-sample SHA-256 provenance and prediction traces saved to `docs/directional_eval_report.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from urban_change.artifacts import artifact_manifest, download_artifact
from urban_change.data.dataset import read_labels
from urban_change.evaluation.metrics import SegmentationMetrics, caption_metrics
from urban_change.inference import PALETTE, ChangePredictor
from urban_change.visual_audit import summarize_mask_regions

# Lexical sets for evaluating urban construction (addition) vs deconstruction (removal) polarity
ADDITION_VERBS_AND_CUES: frozenset[str] = frozenset(
    {
        "built",
        "build",
        "added",
        "add",
        "adds",
        "appear",
        "appears",
        "appeared",
        "constructed",
        "construct",
        "erected",
        "developed",
        "newly",
        "emerged",
        "paved",
        "expanded",
    }
)

REMOVAL_VERBS_AND_CUES: frozenset[str] = frozenset(
    {
        "removed",
        "remove",
        "removes",
        "disappear",
        "disappears",
        "disappeared",
        "demolished",
        "demolish",
        "destroyed",
        "cleared",
        "gone",
        "torn",
        "razed",
        "replaced",
        "vanished",
        "eliminated",
    }
)

URBAN_STRUCTURE_NOUNS: frozenset[str] = frozenset(
    {
        "building",
        "buildings",
        "house",
        "houses",
        "villa",
        "villas",
        "road",
        "roads",
        "street",
        "streets",
        "lane",
        "lanes",
        "highway",
        "factory",
        "structure",
        "structures",
        "residential",
        "roof",
        "roofs",
    }
)

WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)?", re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rgb_mask_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB palette mask (black=0, yellow=1, red=2) back into indexed uint8 labels {0, 1, 2}."""
    arr = np.asarray(mask_rgb, dtype=np.uint8)
    if arr.ndim == 2:
        if not np.isin(arr, (0, 1, 2)).all():
            raise ValueError("2D mask must contain only class labels {0, 1, 2}")
        return arr

    labels = np.zeros(arr.shape[:2], dtype=np.uint8)
    is_road = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 180) & (arr[:, :, 2] < 80)
    is_building = (arr[:, :, 0] > 180) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    labels[is_road] = 1
    labels[is_building] = 2
    return labels


def classify_caption_polarity(caption: str) -> dict[str, Any]:
    """Analyze a change caption's urban structure polarity (addition vs removal vs mixed)."""
    clauses = [
        c.strip() for c in re.split(r"\b(?:and|while|whereas|,|;)\b", caption.lower()) if c.strip()
    ]
    all_tokens = [w.lower() for w in WORD_RE.findall(caption)]

    addition_cues = sorted({t for t in all_tokens if t in ADDITION_VERBS_AND_CUES})
    removal_cues = sorted({t for t in all_tokens if t in REMOVAL_VERBS_AND_CUES})

    # Determine dominant urban-structure action by inspecting clauses mentioning buildings/roads/houses/villas
    urban_add_clauses = 0
    urban_rem_clauses = 0
    for cl in clauses:
        cl_tokens = set(WORD_RE.findall(cl))
        has_urban = bool(cl_tokens & URBAN_STRUCTURE_NOUNS)
        has_add = bool(cl_tokens & ADDITION_VERBS_AND_CUES)
        has_rem = bool(cl_tokens & REMOVAL_VERBS_AND_CUES)
        if has_urban and has_add and not has_rem:
            urban_add_clauses += 1
        elif has_urban and has_rem and not has_add:
            urban_rem_clauses += 1

    if urban_add_clauses > urban_rem_clauses:
        urban_polarity = "urban_addition"
    elif urban_rem_clauses > urban_add_clauses:
        urban_polarity = "urban_removal"
    elif addition_cues and not removal_cues:
        urban_polarity = "urban_addition"
    elif removal_cues and not addition_cues:
        urban_polarity = "urban_removal"
    elif addition_cues and removal_cues:
        urban_polarity = "mixed_transition"
    else:
        urban_polarity = "no_change_or_neutral"

    return {
        "urban_polarity": urban_polarity,
        "addition_cues": addition_cues,
        "removal_cues": removal_cues,
        "urban_addition_clauses": urban_add_clauses,
        "urban_removal_clauses": urban_rem_clauses,
    }


def evaluate_pair_polarity_inversion(
    forward_caption: str,
    reversed_caption: str,
) -> dict[str, Any]:
    """Evaluate whether a (T1 -> T2) and (T2 -> T1) caption pair properly inverts polarity."""
    fwd = classify_caption_polarity(forward_caption)
    rev = classify_caption_polarity(reversed_caption)

    # Polarity inverts properly when forward describes urban addition/construction
    # and reversed describes urban removal/deconstruction (or vice versa)
    inverted = (
        fwd["urban_polarity"] == "urban_addition" and rev["urban_polarity"] == "urban_removal"
    ) or (fwd["urban_polarity"] == "urban_removal" and rev["urban_polarity"] == "urban_addition")

    # Forward-addition bias occurs when the reversed (T2 -> T1) prediction still claims
    # urban structures are only being built/added without any removal/deconstruction cue
    forward_addition_bias = (
        fwd["urban_polarity"] == "urban_addition"
        and rev["urban_polarity"] == "urban_addition"
        and len(rev["removal_cues"]) == 0
    )

    return {
        "forward_polarity": fwd,
        "reversed_polarity": rev,
        "polarity_inverted_correctly": inverted,
        "exhibits_forward_addition_bias": forward_addition_bias,
    }


def compute_mask_symmetry(
    labels_forward: np.ndarray,
    labels_reversed: np.ndarray,
) -> dict[str, Any]:
    """Compute pixel-level symmetry mIoU and per-class IoU between forward and reversed predictions."""
    metrics = SegmentationMetrics(classes=3)
    metrics.update(labels_forward, labels_reversed)
    res = metrics.compute()

    chg_fwd = (labels_forward > 0).astype(np.int64)
    chg_rev = (labels_reversed > 0).astype(np.int64)
    inter_chg = int(np.sum((chg_fwd == 1) & (chg_rev == 1)))
    union_chg = int(np.sum((chg_fwd == 1) | (chg_rev == 1)))
    binary_change_iou = float(inter_chg / union_chg) if union_chg > 0 else 1.0

    return {
        "pixel_agreement_rate": round(float(res["pixel_accuracy"]), 6),
        "mean_symmetry_iou": round(float(res["mean_iou"]), 6),
        "binary_change_symmetry_iou": round(binary_change_iou, 6),
        "class_symmetry_iou": {
            "unchanged_0": round(float(res["class_iou"][0]), 6)
            if res["class_iou"][0] is not None
            else None,
            "road_1": round(float(res["class_iou"][1]), 6)
            if res["class_iou"][1] is not None
            else None,
            "building_2": round(float(res["class_iou"][2]), 6)
            if res["class_iou"][2] is not None
            else None,
        },
        "confusion_matrix_fwd_vs_rev": res["confusion_matrix"],
    }


def evaluate_directional_slice(
    samples: list[dict[str, Any]],
    checkpoint_sha256: str,
    evaluation_mode: str = "verified_checkpoint_predictions",
) -> dict[str, Any]:
    """Evaluate a list of paired (T1, T2) and reversed (T2, T1) sample records."""
    if not samples:
        raise ValueError("Directional evaluation slice requires at least one paired sample")

    global_symmetry = SegmentationMetrics(classes=3)
    global_gt_forward = SegmentationMetrics(classes=3)
    global_gt_reversed = SegmentationMetrics(classes=3)
    has_gt_labels = False

    fwd_refs: list[list[str]] = []
    fwd_hyps: list[str] = []
    rev_refs: list[list[str]] = []
    rev_hyps: list[str] = []

    per_sample_traces: list[dict[str, Any]] = []
    inversion_hits = 0
    addition_bias_hits = 0
    sample_symmetry_mious: list[float] = []
    sample_binary_ious: list[float] = []

    for item in samples:
        sample_id = str(item["sample_id"])
        labels_fwd = np.asarray(item["labels_forward"], dtype=np.uint8)
        labels_rev = np.asarray(item["labels_reversed"], dtype=np.uint8)
        caption_fwd = str(item["caption_forward"])
        caption_rev = str(item["caption_reversed"])

        global_symmetry.update(labels_fwd, labels_rev)
        pair_sym = compute_mask_symmetry(labels_fwd, labels_rev)
        sample_symmetry_mious.append(pair_sym["mean_symmetry_iou"])
        sample_binary_ious.append(pair_sym["binary_change_symmetry_iou"])

        gt_trace: dict[str, Any] | None = None
        if item.get("ground_truth_labels") is not None:
            has_gt_labels = True
            gt_lbl = np.asarray(item["ground_truth_labels"], dtype=np.uint8)
            global_gt_forward.update(gt_lbl, labels_fwd)
            global_gt_reversed.update(gt_lbl, labels_rev)
            m_fwd = SegmentationMetrics(classes=3)
            m_fwd.update(gt_lbl, labels_fwd)
            m_rev = SegmentationMetrics(classes=3)
            m_rev.update(gt_lbl, labels_rev)
            gt_trace = {
                "forward_vs_gt_miou": round(float(m_fwd.compute()["mean_iou"]), 6),
                "reversed_vs_gt_miou": round(float(m_rev.compute()["mean_iou"]), 6),
            }

        if item.get("references_forward"):
            fwd_refs.append(list(item["references_forward"]))
            fwd_hyps.append(caption_fwd)
        if item.get("references_reversed"):
            rev_refs.append(list(item["references_reversed"]))
            rev_hyps.append(caption_rev)

        pol = evaluate_pair_polarity_inversion(caption_fwd, caption_rev)
        if pol["polarity_inverted_correctly"]:
            inversion_hits += 1
        if pol["exhibits_forward_addition_bias"]:
            addition_bias_hits += 1

        fwd_summary = summarize_mask_regions(labels_fwd)
        rev_summary = summarize_mask_regions(labels_rev)

        trace_entry: dict[str, Any] = {
            "sample_id": sample_id,
            "family_id": item.get("family_id", f"FAM-test-{sample_id}"),
            "reversed_partner_id": item.get("reversed_partner_id"),
            "input_t1_sha256": item["input_t1_sha256"],
            "input_t2_sha256": item["input_t2_sha256"],
            "forward_mask_sha256": item.get("forward_mask_sha256"),
            "reversed_mask_sha256": item.get("reversed_mask_sha256"),
            "forward_prediction": {
                "direction": "T1 -> T2 (chronological)",
                "caption": caption_fwd,
                "area_percentages": fwd_summary["area_percentages"],
                "bounding_boxes_count": len(fwd_summary["bounding_boxes"]),
            },
            "reversed_prediction": {
                "direction": "T2 -> T1 (reversed)",
                "caption": caption_rev,
                "area_percentages": rev_summary["area_percentages"],
                "bounding_boxes_count": len(rev_summary["bounding_boxes"]),
            },
            "mask_symmetry": pair_sym,
            "caption_polarity": pol,
        }
        if gt_trace is not None:
            trace_entry["ground_truth_segmentation"] = gt_trace
        per_sample_traces.append(trace_entry)

    corpus_sym = global_symmetry.compute()
    n = len(samples)
    polarity_inversion_rate = round(inversion_hits / n, 6)
    forward_addition_bias_rate = round(addition_bias_hits / n, 6)

    summary_report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "report_type": "directional_bias_and_mask_symmetry_evaluation",
        "evaluation_mode": evaluation_mode,
        "checkpoint_sha256": checkpoint_sha256,
        "paired_samples_evaluated": n,
        "total_directional_inferences": 2 * n,
        "metrics": {
            "mask_symmetry_miou_corpus": round(float(corpus_sym["mean_iou"]), 6),
            "mask_symmetry_miou_macro_avg": round(float(np.mean(sample_symmetry_mious)), 6),
            "binary_change_symmetry_iou_macro_avg": round(float(np.mean(sample_binary_ious)), 6),
            "mask_pixel_agreement_rate": round(float(corpus_sym["pixel_accuracy"]), 6),
            "mask_class_symmetry_iou": {
                "unchanged_0": round(float(corpus_sym["class_iou"][0]), 6)
                if corpus_sym["class_iou"][0] is not None
                else None,
                "road_1": round(float(corpus_sym["class_iou"][1]), 6)
                if corpus_sym["class_iou"][1] is not None
                else None,
                "building_2": round(float(corpus_sym["class_iou"][2]), 6)
                if corpus_sym["class_iou"][2] is not None
                else None,
            },
            "caption_polarity_inversion_rate": polarity_inversion_rate,
            "forward_addition_bias_rate": forward_addition_bias_rate,
        },
        "per_sample_traces": per_sample_traces,
    }

    if has_gt_labels:
        gt_f = global_gt_forward.compute()
        gt_r = global_gt_reversed.compute()
        summary_report["ground_truth_segmentation_metrics"] = {
            "forward_miou": round(float(gt_f["mean_iou"]), 6),
            "reversed_miou": round(float(gt_r["mean_iou"]), 6),
            "miou_directional_gap_abs": round(
                abs(float(gt_f["mean_iou"]) - float(gt_r["mean_iou"])), 6
            ),
        }

    if fwd_refs and rev_refs:
        summary_report["reference_caption_metrics"] = {
            "chronological_slice": caption_metrics(fwd_refs, fwd_hyps),
            "reversed_slice": caption_metrics(rev_refs, rev_hyps),
        }

    return summary_report


def evaluate_committed_checksum_test_slice(
    repo_root: Path,
    checkpoint_path: Path | None = None,
    beam_size: int = 5,
) -> dict[str, Any]:
    """Evaluate the checksum-verified LEVIR-MCI test slice under `docs/assets/examples/`.

    Verifies SHA-256 digests of `earlier.png`, `later.png`, and the predicted masks against
    `docs/assets/examples/provenance.json`. If `--checkpoint` is provided or available locally,
    runs live `ChangePredictor` inference in both directions (`(A, B)` and `(B, A)`); otherwise
    evaluates the checksum-verified predictions captured from `MCI_model_augmented.pth`
    (`5ca0b07c82d633a37e98d0a001247fb0283976209eb6c50cd56afdf33796ee46`).
    """
    prov_path = repo_root / "docs" / "assets" / "examples" / "provenance.json"
    val_path = repo_root / "docs" / "validation-report.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    val_doc = json.loads(val_path.read_text(encoding="utf-8"))

    expected_Hashes = {item["path"]: item["sha256"] for item in prov["files"]}
    for rel_path, exp_sha in expected_Hashes.items():
        full_p = repo_root / rel_path
        actual_sha = sha256_bytes(full_p.read_bytes())
        if actual_sha != exp_sha:
            raise ValueError(
                f"Checksum mismatch for {rel_path}: expected {exp_sha}, got {actual_sha}"
            )

    qual_captions: dict[tuple[str, str], str] = {
        (item["sample"], item["direction"]): item["caption"]
        for item in val_doc.get("qualitative_predictions", [])
    }

    predictor: ChangePredictor | None = None
    eval_mode = "checksum_verified_committed_predictions"
    if checkpoint_path is not None and checkpoint_path.is_file():
        predictor = ChangePredictor(checkpoint_path, device="cpu")
        eval_mode = "live_cpu_checkpoint_inference"

    # Exact reversed partner IDs from `docs/provenance_audit_report.json` (bijective CRC-32 match)
    reversed_partners = {
        "test_000004": ("FAM-test-test_000004", "test_002136.png"),
        "test_000017": ("FAM-test-test_000017", "test_002137.png"),
    }

    slice_samples: list[dict[str, Any]] = []
    for sample_id in ("test_000004", "test_000017"):
        s_dir = repo_root / "docs" / "assets" / "examples" / sample_id
        t1_bytes = (s_dir / "earlier.png").read_bytes()
        t2_bytes = (s_dir / "later.png").read_bytes()
        t1_sha = sha256_bytes(t1_bytes)
        t2_sha = sha256_bytes(t2_bytes)

        if predictor is not None:
            with (
                Image.open(io.BytesIO(t1_bytes)) as img_a,
                Image.open(io.BytesIO(t2_bytes)) as img_b,
            ):
                pred_fwd = predictor.predict(img_a, img_b, beam_size=beam_size)
                pred_rev = predictor.predict(img_b, img_a, beam_size=beam_size)
            labels_fwd = pred_fwd.labels
            labels_rev = pred_rev.labels
            cap_fwd = pred_fwd.caption
            cap_rev = pred_rev.caption
            fwd_mask_sha = sha256_bytes(Image.fromarray(PALETTE[labels_fwd]).tobytes())
            rev_mask_sha = sha256_bytes(Image.fromarray(PALETTE[labels_rev]).tobytes())
            ckpt_sha = predictor.checkpoint_sha256
        else:
            fwd_mask_bytes = (s_dir / "mask-chronological.png").read_bytes()
            rev_mask_bytes = (s_dir / "mask-reversed.png").read_bytes()
            fwd_mask_sha = sha256_bytes(fwd_mask_bytes)
            rev_mask_sha = sha256_bytes(rev_mask_bytes)
            with Image.open(io.BytesIO(fwd_mask_bytes)) as mf:
                labels_fwd = rgb_mask_to_labels(np.array(mf.convert("RGB")))
            with Image.open(io.BytesIO(rev_mask_bytes)) as mr:
                labels_rev = rgb_mask_to_labels(np.array(mr.convert("RGB")))
            cap_fwd = qual_captions[(sample_id, "chronological")]
            cap_rev = qual_captions[(sample_id, "reversed")]
            ckpt_sha = str(prov["model_sha256"])

        fam_id, rev_partner = reversed_partners[sample_id]
        slice_samples.append(
            {
                "sample_id": sample_id,
                "family_id": fam_id,
                "reversed_partner_id": rev_partner,
                "input_t1_sha256": t1_sha,
                "input_t2_sha256": t2_sha,
                "forward_mask_sha256": fwd_mask_sha,
                "reversed_mask_sha256": rev_mask_sha,
                "labels_forward": labels_fwd,
                "labels_reversed": labels_rev,
                "caption_forward": cap_fwd,
                "caption_reversed": cap_rev,
            }
        )

    report = evaluate_directional_slice(
        samples=slice_samples,
        checkpoint_sha256=ckpt_sha,
        evaluation_mode=eval_mode,
    )
    report["provenance"] = {
        "dataset_repository": prov["dataset"],
        "dataset_revision": prov["revision"],
        "source_archive": prov["archive"],
        "sample_split": prov["sample_split"],
        "beam_size": beam_size,
        "scope_boundary": (
            "Evaluated on the 2 checksum-verified real LEVIR-MCI test pairs (4 directional inferences: "
            "test_000004 A->B & B->A, test_000017 A->B & B->A) committed in docs/assets/examples/. "
            "Full 964-pair test-split directional evaluation requires downloading the 4.21 GB archive "
            "and 522 MB checkpoint via --images and --checkpoint."
        ),
    }
    return report


def evaluate_local_dataset_split(
    checkpoint_path: Path,
    images_root: Path,
    captions_path: Path,
    split: str = "test",
    limit: int = 10,
    beam_size: int = 5,
    device: str = "auto",
) -> dict[str, Any]:
    """Run live model directional evaluation on up to `limit` changed pairs in a local dataset split."""
    predictor = ChangePredictor(checkpoint_path, device=device)
    doc = json.loads(captions_path.read_text(encoding="utf-8"))
    entries = [
        e
        for e in doc.get("images", [])
        if e.get("split") == split
        and int(e.get("changeflag", 0)) == 1
        and e.get("augmentation", {}).get("type") != "temporal_reversal"
    ]
    selected = entries[:limit]
    samples: list[dict[str, Any]] = []

    for entry in selected:
        fname = str(entry["filename"])
        p_a = images_root / split / "A" / fname
        p_b = images_root / split / "B" / fname
        p_lbl = images_root / split / "label" / fname
        if not (p_a.is_file() and p_b.is_file()):
            continue

        b_a = p_a.read_bytes()
        b_b = p_b.read_bytes()
        with Image.open(io.BytesIO(b_a)) as img_a, Image.open(io.BytesIO(b_b)) as img_b:
            pred_f = predictor.predict(img_a, img_b, beam_size=beam_size, mask_size="target")
            pred_r = predictor.predict(img_b, img_a, beam_size=beam_size, mask_size="target")

        gt_labels = read_labels(p_lbl) if p_lbl.is_file() else None
        refs_f = [str(s["raw"]) for s in entry.get("sentences", [])]

        samples.append(
            {
                "sample_id": Path(fname).stem,
                "family_id": f"FAM-{split}-{Path(fname).stem}",
                "reversed_partner_id": f"{Path(fname).stem}_reversed.png",
                "input_t1_sha256": sha256_bytes(b_a),
                "input_t2_sha256": sha256_bytes(b_b),
                "forward_mask_sha256": sha256_bytes(pred_f.labels.tobytes()),
                "reversed_mask_sha256": sha256_bytes(pred_r.labels.tobytes()),
                "labels_forward": pred_f.labels,
                "labels_reversed": pred_r.labels,
                "ground_truth_labels": gt_labels,
                "caption_forward": pred_f.caption,
                "caption_reversed": pred_r.caption,
                "references_forward": refs_f,
            }
        )

    return evaluate_directional_slice(
        samples=samples,
        checkpoint_sha256=predictor.checkpoint_sha256,
        evaluation_mode=f"live_dataset_{split}_slice",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Mask Symmetry mIoU and Caption Polarity Inversion Rate on paired (T1, T2) and (T2, T1) samples."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional local path to MCI_model_augmented.pth (or use --download-checkpoint)",
    )
    parser.add_argument(
        "--download-checkpoint",
        action="store_true",
        help="Download and verify MCI_model_augmented.pth from Hugging Face before evaluation",
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Optional local dataset images/ root directory for multi-sample evaluation",
    )
    parser.add_argument(
        "--captions",
        type=Path,
        help="Optional local captions JSON path when --images is supplied",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/directional_eval_report.json"),
        help="Destination JSON report path (default: docs/directional_eval_report.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that freshly computed metrics on the committed test slice match --output",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = args.checkpoint
    if args.download_checkpoint and ckpt_path is None:
        ckpt_path = download_artifact(
            "model", repo_root / "artifacts" / artifact_manifest()["model"]["filename"]
        )

    if args.images and args.captions and ckpt_path:
        report = evaluate_local_dataset_split(
            checkpoint_path=ckpt_path,
            images_root=args.images,
            captions_path=args.captions,
            split=args.split,
            limit=args.limit,
            beam_size=args.beam_size,
        )
    else:
        report = evaluate_committed_checksum_test_slice(
            repo_root=repo_root,
            checkpoint_path=ckpt_path,
            beam_size=args.beam_size,
        )

    if args.check:
        if not args.output.is_file():
            raise FileNotFoundError(f"Missing directional report: {args.output}")
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        assert existing["metrics"] == report["metrics"], "Directional evaluation metrics mismatch"
        assert existing["checkpoint_sha256"] == report["checkpoint_sha256"]
        print(f"OK: Verified directional consistency report at {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    m = report["metrics"]
    print(
        f"Wrote {args.output}: samples={report['paired_samples_evaluated']}, "
        f"mask_symmetry_miou={m['mask_symmetry_miou_corpus']:.4f}, "
        f"polarity_inversion_rate={m['caption_polarity_inversion_rate']:.4f}, "
        f"forward_addition_bias_rate={m['forward_addition_bias_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
