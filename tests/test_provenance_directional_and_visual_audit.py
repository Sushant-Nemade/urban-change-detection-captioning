"""Tests for Split-Family Provenance Auditor, Directional Consistency Harness, and Auditable Visual Renderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import visual_audit as top_level_visual_audit
from tools.audit_split_provenance import audit_manifests_and_signatures
from tools.audit_split_provenance import main as audit_cli_main
from tools.evaluate_directional_consistency import (
    classify_caption_polarity,
    compute_mask_symmetry,
    evaluate_committed_checksum_test_slice,
    evaluate_pair_polarity_inversion,
)
from tools.evaluate_directional_consistency import main as directional_cli_main
from urban_change.inference import PALETTE, Prediction, save_prediction
from urban_change.visual_audit import (
    extract_change_bounding_boxes,
    render_visual_audit_card,
    summarize_mask_regions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_provenance_auditor_maps_families_and_detects_cross_split_leakage() -> None:
    orig_doc = {
        "images": [
            {
                "filename": "train_000001.png",
                "split": "train",
                "imgid": 0,
                "changeflag": 1,
                "sentences": [{"raw": "a road is built"}],
            },
            {
                "filename": "val_000001.png",
                "split": "val",
                "imgid": 1,
                "changeflag": 0,
                "sentences": [{"raw": "no change"}],
            },
            {
                "filename": "test_000001.png",
                "split": "test",
                "imgid": 2,
                "changeflag": 1,
                "sentences": [{"raw": "a villa is built"}],
            },
        ]
    }
    # Clean augmented manifest (reversed stays in same split as source)
    clean_aug_doc = {
        "images": [
            *orig_doc["images"],
            {
                "filename": "train_000001_reversed.png",
                "split": "train",
                "imgid": 3,
                "changeflag": 1,
                "sentences": [{"raw": "a road is removed"}],
            },
            {
                "filename": "test_002000.png",
                "split": "test",
                "imgid": 4,
                "changeflag": 1,
                "sentences": [{"raw": "a villa is removed"}],
            },
        ]
    }
    sigs = {
        ("train", "A", "train_000001.png"): (101, 500),
        ("train", "B", "train_000001.png"): (102, 510),
        ("train", "label", "train_000001.png"): (103, 200),
        ("val", "A", "val_000001.png"): (201, 500),
        ("val", "B", "val_000001.png"): (202, 500),
        ("val", "label", "val_000001.png"): (203, 200),
        ("test", "A", "test_000001.png"): (301, 600),
        ("test", "B", "test_000001.png"): (302, 610),
        ("test", "label", "test_000001.png"): (303, 220),
        # Reversed test_002000 swaps A and B of test_000001
        ("test", "A", "test_002000.png"): (302, 610),
        ("test", "B", "test_002000.png"): (301, 600),
        ("test", "label", "test_002000.png"): (303, 220),
    }

    report = audit_manifests_and_signatures(
        json.dumps(orig_doc).encode("utf-8"),
        json.dumps(clean_aug_doc).encode("utf-8"),
        member_signatures=sigs,
    )
    ver = report["zero_leakage_verification"]
    assert ver["zero_cross_split_leakage_verified"] is True
    assert ver["cross_split_leakage_count"] == 0
    assert ver["mapped_reversal_count"] == 2
    assert report["deterministic_partition_policies"]["train_only_augmentation"]["counts"] == {
        "train": 2,
        "val": 1,
        "test": 1,
    }

    # Now inject a cross-split leak: put the reversed version of test_000001 into 'train'!
    leaky_aug_doc = {
        "images": [
            *orig_doc["images"],
            {
                "filename": "train_leaked_from_test.png",
                "split": "train",
                "imgid": 99,
                "changeflag": 1,
                "sentences": [{"raw": "a villa is removed"}],
            },
        ]
    }
    sigs_leaky = dict(sigs)
    sigs_leaky[("train", "A", "train_leaked_from_test.png")] = (302, 610)
    sigs_leaky[("train", "B", "train_leaked_from_test.png")] = (301, 600)
    sigs_leaky[("train", "label", "train_leaked_from_test.png")] = (303, 220)

    try:
        audit_manifests_and_signatures(
            json.dumps(orig_doc).encode("utf-8"),
            json.dumps(leaky_aug_doc).encode("utf-8"),
            member_signatures=sigs_leaky,
        )
    except ValueError as exc:
        assert "crosses" in str(exc)
    else:
        raise AssertionError("Expected ValueError on cross-split family leakage")


def test_committed_provenance_audit_report_is_valid() -> None:
    report_path = REPO_ROOT / "docs" / "provenance_audit_report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["manifests"]["original_total_entries"] == 10077
    assert data["manifests"]["augmented_total_entries"] == 15115
    assert data["manifests"]["total_reversed_entries"] == 5038
    assert data["zero_leakage_verification"]["zero_cross_split_leakage_verified"] is True
    assert data["zero_leakage_verification"]["cross_split_leakage_count"] == 0
    assert data["zero_leakage_verification"]["all_changed_pairs_bijectively_mapped"] is True
    assert audit_cli_main(["--check", "--output", str(report_path)]) == 0


def test_directional_polarity_and_symmetry_harness() -> None:
    pol_ok = evaluate_pair_polarity_inversion(
        "the woods disappear and a t-shaped road is built with villas built along",
        "trees are added , and a road with villas along it is removed",
    )
    assert pol_ok["polarity_inverted_correctly"] is True
    assert pol_ok["exhibits_forward_addition_bias"] is False

    pol_biased = evaluate_pair_polarity_inversion(
        "a new residential building is constructed along the road",
        "two new buildings are built beside the road",
    )
    assert pol_biased["polarity_inverted_correctly"] is False
    assert pol_biased["exhibits_forward_addition_bias"] is True
    assert (
        classify_caption_polarity("nothing has changed")["urban_polarity"] == "no_change_or_neutral"
    )

    m_a = np.zeros((32, 32), dtype=np.uint8)
    m_a[4:12, 4:12] = 1
    m_a[16:28, 16:28] = 2
    sym = compute_mask_symmetry(m_a, m_a.copy())
    assert sym["mean_symmetry_iou"] == 1.0
    assert sym["pixel_agreement_rate"] == 1.0

    slice_report = evaluate_committed_checksum_test_slice(REPO_ROOT)
    assert slice_report["paired_samples_evaluated"] == 2
    assert slice_report["metrics"]["caption_polarity_inversion_rate"] == 1.0
    assert slice_report["metrics"]["forward_addition_bias_rate"] == 0.0
    assert slice_report["metrics"]["mask_symmetry_miou_corpus"] > 0.95
    assert (
        directional_cli_main(
            ["--check", "--output", str(REPO_ROOT / "docs/directional_eval_report.json")]
        )
        == 0
    )


def test_visual_audit_renderer_and_prediction_export(tmp_path: Path) -> None:
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[8:24, 10:30] = 1  # road block (16 * 20 = 320 px)
    labels[32:56, 32:56] = 2  # building block (24 * 24 = 576 px)

    boxes = extract_change_bounding_boxes(labels, min_area_pixels=8)
    assert len(boxes) == 2
    assert boxes[0]["class_name"] == "building_change"
    assert boxes[0]["area_pixels"] == 576
    assert boxes[0]["bbox_xyxy"] == [32, 32, 56, 56]
    assert boxes[1]["class_name"] == "road_change"
    assert boxes[1]["area_pixels"] == 320

    summary = summarize_mask_regions(labels)
    assert summary["pixel_counts"]["road_change"] == 320
    assert summary["pixel_counts"]["building_change"] == 576
    assert summary["bounding_box_counts"]["total"] == 2

    img_a = Image.new("RGB", (64, 64), (40, 80, 40))
    img_b = Image.new("RGB", (64, 64), (120, 120, 120))
    card_rgb, audit_meta = render_visual_audit_card(
        image_a=img_a,
        image_b=img_b,
        labels=labels,
        caption="a road and a building are constructed",
        checkpoint_sha256="5ca0b07c" * 8,
    )
    assert card_rgb.ndim == 3 and card_rgb.shape[2] == 3
    assert card_rgb.shape[1] > 4 * 64
    assert audit_meta["bounding_box_counts"]["total"] == 2
    assert top_level_visual_audit.summarize_mask_regions(labels)["total_pixels"] == 4096

    pred = Prediction(
        caption="a road and a building are constructed",
        labels=labels,
        mask_rgb=PALETTE[labels],
        elapsed_ms=15.2,
        checkpoint_sha256="5ca0b07c" * 8,
        audit_card_rgb=card_rgb,
        audit_report=audit_meta,
    )
    out_dir = save_prediction(pred, tmp_path / "pred_out", image_a=img_a, image_b=img_b)
    assert (out_dir / "audit_card.png").is_file()
    assert (out_dir / "audit_report.json").is_file()
    saved_pred_json = json.loads((out_dir / "prediction.json").read_text(encoding="utf-8"))
    assert "area_percentages" in saved_pred_json
    assert "bounding_boxes" in saved_pred_json
    assert len(saved_pred_json["bounding_boxes"]) == 2
