"""Auditable Visual Evidence Renderer (`urban_change.visual_audit`).

Extends prediction output across CLI, Streamlit, and FastAPI so every inference exports:
  1. Connected-component bounding boxes (`[x_min, y_min, x_max, y_max]`, pixel area, class label)
     for road (`1`) and building (`2`) changes.
  2. Pixel-area change percentages (`unchanged_pct`, `road_change_pct`, `building_change_pct`,
     `total_changed_pct`) and pixel counts.
  3. An auditable visual report card (`audit_card.png`) displaying side-by-side T1 (Earlier),
     T2 (Later), predicted segmentation mask, and an alpha-blended T2 overlay annotated with
     road/building bounding-box coordinates, area percentages, caption, and checkpoint SHA-256.
  4. Structured JSON metadata (`audit_report.json`).
"""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

PALETTE_RGB = np.array([[0, 0, 0], [255, 255, 0], [255, 0, 0]], dtype=np.uint8)
CLASS_NAMES: dict[int, str] = {
    0: "unchanged",
    1: "road_change",
    2: "building_change",
}


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def extract_change_bounding_boxes(
    labels: np.ndarray,
    min_area_pixels: int = 8,
) -> list[dict[str, Any]]:
    """Extract connected-component bounding boxes for road (1) and building (2) change masks."""
    lbl = np.asarray(labels, dtype=np.uint8)
    if lbl.ndim != 2:
        raise ValueError("labels must be a 2D array of class indices {0, 1, 2}")

    boxes: list[dict[str, Any]] = []
    for class_id in (1, 2):
        binary = (lbl == class_id).astype(np.uint8)
        if not np.any(binary):
            continue
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_idx in range(1, num_labels):
            x = int(stats[comp_idx, cv2.CC_STAT_LEFT])
            y = int(stats[comp_idx, cv2.CC_STAT_TOP])
            w = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area < min_area_pixels:
                continue
            cx, cy = float(centroids[comp_idx][0]), float(centroids[comp_idx][1])
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "bbox_xywh": [x, y, w, h],
                    "area_pixels": area,
                    "centroid_xy": [round(cx, 1), round(cy, 1)],
                }
            )

    boxes.sort(key=lambda b: (-int(b["area_pixels"]), int(b["class_id"]), b["bbox_xyxy"]))
    return boxes


def summarize_mask_regions(
    labels: np.ndarray,
    min_area_pixels: int = 8,
) -> dict[str, Any]:
    """Compute pixel-area change percentages, pixel counts, and connected-component bounding boxes."""
    lbl = np.asarray(labels, dtype=np.uint8)
    if lbl.ndim != 2:
        raise ValueError("labels must be a 2D uint8 array")
    total_pixels = int(lbl.size)
    if total_pixels == 0:
        raise ValueError("labels array cannot be empty")

    unchanged_px = int(np.sum(lbl == 0))
    road_px = int(np.sum(lbl == 1))
    building_px = int(np.sum(lbl == 2))
    changed_px = road_px + building_px

    unchanged_pct = round((unchanged_px / total_pixels) * 100.0, 4)
    road_pct = round((road_px / total_pixels) * 100.0, 4)
    building_pct = round((building_px / total_pixels) * 100.0, 4)
    total_changed_pct = round((changed_px / total_pixels) * 100.0, 4)

    boxes = extract_change_bounding_boxes(lbl, min_area_pixels=min_area_pixels)
    return {
        "total_pixels": total_pixels,
        "pixel_counts": {
            "unchanged": unchanged_px,
            "road_change": road_px,
            "building_change": building_px,
            "total_changed": changed_px,
        },
        "area_percentages": {
            "unchanged_pct": unchanged_pct,
            "road_change_pct": road_pct,
            "building_change_pct": building_pct,
            "total_changed_pct": total_changed_pct,
        },
        "bounding_boxes": boxes,
        "bounding_box_counts": {
            "road_change": sum(1 for b in boxes if b["class_id"] == 1),
            "building_change": sum(1 for b in boxes if b["class_id"] == 2),
            "total": len(boxes),
        },
    }


def render_overlay_with_bboxes(
    image_b_rgb: np.ndarray,
    labels: np.ndarray,
    bounding_boxes: list[dict[str, Any]] | None = None,
    alpha: float = 0.42,
) -> np.ndarray:
    """Render an alpha-blended change mask overlay on T2 (`image_b_rgb`) with bounding boxes."""
    lbl = np.asarray(labels, dtype=np.uint8)
    h, w = lbl.shape[:2]

    base = np.asarray(image_b_rgb, dtype=np.uint8)
    if base.shape[:2] != (h, w):
        base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_rgb = PALETTE_RGB[lbl]
    changed_mask = lbl > 0

    blended = base.copy().astype(np.float32)
    blended[changed_mask] = (1.0 - alpha) * blended[changed_mask] + alpha * mask_rgb[
        changed_mask
    ].astype(np.float32)
    canvas = np.clip(blended, 0, 255).astype(np.uint8)

    boxes = bounding_boxes if bounding_boxes is not None else extract_change_bounding_boxes(lbl)
    for box in boxes[:25]:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        color = (255, 235, 0) if box["class_id"] == 1 else (255, 50, 50)
        cv2.rectangle(canvas, (x1, y1), (max(x1 + 1, x2 - 1), max(y1 + 1, y2 - 1)), color, 2)

    return canvas


def render_visual_audit_card(
    image_a: Image.Image | np.ndarray,
    image_b: Image.Image | np.ndarray,
    labels: np.ndarray,
    caption: str,
    checkpoint_sha256: str,
    min_area_pixels: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a side-by-side 4-panel auditable visual report card (`audit_card.png`) and metadata dict."""
    lbl = np.asarray(labels, dtype=np.uint8)
    h, w = lbl.shape[:2]

    arr_a = np.array(
        image_a.convert("RGB") if isinstance(image_a, Image.Image) else image_a, dtype=np.uint8
    )
    arr_b = np.array(
        image_b.convert("RGB") if isinstance(image_b, Image.Image) else image_b, dtype=np.uint8
    )

    if arr_a.shape[:2] != (h, w):
        arr_a_resized = cv2.resize(arr_a, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        arr_a_resized = arr_a

    if arr_b.shape[:2] != (h, w):
        arr_b_resized = cv2.resize(arr_b, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        arr_b_resized = arr_b

    region_summary = summarize_mask_regions(lbl, min_area_pixels=min_area_pixels)
    mask_rgb = PALETTE_RGB[lbl]
    overlay_rgb = render_overlay_with_bboxes(
        arr_b_resized,
        lbl,
        bounding_boxes=region_summary["bounding_boxes"],
    )

    # Assemble 4-panel horizontal strip with header and footer bars
    pad = 12
    header_h = 34
    footer_h = 68
    card_w = 4 * w + 5 * pad
    card_h = header_h + h + footer_h + 2 * pad

    card_img = Image.new("RGB", (card_w, card_h), (15, 23, 42))
    draw = ImageDraw.Draw(card_img)

    panels = [
        ("01 / T1 (Earlier)", arr_a_resized),
        ("02 / T2 (Later)", arr_b_resized),
        ("03 / Change Mask (Road/Bldg)", mask_rgb),
        ("04 / Overlay + BBoxes", overlay_rgb),
    ]

    for idx, (title, panel_arr) in enumerate(panels):
        x0 = pad + idx * (w + pad)
        y0 = pad + header_h
        draw.text((x0, pad + 8), title, fill=(248, 250, 252))
        card_img.paste(Image.fromarray(panel_arr), (x0, y0))

    pcts = region_summary["area_percentages"]
    bbox_counts = region_summary["bounding_box_counts"]
    stats_line = (
        f"Changed Area: {pcts['total_changed_pct']:.2f}% "
        f"(Road: {pcts['road_change_pct']:.2f}% [{bbox_counts['road_change']} bbox], "
        f"Building: {pcts['building_change_pct']:.2f}% [{bbox_counts['building_change']} bbox], "
        f"Unchanged: {pcts['unchanged_pct']:.2f}%) | Model SHA-256: {checkpoint_sha256[:16]}..."
    )
    cap_display = caption if len(caption) <= 130 else caption[:127] + "..."
    caption_line = f"Caption: {cap_display}"

    footer_y = pad + header_h + h + 10
    draw.text((pad, footer_y), stats_line, fill=(242, 194, 48))
    draw.text((pad, footer_y + 24), caption_line, fill=(226, 232, 240))

    card_rgb = np.array(card_img, dtype=np.uint8)
    audit_metadata = {
        "schema_version": "1.0.0",
        "caption": caption,
        "checkpoint_sha256": checkpoint_sha256,
        "mask_shape": [int(h), int(w)],
        "input_t1_rgb_sha256": _sha256_array(arr_a),
        "input_t2_rgb_sha256": _sha256_array(arr_b),
        "labels_sha256": _sha256_array(lbl),
        "classes": {"0": "unchanged", "1": "road change", "2": "building change"},
        **region_summary,
    }
    return card_rgb, audit_metadata
