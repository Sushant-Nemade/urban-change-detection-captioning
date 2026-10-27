"""Top-level Auditable Visual Evidence Renderer shim (`src/visual_audit.py`).

Re-exports the full visual audit implementation from `urban_change.visual_audit` so both
`import visual_audit` (when `src` is on `PYTHONPATH`) and `from urban_change import visual_audit`
work seamlessly.
"""

from __future__ import annotations

from urban_change.visual_audit import (
    CLASS_NAMES,
    PALETTE_RGB,
    extract_change_bounding_boxes,
    render_overlay_with_bboxes,
    render_visual_audit_card,
    summarize_mask_regions,
)

__all__ = [
    "CLASS_NAMES",
    "PALETTE_RGB",
    "extract_change_bounding_boxes",
    "render_overlay_with_bboxes",
    "render_visual_audit_card",
    "summarize_mask_regions",
]
