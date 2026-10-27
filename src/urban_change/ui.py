"""Streamlit demo backed by the shared predictor."""

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

from urban_change.inference import ChangePredictor


@st.cache_resource(show_spinner=False)
def load_predictor(path, modified_ns, device):
    # File modification time is part of the cache key so replacing a checkpoint
    # at the same path does not silently reuse the previous model.
    return ChangePredictor(path, device=device)


def result_archive(prediction):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        items = [("mask.png", prediction.mask_rgb), ("labels.png", prediction.labels)]
        if prediction.audit_card_rgb is not None:
            items.append(("audit_card.png", prediction.audit_card_rgb))
        for name, array in items:
            image = io.BytesIO()
            Image.fromarray(array).save(image, format="PNG")
            target.writestr(name, image.getvalue())
        target.writestr("caption.txt", prediction.caption + "\n")
        target.writestr("prediction.json", json.dumps(prediction.metadata(), indent=2))
        if prediction.audit_report is not None:
            target.writestr("audit_report.json", json.dumps(prediction.audit_report, indent=2))
    return archive.getvalue()


def main():
    st.set_page_config(page_title="Urban Change", page_icon="🏙️", layout="wide")
    st.markdown(
        "<div style='color:#38bdf8;font-size:0.8rem;letter-spacing:0.16em'>TWO MOMENTS · ONE CHANGING CITY</div>",
        unsafe_allow_html=True,
    )
    st.title("Urban Change Detection & Change Captioning")
    st.write(
        "Compare aligned satellite images to locate changes in roads and buildings and describe what changed."
    )
    with st.sidebar:
        st.header("Model")
        checkpoint = st.text_input(
            "Checkpoint path",
            value=os.environ.get("URBAN_CHANGE_CHECKPOINT", "artifacts/MCI_model_augmented.pth"),
        )
        device = st.selectbox("Device", ["auto", "cpu", "cuda"])
        beam_size = st.select_slider("Caption beam size", options=[1, 3, 5], value=5)
        load = st.button("Load model", type="primary", width="stretch")
        st.divider()
        st.caption(
            "Earlier image → later image. Reversing the order changes the direction described in the caption."
        )
        st.caption(
            "Mask: road change is yellow; building change is red; unchanged pixels are black."
        )
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if load:
        st.session_state.predictor = None
        st.session_state.pop("prediction", None)
        try:
            path = Path(checkpoint).expanduser().resolve()
            with st.spinner("Loading model…"):
                st.session_state.predictor = load_predictor(
                    str(path), path.stat().st_mtime_ns, device
                )
            st.sidebar.success(f"Ready on {st.session_state.predictor.device}")
        except (OSError, ValueError, RuntimeError) as exc:
            st.sidebar.error(str(exc))
    before, after = st.columns(2)
    with before:
        st.subheader("01 / Earlier image")
        image_a = st.file_uploader(
            "Image A", type=["png", "jpg", "jpeg", "tif", "tiff"], key="image_a"
        )
    with after:
        st.subheader("02 / Later image")
        image_b = st.file_uploader(
            "Image B", type=["png", "jpg", "jpeg", "tif", "tiff"], key="image_b"
        )
    a = b = None
    try:
        if image_a is not None:
            a = Image.open(image_a).convert("RGB")
            before.image(a, width="stretch")
        if image_b is not None:
            b = Image.open(image_b).convert("RGB")
            after.image(b, width="stretch")
    except (OSError, ValueError) as exc:
        st.error(f"Could not read image: {exc}")
    input_identity = tuple(
        hashlib.sha256(upload.getvalue()).hexdigest() if upload is not None else None
        for upload in (image_a, image_b)
    )
    if st.session_state.get("prediction_inputs") != input_identity:
        st.session_state.pop("prediction", None)
    if st.button(
        "Detect and describe changes",
        type="primary",
        disabled=a is None or b is None or st.session_state.predictor is None,
    ):
        try:
            with st.spinner("Comparing the two images…"):
                st.session_state.prediction = st.session_state.predictor.predict(a, b, beam_size)
                st.session_state.prediction_inputs = input_identity
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if "prediction" in st.session_state:
        prediction = st.session_state.prediction
        st.divider()
        mask_column, caption_column = st.columns([1, 1])
        mask_column.image(prediction.mask_rgb, caption="Change mask", width="stretch")
        with caption_column:
            st.subheader("What changed")
            st.write(prediction.caption or "The decoder returned no descriptive tokens.")
            meta = prediction.metadata()
            pcts = meta.get("area_percentages", {})
            if pcts:
                st.caption(
                    f"Changed area: {pcts.get('total_changed_pct', 0.0):.2f}% "
                    f"(Road: {pcts.get('road_change_pct', 0.0):.2f}%, "
                    f"Building: {pcts.get('building_change_pct', 0.0):.2f}%)"
                )
            st.caption(f"Inference time: {prediction.elapsed_ms / 1000:.2f} seconds")
            st.download_button(
                "Download results",
                result_archive(prediction),
                "urban-change-results.zip",
                mime="application/zip",
                width="stretch",
            )
        if prediction.audit_card_rgb is not None:
            st.image(
                prediction.audit_card_rgb,
                caption="Auditable visual report card (T1, T2, Change Mask, Overlay + Bounding Boxes)",
                width="stretch",
            )
    elif a is None or b is None:
        st.info("Load a checkpoint and upload both images to begin.")
