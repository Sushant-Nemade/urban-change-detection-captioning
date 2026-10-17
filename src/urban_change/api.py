"""HTTP inference with separate liveness and model readiness endpoints."""

import base64
import io
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from urban_change import __version__
from urban_change.inference import ChangePredictor

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 4096 * 4096


def open_uploaded_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 20 MiB upload limit")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Image exceeds the 4096x4096 pixel budget")
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Upload is not a supported, readable image") from exc


def encode_png(array) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_app(checkpoint=None, vocab=None, device=None):
    @asynccontextmanager
    async def lifespan(application):
        source = checkpoint or os.environ.get("URBAN_CHANGE_CHECKPOINT")
        application.state.predictor = (
            ChangePredictor(
                source,
                vocab or os.environ.get("URBAN_CHANGE_VOCAB"),
                device or os.environ.get("URBAN_CHANGE_DEVICE", "auto"),
            )
            if source
            else None
        )
        application.state.lock = threading.Lock()
        yield
        application.state.predictor = None

    application = FastAPI(
        title="Urban Change Detection & Change Captioning", version=__version__, lifespan=lifespan
    )

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/ready")
    def ready():
        predictor = application.state.predictor
        if predictor is None:
            return JSONResponse(
                {"ready": False, "reason": "No model checkpoint configured"}, status_code=503
            )
        return {
            "ready": True,
            "device": str(predictor.device),
            "checkpoint_sha256": predictor.checkpoint_sha256,
        }

    @application.post("/infer")
    async def infer(
        image_a: UploadFile = File(...),
        image_b: UploadFile = File(...),
        beam_size: int = 5,
        mask_size: str = "a",
    ):
        predictor = application.state.predictor
        if predictor is None:
            raise HTTPException(503, "Set URBAN_CHANGE_CHECKPOINT before starting inference")
        a = open_uploaded_image(await image_a.read(MAX_UPLOAD_BYTES + 1))
        b = open_uploaded_image(await image_b.read(MAX_UPLOAD_BYTES + 1))

        def predict():
            with application.state.lock:
                return predictor.predict(a, b, beam_size, mask_size)

        try:
            result = await run_in_threadpool(predict)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            **result.metadata(),
            "mask_png_base64": encode_png(result.mask_rgb),
            "labels_png_base64": encode_png(result.labels),
        }

    return application


app = create_app()
