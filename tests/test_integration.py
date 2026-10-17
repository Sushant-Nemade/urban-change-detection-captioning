"""Opt-in real model execution; supplied compatibility weights are untrained."""

import base64
import gc
import io
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from urban_change.api import create_app
from urban_change.checkpoints import load_checkpoint
from urban_change.data.prepare import prepare_metadata
from urban_change.data.text import load_vocab
from urban_change.inference import ChangePredictor
from urban_change.training.engine import TrainConfig, configure_stage, train

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def compatibility_checkpoint():
    path = os.environ.get("URBAN_CHANGE_TEST_CHECKPOINT")
    if not path or not Path(path).is_file():
        pytest.skip("Set URBAN_CHANGE_TEST_CHECKPOINT to a compatible tensor checkpoint")
    return path


def test_legacy_model_numerical_parity(compatibility_checkpoint):
    golden_path = os.environ.get("URBAN_CHANGE_TEST_GOLDEN")
    if not golden_path:
        pytest.skip("Set URBAN_CHANGE_TEST_GOLDEN to the captured legacy outputs")
    predictor = ChangePredictor(compatibility_checkpoint, device="cpu")
    golden = np.load(golden_path)
    with torch.inference_mode():
        a, b, segmentation = predictor.model(
            torch.from_numpy(golden["a"]), torch.from_numpy(golden["b"])
        )
        sequence = predictor.model.decoder.sample_beam(a, b, k=3)
    for name, value in [("f1", a), ("f2", b), ("seg", segmentation)]:
        np.testing.assert_allclose(value.numpy(), golden[name], rtol=1e-5, atol=1e-5)
    assert sequence == golden["seq"].tolist()
    result = predictor.predict(
        Image.new("RGB", (300, 280)), Image.new("RGB", (320, 290)), beam_size=1
    )
    assert result.labels.shape == (280, 300)
    assert set(np.unique(result.labels)).issubset({0, 1, 2})


def test_real_api_prediction(compatibility_checkpoint):
    image = io.BytesIO()
    Image.new("RGB", (256, 256), (80, 100, 90)).save(image, format="PNG")
    payload = image.getvalue()
    with TestClient(create_app(checkpoint=compatibility_checkpoint, device="cpu")) as client:
        assert client.get("/ready").status_code == 200
        result = client.post(
            "/infer?beam_size=1",
            files={
                "image_a": ("a.png", payload, "image/png"),
                "image_b": ("b.png", payload, "image/png"),
            },
        )
        assert result.status_code == 200
        data = result.json()
        assert isinstance(data["caption"], str)
        assert Image.open(io.BytesIO(base64.b64decode(data["mask_png_base64"]))).size == (256, 256)
        bad = client.post(
            "/infer?beam_size=0",
            files={"image_a": ("a.png", payload), "image_b": ("b.png", payload)},
        )
        assert bad.status_code == 422


def test_training_modes_reapply_after_validation(compatibility_checkpoint):
    model = ChangePredictor(compatibility_checkpoint, device="cpu").model
    config = TrainConfig(stage="s2", goal="caption", unfreeze_caption_neck=True)
    for _ in range(2):
        configure_stage(model, config)
        assert not model.encoder.training and model.decoder.training
        assert all(module.training for module in model.encoder_trans.cap_modules_list)
        assert all(not module.training for module in model.encoder_trans.CD_modules_list)
        assert any(p.requires_grad for p in model.encoder_trans.Dynamic_DIF_aware_TR.parameters())
        model.eval()
        assert all(not module.training for module in model.modules())
    configure_stage(model, TrainConfig(stage="s2", goal="detection"))
    assert model.encoder.training and not model.decoder.training
    assert all(not module.training for module in model.encoder_trans.cap_modules_list)
    assert all(module.training for module in model.encoder_trans.CD_modules_list)


def test_joint_training_updates_both_heads(compatibility_checkpoint, dataset_fixture, tmp_path):
    import json

    fixture = dataset_fixture
    output = tmp_path / "joint"
    # A smaller training-only vocabulary also exercises transplantation from
    # the legacy 714-token checkpoint before a real full-model backward pass.
    config = TrainConfig(
        images=str(fixture["images"]),
        metadata=str(fixture["metadata"]),
        output=str(output),
        stage="s2",
        goal="joint",
        checkpoint=compatibility_checkpoint,
        device="cpu",
        epochs=1,
        batch_size=1,
        accumulation_steps=3,
        train_limit=1,
        val_limit=1,
        beam_size=1,
        loss_balance=True,
    )
    report = train(config)
    assert report["epochs_completed"] == 1
    epoch = json.loads((output / "history.jsonl").read_text().splitlines()[0])
    assert epoch["optimizer_steps"] == 1
    assert epoch["validation"]["goal"] == "joint"
    assert "captioning" in epoch["validation"] and "segmentation" in epoch["validation"]
    assert np.isfinite(epoch["training_loss"])
    for checkpoint in [output / "last.pth", output / "best.pth"]:
        assert checkpoint.is_file()
        checkpoint.unlink()  # Full optimizer snapshots are unnecessary test artifacts.


def test_training_partial_accumulation_and_resume(
    compatibility_checkpoint, dataset_fixture, tmp_path
):
    fixture = dataset_fixture
    # Match the legacy model's exact token ordering while using tiny local image
    # fixtures. This exercises optimization/resume, not model accuracy.
    vocab_path = tmp_path / "published-vocab.json"
    import json

    vocab_path.write_text(json.dumps(load_vocab(size=714)))
    prepare_metadata(
        fixture["captions"], fixture["metadata"], vocab_path=vocab_path, overwrite=True
    )
    output = tmp_path / "run"
    options = dict(
        images=str(fixture["images"]),
        metadata=str(fixture["metadata"]),
        output=str(output),
        stage="s2",
        goal="caption",
        device="cpu",
        batch_size=1,
        accumulation_steps=3,
        train_limit=2,
        val_limit=1,
        beam_size=1,
    )
    result = train(TrainConfig(**options, checkpoint=compatibility_checkpoint, epochs=1))
    assert result["epochs_completed"] == 1
    history = json.loads((output / "history.jsonl").read_text().splitlines()[0])
    assert history["optimizer_steps"] == 1
    saved = load_checkpoint(output / "last.pth")
    assert saved["epoch"] == 0 and saved["optimizer"]["state"]
    assert saved["rng"]["torch"].numel() > 0
    assert all(torch.isfinite(value).all() for value in saved["decoder_dict"].values())
    del saved
    gc.collect()
    resumed = train(TrainConfig(**options, resume=str(output / "last.pth"), epochs=2))
    assert resumed["epochs_completed"] == 1 and resumed["last_epoch"] == 1
    assert len((output / "history.jsonl").read_text().splitlines()) == 2
