import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from urban_change.api import MAX_UPLOAD_BYTES, create_app, open_uploaded_image
from urban_change.cli import main


def test_cli_exposes_workflows_and_fails_on_missing_checkpoint(capsys):
    import pytest

    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "validate-data" in capsys.readouterr().out
    with pytest.raises(SystemExit) as result:
        main(
            [
                "predict",
                "--checkpoint",
                "/missing/model.pth",
                "--image-a",
                "a.png",
                "--image-b",
                "b.png",
                "--output",
                "outputs/test",
            ]
        )
    assert result.value.code == 1
    assert "Checkpoint not found" in capsys.readouterr().err


def test_api_distinguishes_liveness_and_readiness(monkeypatch):
    monkeypatch.delenv("URBAN_CHANGE_CHECKPOINT", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 503
        result = client.post(
            "/infer", files={"image_a": ("a.png", b"bad"), "image_b": ("b.png", b"bad")}
        )
        assert result.status_code == 503


def test_image_upload_rejects_invalid_or_excessive_content():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as invalid:
        open_uploaded_image(b"not an image")
    assert invalid.value.status_code == 422
    with pytest.raises(HTTPException) as excessive:
        open_uploaded_image(b"x" * (MAX_UPLOAD_BYTES + 1))
    assert excessive.value.status_code == 413
    image = io.BytesIO()
    Image.new("RGB", (12, 12)).save(image, format="PNG")
    assert open_uploaded_image(image.getvalue()).size == (12, 12)


def test_streamlit_ui_and_missing_checkpoint(monkeypatch):
    monkeypatch.setenv("URBAN_CHANGE_CHECKPOINT", "/missing/checkpoint.pth")
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "apps/streamlit_app.py"), default_timeout=30
    ).run()
    assert not app.exception
    assert app.title[0].value == "Urban Change Detection & Change Captioning"
    assert len(app.get("file_uploader")) == 2
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert len(app.sidebar.error) == 1


def test_cli_resume_overrides_template_finetuning_checkpoint(tmp_path, monkeypatch, capsys):
    import json

    configuration = tmp_path / "training.json"
    configuration.write_text(json.dumps({"stage": "s2", "checkpoint": "starting-weights.pth"}))
    monkeypatch.setattr(
        "urban_change.training.engine.train",
        lambda config: {"checkpoint": config.checkpoint, "resume": config.resume},
    )
    main(["train", "--config", str(configuration), "--resume", "run/last.pth"])
    result = json.loads(capsys.readouterr().out)
    assert result == {"checkpoint": None, "resume": "run/last.pth"}
