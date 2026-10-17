import io
import json
from datetime import datetime

import numpy as np
import pytest
import torch
from PIL import Image

from urban_change.artifacts import artifact_manifest, download_artifact
from urban_change.checkpoints import atomic_save, load_checkpoint
from urban_change.evaluation.metrics import SegmentationMetrics, caption_metrics
from urban_change.inference import PALETTE, Prediction, prepare_image, save_prediction
from urban_change.models.decoder import DecoderTransformer
from urban_change.runtime import resolve_device


def test_preprocessing_rgb_grayscale_and_alpha():
    for mode in ("RGB", "L", "RGBA"):
        result = prepare_image(Image.new(mode, (120, 180), 0))
        assert result.shape == (1, 3, 256, 256)
        assert result.dtype == torch.float32 and torch.isfinite(result).all()
    expected = (0 - 0.39073 * 255) / (0.15329 * 255)
    assert prepare_image(Image.new("RGB", (256, 256), 0))[0, 0, 0, 0] == pytest.approx(expected)


def test_metrics_handle_absent_classes_and_validate_labels():
    metrics = SegmentationMetrics()
    with pytest.raises(ValueError, match="No pixels"):
        metrics.compute()
    metrics.update(np.array([0, 1, 1]), np.array([0, 1, 2]))
    report = metrics.compute()
    assert report["pixel_accuracy"] == pytest.approx(2 / 3)
    assert report["class_iou"] == [1.0, 0.5, 0.0]
    metrics = SegmentationMetrics()
    metrics.update(np.zeros(3), np.zeros(3))
    assert metrics.compute()["class_iou"] == [1.0, None, None]
    with pytest.raises(ValueError):
        metrics.update(np.array([0]), np.array([3]))


def test_caption_metrics_match_exact_and_mismatched_captions():
    refs = [["a building was added"], ["a road was removed"]]
    perfect = caption_metrics(refs, [r[0] for r in refs])
    mismatched = caption_metrics(refs, ["trees trees trees trees"] * 2)
    assert perfect["bleu_4"] > 0.99 and perfect["rouge_l"] == 1
    assert perfect["cider"] > mismatched["cider"]
    assert "meteor" not in perfect
    with pytest.raises(ValueError):
        caption_metrics([], [])


def test_decoder_gradients_and_beam_search():
    vocab = {"<NULL>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3, "building": 4, "removed": 5}
    decoder = DecoderTransformer(32, 32, len(vocab), 6, vocab, 4, 1, 0.1).eval()
    a, b = torch.randn(1, 32, 2, 2), torch.randn(1, 32, 2, 2)
    tokens = torch.tensor([[2, 4, 5, 3]])
    scores, _, lengths, _ = decoder(a, b, tokens, torch.tensor([4]))
    loss = scores[:, :3].square().mean()
    loss.backward()
    assert lengths == [3]
    assert decoder.vocab_embedding.weight.grad.abs().sum() > 0
    assert decoder.wdc.weight.data_ptr() == decoder.vocab_embedding.weight.data_ptr()
    with torch.inference_mode():
        sequence = decoder.sample_beam(a, b, k=3)
    assert sequence[0] == 2 and 2 <= len(sequence) <= 7
    assert all(0 <= token < len(vocab) for token in sequence)


def test_prediction_outputs_roundtrip_and_preserve_existing_files(tmp_path):
    labels = np.array([[0, 1, 2]], dtype=np.uint8)
    prediction = Prediction("a building was removed", labels, PALETTE[labels], 12.5, "a" * 64)
    output = save_prediction(prediction, tmp_path / "results")
    assert np.array_equal(np.array(Image.open(output / "mask.png")), PALETTE[labels])
    assert json.loads((output / "prediction.json").read_text())["caption"] == prediction.caption
    with pytest.raises(FileExistsError):
        save_prediction(prediction, output)


def test_tensor_checkpoint_roundtrip_and_rejection_of_pickle_objects(tmp_path):
    state = {
        key: {"x": torch.ones(1)} for key in ("encoder_dict", "encoder_trans_dict", "decoder_dict")
    }
    path = tmp_path / "checkpoint.pth"
    atomic_save(state, path)
    assert load_checkpoint(path)["encoder_dict"]["x"].item() == 1
    state["unsupported_object"] = datetime(2026, 1, 1)
    atomic_save(state, path)
    with pytest.raises(ValueError, match="unsupported pickled"):
        load_checkpoint(path)
    atomic_save({"not_a_model": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(path)


def test_download_integrity_and_existing_artifact_protection(tmp_path, monkeypatch):
    payload = b"incorrect checkpoint"
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    output = tmp_path / "model.pth"
    with pytest.raises(ValueError, match="SHA-256"):
        download_artifact("model", output)
    assert not output.exists() and not list(tmp_path.glob("*.part"))
    output.write_bytes(payload)
    with pytest.raises(FileExistsError):
        download_artifact("model", output)
    assert len(artifact_manifest()["model"]["sha256"]) == 64


def test_device_selection_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_device()) == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_device("cuda")


def test_finetuning_transplants_rows_by_token_identity():
    from types import SimpleNamespace

    from urban_change.training.engine import initialize_finetuning

    source_vocab = {"<NULL>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3, "road": 4, "house": 5}
    target_vocab = {
        "<NULL>": 0,
        "<UNK>": 1,
        "<START>": 2,
        "<END>": 3,
        "house": 4,
        "road": 5,
        "new": 6,
    }

    def model(vocab):
        return SimpleNamespace(
            encoder=torch.nn.Linear(2, 2),
            encoder_trans=torch.nn.Linear(2, 2),
            decoder=DecoderTransformer(32, 32, len(vocab), 6, vocab, 4, 1, 0.1),
        )

    source, target = model(source_vocab), model(target_vocab)
    with torch.no_grad():
        for row in range(len(source_vocab)):
            source.decoder.vocab_embedding.weight[row].fill_(row)
            source.decoder.wdc.bias[row] = row
    new_row = target.decoder.vocab_embedding.weight[6].detach().clone()
    state = {
        "vocab": source_vocab,
        "encoder_dict": source.encoder.state_dict(),
        "encoder_trans_dict": source.encoder_trans.state_dict(),
        "decoder_dict": source.decoder.state_dict(),
    }
    initialize_finetuning(target, state, target_vocab)
    assert (target.decoder.vocab_embedding.weight[4] == 5).all()
    assert (target.decoder.vocab_embedding.weight[5] == 4).all()
    assert target.decoder.wdc.bias[4] == 5
    assert torch.equal(target.decoder.vocab_embedding.weight[6], new_row)
    state["decoder_dict"].pop("Conv1.weight")
    with pytest.raises(ValueError, match="missing decoder"):
        initialize_finetuning(target, state, target_vocab)
