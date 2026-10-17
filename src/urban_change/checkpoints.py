"""Validated tensor checkpoints and atomic experiment snapshots."""

import os
import pickle
import tempfile
from pathlib import Path

import torch


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Never fall back to unrestricted pickle loading. Official legacy snapshots
    # contain tensor state dictionaries and work with weights_only=True.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Checkpoint contains unsupported pickled objects; use a tensor state-dict checkpoint"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint containing model state dictionaries")
    for key in ("encoder_dict", "encoder_trans_dict", "decoder_dict"):
        if not isinstance(checkpoint.get(key), dict) or not checkpoint[key]:
            raise ValueError(f"Checkpoint is missing a nonempty {key}")
    return checkpoint


def atomic_save(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
