"""Staged optimization with deterministic data loading and resumable state."""

import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from urban_change.checkpoints import atomic_save, load_checkpoint
from urban_change.data.dataset import LevirMCIDataset
from urban_change.data.text import load_vocab, validate_vocab
from urban_change.evaluation.runner import evaluate_model
from urban_change.models.system import ChangeModel, ModelConfig
from urban_change.runtime import resolve_device


@dataclass
class TrainConfig:
    images: str = ""
    metadata: str = ""
    output: str = "runs/joint"
    stage: str = "s1"
    goal: str = "joint"
    checkpoint: str | None = None
    source_vocab: str | None = None
    backbone_weights: str | None = None
    random_init: bool = False
    resume: str | None = None
    device: str = "auto"
    epochs: int = 30
    batch_size: int = 8
    accumulation_steps: int = 8
    workers: int = 0
    encoder_lr: float = 1e-4
    decoder_lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    loss_balance: bool = False
    unfreeze_caption_neck: bool = False
    amp: bool = True
    seed: int = 42
    patience: int = 10
    lr_patience: int = 5
    beam_size: int = 1
    train_limit: int | None = None
    val_limit: int | None = None
    model: dict = field(default_factory=dict)

    def validate(self):
        if not self.images or not self.metadata:
            raise ValueError(
                "Training requires --images and --metadata, directly or through --config"
            )
        if self.stage not in {"s1", "s2"} or self.goal not in {"joint", "caption", "detection"}:
            raise ValueError("Invalid training stage or goal")
        if (
            min(
                self.epochs,
                self.batch_size,
                self.accumulation_steps,
                self.patience,
                self.lr_patience,
            )
            < 1
            or self.workers < 0
        ):
            raise ValueError("Epochs, batch sizes, accumulation, and patience must be positive")
        if self.stage == "s2" and not (self.checkpoint or self.resume):
            raise ValueError("Stage s2 requires --checkpoint or --resume")
        if self.resume and self.checkpoint:
            raise ValueError("Use either --resume or --checkpoint")
        if self.stage == "s1" and not (self.backbone_weights or self.random_init or self.resume):
            raise ValueError(
                "Stage s1 requires --backbone-weights; use --random-init only for an explicit scratch experiment"
            )
        if (
            min(self.encoder_lr, self.decoder_lr, self.grad_clip) <= 0
            or self.weight_decay < 0
            or not 0 <= self.label_smoothing < 1
        ):
            raise ValueError("Invalid optimizer or label smoothing configuration")
        return self


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def configure_stage(model, config):
    goal_id = {"detection": 0, "caption": 1, "joint": 2}[config.goal]
    model.train()
    encoder_trainable = config.goal != "caption" or config.stage == "s1"
    model.encoder.fine_tune(encoder_trainable)
    model.encoder.train(encoder_trainable)
    if config.goal == "caption" and config.stage == "s2" and not config.unfreeze_caption_neck:
        model.encoder_trans.requires_grad_(False).eval()
    else:
        model.encoder_trans.fine_tune(goal_id)
    model.decoder.fine_tune(config.goal != "detection")
    model.decoder.train(config.goal != "detection")


def initialize_finetuning(model, state, target_vocab, source_vocab_path=None):
    source_vocab = load_vocab(source_vocab_path) if source_vocab_path else state.get("vocab")
    source_size = int(state["decoder_dict"]["wdc.weight"].shape[0])
    if source_vocab is None:
        source_vocab = load_vocab(size=source_size)
    validate_vocab(source_vocab)
    if len(source_vocab) != source_size:
        raise ValueError("Source vocabulary does not match checkpoint")
    model.encoder.load_state_dict(state["encoder_dict"], strict=True)
    model.encoder_trans.load_state_dict(state["encoder_trans_dict"], strict=True)
    target = model.decoder.state_dict()
    missing = target.keys() - state["decoder_dict"].keys()
    if missing:
        raise ValueError(f"Checkpoint is missing decoder parameters: {sorted(missing)}")
    token_keys = {"vocab_embedding.weight", "wdc.weight", "wdc.bias"}
    for key, value in state["decoder_dict"].items():
        if key not in target:
            raise ValueError(f"Unexpected decoder parameter: {key}")
        if key in token_keys:
            for token in source_vocab.keys() & target_vocab.keys():
                target[key][target_vocab[token]].copy_(value[source_vocab[token]])
        elif key == "position_encoding.pe":
            # Positional encoding is deterministic and can be regenerated for a
            # different maximum caption length without resizing learned weights.
            if value.shape == target[key].shape:
                target[key].copy_(value)
        elif value.shape != target[key].shape:
            raise ValueError(f"Decoder architecture mismatch: {key}")
        else:
            target[key].copy_(value)
    model.decoder.load_state_dict(target, strict=True)


def batch_loss(model, sample, config, device):
    features_a, features_b, logits = model(
        sample["image_a"].to(device), sample["image_b"].to(device)
    )
    detection = caption = None
    if config.goal != "caption":
        detection = torch.nn.functional.cross_entropy(logits, sample["labels"].to(device))
    if config.goal != "detection":
        scores, captions, lengths, _ = model.decoder(
            features_a, features_b, sample["tokens"].to(device), sample["length"].to(device)
        )
        predictions = pack_padded_sequence(scores, lengths, batch_first=True).data
        targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True).data
        caption = torch.nn.functional.cross_entropy(
            predictions, targets, label_smoothing=config.label_smoothing
        )
    terms = [loss for loss in (detection, caption) if loss is not None]
    loss = sum(
        term / (term.detach() + 1e-8) if config.loss_balance and len(terms) == 2 else term
        for term in terms
    )
    return loss, {
        "detection_loss": float(detection.detach()) if detection is not None else None,
        "caption_loss": float(caption.detach()) if caption is not None else None,
    }


def train(config: TrainConfig) -> dict:
    config.validate()
    device = resolve_device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    output = Path(config.output)
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise FileExistsError(f"Run directory is nonempty: {output}; use a new output or --resume")
    vocab = load_vocab(Path(config.metadata) / "vocab.json")
    model_config = ModelConfig(**config.model)
    model = ChangeModel(vocab, model_config, config.backbone_weights if not config.resume else None)
    resumed = load_checkpoint(config.resume) if config.resume else None
    if resumed:
        if resumed.get("vocab") != vocab or resumed.get("model_config") != asdict(model_config):
            raise ValueError("Resume vocabulary/model configuration differs from the saved run")
        previous = resumed.get("training_config", {})
        allowed_changes = {"resume", "checkpoint", "output", "device", "epochs", "workers"}
        for key, value in asdict(config).items():
            if key not in allowed_changes and previous.get(key) != value:
                raise ValueError(f"Resume configuration changed: {key}")
        model.load_legacy_state(resumed)
    elif config.checkpoint:
        initialize_finetuning(model, load_checkpoint(config.checkpoint), vocab, config.source_vocab)
    model.to(device)
    configure_stage(model, config)
    groups = []
    for module, rate in (
        (model.encoder, config.encoder_lr),
        (model.encoder_trans, config.encoder_lr),
        (model.decoder, config.decoder_lr),
    ):
        parameters = [p for p in module.parameters() if p.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": rate})
    optimizer = torch.optim.Adam(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.lr_patience
    )
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    training = LevirMCIDataset(
        config.images, config.metadata, "train", vocab, model_config.max_length, config.train_limit
    )
    validation = LevirMCIDataset(
        config.images, config.metadata, "val", vocab, model_config.max_length, config.val_limit
    )
    loader = DataLoader(
        training,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        drop_last=False,
    )
    start_epoch, best, stale = 0, -math.inf, 0
    if resumed:
        if "optimizer" not in resumed:
            raise ValueError("This is a weights-only checkpoint; use --checkpoint for fine-tuning")
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        start_epoch, best, stale = (
            resumed["epoch"] + 1,
            resumed["best_metric"],
            resumed["stale_epochs"],
        )
        torch.set_rng_state(resumed["rng"]["torch"])
        random.setstate(resumed["rng"]["python"])
        np_state = resumed["rng"]["numpy"]
        np.random.set_state(
            (
                np_state[0],
                np.array(np_state[1], dtype=np.uint32),
                np_state[2],
                np_state[3],
                np_state[4],
            )
        )
        if device.type == "cuda" and "cuda" in resumed["rng"]:
            torch.cuda.set_rng_state_all(resumed["rng"]["cuda"])
    if start_epoch >= config.epochs:
        raise ValueError("No epochs remain; increase --epochs beyond the resumed epoch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    history = []
    for epoch in range(start_epoch, config.epochs):
        # Validation always uses eval mode; reapply fine-grained train/freeze
        # states at the start of every training epoch, including after resume.
        configure_stage(model, config)
        optimizer.zero_grad(set_to_none=True)
        loss_sum, steps = 0.0, 0
        for index, sample in enumerate(loader):
            window_start = (index // config.accumulation_steps) * config.accumulation_steps
            window_size = min(config.accumulation_steps, len(loader) - window_start)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with context:
                loss, _ = batch_loss(model, sample, config, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {index}")
            scaler.scale(loss / window_size).backward()
            loss_sum += float(loss.detach())
            if (index + 1) % config.accumulation_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        metrics = evaluate_model(model, vocab, validation, device, config.beam_size, config.goal)
        score = metrics.get("segmentation", {}).get("mean_iou", 0.0) + metrics.get(
            "captioning", {}
        ).get("bleu_4", 0.0)
        scheduler.step(score)
        improved = score > best
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        np_state = np.random.get_state()
        rng = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
        }
        if device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        state = {
            **model.checkpoint_state(),
            "vocab": vocab,
            "training_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "stale_epochs": stale,
            "rng": rng,
        }
        atomic_save(state, output / "last.pth")
        if improved:
            atomic_save(state, output / "best.pth")
        report = {
            "epoch": epoch,
            "training_loss": loss_sum / len(loader),
            "optimizer_steps": steps,
            "selection_metric": score,
            "best_metric": best,
            "validation": metrics,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        history.append(report)
        if stale >= config.patience:
            break
    return {
        "output": str(output),
        "epochs_completed": len(history),
        "last_epoch": history[-1]["epoch"],
        "best_metric": best,
    }
