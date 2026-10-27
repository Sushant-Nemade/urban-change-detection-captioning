"""Command-line workflows for the project."""

import argparse
import json
import os
from pathlib import Path

from urban_change import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="urban-change", description="Urban Change Detection & Change Captioning"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser(
        "predict", help="Generate a change mask and caption from an ordered image pair"
    )
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image-a", required=True, help="Earlier image")
    predict.add_argument("--image-b", required=True, help="Later image")
    predict.add_argument("--vocab")
    predict.add_argument("--device", default="auto")
    predict.add_argument("--beam-size", type=int, default=5)
    predict.add_argument("--mask-size", choices=["a", "b", "target"], default="a")
    predict.add_argument("--output", required=True)
    predict.add_argument("--overwrite", action="store_true")

    prepare = commands.add_parser(
        "prepare", help="Tokenize annotations and create reproducible split metadata"
    )
    prepare.add_argument("--captions", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--vocab", help="Reuse a checkpoint's vocabulary without rebuilding token indices"
    )
    prepare.add_argument("--min-count", type=int, default=5)
    prepare.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser(
        "validate-data", help="Check dataset files, masks, captions, and split identities"
    )
    validate.add_argument("--images", required=True)
    validate.add_argument("--metadata", required=True)
    validate.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"]
    )
    validate.add_argument("--limit", type=int)

    augment = commands.add_parser(
        "augment", help="Swap A/B and reverse captions into a separate dataset"
    )
    augment.add_argument("--captions", required=True)
    augment.add_argument("--images", required=True)
    augment.add_argument("--output", required=True)
    augment.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train"])
    source = augment.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--caption-map", help="JSON mapping split/filename to reviewed reversed captions"
    )
    source.add_argument(
        "--azure",
        action="store_true",
        help="Generate reversed captions through configured Azure OpenAI",
    )
    augment.add_argument("--env-file", help="Optional local dotenv file for Azure configuration")

    train = commands.add_parser("train", help="Train or fine-tune with resumable checkpoints")
    train.add_argument("--config", help="JSON training configuration")
    for option in [
        "images",
        "metadata",
        "output",
        "checkpoint",
        "source-vocab",
        "backbone-weights",
        "resume",
        "device",
    ]:
        train.add_argument("--" + option, default=argparse.SUPPRESS)
    train.add_argument("--stage", choices=["s1", "s2"], default=argparse.SUPPRESS)
    train.add_argument(
        "--goal", choices=["joint", "caption", "detection"], default=argparse.SUPPRESS
    )
    for option in [
        "epochs",
        "batch-size",
        "accumulation-steps",
        "workers",
        "seed",
        "patience",
        "lr-patience",
        "beam-size",
        "train-limit",
        "val-limit",
    ]:
        train.add_argument("--" + option, type=int, default=argparse.SUPPRESS)
    for option in ["encoder-lr", "decoder-lr", "weight-decay", "grad-clip", "label-smoothing"]:
        train.add_argument("--" + option, type=float, default=argparse.SUPPRESS)
    for option in ["random-init", "amp", "loss-balance", "unfreeze-caption-neck"]:
        train.add_argument(
            "--" + option, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
        )

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a checkpoint on a named dataset split"
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--images", required=True)
    evaluate.add_argument("--metadata", required=True)
    evaluate.add_argument("--vocab")
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--beam-size", type=int, default=5)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument(
        "--meteor-dir", help="Optional official METEOR 1.5 directory; requires Java"
    )
    evaluate.add_argument("--output", required=True, help="Destination JSON report")

    download = commands.add_parser(
        "download", help="Download and verify a pinned model, backbone, or dataset"
    )
    download.add_argument(
        "artifact", choices=["model", "backbone", "dataset-original", "dataset-augmented"]
    )
    download.add_argument("--output", required=True)
    return parser


def dispatch(args):
    if args.command == "predict":
        from PIL import Image

        from urban_change.inference import ChangePredictor, save_prediction

        predictor = ChangePredictor(args.checkpoint, args.vocab, args.device)
        with Image.open(args.image_a) as a, Image.open(args.image_b) as b:
            prediction = predictor.predict(a, b, args.beam_size, args.mask_size)
            save_prediction(prediction, args.output, args.overwrite, image_a=a, image_b=b)
        return prediction.metadata()
    if args.command == "prepare":
        from urban_change.data.prepare import prepare_metadata

        return prepare_metadata(
            args.captions, args.output, args.vocab, args.min_count, args.overwrite
        )
    if args.command == "validate-data":
        from urban_change.data.validate import validate_dataset

        return validate_dataset(args.images, args.metadata, args.splits, args.limit)
    if args.command == "augment":
        from urban_change.data.augment import augment_dataset, azure_rewriter

        mapping = (
            json.loads(Path(args.caption_map).read_text(encoding="utf-8"))
            if args.caption_map
            else None
        )
        rewrite = None
        if args.azure:
            if args.env_file:
                from dotenv import load_dotenv

                load_dotenv(args.env_file)
            required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
            missing = [name for name in required if not os.environ.get(name)]
            if missing:
                raise ValueError("Missing Azure configuration: " + ", ".join(missing))
            rewrite = azure_rewriter(
                os.environ[required[1]], os.environ[required[2]], os.environ[required[0]]
            )
        return augment_dataset(
            args.captions, args.images, args.output, tuple(args.splits), mapping, rewrite
        )
    if args.command == "train":
        from urban_change.training.engine import TrainConfig, train

        config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        overrides = {
            key: value for key, value in vars(args).items() if key not in {"command", "config"}
        }
        if overrides.get("resume") and "checkpoint" not in overrides:
            config["checkpoint"] = None
        config.update(overrides)
        return train(TrainConfig(**config))
    if args.command == "evaluate":
        from urban_change.evaluation.runner import evaluate_checkpoint

        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Evaluation report already exists: {output}")
        report = evaluate_checkpoint(
            args.checkpoint,
            args.images,
            args.metadata,
            args.split,
            args.device,
            args.vocab,
            args.beam_size,
            args.limit,
            args.meteor_dir,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    if args.command == "download":
        from urban_change.artifacts import download_artifact

        return {
            "artifact": args.artifact,
            "path": str(download_artifact(args.artifact, args.output)),
        }
    raise ValueError("Unknown command")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f"urban-change: {exc}\n")
    print(json.dumps(result, indent=2))
