"""COCO caption metrics and explicit segmentation class accounting."""

import contextlib
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from urban_change.evaluation._vendor.bleu.bleu import Bleu
from urban_change.evaluation._vendor.cider.cider import Cider
from urban_change.evaluation._vendor.rouge.rouge import Rouge


class SegmentationMetrics:
    def __init__(self, classes=3):
        self.confusion = np.zeros((classes, classes), dtype=np.int64)

    def update(self, truth, prediction):
        truth, prediction = np.asarray(truth), np.asarray(prediction)
        if truth.shape != prediction.shape:
            raise ValueError("Ground-truth and prediction shapes differ")
        classes = len(self.confusion)
        if (
            not np.isin(truth, range(classes)).all()
            or not np.isin(prediction, range(classes)).all()
        ):
            raise ValueError("Labels must lie within the configured class range")
        indices = classes * truth.astype(np.int64).reshape(-1) + prediction.astype(
            np.int64
        ).reshape(-1)
        self.confusion += np.bincount(indices, minlength=classes * classes).reshape(
            classes, classes
        )

    def compute(self):
        if not self.confusion.sum():
            raise ValueError("No pixels were evaluated")
        true_positive = np.diag(self.confusion).astype(float)
        actual, predicted = self.confusion.sum(1), self.confusion.sum(0)
        union = actual + predicted - true_positive
        iou = np.divide(
            true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0
        )
        recall = np.divide(
            true_positive, actual, out=np.full_like(true_positive, np.nan), where=actual > 0
        )
        return {
            "pixel_accuracy": float(true_positive.sum() / actual.sum()),
            "mean_class_accuracy": float(np.nanmean(recall)),
            "mean_iou": float(np.nanmean(iou)),
            "class_iou": [float(x) if np.isfinite(x) else None for x in iou],
            "frequency_weighted_iou": float(np.nansum(actual / actual.sum() * iou)),
            "confusion_matrix": self.confusion.tolist(),
        }


def meteor_score(references, hypotheses, meteor_dir):
    directory = Path(meteor_dir).resolve()
    jar = directory / "meteor-1.5.jar"
    if shutil.which("java") is None or not jar.is_file():
        raise RuntimeError(
            "METEOR requires Java and a directory containing meteor-1.5.jar and its data/"
        )
    counts = {len(refs) for refs in references}
    if len(counts) != 1:
        raise ValueError("METEOR batch scoring requires equal reference counts")
    with tempfile.TemporaryDirectory(prefix="urban-meteor-") as temporary:
        hypothesis_path, reference_path = (
            Path(temporary) / "hypotheses.txt",
            Path(temporary) / "references.txt",
        )
        hypothesis_path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
        reference_path.write_text(
            "\n".join(ref for refs in references for ref in refs) + "\n", encoding="utf-8"
        )
        result = subprocess.run(
            [
                "java",
                "-Xmx2G",
                "-jar",
                str(jar),
                str(hypothesis_path),
                str(reference_path),
                "-l",
                "en",
                "-norm",
                "-r",
                str(next(iter(counts))),
            ],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
    match = re.search(r"Final score:\s*([\d.eE+-]+)", result.stdout)
    if match is None:
        raise RuntimeError("METEOR did not produce a final score")
    return float(match.group(1))


def caption_metrics(references: list[list[str]], hypotheses: list[str], meteor_dir=None) -> dict:
    if not references or len(references) != len(hypotheses) or any(not refs for refs in references):
        raise ValueError("Caption scoring requires nonempty, matching references and hypotheses")
    wrapped = [[hypothesis] for hypothesis in hypotheses]
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, _ = Bleu(4).compute_score(references, wrapped)
        rouge, _ = Rouge().compute_score(references, wrapped)
        cider, _ = Cider().compute_score(references, wrapped)
    scores = {f"bleu_{index + 1}": float(value) for index, value in enumerate(bleu)}
    scores.update(rouge_l=float(rouge), cider=float(cider))
    if meteor_dir is not None:
        scores["meteor"] = meteor_score(references, hypotheses, meteor_dir)
    return scores
