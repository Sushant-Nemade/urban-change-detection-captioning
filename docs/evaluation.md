# Evaluation protocol

```bash
urban-change evaluate --checkpoint artifacts/MCI_model_augmented.pth \
  --images data/raw/levir_mci/images \
  --metadata data/processed/levir_mci-published \
  --split test --beam-size 5 --device auto --output outputs/test-evaluation.json
```

Evaluation uses batch size one for caption generation. It records checkpoint and split-file SHA-256, selected split, sample limit, beam size, vocabulary size, sample count, elapsed time, and computed metrics. Existing report files are protected. A zero-sample run fails.

## Metrics

- Detection: pixel accuracy, mean class accuracy, per-class IoU, mean IoU, frequency-weighted IoU, and the confusion matrix.
- Captioning: corpus BLEU-1 through BLEU-4, ROUGE-L, and CIDEr using the preserved COCO-caption implementations.
- Optional METEOR: provide `--meteor-dir` pointing to the official METEOR 1.5 directory with its JAR and `data/`, and install Java. The report explicitly lists METEOR as omitted when it is not requested. This optional Java path has not been validated on the current machine.

Caption scores use decoded tokens with START/END/PAD removed. This matters for METEOR: the old evaluator converted word IDs into strings, losing lexical stemming/synonym behavior. Do not silently compare the two METEOR protocols. Pure n-gram metrics preserve the identity structure, but complete benchmark reproduction still requires matching preprocessing and vocabulary.

Classes absent from both truth and prediction have `null` IoU and do not contribute to mean IoU. A class predicted incorrectly where it is absent from truth still contributes a zero IoU. Label ranges and array shapes are validated.

## Reporting removal cases

Report original chronological pairs and reversed pairs separately, with construction/removal cases identified from reviewed annotations. Keep each geographic pair and all its reversals in one split. Reversed examples are correlated views of the same scenes, not additional independent test scenes.

Record dataset revision, checkpoint hash, seed, beam size, vocabulary mapping, normalization, mask encoding, and any test-time augmentation. The new evaluator performs no hidden test-time augmentation. `--limit` is for smoke checks; a limited run must not be labeled a full benchmark.

No benchmark numbers are inferred from checkpoint filenames or reproduced as measured results without executing their exact evaluation protocol.
