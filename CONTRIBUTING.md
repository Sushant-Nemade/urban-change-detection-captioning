# Contributing

Create a branch in the existing checkout. Install `.[demo,api,augmentation,dev]` after choosing matching CPU/CUDA PyTorch builds. Do not commit datasets, pretrained weights, generated tokens, run outputs, `.env` files, or credentials.

```bash
python -m pytest -m 'not integration'
ruff check src tests apps
ruff format --check src tests apps
python -m build
python scripts/check_release.py
python scripts/check_dependencies.py
```

Model changes must preserve existing checkpoint keys or provide an explicit versioned migration. When a compatible local checkpoint and captured legacy outputs are available:

```bash
export URBAN_CHANGE_TEST_CHECKPOINT=/absolute/path/to/compatibility-checkpoint.pth
export URBAN_CHANGE_TEST_GOLDEN=/absolute/path/to/legacy-golden.npz
python -m pytest tests/test_integration.py
```

These checks execute full model inference, HTTP prediction, joint optimization, and CPU caption fine-tuning/resume on temporary image fixtures. Compatibility weights with untrained heads verify execution and regression behavior, not prediction quality. A golden file must come from the original implementation, not from the code under test. The dependency check queries PyPI's advisory metadata for the locked versions and requires network access; it is not a complete security audit.

Keep original/reversed pairs in one split. Report complete experiment settings with metrics, preserve third-party attribution and component licenses, and add regression coverage for behavioral fixes. Changes to training or preprocessing can invalidate comparisons with historical scores.
