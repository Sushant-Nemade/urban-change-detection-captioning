# Inference and serving

Image A is earlier; image B is later. Inputs must show the same aligned ground area. The model does not register misaligned imagery.

## Python

```python
from PIL import Image
from urban_change.inference import ChangePredictor, save_prediction

predictor = ChangePredictor("artifacts/MCI_model_augmented.pth", device="auto")
with Image.open("earlier.png") as before, Image.open("later.png") as after:
    result = predictor.predict(before, after, beam_size=5, mask_size="a")
print(result.caption)
save_prediction(result, "outputs/example")
```

`mask_size` is `a`, `b`, or `target`. Label values are 0/1/2; colors are black/yellow/red. Masks use nearest-neighbor resizing. Nonempty output directories are protected unless overwriting is explicit.

Published legacy checkpoints select a bundled vocabulary by shape. Supply `vocab="path/to/vocab.json"` for custom mappings. New training snapshots embed their vocabulary. Unsupported pickled objects are rejected without unrestricted deserialization fallback.

## HTTP

```bash
python -m pip install -e '.[api]'
export URBAN_CHANGE_CHECKPOINT=artifacts/MCI_model_augmented.pth
export URBAN_CHANGE_DEVICE=auto
python -m uvicorn urban_change.api:app --host 127.0.0.1 --port 8000
```

`GET /health` reports process liveness. `GET /ready` returns 503 without a configured model, and 200 only after loading succeeds. A broken configured checkpoint fails startup.

```bash
curl --fail http://127.0.0.1:8000/ready
curl --fail -X POST 'http://127.0.0.1:8000/infer?beam_size=5&mask_size=a' \
  -F image_a=@earlier.png -F image_b=@later.png
```

Responses contain the caption, base64 PNG mask and indexed labels, checkpoint SHA-256, class definitions, and elapsed inference time. Model execution runs outside the async event loop and is serialized per instance. Each image is limited to 20 MiB and a 4096 × 4096 pixel budget.

Timing covers model execution and mask construction after preparation; it is not a throughput benchmark. Add deployment-specific authentication and routing before exposing this research service externally.
