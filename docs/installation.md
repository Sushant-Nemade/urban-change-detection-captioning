# Installation

Python 3.11–3.12 is supported by the package. Python 3.12 on Linux CPU is validated here. Use a virtual environment and matching PyTorch/torchvision builds.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[demo,api,augmentation,dev]'
```

Windows activation is `.venv\Scripts\Activate.ps1`. Headless OpenCV needs no desktop display. MMCV, MMSegmentation, Transformers, and torchaudio are no longer required.

## Exact Linux CPU development snapshot

```bash
python -m pip install -r requirements/cpu.lock --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
```

The lock includes optional demo/API/augmentation and development dependencies for Python 3.12. It is not a portable CUDA or Windows lockfile. The project is installed separately so the lock contains no machine-specific source path.

## CUDA

Choose matching PyTorch 2.14.1 / torchvision 0.29.1 builds compatible with your driver using [PyTorch's official installation instructions](https://pytorch.org/get-started/locally/). Install the project after the matching wheels. The CPU lock is not a CUDA dependency specification.

GPU execution and mixed precision are implemented but untested on this CPU-only cloud machine. Keep CPU and CUDA snapshots separate.

## Verify

```bash
urban-change --help
python -m pytest -m 'not integration'
python -m streamlit run apps/streamlit_app.py --server.headless=true --browser.gatherUsageStats=false
```

Network destinations include PyPI, the PyTorch wheel index, `huggingface.co`, its redirected file host (`us.aws.cdn.hf.co` for the observed checkpoint), and `raw.githubusercontent.com` for the pinned backbone. GitHub API operations also need `api.github.com`. Keep TLS verification and checksum checks enabled.

Inference needs a full MCI checkpoint, but no dataset, Azure credentials, or separate backbone. Azure augmentation is optional; a reviewed local caption map works offline.
