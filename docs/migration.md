# Migration from the thesis workspace

The project title is **Urban Change Detection & Change Captioning**; the requested repository slug is `urban-change-detection-captioning`. Hosting changes happen separately from these source edits.

| Previous location / script | Replacement |
|---|---|
| `MasterThesis-ChangeAgent/Multi_change/model/` | `src/urban_change/models/` |
| `data/LEVIR_MCI/vocab*.json` | Packaged vocabularies in `src/urban_change/assets/vocab/` |
| Generated split lists and token files | Ignored `data/processed/levir_mci/`; reproduce with `urban-change prepare` |
| `predict.py`, `predict_new.py`, `tools.py` | `urban-change predict` / `ChangePredictor` |
| `streamlit_app.py` | `python -m streamlit run apps/streamlit_app.py` |
| `api_app.py` | `python -m uvicorn urban_change.api:app` |
| `train.py`, `train_optimized.py`, `train_newoptimized.py`, `train_eval_only.py` | `urban-change train` with explicit stage/goal/configuration |
| `test.py`, `test_new.py`, `eval_many.py` | `urban-change evaluate`; invoke separately per checkpoint and report |
| `test_dataset.py`, `OOV_measuring.py` | `urban-change validate-data` |
| `preprocess_data.py` | `urban-change prepare` |
| `augment_dataset_new.py`, `azure_diagnostic.py` | `urban-change augment`, explicit configuration and errors |
| `lagent-main`, `try_chat.py`, `web_demo.py` | Removed experimental agent workflow; use Git history if research requires it |
| `resource/` | One attributed original example figure in `docs/assets/`; unrelated agent graphics removed |
| Reference PDFs and recursive directory dumps | Literature links and concise architecture documentation |
| Tracked pretrained weights and METEOR binaries | Local ignored `artifacts/`; checksummed backbone download and optional external METEOR |

Parameter names and the numerical model graph remain compatible with the original state-dict checkpoint format. New training snapshots add vocabulary, architecture configuration, optimizer/scheduler/scaler state, and RNG state. Old weights-only snapshots are fine-tuning inputs, not complete resumable experiments.

CPU device handling replaces unconditional CUDA calls. Missing model weights now fail strict loading instead of silently leaving random neck parameters. Split/vocabulary preparation and training-mode/accumulation fixes intentionally change broken workflow behavior; old benchmark results require their original protocol.

The reorganization removes files from the current source tree. It does not rewrite Git history. Previous code, binaries, and historical paths remain recoverable there; changing history requires a separate, explicitly reviewed migration.
