# CDCC_MT_01

This workspace contains the code and resources for the Master Thesis project under:

- [MasterThesis-ChangeAgent/](MasterThesis-ChangeAgent/)

The main runnable component is the change detection + captioning project in:

- [MasterThesis-ChangeAgent/Multi_change/](MasterThesis-ChangeAgent/Multi_change/)

## Quickstart (Windows)

### 1) Create a virtual environment

From the repository root:

```powershell
cd MasterThesis-ChangeAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

Core (training + inference + Streamlit demo):

```powershell
pip install -r Multi_change\requirements.txt
```

For exact reproduction (fully pinned snapshot), you can use:

```powershell
pip install -r Multi_change\requirements-lock.txt
```

### 3) Add SegFormer pretrained weights

Place SegFormer weights under:

- `MasterThesis-ChangeAgent/Multi_change/model/pretrained/`

At least `mit_b1.pth` is required for the default backbone.

### 4) Run the Streamlit demo

```powershell
cd Multi_change
python -m streamlit run streamlit_app.py
```

## Training / evaluation

See the detailed guide in:

- [MasterThesis-ChangeAgent/README_MasterThesis-ChangeAgent.md](MasterThesis-ChangeAgent/README_MasterThesis-ChangeAgent.md)

## Optional extras

These are not required for core training/inference:

- Dataset augmentation using Azure/OpenAI: `pip install -r Multi_change\requirements-augmentation.txt`
- Experimental lagent-based agent demo: `pip install -r Multi_change\requirements-agent.txt`

### Which requirements file do I need?

- Core training/inference + Streamlit demo (`streamlit_app.py`): `Multi_change\requirements.txt`
- Exact reproduction (fully pinned): `Multi_change\requirements-lock.txt`
- Dataset augmentation (`augment_dataset_new.py`, `azure_diagnostic.py`): `Multi_change\requirements-augmentation.txt`
- Experimental lagent demo (`web_demo.py`, `try_chat.py`): `Multi_change\requirements-agent.txt`

### Local-only (not uploaded)

Some experimental files are kept untracked on purpose (ignored via `.gitignore`) and will not appear in the remote repository, including:

- `MasterThesis-ChangeAgent/Multi_change/agent_web_demo.py`
- `MasterThesis-ChangeAgent/Multi_change/local_predictor.py`

