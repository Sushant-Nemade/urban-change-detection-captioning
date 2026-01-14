# CDCC_MT_01



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:
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

