# Publication notes

## Current observations

- Hugging Face API responses report both `sushantnemade/MCI_augmented` and `sushantnemade/LEVIR-MCI-augmented` as **public**, despite the intended later publication described in the request. Confirm their visibility in the account settings if that was unintended; this task has not changed visibility.
- Existing model metadata declares Apache-2.0; dataset metadata declares MIT. The SegFormer-derived code has a separate NVIDIA source license with a non-commercial-use limitation. Resolve the scope before presenting the release as unrestricted or commercial-use compatible.
- The model and dataset card replacements under `docs/huggingface/` are local drafts. Publishing them requires authenticated write access and a decision on the model license metadata. No credential value belongs in Git, documentation, or chat.
- The requested GitHub name is `urban-change-detection-captioning`. The repository is currently private. The authenticated rename request was denied with `403 Resource not accessible by integration`; account admin status does not grant this integration repository-administration scope. An owner must complete the rename in GitHub settings. The project title and package use the requested name. Repository hyperlinks point to the current working URL and will continue to redirect after an owner rename; update canonical URLs after the rename succeeds.

## Completed source release

- [PR #1](https://github.com/Sushant-Nemade/ChangeDetection-ChangeCaptioning/pull/1) is merged into `main`. It includes the source reorganization, rewritten README, workflow fixes, attributed qualitative examples, component notices, and local model/dataset card drafts.
- CI passed on Python 3.11 and 3.12 before merging. The exact local validation scope is recorded in [the validation report](validation-report.json).
- The trained checkpoint was downloaded and SHA-256 verified. Real original/reversed pairs, packaged inference, and the HTTP service were exercised. These are qualitative/software checks, not a full benchmark.
- Reusable cloud installation and startup instructions target the reorganized source now available on `main`. Saving these instructions does not publish a new environment snapshot.

## Remaining external work

1. Rename the repository to `urban-change-detection-captioning` in [GitHub repository settings](https://github.com/Sushant-Nemade/ChangeDetection-ChangeCaptioning/settings). Retrying the API request after the source changes still returned `403 Resource not accessible by integration`; repository-administration access is required. Update canonical URLs after the rename succeeds.
2. Supply the removal-case montage from the author’s Windows machine and verify its claims against the actual checkpoint and images. That local file is unavailable in the cloud checkout.
3. Confirm split grouping, generated-caption provenance, and applicable asset rights. Archive indices and caption JSON were inspected; complete archive hashes and all image contents were not audited. The cards identify these evidence gaps.
4. Run the documented full evaluation on the actual dataset before adding benchmark numbers. GPU/AMP execution remains unverified in this CPU environment.
5. Configure an `HF_TOKEN` with write access to the two Hugging Face repositories through secure environment settings, then upload each reviewed card as that repository’s `README.md`. No Hugging Face token or cached login is available in this environment. Resolve the model-license metadata against upstream component terms before publishing that replacement card; preserve artifact files and visibility settings.

## Git history

Current-tree cleanup does not erase previous commits. Old binaries, copied documents, local path strings, and obsolete source remain in history. Do not force-push or rewrite history merely to make a clean current tree appear small. A separate history migration would need an explicit scope, a verified backup, and coordination with other clones.

Before changing a repository from private to public, review its full history for credentials and materials that should not be redistributed. A current-tree pattern scan is useful evidence but is not a complete secret/history audit.
