# Project reorganization plan

## 1. Establish a recoverable baseline

- Inspect live architecture, training stages, data encodings, checkpoints, optional demos, and dependencies.
- Preserve all original tracked files before restructuring.
- Capture legacy model outputs and a compatibility checkpoint with explicitly untrained task heads.
- Inspect actual Hub cards, artifact revisions/checksums, and source licenses.

## 2. Separate reusable code from research artifacts

- Installable `src/urban_change` package with one model graph and one inference implementation.
- Clear data, training, evaluation, UI, and API boundaries.
- Keep published vocabularies; regenerate bulk metadata outside version control.
- Move datasets, weights, METEOR resources, and run outputs to ignored local directories.
- Remove vendored lagent framework, duplicate entry points, obsolete commented implementations, unrelated diagrams, copied papers, and directory dumps.

## 3. Correct workflow defects

- Device selection, strict checkpoint loading, and no unrestricted pickle fallback.
- Portable path construction, validated labels, deterministic metadata, training-only vocabulary construction, and explicit vocabulary transplantation.
- Source-preserving temporal reversal with split provenance and no silent caption fallback.
- Correct train/eval modes, partial accumulation flush, atomic checkpoints, and complete resume state.
- Explicit metric omissions, sample counts, report provenance, and API readiness.

## 4. Write a distinct project identity

- New README centered on urban change interpretation and temporal reversal.
- Attributed original dataset illustration, with no borrowed architecture graphic presented as original work.
- Model and dataset card drafts describing provenance, intended use, limits, and artifact identity.
- Mixed licensing documented accurately; preserve NVIDIA non-commercial restrictions and research attribution.

## 5. Validate and prepare review

- Unit/functional tests for data, augmentation, decoding, metrics, CLI, UI, artifacts, and API behavior.
- Numerical comparison with the original graph and real full-model execution.
- CPU fine-tuning, a partial accumulation window, and checkpoint resume.
- Package build, lint/format checks, live HTTP requests, documentation/asset checks, and current-tree credential-pattern inspection.
- Prepare a review branch and PR if hosting access permits; rename the remote repository as requested after the changes are concrete.

## External prerequisites

- Save network additions for GitHub API and the observed Hugging Face file host to enable remote updates and real-checkpoint downloads.
- Supply the removal-case screenshot from the user’s local Windows machine.
- Confirm artifact licensing and annotation provenance before unrestricted publication.
- Run the full training/evaluation protocol on the actual dataset and a suitable GPU; do not replace it with smoke-test numbers.

## Execution status

The source reorganization, workflow fixes, documentation/cards, numerical regression, full-model CPU inference, training/resume checks, distribution build, and fresh-wheel validation are implemented. The trained checkpoint is now downloaded and verified; real chronological/reversed pairs and live HTTP inference have run. ZIP metadata and caption JSON were inspected for both hosted archives.

The source changes are merged into `main` through [PR #1](https://github.com/Sushant-Nemade/urban-change-detection-captioning/pull/1). The pull request passed CI on Python 3.11 and 3.12. The default checkout now contains the reorganized project; no review-branch adoption is required for new tasks.

The owner renamed the repository to `urban-change-detection-captioning` and made it public; both settings have been verified through the GitHub API. Canonical project links use the new URL. GitHub authentication and repository-local commit identity are verified as Sushant-Nemade / Sushant Nemade.

Hugging Face write authentication is absent. The author’s local removal-case montage, licensing decision, full benchmark, and GPU validation remain outside the completed source changes. See the final validation report and publication notes for the exact evidence and pending owner actions.
