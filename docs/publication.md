# Publication notes

## Current observations

- Hugging Face API responses report both `sushantnemade/MCI_augmented` and `sushantnemade/LEVIR-MCI-augmented` as **public**, despite the intended later publication described in the request. Confirm their visibility in the account settings if that was unintended; this task has not changed visibility.
- Existing model metadata declares Apache-2.0; dataset metadata declares MIT. The SegFormer-derived code has a separate NVIDIA source license with a non-commercial-use limitation. Resolve the scope before presenting the release as unrestricted or commercial-use compatible.
- The model and dataset card replacements under `docs/huggingface/` are local drafts. Publishing them requires authenticated write access and a decision on the model license metadata. No credential value belongs in Git, documentation, or chat.
- The requested GitHub name is `urban-change-detection-captioning`. The repository is currently private. The authenticated rename request was denied with `403 Resource not accessible by integration`; account admin status does not grant this integration repository-administration scope. An owner must complete the rename in GitHub settings. The project title and package use the requested name. Repository hyperlinks point to the current working URL and will continue to redirect after an owner rename; update canonical URLs after the rename succeeds.

## Concrete release work

1. Review the source diff, component notices, benchmark protocol, and local model/dataset cards.
2. Supply the removal-case montage and verify any claims against the actual checkpoint and images.
3. Confirm archive contents, split grouping, generated-caption provenance, and applicable asset rights. Current cards explicitly identify these evidence gaps.
4. Enable required network destinations and verify the trained checkpoint against the pinned checksum. Run inference on real A/B pairs and the documented full evaluation before adding benchmark numbers.
5. Review and merge the source changes; apply the requested repository rename through GitHub access when available.
6. Upload each card as its Hugging Face repository’s `README.md` after licensing review and authenticated approval of the concrete drafts. Preserve artifact files and visibility settings.

## Git history

Current-tree cleanup does not erase previous commits. Old binaries, copied documents, local path strings, and obsolete source remain in history. Do not force-push or rewrite history merely to make a clean current tree appear small. A separate history migration would need an explicit scope, a verified backup, and coordination with other clones.

Before changing a repository from private to public, review its full history for credentials and materials that should not be redistributed. A current-tree pattern scan is useful evidence but is not a complete secret/history audit.
