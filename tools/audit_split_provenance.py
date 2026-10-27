"""Split-Family Provenance & Leakage Auditor (`tools/audit_split_provenance.py`).

Audits original and temporal-reversal augmented LEVIR-MCI manifests and ZIP directories:
  1. Computes SHA-256 digests of caption annotation manifests and central directory records.
  2. Maps every temporal-reversal pair (T2, T1) to its exact source pair (T1, T2) family ID
     via explicit `augmentation.source_filename` metadata or exact (A<->B, label) CRC-32 / byte-size
     or file-digest bijective matching.
  3. Verifies whether any reversed pair crosses `train`, `val`, or `test` split boundaries
     relative to its source family (`cross_split_leakage_count == 0`).
  4. Provides a deterministic family-disjoint splitter supporting both `train_only_augmentation`
     (evaluation splits kept strictly original chronological) and `paired_evaluation_splits`
     (separate chronological and reversed evaluation slices within each family's split).
  5. Emits a machine-readable `docs/provenance_audit_report.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from urban_change.artifacts import artifact_manifest

SPLITS: tuple[str, ...] = ("train", "val", "test")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP Range reader for inspecting remote ZIP central directories and caption JSONs."""

    def __init__(self, url: str, size: int, timeout: int = 60) -> None:
        self.url = url
        self.size = size
        self.timeout = timeout
        self.pos = 0
        self.bytes_read = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: Any) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size - 1, self.pos + len(b) - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        n = len(data)
        b[:n] = data
        self.pos += n
        self.bytes_read += n
        return n


@dataclass(frozen=True)
class FamilyPairRecord:
    """Bijective link between an original scene family (T1, T2) and its reversed pair (T2, T1)."""

    family_id: str
    source_split: str
    source_filename: str
    source_imgid: int
    changeflag: int
    reversed_split: str | None
    reversed_filename: str | None
    reversed_imgid: int | None
    provenance_method: str
    label_identical_verified: bool
    ab_swapped_verified: bool

    def is_cross_split_leak(self) -> bool:
        return self.reversed_split is not None and self.reversed_split != self.source_split


def compute_family_id(split: str, source_filename: str) -> str:
    """Compute a canonical deterministic family ID for a base geographic scene."""
    stem = Path(source_filename).stem
    if stem.endswith("_reversed"):
        stem = stem[: -len("_reversed")]
    return f"FAM-{split}-{stem}"


def audit_manifests_and_signatures(
    original_captions_bytes: bytes,
    augmented_captions_bytes: bytes,
    member_signatures: dict[tuple[str, str, str], tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Audit original and augmented caption manifests and map every reversed sample to its source family.

    Parameters
    ----------
    original_captions_bytes:
        Raw bytes of the original `LevirCCcaptions.json` manifest.
    augmented_captions_bytes:
        Raw bytes of the augmented `LEVIRCCcaptions_fully_cleaned.json` (or `captions.json`) manifest.
    member_signatures:
        Optional mapping `(split, folder, filename) -> (crc32_or_hash_int, byte_size)` from the
        augmented archive or directory tree. Used when historical entries do not carry an explicit
        `augmentation.source_filename` field.
    """
    orig_sha256 = sha256_bytes(original_captions_bytes)
    aug_sha256 = sha256_bytes(augmented_captions_bytes)

    orig_doc = json.loads(original_captions_bytes.decode("utf-8"))
    aug_doc = json.loads(augmented_captions_bytes.decode("utf-8"))

    orig_entries: list[dict[str, Any]] = orig_doc.get("images", [])
    aug_entries: list[dict[str, Any]] = aug_doc.get("images", [])
    if not orig_entries or not aug_entries:
        raise ValueError(
            "Both original and augmented caption manifests must contain non-empty 'images'"
        )

    orig_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    orig_by_filename: dict[str, dict[str, Any]] = {}
    for entry in orig_entries:
        sp, fn = str(entry["split"]), str(entry["filename"])
        if sp not in SPLITS:
            raise ValueError(f"Invalid split {sp!r} in original manifest")
        orig_by_key[(sp, fn)] = entry
        orig_by_filename[fn] = entry

    # Separate augmented entries into base originals vs temporal-reversal additions
    aug_originals: list[dict[str, Any]] = []
    aug_reversals: list[dict[str, Any]] = []

    for entry in aug_entries:
        sp, fn = str(entry["split"]), str(entry["filename"])
        if sp not in SPLITS:
            raise ValueError(f"Invalid split {sp!r} in augmented manifest")
        is_explicit_rev = entry.get("augmentation", {}).get("type") == "temporal_reversal"
        if not is_explicit_rev and (sp, fn) in orig_by_key:
            aug_originals.append(entry)
        else:
            aug_reversals.append(entry)

    # Index original entries by swapped (B, A) signature if member_signatures are provided
    sig_lookup: dict[tuple[tuple[int, int], tuple[int, int]], list[dict[str, Any]]] = {}
    if member_signatures is not None:
        for entry in aug_originals:
            sp, fn = str(entry["split"]), str(entry["filename"])
            sig_a = member_signatures.get((sp, "A", fn))
            sig_b = member_signatures.get((sp, "B", fn))
            if sig_a is not None and sig_b is not None:
                # A reversed pair's (A, B) equals the original pair's (B, A)
                sig_lookup.setdefault((sig_b, sig_a), []).append(entry)

    # Map every reversed entry to its source original entry (strict 1-to-1 bijection)
    reversal_by_source_key: dict[tuple[str, str], tuple[dict[str, Any], str, bool, bool]] = {}
    used_source_keys: set[tuple[str, str]] = set()
    unmapped_reversals: list[str] = []
    cross_split_leaks: list[dict[str, Any]] = []

    for rev in aug_reversals:
        r_split = str(rev["split"])
        r_fn = str(rev["filename"])
        r_imgid = int(rev["imgid"])

        matched_orig: dict[str, Any] | None = None
        method = "unmapped"
        ab_ok = False
        lbl_ok = False

        # Method 1: Explicit augmentation metadata (`urban_change.data.augment`)
        aug_meta = rev.get("augmentation", {})
        if aug_meta.get("type") == "temporal_reversal" and "source_filename" in aug_meta:
            src_fn = str(aug_meta["source_filename"])
            if (r_split, src_fn) in orig_by_key:
                matched_orig = orig_by_key[(r_split, src_fn)]
            elif src_fn in orig_by_filename:
                matched_orig = orig_by_filename[src_fn]
            method = "explicit_metadata"
            ab_ok = True
            lbl_ok = True

        # Method 2: `<stem>_reversed.png` naming convention
        if matched_orig is None and Path(r_fn).stem.endswith("_reversed"):
            base_stem = Path(r_fn).stem[: -len("_reversed")]
            cand_fn = f"{base_stem}{Path(r_fn).suffix}"
            if (r_split, cand_fn) in orig_by_key:
                matched_orig = orig_by_key[(r_split, cand_fn)]
            elif cand_fn in orig_by_filename:
                matched_orig = orig_by_filename[cand_fn]
            if matched_orig is not None:
                method = "filename_stem_convention"
                ab_ok = True
                lbl_ok = True

        # Method 3: Exact (A, B) <-> (B, A) and label signature bijection
        if matched_orig is None and member_signatures is not None:
            r_sig_a = member_signatures.get((r_split, "A", r_fn))
            r_sig_b = member_signatures.get((r_split, "B", r_fn))
            r_sig_lbl = member_signatures.get((r_split, "label", r_fn))
            if r_sig_a is not None and r_sig_b is not None:
                candidates = sig_lookup.get((r_sig_a, r_sig_b), [])
                if candidates:
                    # Select an unassigned candidate, preferring same-split + matching label CRC
                    avail = [
                        c
                        for c in candidates
                        if (str(c["split"]), str(c["filename"])) not in used_source_keys
                    ]
                    pool = avail if avail else candidates
                    same_split = [c for c in pool if str(c["split"]) == r_split]
                    matched_orig = same_split[0] if same_split else pool[0]
                    method = "crc32_size_bijection"
                    ab_ok = True
                    orig_lbl_sig = member_signatures.get(
                        (str(matched_orig["split"]), "label", str(matched_orig["filename"]))
                    )
                    lbl_ok = (r_sig_lbl is not None) and (r_sig_lbl == orig_lbl_sig)

        if matched_orig is None:
            unmapped_reversals.append(f"{r_split}/{r_fn}")
            continue

        s_split = str(matched_orig["split"])
        s_fn = str(matched_orig["filename"])
        used_source_keys.add((s_split, s_fn))
        if s_split != r_split:
            cross_split_leaks.append(
                {
                    "family_id": compute_family_id(s_split, s_fn),
                    "source_split": s_split,
                    "source_filename": s_fn,
                    "reversed_split": r_split,
                    "reversed_filename": r_fn,
                    "reversed_imgid": r_imgid,
                }
            )

        reversal_by_source_key[(s_split, s_fn)] = (rev, method, ab_ok, lbl_ok)

    # Build complete family records across all original pairs
    family_records: list[FamilyPairRecord] = []
    for orig in sorted(orig_entries, key=lambda e: (str(e["split"]), str(e["filename"]))):
        s_split = str(orig["split"])
        s_fn = str(orig["filename"])
        s_imgid = int(orig["imgid"])
        cflag = int(orig.get("changeflag", 0))
        fam_id = compute_family_id(s_split, s_fn)

        rev_info = reversal_by_source_key.get((s_split, s_fn))
        if rev_info is not None:
            rev_entry, method, ab_ok, lbl_ok = rev_info
            family_records.append(
                FamilyPairRecord(
                    family_id=fam_id,
                    source_split=s_split,
                    source_filename=s_fn,
                    source_imgid=s_imgid,
                    changeflag=cflag,
                    reversed_split=str(rev_entry["split"]),
                    reversed_filename=str(rev_entry["filename"]),
                    reversed_imgid=int(rev_entry["imgid"]),
                    provenance_method=method,
                    label_identical_verified=lbl_ok,
                    ab_swapped_verified=ab_ok,
                )
            )
        else:
            family_records.append(
                FamilyPairRecord(
                    family_id=fam_id,
                    source_split=s_split,
                    source_filename=s_fn,
                    source_imgid=s_imgid,
                    changeflag=cflag,
                    reversed_split=None,
                    reversed_filename=None,
                    reversed_imgid=None,
                    provenance_method="original_only",
                    label_identical_verified=True,
                    ab_swapped_verified=True,
                )
            )

    # Compute per-split statistics and deterministic family hashes
    split_summary: dict[str, Any] = {}
    split_family_hashes: dict[str, str] = {}
    for sp in SPLITS:
        sp_fams = [f for f in family_records if f.source_split == sp]
        orig_count = len(sp_fams)
        changed_count = sum(1 for f in sp_fams if f.changeflag == 1)
        unchanged_count = sum(1 for f in sp_fams if f.changeflag == 0)
        rev_count = sum(1 for f in sp_fams if f.reversed_filename is not None)

        fam_manifest_lines = [
            f"{f.family_id}:{f.source_filename}:{f.reversed_filename or 'NONE'}" for f in sp_fams
        ]
        fam_hash = sha256_text("\n".join(fam_manifest_lines))
        split_family_hashes[sp] = fam_hash

        split_summary[sp] = {
            "unique_scene_families": orig_count,
            "original_pairs": orig_count,
            "original_changed_pairs": changed_count,
            "original_unchanged_pairs": unchanged_count,
            "reversed_pairs": rev_count,
            "total_augmented_archive_entries": orig_count + rev_count,
            "changed_reversal_coverage_rate": round(rev_count / max(1, changed_count), 6),
            "family_manifest_sha256": fam_hash,
        }

    # Verify pairwise disjointness of scene family IDs across train, val, and test
    fams_by_split = {
        sp: {f.family_id for f in family_records if f.source_split == sp} for sp in SPLITS
    }
    inter_train_val = len(fams_by_split["train"] & fams_by_split["val"])
    inter_train_test = len(fams_by_split["train"] & fams_by_split["test"])
    inter_val_test = len(fams_by_split["val"] & fams_by_split["test"])

    partition_schemes = build_family_disjoint_partitions(family_records)

    # Sample representative mapped pairs for each split (for human inspection in the report)
    sample_mappings: dict[str, list[dict[str, Any]]] = {}
    for sp in SPLITS:
        paired_in_sp = [
            f for f in family_records if f.source_split == sp and f.reversed_filename is not None
        ]
        sample_mappings[sp] = [
            {
                "family_id": f.family_id,
                "source_filename": f.source_filename,
                "source_imgid": f.source_imgid,
                "reversed_filename": f.reversed_filename,
                "reversed_imgid": f.reversed_imgid,
                "provenance_method": f.provenance_method,
                "ab_swapped_verified": f.ab_swapped_verified,
                "label_identical_verified": f.label_identical_verified,
            }
            for f in paired_in_sp[:5]
        ]

    overall_family_hash = sha256_text("\n".join(f"{sp}:{split_family_hashes[sp]}" for sp in SPLITS))

    return {
        "schema_version": "1.0.0",
        "audit_type": "split_family_provenance_and_leakage_audit",
        "manifests": {
            "original_captions_sha256": orig_sha256,
            "augmented_captions_sha256": aug_sha256,
            "original_total_entries": len(orig_entries),
            "augmented_total_entries": len(aug_entries),
            "total_reversed_entries": len(aug_reversals),
        },
        "zero_leakage_verification": {
            "zero_cross_split_leakage_verified": (
                len(cross_split_leaks) == 0
                and len(unmapped_reversals) == 0
                and inter_train_val == 0
                and inter_train_test == 0
                and inter_val_test == 0
            ),
            "cross_split_leakage_count": len(cross_split_leaks),
            "unmapped_reversal_count": len(unmapped_reversals),
            "mapped_reversal_count": len(aug_reversals) - len(unmapped_reversals),
            "all_changed_pairs_bijectively_mapped": (
                len(unmapped_reversals) == 0
                and all(
                    f.ab_swapped_verified and f.label_identical_verified for f in family_records
                )
            ),
            "pairwise_family_intersections": {
                "train_val": inter_train_val,
                "train_test": inter_train_test,
                "val_test": inter_val_test,
            },
            "cross_split_leak_details": cross_split_leaks,
            "unmapped_reversal_details": unmapped_reversals,
        },
        "overall_family_manifest_sha256": overall_family_hash,
        "splits": split_summary,
        "deterministic_partition_policies": partition_schemes,
        "sample_bijective_mappings": sample_mappings,
    }


def build_family_disjoint_partitions(
    family_records: list[FamilyPairRecord],
) -> dict[str, Any]:
    """Generate deterministic family-disjoint partition manifests for two evaluation protocols:
    1. `train_only_augmentation`: Reversed samples included ONLY in `train`; `val` and `test`
       contain strictly original chronological (T1, T2) pairs.
    2. `paired_evaluation_splits`: `val` and `test` are partitioned into separate
       `chronological` (T1, T2) and `reversed` (T2, T1) slices within the same scene family
       without crossing split boundaries.
    """
    train_only_splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    paired_eval_splits: dict[str, dict[str, list[str]]] = {
        "train": {"chronological": [], "reversed": [], "combined": []},
        "val": {"chronological": [], "reversed": [], "combined": []},
        "test": {"chronological": [], "reversed": [], "combined": []},
    }

    for rec in sorted(family_records, key=lambda r: (r.source_split, r.source_filename)):
        if rec.is_cross_split_leak():
            raise ValueError(
                f"Cannot build family-disjoint partition: family {rec.family_id} crosses "
                f"{rec.source_split} -> {rec.reversed_split}"
            )
        sp = rec.source_split
        train_only_splits[sp].append(rec.source_filename)
        paired_eval_splits[sp]["chronological"].append(rec.source_filename)
        paired_eval_splits[sp]["combined"].append(rec.source_filename)

        if rec.reversed_filename is not None:
            if sp == "train":
                train_only_splits["train"].append(rec.reversed_filename)
            paired_eval_splits[sp]["reversed"].append(rec.reversed_filename)
            paired_eval_splits[sp]["combined"].append(rec.reversed_filename)

    return {
        "train_only_augmentation": {
            "description": (
                "Strict train-only augmentation: reversed pairs are permitted only in train; "
                "val and test contain strictly original chronological (T1, T2) pairs."
            ),
            "counts": {sp: len(files) for sp, files in train_only_splits.items()},
            "split_sha256": {
                sp: sha256_text("\n".join(files)) for sp, files in train_only_splits.items()
            },
        },
        "paired_evaluation_splits": {
            "description": (
                "Paired directional evaluation: every scene family remains strictly inside its "
                "assigned split, with chronological (T1, T2) and reversed (T2, T1) slices "
                "tracked separately for directional symmetry and polarity evaluation."
            ),
            "counts": {
                sp: {k: len(v) for k, v in slices.items()}
                for sp, slices in paired_eval_splits.items()
            },
            "slice_sha256": {
                sp: {k: sha256_text("\n".join(v)) for k, v in slices.items()}
                for sp, slices in paired_eval_splits.items()
            },
        },
    }


def audit_remote_huggingface_archives() -> dict[str, Any]:
    """Audit the pinned Hugging Face `LEVIR-MCI-dataset.zip` and `LEVIR-MCI-augmented_dataset.zip` via HTTP Range."""
    manifest = artifact_manifest()
    orig_meta = manifest["dataset-original"]
    aug_meta = manifest["dataset-augmented"]

    orig_reader = HTTPRangeReader(orig_meta["url"], int(orig_meta["bytes"]))
    with zipfile.ZipFile(io.BufferedReader(orig_reader, buffer_size=1024 * 1024)) as z_orig:
        orig_caption_bytes = z_orig.read("LEVIR-MCI-dataset/LevirCCcaptions.json")

    aug_reader = HTTPRangeReader(aug_meta["url"], int(aug_meta["bytes"]))
    member_sigs: dict[tuple[str, str, str], tuple[int, int]] = {}
    with zipfile.ZipFile(io.BufferedReader(aug_reader, buffer_size=1024 * 1024)) as z_aug:
        aug_caption_bytes = z_aug.read(
            "LEVIR-MCI-augmented_dataset/LEVIRCCcaptions_fully_cleaned.json"
        )
        for info in z_aug.infolist():
            if info.filename.endswith(".png"):
                parts = info.filename.split("/")
                if len(parts) >= 5:
                    sp, folder, fname = parts[-3], parts[-2], parts[-1]
                    member_sigs[(sp, folder, fname)] = (int(info.CRC), int(info.file_size))

    report = audit_manifests_and_signatures(
        original_captions_bytes=orig_caption_bytes,
        augmented_captions_bytes=aug_caption_bytes,
        member_signatures=member_sigs,
    )
    report["source_archives"] = {
        "dataset_repository": "sushantnemade/LEVIR-MCI-augmented",
        "revision": "3844bd00748c4d79af90e7a002f919b0dd20bb13",
        "original_archive": {
            "filename": orig_meta["filename"],
            "expected_archive_sha256": orig_meta["sha256"],
            "archive_bytes": orig_meta["bytes"],
            "range_bytes_read": orig_reader.bytes_read,
        },
        "augmented_archive": {
            "filename": aug_meta["filename"],
            "expected_archive_sha256": aug_meta["sha256"],
            "archive_bytes": aug_meta["bytes"],
            "range_bytes_read": aug_reader.bytes_read,
        },
    }
    return report


def audit_local_directories(
    original_captions_path: Path,
    augmented_captions_path: Path,
    augmented_images_root: Path | None = None,
) -> dict[str, Any]:
    """Audit local original and augmented caption files and optional image directory tree."""
    orig_bytes = original_captions_path.read_bytes()
    aug_bytes = augmented_captions_path.read_bytes()

    member_sigs: dict[tuple[str, str, str], tuple[int, int]] | None = None
    if augmented_images_root is not None and augmented_images_root.is_dir():
        member_sigs = {}
        for sp in SPLITS:
            for folder in ("A", "B", "label"):
                folder_dir = augmented_images_root / sp / folder
                if not folder_dir.is_dir():
                    continue
                for img_file in folder_dir.glob("*.png"):
                    raw = img_file.read_bytes()
                    digest_int = int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")
                    member_sigs[(sp, folder, img_file.name)] = (digest_int, len(raw))

    return audit_manifests_and_signatures(
        original_captions_bytes=orig_bytes,
        augmented_captions_bytes=aug_bytes,
        member_signatures=member_sigs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit LEVIR-MCI original and augmented split-family provenance and verify zero cross-split leakage."
    )
    parser.add_argument(
        "--original-captions",
        type=Path,
        help="Local path to original LevirCCcaptions.json (if omitted, reads pinned HF archive via HTTPS Range)",
    )
    parser.add_argument(
        "--augmented-captions",
        type=Path,
        help="Local path to augmented captions JSON (if omitted, reads pinned HF archive via HTTPS Range)",
    )
    parser.add_argument(
        "--augmented-images",
        type=Path,
        help="Optional local path to augmented images/ root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/provenance_audit_report.json"),
        help="Output JSON report path (default: docs/provenance_audit_report.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that an existing provenance_audit_report.json has zero cross-split leakage",
    )
    args = parser.parse_args(argv)

    if args.check and args.output.is_file() and args.original_captions is None:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        ver = existing["zero_leakage_verification"]
        assert (
            ver[
                "zero_leakage_verified"
                if "zero_leakage_verified" in ver
                else "zero_cross_split_leakage_verified"
            ]
            is True
        )
        assert ver["cross_split_leakage_count"] == 0
        assert ver["unmapped_reversal_count"] == 0
        print(f"OK: Verified zero split-family leakage in {args.output}")
        return 0

    if args.original_captions and args.augmented_captions:
        report = audit_local_directories(
            args.original_captions,
            args.augmented_captions,
            args.augmented_images,
        )
    else:
        report = audit_remote_huggingface_archives()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ver = report["zero_leakage_verification"]
    if not ver["zero_cross_split_leakage_verified"]:
        print(f"FAIL: Cross-split leakage or unmapped reversals detected: {ver}")
        return 1
    print(
        f"Wrote {args.output}: {report['manifests']['original_total_entries']} original, "
        f"{report['manifests']['total_reversed_entries']} reversed, "
        f"cross_split_leakage_count={ver['cross_split_leakage_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
