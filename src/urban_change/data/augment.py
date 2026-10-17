"""Create temporal-reversal samples while preserving split membership."""

import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Callable

from urban_change.data.text import tokenize


def azure_rewriter(
    endpoint: str, deployment: str, api_key: str, api_version: str = "2024-08-01-preview"
) -> Callable[[str], str]:
    from openai import AzureOpenAI

    if not endpoint.startswith("https://"):
        raise ValueError("Azure endpoint must use HTTPS")
    client = AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version=api_version, max_retries=2, timeout=45
    )

    def rewrite(caption: str) -> str:
        response = client.chat.completions.create(
            model=deployment,
            temperature=0.2,
            max_tokens=160,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite a remote-sensing change caption after reversing the before and after images. Reverse additions into removals and removals into additions. Preserve object types, counts, spatial relationships, and unchanged content. Do not invent observations. Return only the rewritten caption.",
                },
                {"role": "user", "content": caption},
            ],
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise ValueError("Caption rewriting returned an empty response")
        return text.strip().strip('"')

    return rewrite


def augment_dataset(
    captions_path: str | Path,
    image_root: str | Path,
    output: str | Path,
    splits: tuple[str, ...] = ("train",),
    caption_map: dict | None = None,
    rewrite: Callable[[str], str] | None = None,
) -> dict:
    if (caption_map is None) == (rewrite is None):
        raise ValueError("Provide exactly one of a reviewed caption map or a caption rewriter")
    if not splits or set(splits) - {"train", "val", "test"}:
        raise ValueError("Invalid augmentation splits")
    source_root, destination = Path(image_root).resolve(), Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"Augmentation output already exists: {destination}")
    if source_root == destination or source_root in destination.parents:
        raise ValueError("Augmentation output must be outside the source image tree")
    document = json.loads(Path(captions_path).read_text(encoding="utf-8"))
    entries = document.get("images", [])
    if not entries:
        raise ValueError("Caption JSON has no images")
    identities = set()
    for entry in entries:
        split, filename = entry["split"], entry["filename"]
        if (
            split not in {"train", "val", "test"}
            or Path(filename).name != filename
            or "\\" in filename
        ):
            raise ValueError("Invalid split or filename in caption annotations")
        if (split, filename) in identities:
            raise ValueError(f"Duplicate sample: {split}/{filename}")
        identities.add((split, filename))
    if any(entry.get("augmentation", {}).get("type") == "temporal_reversal" for entry in entries):
        raise ValueError(
            "Input already contains temporal-reversal annotations; use the original dataset"
        )
    next_imgid = max(int(e["imgid"]) for e in entries) + 1
    next_sentid = max(int(s["sentid"]) for e in entries for s in e["sentences"]) + 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=".augmentation-", dir=destination.parent))
    mapping = []
    used_map = set()
    try:
        for entry in entries:
            split, filename = entry["split"], entry["filename"]
            for folder in ("A", "B", "label", "label_rgb"):
                source = source_root / split / folder / filename
                if not source.exists() and folder == "label_rgb":
                    continue
                target = staged / "images" / split / folder / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        added = []
        for entry in entries:
            if entry["split"] not in splits or int(entry.get("changeflag", 0)) != 1:
                continue
            split, filename = entry["split"], entry["filename"]
            key = f"{split}/{filename}"
            new_name = f"{Path(filename).stem}_reversed{Path(filename).suffix}"
            if (split, new_name) in identities:
                raise ValueError(f"Reversed filename already exists: {new_name}")
            original_captions = entry["sentences"]
            if caption_map is not None:
                texts = caption_map[key]
                used_map.add(key)
            else:
                texts = [rewrite(sentence["raw"]) for sentence in original_captions]
            if (
                not isinstance(texts, list)
                or len(texts) != len(original_captions)
                or any(not isinstance(text, str) or not text.strip() for text in texts)
            ):
                raise ValueError(f"Reversed captions must match the source caption count: {key}")
            new_entry = deepcopy(entry)
            new_entry.update(filename=new_name, imgid=next_imgid, filepath=split)
            new_entry["sentences"] = []
            for text in texts:
                new_entry["sentences"].append(
                    {
                        "raw": text,
                        "tokens": tokenize(text, False, False),
                        "imgid": next_imgid,
                        "sentid": next_sentid,
                    }
                )
                next_sentid += 1
            new_entry["sentids"] = [sentence["sentid"] for sentence in new_entry["sentences"]]
            new_entry["augmentation"] = {
                "type": "temporal_reversal",
                "source_filename": filename,
                "source_imgid": entry["imgid"],
                "caption_source": "reviewed_map"
                if caption_map is not None
                else "azure_generated_requires_review",
            }
            for folder, original_folder in (
                ("A", "B"),
                ("B", "A"),
                ("label", "label"),
                ("label_rgb", "label_rgb"),
            ):
                source = source_root / split / original_folder / filename
                if source.exists():
                    shutil.copy2(source, staged / "images" / split / folder / new_name)
            added.append(new_entry)
            mapping.append(
                {
                    "split": split,
                    "source": filename,
                    "reversed": new_name,
                    "source_imgid": entry["imgid"],
                    "reversed_imgid": next_imgid,
                }
            )
            next_imgid += 1
        if caption_map is not None and set(caption_map) != used_map:
            raise ValueError(
                "Caption map contains unused sample keys; check selected splits/change flags"
            )
        if not added:
            raise ValueError("No changed samples were selected for reversal")
        document["images"].extend(added)
        report = {
            "original_samples": len(entries) - len(added),
            "reversed_samples": len(added),
            "augmented_splits": list(splits),
            "pairs": mapping,
            "caption_review_required": rewrite is not None,
        }
        (staged / "captions.json").write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        (staged / "augmentation.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        staged.replace(destination)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return report
