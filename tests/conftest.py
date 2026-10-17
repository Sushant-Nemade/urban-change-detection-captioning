import json

import numpy as np
import pytest
import torch
from PIL import Image

from urban_change.data.prepare import prepare_metadata


@pytest.fixture(scope="session", autouse=True)
def torch_threads():
    torch.set_num_threads(2)


@pytest.fixture
def dataset_fixture(tmp_path):
    images = tmp_path / "Aerial Area" / "images"
    entries = []
    for index, split in enumerate(("train", "train", "val", "test")):
        filename = f"{split}_{index:06d}.png"
        for folder in ("A", "B", "label"):
            path = images / split / folder / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            array = np.full((256, 256, 3), 50 if folder == "A" else 150, dtype=np.uint8)
            if folder == "label":
                array = np.zeros((256, 256), dtype=np.uint8)
                array[:128] = 128
                array[128:, :128] = 255
            Image.fromarray(array).save(path)
        word = "building" if split == "train" else "evaluationonly"
        entries.append(
            {
                "split": split,
                "filepath": split,
                "filename": filename,
                "imgid": 200 + index,
                "changeflag": 1 if index != 1 else 0,
                "sentences": [
                    {
                        "raw": f"a {word} was added .",
                        "sentid": 1000 + index * 5 + j,
                        "imgid": 200 + index,
                    }
                    for j in range(5)
                ],
            }
        )
    captions = tmp_path / "captions.json"
    captions.write_text(json.dumps({"images": entries}))
    metadata = tmp_path / "metadata"
    prepare_metadata(captions, metadata, min_count=0)
    return {"images": images, "captions": captions, "metadata": metadata, "entries": entries}
