"""Download pinned artifacts with TLS and mandatory SHA-256 verification."""

import hashlib
import json
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path


def artifact_manifest():
    return json.loads(files("urban_change").joinpath("assets", "artifacts.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_artifact(name: str, output: str | Path) -> Path:
    manifest = artifact_manifest()
    if name not in manifest:
        raise ValueError(f"Unknown artifact: {name}")
    artifact = manifest[name]
    output = Path(output)
    if output.exists():
        if sha256(output) == artifact["sha256"]:
            return output
        raise FileExistsError(f"Existing artifact has a different checksum: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".download-", suffix=".part", dir=output.parent, delete=False
        ) as target:
            temporary = Path(target.name)
            with urllib.request.urlopen(artifact["url"], timeout=60) as response:
                while block := response.read(1024 * 1024):
                    target.write(block)
        if sha256(temporary) != artifact["sha256"]:
            raise ValueError(f"Downloaded {name} failed SHA-256 verification")
        if temporary.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Downloaded {name} has an unexpected byte length")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
