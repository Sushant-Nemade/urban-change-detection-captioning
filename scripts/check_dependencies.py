"""Query PyPI's published vulnerability metadata for the CPU lock snapshot."""

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def check_package(pin):
    name, version = pin
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{name}/{version}/json", timeout=15
        ) as response:
            metadata = json.load(response)
        advisories = [
            entry["id"]
            for entry in metadata.get("vulnerabilities", [])
            if not entry.get("withdrawn")
        ]
        return {"name": name, "version": version, "advisories": advisories}
    except Exception as exc:
        return {"name": name, "version": version, "error": type(exc).__name__}


def main():
    root = Path(__file__).resolve().parents[1]
    pins = []
    for line in (root / "requirements/cpu.lock").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ;]+)", line.strip())
        if match:
            pins.append((match[1], match[2].split("+")[0]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        checked = list(pool.map(check_package, pins))
    report = {
        "source": "PyPI version metadata",
        "packages_checked": len(checked),
        "active_advisories": [entry for entry in checked if entry.get("advisories")],
        "unavailable": [entry for entry in checked if "error" in entry],
        "scope": "Known advisories reported by PyPI for locked versions; not a complete security audit",
    }
    print(json.dumps(report, indent=2))
    return not pins or bool(report["active_advisories"] or report["unavailable"])


if __name__ == "__main__":
    raise SystemExit(main())
