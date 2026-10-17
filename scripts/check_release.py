"""Check the current source tree without traversing local datasets or secrets."""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    paths = sorted({ROOT / name for name in names if name and (ROOT / name).is_file()})
    problems = []
    patterns = {
        "github_token": re.compile(
            rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
        ),
        "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "openai_token": re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b"),
    }
    for path in paths:
        relative = str(path.relative_to(ROOT))
        if path.suffix in {".pth", ".pt", ".ckpt", ".pdf", ".jar", ".zip"}:
            problems.append(
                {"path": relative, "problem": "Binary research artifact in source tree"}
            )
        data = path.read_bytes()
        if b"\0" in data[:4096]:
            continue
        for name, pattern in patterns.items():
            if pattern.search(data):
                problems.append({"path": relative, "problem": name})
        if path.suffix == ".md":
            text = data.decode("utf-8")
            for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
                if target.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                local = target.split("#", 1)[0]
                if local and not (path.parent / local).exists():
                    problems.append(
                        {"path": relative, "problem": f"Missing relative link: {target}"}
                    )
    report = {
        "source_files": len(paths),
        "source_bytes": sum(path.stat().st_size for path in paths),
        "problems": problems,
        "scope": "Current source tree and selected credential patterns; not a complete secret or license audit",
    }
    print(json.dumps(report, indent=2))
    return bool(problems)


if __name__ == "__main__":
    raise SystemExit(main())
