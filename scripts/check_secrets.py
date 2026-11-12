# SPDX-License-Identifier: Apache-2.0
# Derived from a secret-check script in a private Egzakta repository and modified for
# this project (private addresses are errors; --tree mode). Relicensed Apache-2.0 by
# the copyright holder. See NOTICE.
"""Secret check. Stdlib only, so a missing binary never blocks a commit.

Keys, tokens, private key blocks and private network addresses (RFC 1918 and
100.64.0.0/10) are errors.

    python scripts/check_secrets.py                    # lines added in the staged changes
    python scripts/check_secrets.py --range main..HEAD # lines added in a commit range
    python scripts/check_secrets.py --tree             # every tracked file, in full

When gitleaks is installed it runs as well, in the diff modes; it never replaces
this scan (gitleaks does not flag private addresses).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

SECRET_PATTERNS = {
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "OpenAI/Anthropic-style key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Slack token": re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}"),
    "bearer token": re.compile(r"(?i)\bauthorization:\s*bearer\s+[A-Za-z0-9._-]{20,}"),
    "assigned secret": re.compile(r"(?i)\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
    "private network address": re.compile(
        r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))"
        r"\.\d{1,3}\.\d{1,3}\b"),
}
# This file and its tests contain the patterns themselves.
SKIP_FILES = ("scripts/check_secrets.py", "tests/test_scripts.py")
BINARY_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".whl", ".gz", ".zip")


def added_lines(diff: str):
    """Yield (path, line) for every added line of a unified diff."""
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and not line.startswith("+++") and path:
            yield path, line[1:]


def scan_lines(lines) -> list[str]:
    """lines: iterable of (path, text). Returns sorted unique findings."""
    errors = set()
    for path, line in lines:
        if path in SKIP_FILES:
            continue
        for name, pat in SECRET_PATTERNS.items():
            if pat.search(line):
                errors.add(f"{path}: {name}")
    return sorted(errors)


def scan(diff: str) -> list[str]:
    return scan_lines(added_lines(diff))


def tracked_lines(cwd: str | None = None):
    """Yield (path, line) for every line of every tracked text file."""
    files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, cwd=cwd,
                           check=True).stdout.decode("utf-8").split("\0")
    for path in filter(None, files):
        if path.lower().endswith(BINARY_SUFFIXES):
            continue
        try:
            with open(path if cwd is None else f"{cwd}/{path}", "rb") as f:
                data = f.read()
        except OSError:
            continue  # tracked but deleted in the working tree
        if b"\0" in data:
            continue
        for line in data.decode("utf-8", "replace").splitlines():
            yield path, line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--range", help="git range to scan instead of the staged changes")
    mode.add_argument("--tree", action="store_true", help="scan every tracked file in full")
    args = ap.parse_args(argv)

    rc = 0
    if args.tree:
        errors = scan_lines(tracked_lines())
    else:
        if shutil.which("gitleaks"):
            cmd = ["gitleaks", "git", "--no-banner"] + (
                [f"--log-opts={args.range}"] if args.range else ["--staged"])
            rc = subprocess.call(cmd)
        git = ["git", "diff", "--unified=0"] + ([args.range] if args.range else ["--cached"])
        diff = subprocess.run(git, capture_output=True, text=True, encoding="utf-8",
                              errors="replace").stdout
        errors = scan(diff)
    for e in errors:
        print("SECRET:", e)
    if errors:
        print("check_secrets: blocked; remove the secret (and rotate it if it was ever pushed)")
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
