#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail if a tracked file, or a tracked file's name, contains a denied term.

The denylist is itself confidential, so it must live OUTSIDE the repository;
the script refuses a denylist path inside the work tree.

    python scripts/check_public_tree.py --denylist ~/.rashomon-denylist.txt

Denylist format (UTF-8, one entry per line):

    # comment
    some-term                     a denied term (case-insensitive)
    allow <path-glob> <term>      the term is allowed in files matching the glob

A term is matched as a whole word on each side that starts or ends with a
letter or digit, so "abc" does not match inside "abcd". PNG files are scanned
through their text/metadata chunks, other binary files through their printable
runs, so names embedded in metadata are caught too.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Denylist:
    terms: tuple[str, ...]
    allows: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (glob, term lowercased)


def parse_denylist(text: str) -> Denylist:
    terms: list[str] = []
    allows: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("allow "):
            parts = line.split(None, 2)
            if len(parts) != 3:
                raise ValueError(f"malformed allow line: {raw!r}")
            allows.append((parts[1], parts[2].lower()))
        else:
            terms.append(line)
    if not terms:
        raise ValueError("denylist has no terms")
    return Denylist(tuple(terms), tuple(allows))


def term_pattern(term: str) -> re.Pattern:
    left = r"(?<![A-Za-z0-9])" if term[0].isalnum() else ""
    right = r"(?![A-Za-z0-9])" if term[-1].isalnum() else ""
    return re.compile(left + re.escape(term) + right, re.IGNORECASE)


def _allowed(deny: Denylist, path: str, term: str) -> bool:
    return any(t == term.lower() and fnmatch.fnmatch(path, g) for g, t in deny.allows)


def scan_text(deny: Denylist, path: str, text: str) -> list[str]:
    hits = []
    for term in deny.terms:
        if _allowed(deny, path, term):
            continue
        if term_pattern(term).search(path):
            hits.append(f"{path}: file name contains a denied term")
        for n, line in enumerate(text.splitlines(), 1):
            if term_pattern(term).search(line):
                hits.append(f"{path}:{n}: denied term")
    return hits


def tracked_files(root: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, cwd=root, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_TEXT_CHUNKS = (b"tEXt", b"zTXt", b"iTXt", b"eXIf")
_STRINGS = re.compile(rb"[\x20-\x7e]{6,}")


def png_text(data: bytes) -> str:
    """Only the PNG chunks that can carry text (metadata); pixel data is noise."""
    import zlib

    out, pos = [], len(_PNG_SIG)
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        if ctype == b"zTXt" and b"\0" in body:
            key, _, rest = body.partition(b"\0")
            try:
                body = key + b" " + zlib.decompress(rest[1:])
            except zlib.error:
                pass
        if ctype in _PNG_TEXT_CHUNKS:
            out.append(body.replace(b"\0", b" ").decode("latin-1"))
        pos += 12 + length
    return "\n".join(out)


def read_any(path: str) -> str:
    """Text files in full; PNG metadata chunks; other binaries as printable runs."""
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(_PNG_SIG):
        return png_text(data)
    if b"\0" not in data:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return "\n".join(m.group().decode("ascii") for m in _STRINGS.finditer(data))


def check_tree(root: str, deny: Denylist) -> list[str]:
    hits: list[str] = []
    for rel in tracked_files(root):
        full = os.path.join(root, rel)
        text = read_any(full) if os.path.isfile(full) else ""
        hits.extend(scan_text(deny, rel, text))
    return hits


def _inside(path: str, root: str) -> bool:
    path = os.path.normcase(os.path.realpath(path))
    root = os.path.normcase(os.path.realpath(root))
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--denylist", required=True, help="denylist file outside the repository")
    ap.add_argument("--root", default=".", help="repository root (default: current directory)")
    args = ap.parse_args(argv)
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                          cwd=args.root, check=True).stdout.strip()
    if _inside(args.denylist, root):
        print("check_public_tree: the denylist must live outside the repository", file=sys.stderr)
        return 2
    with open(args.denylist, encoding="utf-8") as f:
        deny = parse_denylist(f.read())
    hits = check_tree(root, deny)
    # Never echo the matched term: the output may end up in a public CI log.
    for h in hits:
        print(h)
    print(f"check_public_tree: {len(hits)} finding(s) in {len(tracked_files(root))} tracked files")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
