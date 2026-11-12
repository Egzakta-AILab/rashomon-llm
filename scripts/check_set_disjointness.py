#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-release check: the public prompt sets must not contain any item from a
private benchmark manifest.

The private side never enters the repo. Pass it as a file with one sha256 per
line (hex digest of the canonical item body), produced on the machine that
holds the private manifests:

    python scripts/check_set_disjointness.py --hashes private_hashes.txt

Exit 0 when disjoint, 1 on any overlap (prints only the public item id).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

SETS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "rashomon_llm", "data", "sets")


def _item_sha(item: dict) -> str:
    body = json.dumps({"id": item["id"], "messages": item["messages"],
                       "tools": item.get("tools")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hashes", required=True, help="file with one private sha256 per line")
    args = ap.parse_args()

    with open(args.hashes, encoding="utf-8") as f:
        private = {line.strip() for line in f if line.strip()}
    if not private:
        print("no private hashes supplied; refusing to certify disjointness", file=sys.stderr)
        return 1

    collisions = []
    for name in sorted(os.listdir(SETS_DIR)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(SETS_DIR, name), encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if _item_sha(item) in private:
                    collisions.append((name, item.get("id")))
    if collisions:
        for name, item_id in collisions:
            print(f"OVERLAP: {name}: {item_id}", file=sys.stderr)
        return 1
    print(f"disjoint: {len(private)} private hashes checked against the public sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
