#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the long24 validation set used by the exploratory long-task probe
(docs/validation.md). It is NOT one of the packaged sets.

Six families x four instances of tasks that force long reasoning with a numeric
outcome the generator computes exactly. The generated file lands under runs/
(gitignored); it is a validation instrument, not a shipped set.

    python scripts/gen_long_set.py --out runs/long24/long24.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time

SEED = 20261029
M = 1_000_000_007


def _items(rng: random.Random) -> list[dict]:
    items: list[dict] = []

    for i in range(4):
        a = rng.randrange(10**8, 10**9)
        b = rng.randrange(10**8, 10**9)
        ans = (a * b) % 1_000_000
        items.append({"id": f"long-lastdigits-{i}", "family": "long_lastdigits",
                      "messages": [{"role": "user", "content": (
                          f"Compute {a} * {b}. Work it out step by step. "
                          f"At the end, output exactly one line: Answer: <the last 6 digits of the product>.")}],
                      "expected": str(a * b)[-6:]})

    for i in range(4):
        m, n = rng.randrange(5, 9), rng.randrange(5, 9)
        ans = math.comb(m + n, m)
        items.append({"id": f"long-paths-{i}", "family": "long_paths",
                      "messages": [{"role": "user", "content": (
                          f"On a grid, how many monotone paths (only right or up steps) lead from (0,0) "
                          f"to ({m},{n})? Derive it step by step. At the end, output exactly one line: "
                          f"Answer: <number>.")}],
                      "expected": str(ans)})

    for i in range(4):
        n = rng.randrange(14, 24)
        f = [1, 1]
        for _ in range(n):
            f.append(f[-1] + f[-2])
        ans = f[n]  # domino tilings of a 2 x n board
        items.append({"id": f"long-tiling-{i}", "family": "long_tiling",
                      "messages": [{"role": "user", "content": (
                          f"In how many distinct ways can a 2 x {n} board be tiled with 2 x 1 dominoes? "
                          f"Build the recurrence and compute it step by step. At the end, output exactly "
                          f"one line: Answer: <number>.")}],
                      "expected": str(ans)})

    for i in range(4):
        a = rng.randrange(10**6, 10**7)
        b = rng.randrange(30, 60)
        ans = pow(a, b, M)
        items.append({"id": f"long-modpow-{i}", "family": "long_modpow",
                      "messages": [{"role": "user", "content": (
                          f"Compute {a}^{b} mod {M}. Use fast exponentiation and show every squaring "
                          f"step. At the end, output exactly one line: Answer: <number>.")}],
                      "expected": str(ans)})

    for i in range(4):
        n = rng.randrange(6, 10)
        ans = math.factorial(n) - 2 * math.factorial(n - 1)
        items.append({"id": f"long-seating-{i}", "family": "long_seating",
                      "messages": [{"role": "user", "content": (
                          f"{n} people stand in a row. Ana and Bojan refuse to stand next to each other. "
                          f"In how many distinct orders can they stand? Reason step by step. At the end, "
                          f"output exactly one line: Answer: <number>.")}],
                      "expected": str(ans)})

    for i in range(4):
        n = rng.randrange(150, 400)
        s = (n * (n + 1) // 2) ** 2
        ans = s % 1_000_003
        items.append({"id": f"long-cubes-{i}", "family": "long_cubes",
                      "messages": [{"role": "user", "content": (
                          f"Compute the sum of the first {n} cubes, then take it modulo 1000003. "
                          f"Work carefully and show the steps. At the end, output exactly one line: "
                          f"Answer: <number>.")}],
                      "expected": str(ans)})

    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(SEED)
    items = _items(rng)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    manifest = {
        "name": "long24", "generator": {"name": "gen_long_set.py", "version": "1.0.0"},
        "seed": SEED, "generated_at": time.strftime("%Y-%m-%d"), "n_items": len(items),
        "families": sorted({it["family"] for it in items}),
        "items_sha256": {it["id"]: hashlib.sha256(json.dumps(
            {"id": it["id"], "messages": it["messages"], "tools": None},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest() for it in items},
        "scope": "internal validation set; not part of the public packaged sets",
        "license": "Apache-2.0", "llm_involvement": "none — deterministic script",
    }
    with open(args.out + ".manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"long24: {len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
