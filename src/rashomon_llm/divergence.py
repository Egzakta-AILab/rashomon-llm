# SPDX-License-Identifier: Apache-2.0
"""Textual divergence between two responses (docs/protocol.md, metrics).

Comparison unit: token ids when BOTH responses carry them, otherwise characters.
The unit is always reported, and the two units are never mixed on one card.

Pair classification:
- `diverged`: first differing position exists (position is in the reported unit);
- `identical`: equal over the whole observed length;
- `identical_capped`: equal AND both hit the token cap — equal *as observed*;
  their unrestricted continuations are unknown, so they are censored out of the
  divergence curve and counted separately.
"""
from __future__ import annotations

from typing import Any

CURVE_POINTS = (64, 256, 512, 1024)


def _unit(a: dict, b: dict) -> tuple[list, list, str]:
    ta, tb = a.get("token_ids"), b.get("token_ids")
    if ta and tb:
        return list(ta), list(tb), "token"
    return list(a.get("text") or ""), list(b.get("text") or ""), "char"


def first_diff_pos(xs: list, ys: list) -> int | None:
    for i, (x, y) in enumerate(zip(xs, ys)):
        if x != y:
            return i
    if len(xs) != len(ys):
        return min(len(xs), len(ys))
    return None


def classify_pair(a: dict, b: dict) -> dict:
    """a and b are request records (or any dicts with text/token_ids/finish_reason)."""
    xs, ys, unit = _unit(a, b)
    pos = first_diff_pos(xs, ys)
    capped = a.get("finish_reason") == "length" and b.get("finish_reason") == "length"
    if pos is None:
        return {"kind": "identical_capped" if capped else "identical", "pos": None, "unit": unit}
    return {"kind": "diverged", "pos": pos, "unit": unit}


def pair_matrix(records_by_prompt: dict[str, list[dict]]) -> dict[str, dict]:
    """Per prompt: all-pairs classification. Returns prompt_id -> pair summary."""
    out: dict[str, dict] = {}
    for pid, recs in records_by_prompt.items():
        pairs = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                pairs.append(classify_pair(recs[i], recs[j]))
        out[pid] = {
            "n_runs": len(recs),
            "n_pairs": len(pairs),
            "diverged": sum(1 for p in pairs if p["kind"] == "diverged"),
            "identical": sum(1 for p in pairs if p["kind"] == "identical"),
            "identical_capped": sum(1 for p in pairs if p["kind"] == "identical_capped"),
            "positions": [p["pos"] for p in pairs if p["kind"] == "diverged"],
            "unit": pairs[0]["unit"] if pairs else None,
        }
    return out


def divergence_curve(matrix: dict[str, dict], points: tuple = CURVE_POINTS) -> dict:
    """Censored divergence curve: share of non-censored pairs diverged at <= point."""
    curve: dict[str, Any] = {"points": [], "censored_pairs": 0, "total_pairs": 0}
    all_pos: list[int] = []
    non_censored_pairs = 0
    for summary in matrix.values():
        non_censored_pairs += summary["n_pairs"] - summary["identical_capped"]
        all_pos.extend(summary["positions"])
        curve["censored_pairs"] += summary["identical_capped"]
    curve["total_pairs"] = non_censored_pairs
    for t in points:
        share = (sum(1 for p in all_pos if p <= t) / non_censored_pairs) if non_censored_pairs else None
        curve["points"].append({"le": t, "share": share})
    if all_pos:
        s = sorted(all_pos)
        curve["median_pos_diverged_only"] = s[len(s) // 2]
        curve["n_diverged_pairs"] = len(s)
    else:
        curve["median_pos_diverged_only"] = None
        curve["n_diverged_pairs"] = 0
    return curve
