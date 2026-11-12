# SPDX-License-Identifier: Apache-2.0
"""Interval assembly for rashomon-llm.

The cluster bootstrap itself lives in `stats.py` (code unchanged from its source; do not edit).
This module adds the one-sample encodings and small exact intervals:

- pair rate / any-of-N headline: per-prompt value vs an all-zero vector through
  `cluster_paired_bootstrap` — the same indicator-vs-zero encoding the originating
  private study used, so intervals reproduce its conventions;
- Wilson interval for the "prompts with disagreement among N runs" companion;
- Clopper-Pearson by bisection on the binomial tail (stdlib only), used whenever
  the bootstrap is degenerate (every cluster identical, e.g. 0/20) so a card never
  prints `0% [0.0, 0.0]` as if it were proof of stability.
"""
from __future__ import annotations

import math

from .stats import cluster_paired_bootstrap

_Z_95 = 1.959963984540054


def wilson_interval(x: int, n: int, z: float = _Z_95) -> dict:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"estimate": p, "ci_low": max(0.0, center - half), "ci_high": min(1.0, center + half),
            "n": n, "x": x, "method": "wilson"}


def _binom_tail_ge(x: int, n: int, p: float) -> float:
    """P(X >= x) for X ~ Binomial(n, p)."""
    if p <= 0.0:
        return 0.0 if x > 0 else 1.0
    if p >= 1.0:
        return 1.0 if x <= n else 0.0
    total = 0.0
    for k in range(x, n + 1):
        total += math.comb(n, k) * p**k * (1 - p) ** (n - k)
    return total


def clopper_pearson(x: int, n: int, alpha: float = 0.05) -> dict:
    """Two-sided Clopper-Pearson interval by bisection on the binomial tails.

    lo solves P(X >= x | p=lo) = alpha/2; hi solves P(X <= x | p=hi) = alpha/2.
    Degenerate ends: x=0 gives lo=0 and hi=1-(alpha/2)^(1/n); x=n mirrors it.
    """
    if not 0 <= x <= n or n <= 0:
        raise ValueError("need 0 <= x <= n, n > 0")

    def solve(target_tail: float, lower_tail: bool) -> float:
        # lower bound: tail = P(X >= x | p), increasing in p;
        # upper bound: tail = P(X <= x | p), decreasing in p.
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if lower_tail:
                tail = _binom_tail_ge(x, n, mid)
            else:
                tail = 1.0 - _binom_tail_ge(x + 1, n, mid) if x < n else 1.0
            if (tail < target_tail) == lower_tail:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lo = 0.0 if x == 0 else solve(alpha / 2, lower_tail=True)
    hi = 1.0 if x == n else solve(alpha / 2, lower_tail=False)
    return {"estimate": x / n, "ci_low": lo, "ci_high": hi, "n": n, "x": x,
            "method": "clopper-pearson"}


def cluster_rate_interval(values: dict[str, list[float]], clusters: dict[str, str] | None = None,
                          n_boot: int = 10000, seed: int = 0) -> dict:
    """One-sample cluster-bootstrap interval for a per-prompt rate.

    `values` maps prompt id -> [per-prompt rate] (a single-element list; the cluster
    bootstrap averages per item first). Uses the originating study's encoding: the prompt values
    are compared against an all-zero vector, and `diff` is the rate.
    """
    zeros = {k: [0.0] for k in values}
    res = cluster_paired_bootstrap(values, zeros, clusters=clusters, n_boot=n_boot, seed=seed)
    out = {"estimate": res["diff"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
           "n_items": res["n_items"], "n_clusters": res["n_clusters"],
           "method": "cluster bootstrap (percentile, 10000 resamples, seed 0)"}
    distinct = {round(v[0], 12) for v in values.values() if v}
    if len(distinct) <= 1:
        # Every prompt agrees: the bootstrap collapses to [v, v]. Report the degenerate
        # interval but attach the exact binomial bound so a zero never reads as certainty.
        x = sum(1 for v in values.values() if v and v[0] > 0)
        out["degenerate"] = True
        out["exact_bound"] = {**clopper_pearson(x, len(values)), "level": "prompt",
                              "assumes": "independent prompts"}
        out["exact_bound_family"] = family_level_bound(values, clusters)
    return out


FAMILY_BOUND_NOTE = ("The prompt-level bound assumes independent prompts; prompts of one family "
                     "are close variants, so the family-level bound is the conservative reading.")


def family_level_bound(values: dict[str, list[float]], clusters: dict[str, str] | None) -> dict:
    """Clopper-Pearson over families (the bootstrap cluster unit) instead of prompts.

    A family counts as disagreeing if any of its prompts disagrees. With 0 of 5
    families this gives an upper bound of 52.2%, against 8.8% for 0 of 40 prompts.
    """
    fam_hits: dict[str, bool] = {}
    for pid, v in values.items():
        fam = (clusters or {}).get(pid) or pid
        fam_hits[fam] = fam_hits.get(fam, False) or bool(v and v[0] > 0)
    x = sum(1 for hit in fam_hits.values() if hit)
    return {**clopper_pearson(x, len(fam_hits)), "level": "family"}
