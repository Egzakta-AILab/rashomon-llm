# SPDX-License-Identifier: Apache-2.0
# Taken from a private Egzakta repository (source commit recorded in the internal repository).
# Code unchanged, comments edited for publication. Relicensed Apache-2.0 by the copyright
# holder. See NOTICE.
# Do not restyle or "improve": the percentile indexing reproduces the intervals of the
# originating private study.
"""Paired item-level bootstrap with seeds clustered per item.

Inputs map item_id -> list of per-seed scores (0/1 or floats). Seeds of one item
are averaged first, because repeated samples of the same task are not independent
tasks. Only items present in both arms are compared.

Cluster versions (from the originating private study): close variants of one
template are resampled as one cluster (item -> cluster id; default one item = one cluster).
cluster_paired_bootstrap is paired_bootstrap with clusters; paired_rate_bootstrap gives the
ratio and the difference of two rates (count / seconds) with the same cluster resampling.
"""
from __future__ import annotations

import random
from statistics import fmean
from typing import Dict, List, Optional


def item_means(results: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: fmean(v) for k, v in results.items() if v}


def paired_bootstrap(a: Dict[str, List[float]], b: Dict[str, List[float]],
                     n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of a minus mean of b over common items, with a percentile CI."""
    ma, mb = item_means(a), item_means(b)
    items = sorted(set(ma) & set(mb))
    if not items:
        raise ValueError("no common items")
    diffs = [ma[i] - mb[i] for i in items]
    n = len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {
        "n_items": n,
        "mean_a": fmean(ma[i] for i in items),
        "mean_b": fmean(mb[i] for i in items),
        "diff": fmean(diffs),
        "ci_low": lo,
        "ci_high": hi,
    }


def to_pp(x: float) -> float:
    """Fraction to percentage points."""
    return 100.0 * x


# -- cluster bootstrap (tasks resampled with close variants as one cluster) ---------------------

def _cluster_groups(items: List[str], clusters: Optional[Dict[str, str]]) -> List[List[str]]:
    """Items grouped by cluster id (default: one item = one cluster), in sorted cluster-id order."""
    groups: Dict[str, List[str]] = {}
    for i in items:
        groups.setdefault((clusters or {}).get(i) or i, []).append(i)
    return [groups[k] for k in sorted(groups)]


def _percentiles(boots: List[float], alpha: float) -> tuple:
    boots = sorted(boots)
    n = len(boots)
    return boots[int((alpha / 2) * n)], boots[min(n - 1, int((1 - alpha / 2) * n))]


def cluster_paired_bootstrap(a: Dict[str, List[float]], b: Dict[str, List[float]],
                             clusters: Optional[Dict[str, str]] = None, n_boot: int = 10000,
                             seed: int = 0, alpha: float = 0.05) -> dict:
    """paired_bootstrap with clusters of items resampled as units (item -> cluster id).

    Each resample draws clusters with replacement and takes the mean item difference over the
    drawn clusters' items. With one item per cluster (the default) the draws and the result are
    identical to paired_bootstrap."""
    ma, mb = item_means(a), item_means(b)
    items = sorted(set(ma) & set(mb))
    if not items:
        raise ValueError("no common items")
    groups = _cluster_groups(items, clusters)
    sums = [sum(ma[i] - mb[i] for i in g) for g in groups]
    sizes = [len(g) for g in groups]
    k = len(groups)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        m = 0
        for _ in range(k):
            j = rng.randrange(k)
            s += sums[j]
            m += sizes[j]
        boots.append(s / m)
    lo, hi = _percentiles(boots, alpha)
    return {"n_items": len(items), "n_clusters": k, "mean_a": fmean(ma[i] for i in items),
            "mean_b": fmean(mb[i] for i in items), "diff": fmean(ma[i] - mb[i] for i in items),
            "ci_low": lo, "ci_high": hi}


def _rate(count: float, seconds: float, scale: float) -> float:
    return scale * count / seconds if seconds > 0 else (float("inf") if count > 0 else 0.0)


def _ratio(rx: float, ry: float) -> float:
    return rx / ry if ry > 0 else (float("inf") if rx > 0 else 1.0)


def paired_rate_bootstrap(x: Dict[str, tuple], y: Dict[str, tuple],
                          clusters: Optional[Dict[str, str]] = None, n_boot: int = 10000,
                          seed: int = 0, alpha: float = 0.05, scale: float = 3600.0) -> dict:
    """Two rates as ratios of sums, x vs y, paired over common items, clusters resampled as units.

    x, y: item -> (count, seconds), e.g. (correct runs of the task, the task's share of the arm's
    wall time). rate = scale * sum(count) / sum(seconds) over the (resampled) items. Returns the
    point estimates and percentile CIs of rate_x / rate_y and of rate_x - rate_y (per `scale`)."""
    items = sorted(set(x) & set(y))
    if not items:
        raise ValueError("no common items")
    groups = _cluster_groups(items, clusters)
    gx = [(sum(x[i][0] for i in g), sum(x[i][1] for i in g)) for g in groups]
    gy = [(sum(y[i][0] for i in g), sum(y[i][1] for i in g)) for g in groups]
    k = len(groups)
    rng = random.Random(seed)
    ratios, diffs = [], []
    for _ in range(n_boot):
        cx = sx = cy = sy = 0.0
        for _ in range(k):
            j = rng.randrange(k)
            cx += gx[j][0]
            sx += gx[j][1]
            cy += gy[j][0]
            sy += gy[j][1]
        rx, ry = _rate(cx, sx, scale), _rate(cy, sy, scale)
        ratios.append(_ratio(rx, ry))
        diffs.append(rx - ry)
    rx = _rate(sum(c for c, _ in gx), sum(s for _, s in gx), scale)
    ry = _rate(sum(c for c, _ in gy), sum(s for _, s in gy), scale)
    r_lo, r_hi = _percentiles(ratios, alpha)
    d_lo, d_hi = _percentiles(diffs, alpha)
    return {"n_items": len(items), "n_clusters": k, "rate_x": rx, "rate_y": ry,
            "ratio": _ratio(rx, ry), "ratio_ci_low": r_lo, "ratio_ci_high": r_hi,
            "diff": rx - ry, "diff_ci_low": d_lo, "diff_ci_high": d_hi}


# -- stratified cluster bootstrap --------------------------------------------------------------
# Strata are groups of task families. Each cluster lies in
# exactly one stratum and is resampled once, with all its items. In every replicate each stratum s
# draws k_s clusters with replacement from its own k_s clusters (strata in sorted key order, clusters
# in sorted id order, one random.Random(seed) stream for all strata). No extra weights: the
# estimators are those of cluster_paired_bootstrap and paired_rate_bootstrap over the pooled draw,
# so the point estimates equal the unstratified ones. With one stratum the draws and the CIs equal
# the unstratified functions exactly.

def _stratified_groups(items: List[str], clusters: Optional[Dict[str, str]],
                       strata: Optional[Dict[str, str]]) -> List[tuple]:
    """[(stratum, [cluster item lists in sorted cluster-id order])] in sorted stratum order.

    strata: item -> stratum (None: one stratum). Raises ValueError for an item without a stratum
    and for a cluster whose items lie in two strata."""
    by_stratum: Dict[str, List[str]] = {}
    of_cluster: Dict[str, str] = {}
    for i in items:
        if strata is None:
            s = ""
        elif strata.get(i) is None:
            raise ValueError(f"item {i!r} has no stratum")
        else:
            s = strata[i]
        c = (clusters or {}).get(i) or i
        if of_cluster.setdefault(c, s) != s:
            raise ValueError(f"cluster {c!r} lies in two strata: {of_cluster[c]!r} and {s!r}")
        by_stratum.setdefault(s, []).append(i)
    return [(s, _cluster_groups(by_stratum[s], clusters)) for s in sorted(by_stratum)]


def _strata_info(sgroups: List[tuple]) -> dict:
    return {s: {"n_clusters": len(gs), "n_items": sum(len(g) for g in gs)} for s, gs in sgroups}


def stratified_cluster_paired_bootstrap(a: Dict[str, List[float]], b: Dict[str, List[float]],
                                        clusters: Optional[Dict[str, str]] = None,
                                        strata: Optional[Dict[str, str]] = None, n_boot: int = 10000,
                                        seed: int = 0, alpha: float = 0.05) -> dict:
    """cluster_paired_bootstrap with the clusters resampled within strata (item -> stratum).

    Each replicate draws, per stratum, as many clusters as it has, with replacement, and takes the
    mean item difference over all drawn items of all strata."""
    ma, mb = item_means(a), item_means(b)
    items = sorted(set(ma) & set(mb))
    if not items:
        raise ValueError("no common items")
    sgroups = _stratified_groups(items, clusters, strata)
    parts = [([sum(ma[i] - mb[i] for i in g) for g in gs], [len(g) for g in gs]) for _, gs in sgroups]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        m = 0
        for sums, sizes in parts:
            k = len(sums)
            for _ in range(k):
                j = rng.randrange(k)
                s += sums[j]
                m += sizes[j]
        boots.append(s / m)
    lo, hi = _percentiles(boots, alpha)
    return {"n_items": len(items), "n_clusters": sum(len(gs) for _, gs in sgroups), "n_strata": len(sgroups),
            "strata": _strata_info(sgroups), "mean_a": fmean(ma[i] for i in items),
            "mean_b": fmean(mb[i] for i in items), "diff": fmean(ma[i] - mb[i] for i in items),
            "ci_low": lo, "ci_high": hi}


def stratified_paired_rate_bootstrap(x: Dict[str, tuple], y: Dict[str, tuple],
                                     clusters: Optional[Dict[str, str]] = None,
                                     strata: Optional[Dict[str, str]] = None, n_boot: int = 10000,
                                     seed: int = 0, alpha: float = 0.05, scale: float = 3600.0) -> dict:
    """paired_rate_bootstrap with the clusters resampled within strata (item -> stratum).

    rate = scale * sum(count) / sum(seconds) over the pooled draw of all strata (ratio of sums)."""
    items = sorted(set(x) & set(y))
    if not items:
        raise ValueError("no common items")
    sgroups = _stratified_groups(items, clusters, strata)
    parts = [([(sum(x[i][0] for i in g), sum(x[i][1] for i in g)) for g in gs],
              [(sum(y[i][0] for i in g), sum(y[i][1] for i in g)) for g in gs]) for _, gs in sgroups]
    rng = random.Random(seed)
    ratios, diffs = [], []
    for _ in range(n_boot):
        cx = sx = cy = sy = 0.0
        for gx, gy in parts:
            k = len(gx)
            for _ in range(k):
                j = rng.randrange(k)
                cx += gx[j][0]
                sx += gx[j][1]
                cy += gy[j][0]
                sy += gy[j][1]
        rx, ry = _rate(cx, sx, scale), _rate(cy, sy, scale)
        ratios.append(_ratio(rx, ry))
        diffs.append(rx - ry)
    gx_all = [u for gx, _ in parts for u in gx]
    gy_all = [u for _, gy in parts for u in gy]
    rx = _rate(sum(c for c, _ in gx_all), sum(s for _, s in gx_all), scale)
    ry = _rate(sum(c for c, _ in gy_all), sum(s for _, s in gy_all), scale)
    r_lo, r_hi = _percentiles(ratios, alpha)
    d_lo, d_hi = _percentiles(diffs, alpha)
    return {"n_items": len(items), "n_clusters": len(gx_all), "n_strata": len(sgroups),
            "strata": _strata_info(sgroups), "rate_x": rx, "rate_y": ry,
            "ratio": _ratio(rx, ry), "ratio_ci_low": r_lo, "ratio_ci_high": r_hi,
            "diff": rx - ry, "diff_ci_low": d_lo, "diff_ci_high": d_hi}
