# SPDX-License-Identifier: Apache-2.0
"""Metrics and report assembly from a run file (docs/protocol.md).

Everything is recomputed from run.json. The headline is the *pair rate*; the
any-of-N companion is labelled with N; a degenerate bootstrap attaches a
Clopper-Pearson bound; above 5% unscorable measured requests no headline is
printed. Cross-arm (K1xK8) disagreement is reported whenever both arms exist —
two arms can each be internally stable and still disagree with each other.
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from .divergence import divergence_curve, pair_matrix
from .intervals import FAMILY_BOUND_NOTE, cluster_rate_interval, wilson_interval
from .outcomes import pick_outcome_fn
from .privacy import public_server_meta

HEADLINE_UNSCORABLE_LIMIT = 0.05


def unit_label(unit: str | None) -> str:
    """Threshold label for the divergence curve: 'chars' or 'tokens', never mixed."""
    return {"char": "chars", "token": "tokens"}.get(unit or "", unit or "units")


def display_model(rep: dict) -> str:
    """Model name to show: the report-time display label if one was given."""
    label = (rep.get("display") or {}).get("model_label")
    return label if label else str((rep.get("config") or {}).get("model"))


def textual_mean(matrix: dict[str, dict], families: dict[str, str]) -> dict | None:
    """Per-prompt share of differing pairs, averaged over prompts, with the same
    cluster bootstrap (by family) as the headline. Censored pairs leave a prompt's
    denominator; a prompt with only censored pairs leaves the mean."""
    values: dict[str, list[float]] = {}
    for pid, m in matrix.items():
        usable = m["n_pairs"] - m["identical_capped"]
        if usable > 0:
            values[pid] = [m["diverged"] / usable]
    if not values:
        return None
    out = cluster_rate_interval(values, clusters={pid: families.get(pid, pid) for pid in values})
    out["n_prompts_total"] = len(matrix)
    return out


def _scorable_ok(record: dict) -> bool:
    return record.get("status") == "ok"


def compute_report(run: dict, grader: str | None = None) -> dict:
    set_name = (run.get("set") or {}).get("name", "set")
    requests = run.get("requests") or []
    cfg = run.get("config") or {}
    n_req = cfg.get("n")

    outcome_fn, outcome_label = pick_outcome_fn(set_name, grader,
                                                outcome=cfg.get("outcome_function"))

    by_prompt_arm: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    unscorable: list[dict] = []
    for rec in requests:
        if _scorable_ok(rec):
            rec = dict(rec)
            rec["outcome"] = outcome_fn(rec) if outcome_fn else None
            by_prompt_arm[rec["prompt_id"]][rec["arm"]].append(rec)
        else:
            unscorable.append(rec)

    total = len(requests)
    unshare = len(unscorable) / total if total else 0.0

    # ---- per-prompt outcome structure and pair rates
    per_prompt: dict[str, Any] = {}
    pair_values: dict[str, list[float]] = {}
    families: dict[str, str] = {}
    any_disagree: dict[str, int] = {}
    n_scorable_prompts = 0

    for pid, arms in by_prompt_arm.items():
        all_recs = [r for recs in arms.values() for r in recs]
        families[pid] = all_recs[0].get("family") or pid
        outcomes = [r["outcome"] for r in all_recs] if outcome_fn else None
        classes = Counter(o["class"] for o in outcomes) if outcomes else None
        values = [o["value"] for o in outcomes if o["class"] != "unscorable"] if outcomes else None

        entry: dict[str, Any] = {
            "family": families[pid],
            "n_scorable": len(all_recs),
            "outcome_classes": dict(classes) if classes else None,
        }

        if len(all_recs) >= 2:
            n_scorable_prompts += 1
            if outcome_fn:
                pairs = list(combinations(range(len(all_recs)), 2))
                disagree = sum(
                    1 for i, j in pairs
                    if (all_recs[i]["outcome"]["class"], all_recs[i]["outcome"]["value"])
                       != (all_recs[j]["outcome"]["class"], all_recs[j]["outcome"]["value"])
                )
                rate = disagree / len(pairs)
                pair_values[pid] = [rate]
                any_disagree[pid] = 1 if disagree else 0
                entry["pair_disagreement"] = round(rate, 6)
                entry["distinct_outcomes"] = len(set(values))
        else:
            entry["unresolved"] = True
        per_prompt[pid] = entry

    # ---- headline: pair rate with cluster bootstrap by template family
    headline: dict[str, Any]
    if unshare > HEADLINE_UNSCORABLE_LIMIT:
        headline = {"suppressed": True,
                    "reason": f"unscorable share {unshare:.1%} exceeds 5%"}
    elif len(pair_values) < 2:
        headline = {"suppressed": True, "reason": "fewer than two scorable prompts"}
    elif outcome_fn is None:
        headline = {"suppressed": True, "reason": "no outcome function; textual divergence only"}
    else:
        headline = cluster_rate_interval(pair_values, clusters=families)
        headline["suppressed"] = False

    # ---- companion: prompts with disagreement among N runs (Wilson, N in label)
    x = sum(any_disagree.values())
    companion = wilson_interval(x, n_scorable_prompts) if n_scorable_prompts else None
    if companion is not None:
        companion["label"] = f"prompts with disagreement among {n_req} runs (R@{n_req})"

    # ---- textual divergence (pair share + censored curve)
    textual: dict[str, Any] = {}
    for arm in ("k8", "k1"):
        recs_by_prompt = {pid: arms[arm] for pid, arms in by_prompt_arm.items() if arms.get(arm)}
        if not recs_by_prompt:
            continue
        matrix = pair_matrix(recs_by_prompt)
        curve = divergence_curve(matrix)
        units = {m["unit"] for m in matrix.values() if m["unit"]}
        curve["descriptive"] = True  # curve points carry no interval
        textual[arm] = {
            "mean_over_prompts": textual_mean(matrix, families),
            # secondary: pooled over all non-censored pairs (no interval)
            "pair_share": (curve["n_diverged_pairs"] / curve["total_pairs"]
                           if curve["total_pairs"] else None),
            "curve": curve,
            "unit": units.pop() if len(units) == 1 else ("mixed" if units else None),
        }

    # ---- cross-arm disagreement (K1 x K8): arms can be internally stable and
    # still disagree with each other
    cross: dict[str, Any] | None = None
    if any(arms.get("k8") and arms.get("k1") for arms in by_prompt_arm.values()):
        cross_pairs = 0
        cross_dis = 0
        for pid, arms in by_prompt_arm.items():
            for a in arms.get("k8", []):
                for b in arms.get("k1", []):
                    cross_pairs += 1
                    if outcome_fn:
                        if (a["outcome"]["class"], a["outcome"]["value"]) != \
                           (b["outcome"]["class"], b["outcome"]["value"]):
                            cross_dis += 1
                    else:
                        if (a.get("text") or "") != (b.get("text") or ""):
                            cross_dis += 1
        if cross_pairs:
            cross = {"pairs": cross_pairs, "disagree": cross_dis,
                     "rate": cross_dis / cross_pairs}

    # ---- hygiene flags
    fps = [r.get("system_fingerprint") for r in requests if r.get("system_fingerprint")]
    fingerprint_changed = len(set(fps)) > 1
    inflights = [r["inflight_at_dispatch"] + 1 for r in requests if r.get("arm") == "k8"]
    k_target = cfg.get("k") or 0
    achieved = {
        "max": max(inflights) if inflights else None,
        "mean": (sum(inflights) / len(inflights)) if inflights else None,
        "share_below_0.75k": (sum(1 for i in inflights if i < 0.75 * k_target) / len(inflights))
        if inflights and k_target else None,
    }

    return {
        "schema": "rashomon-report/1",
        "generated_at": time.time(),
        "tool": run.get("tool"),
        "run_sha256": run.get("run_sha256"),
        "run_finished_at": run.get("finished_at"),
        "config": {k: cfg.get(k) for k in ("model", "temperature", "sampling", "mode",
                                           "max_tokens", "n", "k", "quick")},
        "set": {"name": set_name, "n_items": (run.get("set") or {}).get("n_items")},
        "server": public_server_meta(run.get("server")),
        "outcome_function": outcome_label,
        "counts": {"requests": total, "scorable": total - len(unscorable),
                   "unscorable": len(unscorable), "unscorable_share": unshare,
                   "unscorable_by_reason": dict(Counter(
                       r.get("unscorable_reason") for r in unscorable))},
        "headline_pair_rate": headline,
        "companion": companion,
        "textual": textual,
        "cross_arm_k1xk8": cross,
        "hygiene": {"fingerprint_changed": fingerprint_changed,
                    "achieved_concurrency": achieved},
        "per_prompt": per_prompt,
        "disclaimers": [
            "No divergence observed does not mean the server is deterministic.",
            "K=8 vs K=1 is a client-concurrency contrast, not proof of a specific cause.",
            ("Intervals are descriptive (cluster bootstrap, 10000 resamples, seed 0; "
             "Wilson for the companion)."),
        ],
    }


def textual_summary(t: dict) -> str:
    """'84.8% [72.4%, 97.4%], mean over 23 of 24 prompts' — the number the card,
    report.md and README all show."""
    tm = t["mean_over_prompts"]
    return (f"{tm['estimate']:.1%} [{tm['ci_low']:.1%}, {tm['ci_high']:.1%}], "
            f"mean over {tm['n_items']} of {tm['n_prompts_total']} prompts")


def to_markdown(rep: dict) -> str:
    cfg = rep["config"]
    lines = [
        f"# rashomon-llm report — {time.strftime('%Y-%m-%d %H:%M', time.localtime(rep['generated_at']))}",
        "",
        (f"- model: `{display_model(rep)}` (requested temperature: {cfg.get('temperature')}, "
         f"sampling: {cfg.get('sampling')})"),
        (f"- set: `{rep['set']['name']}` ({rep['set']['n_items']} prompts), mode: {cfg.get('mode')}, "
         f"N={cfg.get('n')}, K={cfg.get('k')}"),
        f"- outcome function: {rep.get('outcome_function') or 'none — textual divergence only'}",
        (f"- requests: {rep['counts']['requests']} "
         f"({rep['counts']['unscorable']} unscorable, {rep['counts']['unscorable_share']:.1%})"),
        "",
    ]
    h = rep["headline_pair_rate"]
    if h.get("suppressed"):
        lines.append(f"**Headline pair rate: not printed — {h['reason']}.**")
    else:
        lines.append(f"**Pair-disagreement rate: {h['estimate']:.1%} "
                     f"[{h['ci_low']:.1%}, {h['ci_high']:.1%}]** "
                     f"({h['n_items']} prompts, {h['n_clusters']} clusters)")
        if h.get("degenerate"):
            eb = h["exact_bound"]
            lines.append(f"  - degenerate bootstrap (all prompts agree); exact "
                         f"Clopper-Pearson bound, prompt-level ({eb['x']}/{eb['n']} prompts): "
                         f"[{eb['ci_low']:.1%}, {eb['ci_high']:.1%}] — this is not proof of determinism")
            fb = h.get("exact_bound_family")
            if fb:
                lines.append(f"  - family-level bound ({fb['x']}/{fb['n']} families): "
                             f"[{fb['ci_low']:.1%}, {fb['ci_high']:.1%}]")
                lines.append(f"  - {FAMILY_BOUND_NOTE}")
    if rep.get("companion"):
        c = rep["companion"]
        lines.append(f"- companion ({c['label']}): {c['estimate']:.1%} "
                     f"[{c['ci_low']:.1%}, {c['ci_high']:.1%}]")
    if rep.get("cross_arm_k1xk8"):
        x = rep["cross_arm_k1xk8"]
        lines.append(f"- cross-arm K1xK8 disagreement: {x['rate']:.1%} of {x['pairs']} pairs")
    lines.append("")
    for arm, t in (rep.get("textual") or {}).items():
        lines.append(f"## textual divergence ({arm}, unit: {t['unit']})")
        tm = t.get("mean_over_prompts")
        if tm is None:
            lines.append(f"- no non-censored pairs; censored: {t['curve']['censored_pairs']}")
        else:
            lines.append(f"- visible text differing: {textual_summary(t)} "
                         f"(censored pairs: {t['curve']['censored_pairs']})")
            if tm.get("degenerate"):
                eb = tm["exact_bound"]
                lines.append(f"  - degenerate bootstrap; exact Clopper-Pearson bound, prompt-level "
                             f"({eb['x']}/{eb['n']} prompts): [{eb['ci_low']:.1%}, {eb['ci_high']:.1%}]")
            lines.append(f"- pooled share over all non-censored pairs (secondary): {t['pair_share']:.1%}")
            ul = unit_label(t["unit"])
            pts = ", ".join(f"{ul} <= {p['le']}: {p['share']:.1%}" for p in t["curve"]["points"]
                            if p["share"] is not None)
            lines.append(f"- divergence curve (descriptive, pooled, no interval): {pts}")
        lines.append("")
    if rep.get("per_prompt"):
        lines.append("## per-prompt matrix")
        lines.append("| prompt | family | scorable runs | pair disagreement | outcome classes |")
        lines.append("|---|---|---|---|---|")
        for pid, entry in sorted(rep["per_prompt"].items()):
            if entry.get("unresolved"):
                lines.append(f"| {pid} | {entry['family']} | {entry['n_scorable']} | "
                             f"unresolved | — |")
                continue
            classes = ", ".join(f"{k}:{v}" for k, v in (entry.get("outcome_classes") or {}).items())
            pd = entry.get("pair_disagreement")
            lines.append(f"| {pid} | {entry['family']} | {entry['n_scorable']} | "
                         f"{'n/a' if pd is None else f'{pd:.2f}'} | {classes} |")
        lines.append("")
    lines.append("## disclaimers")
    lines.extend(f"- {d}" for d in rep["disclaimers"])
    lines.append("")
    lines.append(f"run sha256: `{rep.get('run_sha256')}`")
    return "\n".join(lines) + "\n"


def comparison_markdown(run1: dict, rep1: dict, run2: dict, rep2: dict) -> str:
    """A labelled side-by-side of two run files (e.g. --compare-endpoint output).

    Descriptive only: never a quality ranking, and never a 'cost of determinism'
    claim unless model, hardware, cache and output length are matched.
    """
    def tok_s(run: dict):
        recs = [r for r in run.get("requests", [])
                if r.get("status") == "ok" and r.get("arm") == "k8"]
        toks = sum(r.get("completion_tokens") or 0 for r in recs)
        dur = sum(r.get("duration_s") or 0 for r in recs)
        return (toks / dur) if dur > 0 and toks > 0 else None

    def row(run: dict, rep: dict, label: str) -> str:
        h = rep["headline_pair_rate"]
        hstr = (f"suppressed ({h['reason']})" if h.get("suppressed")
                else f"{h['estimate']:.1%} [{h['ci_low']:.1%}, {h['ci_high']:.1%}]")
        t = tok_s(run)
        cfg = rep["config"]
        return (f"| {label} | {cfg.get('model')} | T={cfg.get('temperature')} | "
                f"{hstr} | {f'{t:.1f}' if t else 'n/a'} | "
                f"{rep['counts']['unscorable_share']:.1%} |")

    return "\n".join([
        "",
        "## side-by-side comparison (labelled; not a quality ranking)",
        "",
        "| run | model | sampling T | pair-disagreement rate | tok/s (k8 arm) | unscorable |",
        "|---|---|---|---|---|---|",
        row(run1, rep1, "A"),
        row(run2, rep2, "B"),
        "",
        ("Compare only runs with the same set, N, K, sampling and token budget. "
         "Differences are descriptive, not a verdict."),
    ])
