# SPDX-License-Identifier: Apache-2.0
"""`rashomon report` and `rashomon verify` (docs/protocol.md).

Everything is recomputed from the run file. A run file is data, never a code
carrier: a grader it records is never executed on its own. The user names the
grader explicitly with `--grader`, and the recorded sha256 is used only to check
that the user's script is the one the run used.
"""
from __future__ import annotations

import json

from .card import render_card
from .model import load_run, sha256_text, verify_run
from .report import comparison_markdown, compute_report, to_markdown


def _grader_for(run: dict, explicit: str | None) -> str | None:
    """The grader to run: only one the user passed with --grader, never one the
    run file names. If the run recorded a grader hash, the user's script must match it."""
    recorded = (run.get("config") or {}).get("grader") or {}
    if not explicit:
        if recorded:
            raise SystemExit(
                "this run used a grader; pass --grader <your copy of the script> to recompute "
                "its outcomes (a grader named inside a run file is never executed)")
        return None
    want = recorded.get("sha256")
    if want:
        with open(explicit, "rb") as f:
            got = sha256_text(f.read().decode("utf-8"))
        if got != want:
            raise SystemExit(f"--grader {explicit!r} does not match the sha256 recorded in the run")
    return explicit


# Keys that describe how a report is shown, not what it measured.
_DISPLAY_KEYS = ("report_sha256", "display")


def report_sha256(rep: dict) -> str:
    """Digest of the measured content; display-only keys are excluded."""
    return sha256_text(json.dumps({k: v for k, v in rep.items() if k not in _DISPLAY_KEYS},
                                  sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _out_base(run_json: str) -> str:
    return run_json.removesuffix(".json")


def report_command(args) -> int:
    run = load_run(args.run_json)
    grader = _grader_for(run, getattr(args, "grader", None))
    rep = compute_report(run, grader=grader)
    rep["report_sha256"] = report_sha256(rep)
    label = getattr(args, "model_label", None)
    if label:
        # Display only: added after the digest, so it never changes report_sha256.
        rep["display"] = {"model_label": label}

    markdown = to_markdown(rep)
    compare_path = getattr(args, "compare", None)
    if compare_path:
        run2 = load_run(compare_path)
        rep2 = compute_report(run2, grader=grader)
        rep["comparison"] = {"run_sha256": run2.get("run_sha256"),
                             "model": rep2["config"].get("model"),
                             "headline_pair_rate": rep2["headline_pair_rate"]}
        markdown += comparison_markdown(run, rep, run2, rep2) + "\n"

    base = _out_base(args.run_json)
    with open(base + ".report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(base + ".report.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(markdown)
    card_path = base + ".card.png"
    try:
        render_card(rep, card_path)
        card_note = card_path
    except SystemExit as e:
        card_note = f"skipped ({e})"

    print(f"report: {base}.report.json / {base}.report.md")
    print(f"card: {card_note}")
    h = rep["headline_pair_rate"]
    if h.get("suppressed"):
        print(f"headline: not printed ({h['reason']})")
    else:
        print(f"headline pair rate: {h['estimate']:.1%} [{h['ci_low']:.1%}, {h['ci_high']:.1%}]")
        if h.get("degenerate"):
            print(f"  degenerate bootstrap; exact Clopper-Pearson upper bound: "
                  f"{h['exact_bound']['ci_high']:.1%} — not proof of determinism")
    return 0


def verify_command(args) -> int:
    ok = verify_run(args.run_json)
    run = load_run(args.run_json)
    rep = compute_report(run)
    print(f"run sha256: {'OK' if ok else 'MISMATCH — the file was modified after the run'}")
    print(f"recomputed report sha256: {rep and sha256_text(json.dumps(rep, sort_keys=True, separators=(',', ':'), ensure_ascii=False))[:32]}…")
    return 0 if ok else 1
