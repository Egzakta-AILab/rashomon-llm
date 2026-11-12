# SPDX-License-Identifier: Apache-2.0
"""P2 tests: outcome functions, report metrics, card, verify, end-to-end."""
from __future__ import annotations

import json

import pytest

from rashomon_llm.cli import main
from rashomon_llm.mock import serve
from rashomon_llm.model import load_run
from rashomon_llm.outcomes import agent_outcome, grader_outcome, math_outcome
from rashomon_llm.report import compute_report

PIL = pytest.importorskip("PIL", reason="card extra not installed")


# ---------------------------------------------------------------- outcomes

def _ok_record(text=None, calls=None, finish="stop"):
    return {"status": "ok", "text": text, "tool_calls_raw": calls, "finish_reason": finish}


def test_math_outcome_normalizes_numbers():
    assert math_outcome(_ok_record("The answer is 1,000."))["value"] == repr(1000.0)
    assert math_outcome(_ok_record("so we get \\boxed{1/2}"))["value"] == repr(0.5)
    assert math_outcome(_ok_record("result: -0"))["value"] == repr(0.0)
    assert math_outcome(_ok_record("nothing numeric here"))["class"] == "no_answer"
    assert math_outcome({"status": "unscorable"})["class"] == "unscorable"


def _call(name, args: dict) -> dict:
    return {"id": "call_1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_agent_outcome_normalization_is_key_and_type_stable():
    a = agent_outcome(_ok_record(calls=[_call("check_inventory", {"b": 1, "a": 2})]))
    b = agent_outcome(_ok_record(calls=[_call("check_inventory", {"a": 2.0, "b": 1.0})]))
    assert a["class"] == "ok" and a["value"] == b["value"]
    c = agent_outcome(_ok_record(calls=[_call("check_inventory", {"a": 3, "b": 1})]))
    assert c["value"] != a["value"]


def test_agent_outcome_classes():
    assert agent_outcome(_ok_record(calls=None))["class"] == "no_tool_call"
    capped = agent_outcome(_ok_record(calls=None, finish="length"))
    assert capped["class"] == "no_tool_call" and capped["capped"] is True
    bad = agent_outcome(_ok_record(calls=[{"function": {"name": "t", "arguments": "{bad"}}]))
    assert bad["class"] == "parse_failed"


def test_grader_outcome_runs_local_script(tmp_path):
    script = tmp_path / "grader.py"
    script.write_text("import sys, json\nprint('v:' + (json.load(sys.stdin).get('text') or 'none'))\n",
                      encoding="utf-8")
    res = grader_outcome(str(script), _ok_record("abc"))
    assert res == {"class": "ok", "value": "v:abc"}


# ---------------------------------------------------------------- report

def _synthetic_run(divergent: bool, n: int = 4) -> dict:
    requests = []
    for p in range(4):
        for s in range(n):
            value = "A" if (s % 2 == 0 or not divergent) else "B"
            requests.append({
                "id": f"k8:p{p}:s{s}", "prompt_id": f"p{p}", "family": "fam",
                "arm": "k8", "mode": "sweep", "sweep": s, "status": "ok",
                "inflight_at_dispatch": 7, "finish_reason": "tool_calls",
                "tool_calls_raw": [_call("check_inventory", {"sku": value})],
                "text": f"text-{value}", "system_fingerprint": "fp1",
            })
    return {
        "schema": "rashomon-run/1",
        "tool": {"name": "rashomon-llm", "version": "0.1.0a0"},
        "config": {"model": "m", "temperature": 0.0, "sampling": "fixed", "mode": "sweep",
                   "max_tokens": 2048, "n": n, "k": 8, "quick": False},
        "set": {"name": "agent40.jsonl", "n_items": 4},
        "server": {"probed": False, "fields": {}},
        "requests": requests,
        "scheduling": {},
        "run_sha256": "x",
    }


def test_report_stable_run_has_degenerate_bound():
    rep = compute_report(_synthetic_run(divergent=False))
    h = rep["headline_pair_rate"]
    assert h["suppressed"] is False
    assert h["estimate"] == 0.0
    assert h["degenerate"] is True
    assert h["exact_bound"]["ci_high"] > 0.0  # never "0% [0, 0]"


def test_report_divergent_run_pair_rate():
    rep = compute_report(_synthetic_run(divergent=True))
    h = rep["headline_pair_rate"]
    # every prompt: 2 A + 2 B -> 4 of 6 pairs disagree
    assert h["estimate"] == pytest.approx(4 / 6)
    assert h["ci_low"] <= h["estimate"] <= h["ci_high"]
    assert rep["companion"]["estimate"] == 1.0
    assert rep["companion"]["label"].endswith("(R@4)")


def test_report_suppresses_headline_above_five_percent_unscorable():
    run = _synthetic_run(divergent=True)
    for i, r in enumerate(run["requests"]):
        if i % 2 == 0:
            r["status"] = "unscorable"
            r["unscorable_reason"] = "http_429"
    rep = compute_report(run)
    assert rep["headline_pair_rate"]["suppressed"] is True


def test_report_without_outcome_function_is_textual_only():
    run = _synthetic_run(divergent=True)
    run["set"]["name"] = "custom.jsonl"
    rep = compute_report(run)
    assert rep["outcome_function"] is None
    assert rep["headline_pair_rate"]["suppressed"] is True
    assert rep["textual"]["k8"]["pair_share"] > 0


# ---------------------------------------------------------------- end-to-end

def test_end_to_end_run_report_verify_on_stable_mock(tmp_path):
    server, url = serve("stable", chunk_delay_s=0.003)
    out = tmp_path / "run.json"
    try:
        rc = main(["run", "--base-url", url, "--model", "mock-model", "--set", "agent40",
                   "--n", "3", "--k", "8", "--no-probe", "--out", str(out)])
        assert rc == 0
        run = load_run(str(out))
        assert len(run["requests"]) == 120
        assert all(r["status"] == "ok" for r in run["requests"])
        assert all(r["stopped_early"] for r in run["requests"])  # agent sets stop at tool close
        rc = main(["report", str(out)])
        assert rc == 0
        assert (tmp_path / "run.report.json").is_file()
        assert (tmp_path / "run.report.md").is_file()
        card = tmp_path / "run.card.png"
        assert card.is_file()
        with open(card, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"
        assert main(["verify", str(out)]) == 0
    finally:
        server.shutdown()


def test_verify_detects_tampering(tmp_path):
    server, url = serve("stable", chunk_delay_s=0.003)
    out = tmp_path / "run.json"
    try:
        assert main(["run", "--base-url", url, "--model", "mock-model", "--set", "math40",
                     "--n", "2", "--k", "4", "--no-probe", "--out", str(out)]) == 0
    finally:
        server.shutdown()
    run = load_run(str(out))
    run["requests"][0]["text"] = "tampered"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f)
    assert main(["verify", str(out)]) == 1
