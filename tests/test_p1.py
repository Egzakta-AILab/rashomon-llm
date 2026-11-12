# SPDX-License-Identifier: Apache-2.0
"""P1 tests: statistics copy, intervals, scheduler, divergence, mock integration."""
from __future__ import annotations

import json

import pytest

from rashomon_llm import divergence, intervals, stats
from rashomon_llm.cli import main
from rashomon_llm.mock import serve
from rashomon_llm.model import load_run, verify_run
from rashomon_llm.scheduler import arm_order, build_burst_jobs, build_sweep_jobs, run_jobs

# ---------------------------------------------------------------- stats copy

def test_cluster_bootstrap_matches_paired_without_clusters():
    a = {"t1": [1.0], "t2": [0.0], "t3": [1.0], "t4": [0.0], "t5": [1.0]}
    b = {k: [0.0] for k in a}
    plain = stats.paired_bootstrap(a, b, n_boot=2000, seed=0)
    clustered = stats.cluster_paired_bootstrap(a, b, n_boot=10000, seed=0)
    assert plain["diff"] == pytest.approx(0.6)
    assert clustered["diff"] == pytest.approx(plain["diff"])
    assert clustered["ci_low"] == pytest.approx(plain["ci_low"])
    assert clustered["ci_high"] == pytest.approx(plain["ci_high"])


def test_cluster_bootstrap_resamples_clusters_as_units():
    # Two clusters, each internally unanimous: the only possible bootstrap means
    # are 0, 0.5 and 1, so both CI ends must be one of those values.
    a = {"a1": [1.0], "a2": [1.0], "b1": [0.0], "b2": [0.0]}
    z = {k: [0.0] for k in a}
    res = stats.cluster_paired_bootstrap(a, z, clusters={"a1": "A", "a2": "A",
                                                         "b1": "B", "b2": "B"},
                                         n_boot=2000, seed=0)
    assert res["n_clusters"] == 2
    assert res["ci_low"] in (0.0, 0.5, 1.0)
    assert res["ci_high"] in (0.0, 0.5, 1.0)


def test_stats_percentile_indexing_is_unchanged():
    # Hand-computed percentile check on the recycled indexing convention:
    # boots sorted, lo at int(alpha/2 * n), hi at min(n-1, int((1-alpha/2)*n)).
    assert stats._percentiles(list(range(10000)), 0.05) == (250, 9750)


# ---------------------------------------------------------------- intervals

def test_wilson_known_value():
    res = intervals.wilson_interval(4, 20)
    assert res["estimate"] == pytest.approx(0.2)
    assert res["ci_low"] == pytest.approx(0.081, abs=0.01)
    assert res["ci_high"] == pytest.approx(0.416, abs=0.01)


def test_clopper_pearson_degenerate_zero_of_twenty():
    res = intervals.clopper_pearson(0, 20)
    assert res["ci_low"] == 0.0
    assert res["ci_high"] == pytest.approx(0.1684, abs=0.001)


def test_clopper_pearson_symmetry():
    lo = intervals.clopper_pearson(3, 20)
    hi = intervals.clopper_pearson(17, 20)
    assert lo["ci_low"] == pytest.approx(1 - hi["ci_high"], abs=1e-9)


def test_cluster_rate_interval_degenerate_attaches_exact_bound():
    values = {f"p{i}": [0.0] for i in range(20)}
    res = intervals.cluster_rate_interval(values)
    assert res["estimate"] == 0.0
    assert res["degenerate"] is True
    assert res["exact_bound"]["ci_high"] == pytest.approx(0.1684, abs=0.001)


def test_cluster_rate_interval_basic():
    values = {f"p{i}": [0.5] for i in range(10)} | {f"q{i}": [0.0] for i in range(10)}
    res = intervals.cluster_rate_interval(values, n_boot=2000)
    assert res["estimate"] == pytest.approx(0.25)
    assert res["ci_low"] <= res["estimate"] <= res["ci_high"]


# ---------------------------------------------------------------- scheduler

def test_sweep_never_two_copies_of_one_prompt_inflight():
    import asyncio

    ids = [f"p{i}" for i in range(6)]
    jobs, perms = build_sweep_jobs(ids, ["f"] * 6, n=4, arm="k8", seed=1)
    assert len(jobs) == 24 and len(perms) == 4

    active: set[int] = set()
    overlap_violations = []

    async def worker(job, inflight_at_dispatch, dispatched_at):
        if job.prompt_idx in active:
            overlap_violations.append(job.id)
        active.add(job.prompt_idx)
        await asyncio.sleep(0.005)
        active.discard(job.prompt_idx)
        return job.id

    out = asyncio.run(run_jobs(jobs, k=4, worker=worker))
    # dispatch order may differ from jobs order (a later round fills a freed slot
    # while the earlier round drains); completeness is what the runner relies on
    assert sorted(out) == sorted(j.id for j in jobs)
    assert overlap_violations == []


def test_burst_allows_same_prompt_overlap():
    import asyncio

    ids = [f"p{i}" for i in range(3)]
    jobs = build_burst_jobs(ids, ["f"] * 3, n=3, arm="k8")
    active: set[int] = set()
    overlaps = []

    async def worker(job, inflight_at_dispatch, dispatched_at):
        if job.prompt_idx in active:
            overlaps.append(job.id)
        active.add(job.prompt_idx)
        await asyncio.sleep(0.01)
        active.discard(job.prompt_idx)
        return job.id

    asyncio.run(run_jobs(jobs, k=3, worker=worker, allow_same_prompt_inflight=True))
    assert overlaps  # burst is the labelled diagnostic: copies DO share the batch


def test_arm_order_is_counterbalanced_by_seed():
    assert arm_order(2) == ["k8", "k1"]
    assert arm_order(3) == ["k1", "k8"]


# ---------------------------------------------------------------- divergence

def _rec(text, finish="stop", token_ids=None):
    return {"text": text, "finish_reason": finish, "token_ids": token_ids}


def test_first_diff_char_and_token_units():
    d = divergence.classify_pair(_rec("abcdef"), _rec("abcXef"))
    assert d["kind"] == "diverged" and d["pos"] == 3 and d["unit"] == "char"
    d = divergence.classify_pair(_rec("same", token_ids=[1, 2, 3]), _rec("diff", token_ids=[1, 9, 3]))
    assert d["kind"] == "diverged" and d["pos"] == 1 and d["unit"] == "token"


def test_identical_capped_pairs_are_censored():
    a = _rec("x" * 100, finish="length")
    b = _rec("x" * 100, finish="length")
    assert divergence.classify_pair(a, b)["kind"] == "identical_capped"
    matrix = divergence.pair_matrix({"p": [a, b]})
    curve = divergence.divergence_curve(matrix)
    assert curve["censored_pairs"] == 1
    assert curve["total_pairs"] == 0
    assert curve["points"][0]["share"] is None


def test_divergence_curve_counts():
    recs = [_rec("aaaa"), _rec("aaxa"), _rec("aaaa")]
    matrix = divergence.pair_matrix({"p": recs})
    curve = divergence.divergence_curve(matrix)
    assert curve["n_diverged_pairs"] == 2
    assert curve["points"][1]["share"] == pytest.approx(2 / 3)  # by position 256


# ---------------------------------------------------------------- mock + cli

def _write_set(tmp_path, n=4):
    path = tmp_path / "set.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"id": f"p{i}", "family": "fam",
                                "messages": [{"role": "user", "content": f"task {i}"}]}) + "\n" for i in range(n))
    return str(path)


def _run(tmp_path, url, extra, n=4, k=4):
    out = tmp_path / "run.json"
    rc = main(["run", "--base-url", url, "--model", "mock-model", "--set", _write_set(tmp_path),
               "--n", str(n), "--k", str(k), "--no-probe", "--out", str(out)] + extra)
    assert rc == 0
    assert verify_run(str(out))
    return load_run(str(out))


def test_mock_stable_server_scores_zero(tmp_path):
    server, url = serve("stable", chunk_delay_s=0.005)
    try:
        run = _run(tmp_path, url, ["--control"])
    finally:
        server.shutdown()
    recs = run["requests"]
    assert all(r["status"] == "ok" for r in recs)
    texts = {}
    for r in recs:
        texts.setdefault(r["prompt_id"], set()).add(r["text"])
    assert all(len(t) == 1 for t in texts.values())


def test_cotenant_server_caught_by_sweep_missed_by_burst(tmp_path):
    """The G2 property: co-tenancy-driven divergence is visible in sweep mode and
    invisible in burst mode, and the K=1 control stays stable."""
    server, url = serve("cotenant", chunk_delay_s=0.01)
    try:
        sweep = _run(tmp_path, url, ["--control"], n=4, k=4)
        burst = _run(tmp_path, url, ["--mode", "burst"], n=4, k=4)
    finally:
        server.shutdown()

    def per_arm(run, arm):
        by_prompt: dict[str, set] = {}
        for r in run["requests"]:
            if r["arm"] == arm and r["status"] == "ok":
                by_prompt.setdefault(r["prompt_id"], set()).add(r["text"])
        return by_prompt

    sweep_k8 = per_arm(sweep, "k8")
    sweep_k1 = per_arm(sweep, "k1")
    burst_k8 = per_arm(burst, "k8")

    assert any(len(t) > 1 for t in sweep_k8.values())       # sweep detects it
    assert all(len(t) == 1 for t in sweep_k1.values())      # control stays stable
    assert all(len(t) == 1 for t in burst_k8.values())      # burst misses it


def test_flaky_server_marks_unscorable_and_never_retries(tmp_path):
    server, url = serve("flaky", chunk_delay_s=0.005)
    try:
        run = _run(tmp_path, url, [], n=10, k=4)  # 40 requests, every 10th is a planted 429
    finally:
        server.shutdown()
    bad = [r for r in run["requests"] if r["status"] == "unscorable"]
    assert len(bad) == 4
    assert all(r["unscorable_reason"] == "http_429" for r in bad)


def test_run_file_schema_and_frozen_config(tmp_path):
    server, url = serve("stable", chunk_delay_s=0.005)
    try:
        run = _run(tmp_path, url, [])
    finally:
        server.shutdown()
    assert run["schema"] == "rashomon-run/1"
    assert run["config"]["temperature"] == 0.0
    assert run["config_sha256"]
    assert {r["arm"] for r in run["requests"]} == {"k8"}
    assert run["scheduling"]["arm_order"] in (["k8", "k1"], ["k1", "k8"])
    assert all(r["inflight_at_dispatch"] >= 0 for r in run["requests"])
