# SPDX-License-Identifier: Apache-2.0
"""`rashomon run` orchestration (docs/protocol.md).

Freezes the request profile, runs the counterbalanced arms (concurrent sweep or
burst diagnostic; optional K=1 control), records everything, writes run.json.
Measured requests are never retried; the only retrying code is the unscored
preflight probe.
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import asdict

import httpx

from . import __version__
from .model import PromptItem, finish_run, new_run, save_run
from .runner import Runner, api_key_from_env, make_client
from .scheduler import arm_order, build_burst_jobs, build_sweep_jobs, run_jobs
from .sets import SetError, load_set


def _frozen_config(args) -> dict:
    cfg = {
        "model": args.model,
        "max_tokens": 512 if args.quick else args.max_tokens,
        "mode": args.mode,
        "sampling": args.sampling,
        "n": args.n,
        "k": args.k,
        "control_n": args.control_n,
        "seed": args.seed,
        "quick": bool(args.quick),
        "outcome_function": getattr(args, "outcome", None),
    }
    if args.sampling == "production":
        # The server's own defaults: temperature/top-p are omitted, never mixed
        # into the requested-temperature-0 arm.
        cfg["temperature"] = None
    else:
        cfg["temperature"] = args.temperature
    extra = {}
    for cli_name, field in (("top_p", "top_p"), ("top_k", "top_k"), ("min_p", "min_p")):
        value = getattr(args, cli_name, None)
        if value is not None:
            extra[field] = value
    if extra:
        cfg["sampling_extra"] = extra
    return cfg


async def _preflight_generation(runner, client, sample_item) -> list[str]:
    """One unscored probe generation with the REAL request shape (including tools
    when the set carries them — a field may be legal alone and rejected in
    combination, e.g. llama.cpp refuses logprobs with tools + stream).

    Drops rejected optional fields in a fixed order and returns the dropped
    names, which become a recorded configuration change in the run file."""
    from .runner import build_request_body

    dropped: list[str] = []
    for _attempt in range(4):
        probe_cfg = {**runner.config, "max_tokens": 1, "dropped_fields": dropped}
        body = build_request_body(sample_item, probe_cfg)
        try:
            async with client.stream("POST", f"{runner.base_url}/chat/completions",
                                     json=body, timeout=60.0) as resp:
                if resp.status_code == 200:
                    async for _ in resp.aiter_lines():
                        pass
                    return dropped
                if resp.status_code == 400:
                    # Drop the least essential optional field and try once more.
                    dropped.append("logprobs" if "logprobs" not in dropped else "stream_options")
                    continue
                raise SystemExit(
                    f"preflight generation failed: HTTP {resp.status_code} from "
                    f"{runner.base_url}; fix the endpoint before measuring")
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise SystemExit(f"preflight cannot reach {runner.base_url}: {e}") from e
    raise SystemExit("preflight generation: server rejected the request profile repeatedly")


async def run_async(args) -> dict:
    items, set_info = load_set(args.set_name)
    if args.quick:
        set_info = {**set_info, "quick_preset": "512 tokens, no control; labelled not comparable"}
    config = _frozen_config(args)
    if args.grader:
        config["grader"] = {"path": os.path.abspath(args.grader), "sha256": args.grader_sha256}
    key = api_key_from_env(args.api_key_env)
    runner = Runner(args.base_url, config, key)

    async with make_client(key) as client:
        server_meta = await runner.preflight(client, probe=not args.no_probe,
                                             record_base_url=getattr(args, "record_base_url", False))
        dropped = await _preflight_generation(runner, client, items[0])
        config["dropped_fields"] = dropped

        run = new_run(config, set_info, server_meta, __version__)
        prompt_ids = [it.id for it in items]
        families = [it.family for it in items]
        order = arm_order(args.seed)
        run["scheduling"]["arm_order"] = order

        async def worker(job, inflight_at_dispatch, dispatched_at):
            item: PromptItem = items[job.prompt_idx]
            return await runner.run_job(client, job, item, inflight_at_dispatch, dispatched_at)

        def progress(done: int, total: int) -> None:
            if done % 10 == 0 or done == total:
                print(f"\r  {done}/{total} requests", end="", file=sys.stderr, flush=True)

        for arm in order:
            if arm == "k1" and (not args.control or args.quick):
                continue
            k = 1 if arm == "k1" else args.k
            if args.mode == "burst" and arm == "k8":
                # Burst diagnostic: the N copies of one prompt share one wave, and
                # the wave fully drains before the next prompt starts — otherwise
                # leftover co-tenants from the previous wave would contaminate the
                # very batch homogeneity this mode is meant to demonstrate.
                run["scheduling"]["mode"] = "burst"
                all_jobs = build_burst_jobs(prompt_ids, families, args.n, arm)
                records = []
                for i in range(len(prompt_ids)):
                    wave = [j for j in all_jobs if j.prompt_idx == i]
                    records.extend(await run_jobs(wave, k, worker,
                                                  allow_same_prompt_inflight=True,
                                                  on_progress=progress))
            else:
                n_arm = (args.control_n or args.n) if arm == "k1" else args.n
                jobs, perms = build_sweep_jobs(prompt_ids, families, n_arm, arm, args.seed)
                run["scheduling"][f"permutations_{arm}"] = (
                    None if args.mode == "burst" else [[items[i].id for i in p] for p in perms])
                records = await run_jobs(jobs, k, worker, on_progress=progress)
            for rec in records:
                d = asdict(rec)
                d["ttft_s"] = rec.ttft_s
                d["duration_s"] = rec.duration_s
                run["requests"].append(d)
        print(file=sys.stderr)

    finish_run(run)
    return run


def run_command(args) -> int:
    if args.grader:
        # Read and hash the grader here (sync), never inside the event loop.
        from .model import sha256_text
        with open(args.grader, "rb") as f:
            args.grader_sha256 = sha256_text(f.read().decode("utf-8"))
    try:
        run = asyncio.run(run_async(args))
    except SetError as e:
        print(f"set error: {e}", file=sys.stderr)
        return 2
    save_run(run, args.out)

    total = len(run["requests"])
    ok = sum(1 for r in run["requests"] if r["status"] == "ok")
    unscorable = total - ok
    inflights = [r["inflight_at_dispatch"] + 1 for r in run["requests"] if r["arm"] == "k8"]
    duration = run["finished_at"] - run["started_at"]
    print(f"run.json written: {args.out}")
    print(f"  requests: {total} ({ok} ok, {unscorable} unscorable) in {duration:.0f}s")
    if unscorable / max(1, total) > 0.05:
        print("  warning: >5% of measured requests are unscorable; no headline rate "
              "should be printed from this run", file=sys.stderr)
    if inflights:
        print(f"  max achieved in-flight (k8 arm): {max(inflights)}")
    print(f"  run sha256: {run['run_sha256'][:16]}…")

    if args.compare_endpoint:
        # Same frozen config against the second endpoint, into a sibling run
        # file. The side-by-side is composed at report time (`report --compare`).
        import copy

        args2 = copy.copy(args)
        args2.base_url = args.compare_endpoint
        print("comparison endpoint: running the same profile against it now…",
              file=sys.stderr)
        run2 = asyncio.run(run_async(args2))
        out2 = args.out[:-5] + ".compare.json" if args.out.endswith(".json") \
            else args.out + ".compare.json"
        save_run(run2, out2)
        total2 = len(run2["requests"])
        ok2 = sum(1 for r in run2["requests"] if r["status"] == "ok")
        print(f"comparison run.json written: {out2}")
        print(f"  requests: {total2} ({ok2} ok, {total2 - ok2} unscorable)")
    return 0
