# SPDX-License-Identifier: Apache-2.0
"""Sweep scheduler for rashomon-llm (docs/protocol.md, experiment design).

The estimand is the *sweep*: N shuffled passes over the prompt set at K measured
requests in flight, where two copies of one prompt are never in flight together.
Round r+1 may fill slots while round r drains, so achieved concurrency stays near K.

`burst` is kept only as a labelled diagnostic: N copies of one prompt fired
together, prompts processed one after another.

The K=1 control runs the same copies sequentially. Arm order is counterbalanced
by the scheduling seed (half of runs K=8 first, half K=1 first) and recorded.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Job:
    prompt_idx: int
    prompt_id: str
    family: str
    arm: str        # "k8" | "k1"
    mode: str       # "sweep" | "burst"
    sweep: int

    @property
    def id(self) -> str:
        return f"{self.arm}:{self.prompt_id}:s{self.sweep}"


def build_sweep_jobs(prompt_ids: list[str], families: list[str], n: int, arm: str,
                     seed: int) -> tuple[list[Job], list[list[int]]]:
    """N shuffled rounds over the set; the permutations are returned for the run record.

    The scheduling seed is deliberately a different stream from any inference seed.
    """
    rng = random.Random(f"{seed}:{arm}")  # a stream separate from any inference seed
    jobs: list[Job] = []
    permutations: list[list[int]] = []
    for s in range(n):
        perm = list(range(len(prompt_ids)))
        rng.shuffle(perm)
        permutations.append(perm)
        for idx in perm:
            jobs.append(Job(prompt_idx=idx, prompt_id=prompt_ids[idx], family=families[idx],
                            arm=arm, mode="sweep", sweep=s))
    return jobs, permutations


def build_burst_jobs(prompt_ids: list[str], families: list[str], n: int, arm: str) -> list[Job]:
    """Diagnostic: all N copies of one prompt back to back, then the next prompt."""
    return [Job(prompt_idx=i, prompt_id=prompt_ids[i], family=families[i], arm=arm,
                mode="burst", sweep=s)
            for i in range(len(prompt_ids)) for s in range(n)]


async def run_jobs(jobs: list[Job], k: int,
                   worker: Callable[[Job, int, float], Awaitable[Any]],
                   allow_same_prompt_inflight: bool = False,
                   on_progress: Callable[[int, int], None] | None = None) -> list[Any]:
    """Run jobs with at most k in flight; returns records in dispatch order.

    Unless `allow_same_prompt_inflight` (burst diagnostic), two jobs for the same
    prompt are never in flight together; a later round may fill slots while an
    earlier round drains.
    """
    pending = list(jobs)
    job_by_id = {j.id: j for j in jobs}
    inflight: dict[str, asyncio.Task] = {}      # job.id -> task producing (job.id, record)
    inflight_prompts: set[int] = set()
    results: dict[str, Any] = {}
    order: list[str] = []
    done_count = 0

    async def run_one(job: Job, inflight_at_dispatch: int):
        dispatched = time.time()
        record = await worker(job, inflight_at_dispatch, dispatched)
        return job.id, record

    while pending or inflight:
        if len(inflight) < k:
            still_pending: list[Job] = []
            for job in pending:
                if len(inflight) < k and (allow_same_prompt_inflight
                                          or job.prompt_idx not in inflight_prompts):
                    task = asyncio.ensure_future(run_one(job, len(inflight)))
                    inflight[job.id] = task
                    inflight_prompts.add(job.prompt_idx)
                    order.append(job.id)
                else:
                    still_pending.append(job)
            pending = still_pending
        if not inflight:
            if pending:
                # Unreachable by construction (an empty in-flight set frees every
                # prompt), but fail loudly instead of hanging.
                raise AssertionError("scheduler stalled with pending jobs and nothing in flight")
            break
        done, _ = await asyncio.wait(set(inflight.values()), return_when=asyncio.FIRST_COMPLETED)
        done_ids = set()
        for task in done:
            jid, record = task.result()
            results[jid] = record
            done_ids.add(jid)
            done_count += 1
            if on_progress:
                on_progress(done_count, len(jobs))
        for jid in done_ids:
            del inflight[jid]
            inflight_prompts.discard(job_by_id[jid].prompt_idx)
    return [results[jid] for jid in order]


def arm_order(seed: int) -> list[str]:
    """Counterbalanced arm order: derived from the scheduling seed, recorded on the card."""
    return ["k1", "k8"] if seed % 2 else ["k8", "k1"]
