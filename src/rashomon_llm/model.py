# SPDX-License-Identifier: Apache-2.0
"""Run file model for rashomon-llm (`rashomon-run/1`).

A run file is the single source of truth: every card and report is recomputed
from it (`rashomon verify`). Records are append-only facts; metrics never mutate
them. The request body sent to the server is frozen and hashed before the first
scored call (`config_sha256`).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

SCHEMA = "rashomon-run/1"

# Request lifecycle states. A measured request is never retried (docs/protocol.md):
# anything that is not a clean completion is unscorable and stays visible.
STATUS_OK = "ok"
STATUS_UNSCORABLE = "unscorable"

UNSCORABLE_REASONS = (
    "http_429",
    "http_5xx",
    "http_other",
    "connect_error",
    "timeout",
    "cancel_unconfirmed",  # timed out and we cannot prove the server stopped the work
    "extraction_failed",
)


@dataclass
class PromptItem:
    id: str
    family: str                 # template family; the bootstrap cluster unit
    messages: list[dict]
    tools: list | None = None  # OpenAI tool schema, offered but never executed in v1


@dataclass
class RequestRecord:
    id: str                     # f"{arm}:{prompt_id}:s{sweep}"
    prompt_id: str
    family: str
    arm: str                    # "k8" (concurrent) | "k1" (control)
    mode: str                   # "sweep" | "burst"
    sweep: int
    dispatched_at: float = 0.0
    first_byte_at: float | None = None
    finished_at: float | None = None
    inflight_at_dispatch: int = 0
    status: str = STATUS_OK
    unscorable_reason: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    stopped_early: bool = False       # agent sets: stream closed after the first tool call
    text: str | None = None
    token_ids: list[int] | None = None
    tool_calls_raw: list | None = None
    model_returned: str | None = None
    system_fingerprint: str | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    @property
    def ttft_s(self) -> float | None:
        if self.first_byte_at is None or not self.dispatched_at:
            return None
        return self.first_byte_at - self.dispatched_at

    @property
    def duration_s(self) -> float | None:
        if self.finished_at is None or not self.dispatched_at:
            return None
        return self.finished_at - self.dispatched_at


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def new_run(config: dict, set_info: dict, server: dict, tool_version: str) -> dict:
    """Create the run document; `config` must already be the frozen request profile."""
    return {
        "schema": SCHEMA,
        "tool": {"name": "rashomon-llm", "version": tool_version},
        "started_at": time.time(),
        "finished_at": None,
        "config": config,
        "config_sha256": sha256_text(canonical_json(config)),
        "server": server,
        "set": set_info,
        "requests": [],
        "scheduling": {},
    }


def finish_run(run: dict) -> None:
    run["finished_at"] = time.time()


def run_sha256(run: dict) -> str:
    """Digest over everything except the digest field itself."""
    body = {k: v for k, v in run.items() if k != "run_sha256"}
    return sha256_text(canonical_json(body))


def save_run(run: dict, path: str) -> None:
    run["run_sha256"] = run_sha256(run)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f, ensure_ascii=False, indent=1)
        f.write("\n")


def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        run = json.load(f)
    if run.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} file: {path}")
    return run


def verify_run(path: str) -> bool:
    run = load_run(path)
    return run.get("run_sha256") == run_sha256(run)
