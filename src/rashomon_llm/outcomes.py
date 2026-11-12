# SPDX-License-Identifier: Apache-2.0
"""Outcome functions (docs/protocol.md, metrics).

An outcome is NEVER accuracy. It is a value with an explicit class, and the
classes are part of the data:

- `ok`: a normalized outcome value (a number, or first tool name + arguments);
- `no_tool_call`: the model answered without calling a tool (agent sets);
- `no_answer`: no numeric answer found (math sets);
- `parse_failed`: the tool-call arguments were not valid JSON;
- `unscorable`: the request itself failed (never an outcome).

Early-stop note: agent runs stop as soon as the first tool call closes
(`stopped_early`); the outcome is fully observed at that point by construction.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from typing import Any

OUTCOME_CLASSES = ("ok", "no_tool_call", "no_answer", "parse_failed", "unscorable")

# ---------------------------------------------------------------------------
# normalization


def normalize_value(obj: Any) -> Any:
    """Canonical form: object keys sorted, array order preserved, numbers as
    float (1 and 1.0 are the same), call ids already dropped by the caller."""
    if isinstance(obj, dict):
        return {k: normalize_value(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [normalize_value(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        v = float(obj)
        return 0.0 if v == 0 else v  # -0.0 == 0.0
    return obj


def _canon(obj: Any) -> str:
    return json.dumps(normalize_value(obj), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


# ---------------------------------------------------------------------------
# agent outcome: first tool call, name + normalized arguments


def agent_outcome(record: dict) -> dict:
    """Outcome = first tool call (name + normalized args), as an ordered list of
    one entry; parallel calls in the same message are an ordered list (order
    preserved, call ids dropped)."""
    if record.get("status") != "ok":
        return {"class": "unscorable", "value": None}
    calls = record.get("tool_calls_raw") or []
    if not calls:
        if record.get("finish_reason") == "length":
            return {"class": "no_tool_call", "value": None, "capped": True}
        return {"class": "no_tool_call", "value": None}
    first = calls[0]
    fn = (first or {}).get("function") or {}
    name = fn.get("name") or ""
    raw_args = fn.get("arguments") or ""
    try:
        args = json.loads(raw_args) if raw_args else {}
    except ValueError:
        return {"class": "parse_failed", "value": name}
    return {"class": "ok", "value": f"{name}({_canon(args)})"}


# ---------------------------------------------------------------------------
# math outcome: the final numeric answer

_NUMBER = re.compile(r"""
    [-+]?                       # sign
    (?:
        \d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?   # 1,000 or 1 000, optional decimals
        | \d+\.\d+                       # decimal
        | \d+                            # integer
    )
    (?:\s*/\s*[-+]?\d+(?:\.\d+)?)?         # optional fraction part
""", re.VERBOSE)


def _parse_number(token: str) -> float | None:
    token = token.strip().replace(",", "").replace(" ", "")
    try:
        if "/" in token:
            num, den = token.split("/", 1)
            den_v = float(den)
            if den_v == 0:
                return None
            v = float(num) / den_v
        else:
            v = float(token)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return 0.0 if v == 0 else round(v, 9)


def math_outcome(record: dict) -> dict:
    """Outcome = the last number-like token in the text, normalized
    (thousands separators, fractions, -0)."""
    if record.get("status") != "ok":
        return {"class": "unscorable", "value": None}
    text = record.get("text") or ""
    # Prefer a \boxed{...} or "answer is X" form when present.
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    candidates = boxed if boxed else [text]
    for scope in candidates:
        found = None
        for m in _NUMBER.finditer(scope):
            v = _parse_number(m.group(0))
            if v is not None:
                found = v
        if found is not None:
            return {"class": "ok", "value": repr(found)}
    return {"class": "no_answer", "value": None}


# ---------------------------------------------------------------------------
# user grader: local script, stdin JSON -> stdout string, timed and hashed


def grader_outcome(script: str, record: dict, timeout_s: float = 10.0) -> dict:
    if record.get("status") != "ok":
        return {"class": "unscorable", "value": None}
    payload = json.dumps(record, ensure_ascii=False)
    try:
        proc = subprocess.run([sys.executable, script], input=payload, capture_output=True,
                              text=True, timeout=timeout_s, encoding="utf-8", check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"class": "unscorable", "value": None, "grader_error": str(e)}
    if proc.returncode != 0:
        return {"class": "unscorable", "value": None,
                "grader_error": proc.stderr.strip()[:500]}
    return {"class": "ok", "value": proc.stdout.strip()}


def pick_outcome_fn(set_name: str, grader: str | None, outcome: str | None = None):
    """Outcome function selection: explicit --outcome wins, then a grader, then
    the set-name prefix (agent*/math*/long*)."""
    if outcome == "none":
        return None, None
    if grader:
        return lambda rec: grader_outcome(grader, rec), f"grader:{grader}"
    if outcome == "agent":
        return agent_outcome, "agent:first-tool-call"
    if outcome == "math":
        return math_outcome, "math:final-number"
    lowered = set_name.lower()
    if lowered.startswith("agent"):
        return agent_outcome, "agent:first-tool-call"
    if lowered.startswith(("math", "long")):
        return math_outcome, "math:final-number"
    return None, None
