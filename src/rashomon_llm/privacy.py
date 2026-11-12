# SPDX-License-Identifier: Apache-2.0
"""What a run file and a report may say about the server.

A run file is meant to be shareable, so by default it does not carry the full
base URL: only the scheme and a short sha256 of the host (enough to tell two
endpoints apart, not enough to name one). `--record-base-url` stores the full
URL as an explicit opt-in.

Probed server metadata is reduced to an allowlist of keys. Endpoints such as
`/models` (`root`), llama.cpp `/props` (`model_path`, `chat_template`,
`system_prompt`) or SGLang `/get_server_info` (launch arguments) can expose disk
paths or a system prompt; none of that is kept.
"""
from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import urlsplit

HOST_HASH_CHARS = 16

# source -> allowed top-level keys. `models` is filtered per entry of `data`.
_MODEL_ENTRY_KEYS = ("id", "object", "owned_by", "max_model_len")
_ALLOWED_KEYS: dict[str, tuple[str, ...]] = {
    "version": ("version",),
    "props": ("build_info", "total_slots"),
    "server_info": ("version",),
}


def describe_base_url(base_url: str, record_full: bool = False) -> dict:
    """Scheme + truncated host hash; the full URL only on explicit opt-in."""
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    out: dict[str, Any] = {
        "scheme": parts.scheme or None,
        "host_sha256": hashlib.sha256(host.encode("utf-8")).hexdigest()[:HOST_HASH_CHARS],
    }
    if record_full:
        out["base_url"] = base_url
    return out


def _filter_models(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    entries = value.get("data")
    if not isinstance(entries, list):
        return None
    kept = [{k: e[k] for k in _MODEL_ENTRY_KEYS if k in e} for e in entries if isinstance(e, dict)]
    return {"data": kept}


def sanitize_fields(fields: dict | None) -> dict:
    """Reduce probed metadata to the allowlist; unknown sources are dropped."""
    out: dict[str, Any] = {}
    for source, entry in (fields or {}).items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if source == "models":
            kept = _filter_models(value)
        elif source in _ALLOWED_KEYS and isinstance(value, dict):
            kept = {k: value[k] for k in _ALLOWED_KEYS[source] if k in value}
        else:
            kept = None
        if kept:
            out[source] = {"value": kept, "provenance": entry.get("provenance", "unknown")}
    return out


def public_server_meta(server: dict | None) -> dict | None:
    """Server block safe to put in a report, also for run files written before
    the allowlist existed (their full base_url is reduced here, at report time)."""
    if server is None:
        return None
    out: dict[str, Any] = {"probed": server.get("probed")}
    endpoint = server.get("endpoint")
    if endpoint is None and server.get("base_url"):
        endpoint = describe_base_url(server["base_url"])
    if endpoint is not None:
        out["endpoint"] = {k: v for k, v in endpoint.items() if k != "base_url"}
    out["fields"] = sanitize_fields(server.get("fields"))
    return out
