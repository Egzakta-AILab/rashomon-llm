# SPDX-License-Identifier: Apache-2.0
"""HTTP runner for rashomon-llm (docs/protocol.md, measurement hygiene).

Rules baked in here, not in callers:

- a measured request is NEVER retried — not even before the first byte, because
  the server may already be generating; any failure is an unscorable record;
- streaming is the default (TTFT is a recorded quantity);
- timeout clocks are connect 30 s / TTFT 300 s / per-chunk idle 120 s /
  total 1800 s; a timeout that cannot be confirmed cancelled marks the record
  `cancel_unconfirmed`;
- every response carries `model`, `system_fingerprint`, usage and
  `cached_tokens` when the server returns them; a fingerprint change mid-run
  is flagged by the report layer;
- the API key is read from an environment variable, never from argv or a file.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from .model import STATUS_UNSCORABLE, PromptItem, RequestRecord
from .privacy import describe_base_url, sanitize_fields
from .scheduler import Job

# Timeout clocks (seconds). httpx `read` is per-chunk idle, which matches the
# per-chunk idle clock; the total deadline is enforced with asyncio, and the
# wait for response headers is capped at TTFT_LIMIT.
CONNECT_TIMEOUT = 30.0
TTFT_LIMIT = 300.0
CHUNK_IDLE_LIMIT = 120.0
TOTAL_DEADLINE = 1800.0


def make_client(api_key: str | None) -> httpx.AsyncClient:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, read=CHUNK_IDLE_LIMIT,
                            write=30.0, pool=CONNECT_TIMEOUT)
    return httpx.AsyncClient(headers=headers, timeout=timeout)


def build_request_body(item: PromptItem, config: dict) -> dict:
    """Assemble the exact request body from the frozen profile.

    Fields the server rejects are a recorded configuration change (see
    `preflight`), never silently dropped mid-run.
    """
    body: dict[str, Any] = {
        "model": config["model"],
        "messages": item.messages,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "stream": True,
    }
    dropped = set(config.get("dropped_fields") or [])
    if "stream_options" not in dropped:
        body["stream_options"] = {"include_usage": True}
    if "logprobs" not in dropped:
        body["logprobs"] = True
    if config.get("seed") is not None:
        body["seed"] = config["seed"]
    if item.tools:
        body["tools"] = item.tools
    if config.get("sampling_extra"):
        body.update(config["sampling_extra"])
    return body


def _extract_usage(usage: dict | None) -> tuple[int | None, int | None]:
    if not usage:
        return None, None
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens"), usage.get("completion_tokens")


def _token_ids_from_logprobs(content: list | None) -> list[int] | None:
    """vLLM returns token strings like 'token_id:1234' in completions logprobs."""
    if not content:
        return None
    ids: list[int] = []
    for entry in content:
        token = (entry or {}).get("token", "")
        if token.startswith("token_id:"):
            try:
                ids.append(int(token.split(":", 1)[1]))
            except ValueError:
                return None
        else:
            return None
    return ids or None


class Runner:
    def __init__(self, base_url: str, config: dict, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.config = config
        self.api_key = api_key

    async def _one_request(self, client: httpx.AsyncClient, job: Job, item: PromptItem,
                           inflight_at_dispatch: int, dispatched_at: float) -> RequestRecord:
        rec = RequestRecord(
            id=job.id, prompt_id=job.prompt_id, family=job.family, arm=job.arm,
            mode=job.mode, sweep=job.sweep, dispatched_at=dispatched_at,
            inflight_at_dispatch=inflight_at_dispatch,
        )
        body = build_request_body(item, self.config)
        url = f"{self.base_url}/chat/completions"
        try:
            async with asyncio.timeout(TOTAL_DEADLINE):
                async with client.stream("POST", url, json=body) as resp:
                    rec.http_status = resp.status_code
                    if resp.status_code == 429:
                        return self._fail(rec, "http_429", "HTTP 429")
                    if resp.status_code >= 500:
                        return self._fail(rec, "http_5xx", f"HTTP {resp.status_code}")
                    if resp.status_code != 200:
                        # Record the server's own message; a bare "HTTP 400" is undiagnosable.
                        detail = ""
                        try:
                            raw = await resp.aread()
                            detail = raw.decode("utf-8", "replace")[:300].replace("\n", " ")
                        except httpx.HTTPError:
                            pass
                        return self._fail(rec, "http_other",
                                          f"HTTP {resp.status_code}: {detail}".strip())
                    ctype = resp.headers.get("content-type", "")
                    if "text/event-stream" in ctype:
                        await self._consume_sse(resp, rec, early_stop_tool=bool(item.tools))
                    else:
                        await self._read_json_body(resp, rec)
        except TimeoutError:
            # Cannot prove the server cancelled the in-flight request.
            return self._fail(rec, "cancel_unconfirmed", f"total deadline {TOTAL_DEADLINE}s hit")
        except httpx.ConnectTimeout:
            return self._fail(rec, "timeout", "connect timeout")
        except httpx.ReadTimeout:
            return self._fail(rec, "timeout", f"no chunk for {CHUNK_IDLE_LIMIT}s")
        except (httpx.ConnectError, httpx.NetworkError) as e:
            return self._fail(rec, "connect_error", str(e))
        except httpx.HTTPError as e:
            return self._fail(rec, "http_other", str(e))
        return rec

    def _fail(self, rec: RequestRecord, reason: str, error: str) -> RequestRecord:
        rec.status = STATUS_UNSCORABLE
        rec.unscorable_reason = reason
        rec.error = error
        rec.finished_at = time.time()
        return rec

    async def _read_json_body(self, resp: httpx.Response, rec: RequestRecord) -> None:
        data = await resp.aread()
        rec.first_byte_at = rec.first_byte_at or time.time()
        rec.finished_at = time.time()
        try:
            payload = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            rec.status = STATUS_UNSCORABLE
            rec.unscorable_reason = "extraction_failed"
            rec.error = f"non-JSON body: {e}"
            return
        self._fill_from_payload(payload, rec)

    async def _consume_sse(self, resp: httpx.Response, rec: RequestRecord,
                           early_stop_tool: bool = False) -> None:
        text_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish_reason: str | None = None
        usage: dict | None = None
        token_ids: list[int] = []
        stopped_early = False
        async for line in resp.aiter_lines():
            if rec.first_byte_at is None and line:
                rec.first_byte_at = time.time()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            rec.model_returned = chunk.get("model", rec.model_returned)
            rec.system_fingerprint = chunk.get("system_fingerprint", rec.system_fingerprint)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                ids = _token_ids_from_logprobs((choice.get("logprobs") or {}).get("content"))
                if ids:
                    token_ids.extend(ids)
                for tc in delta.get("tool_calls") or []:
                    slot = tc.get("index", 0)
                    entry = tool_calls.setdefault(slot, {"id": tc.get("id"), "type": tc.get("type"),
                                                         "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        entry["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]
            # Agent sets stop as soon as the first tool call has closed: the
            # outcome is fully observed at that point and the rest of the stream
            # is deliberately not collected (docs/protocol.md, metrics).
            if early_stop_tool and 0 in tool_calls:
                args_so_far = tool_calls[0]["function"]["arguments"]
                if args_so_far:
                    try:
                        json.loads(args_so_far)
                        stopped_early = True
                        break
                    except ValueError:
                        pass
        if stopped_early:
            rec.stopped_early = True
            await resp.aclose()
        rec.finished_at = time.time()
        rec.text = "".join(text_parts) or None
        rec.tool_calls_raw = [tool_calls[i] for i in sorted(tool_calls)] or None
        rec.finish_reason = finish_reason
        rec.token_ids = token_ids or None
        rec.cached_tokens, rec.completion_tokens = _extract_usage(usage)

    def _fill_from_payload(self, payload: dict, rec: RequestRecord) -> None:
        rec.model_returned = payload.get("model")
        rec.system_fingerprint = payload.get("system_fingerprint")
        choices = payload.get("choices") or []
        if choices:
            choice = choices[0]
            rec.finish_reason = choice.get("finish_reason")
            msg = choice.get("message") or {}
            rec.text = msg.get("content") or None
            rec.tool_calls_raw = msg.get("tool_calls") or None
            ids = _token_ids_from_logprobs((choice.get("logprobs") or {}).get("content"))
            rec.token_ids = ids or None
        rec.cached_tokens, rec.completion_tokens = _extract_usage(payload.get("usage"))

    async def run_job(self, client: httpx.AsyncClient, job: Job, item: PromptItem,
                      inflight_at_dispatch: int, dispatched_at: float) -> RequestRecord:
        return await self._one_request(client, job, item, inflight_at_dispatch, dispatched_at)

    async def preflight(self, client: httpx.AsyncClient, probe: bool = True,
                        record_base_url: bool = False) -> dict:
        """Unscored capability probe. May retry (it is not a measured request).

        Returns server metadata with per-field provenance labels, reduced to an
        allowlist (see privacy.py). The endpoint is recorded as scheme + host
        hash unless `record_base_url` is set.
        """
        meta: dict[str, Any] = {"endpoint": describe_base_url(self.base_url, record_base_url),
                                "probed": probe, "fields": {}}
        if not probe:
            return meta
        for path, source in (("/models", "models"), ("/version", "version"),
                             ("/props", "props"), ("/get_server_info", "server_info")):
            try:
                resp = await client.get(f"{self.base_url}{path}", timeout=CONNECT_TIMEOUT)
                if resp.status_code == 200:
                    meta["fields"][source] = {"value": resp.json(), "provenance": "server-reported"}
            except (httpx.HTTPError, ValueError):
                continue
        meta["fields"] = sanitize_fields(meta["fields"])
        return meta


def api_key_from_env(env_name: str | None) -> str | None:
    if not env_name:
        return None
    key = os.environ.get(env_name)
    if not key:
        raise SystemExit(f"environment variable {env_name} is not set")
    return key
