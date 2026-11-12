# SPDX-License-Identifier: Apache-2.0
"""Mock OpenAI-compatible server for the G2 gate (docs/protocol.md).

Stdlib-only threaded server with planted behavior:

- `stable`: every request for a prompt returns the same response. A correct tool
  measures zero divergence — and must print the degenerate-interval warning, not
  a clean bill of determinism.
- `cotenant`: the response depends on the *set of prompts currently in flight*.
  A sweep run (heterogeneous co-tenancy, changing over time) detects it; a burst
  run (N clones of the same prompt sharing one batch) misses it. This is the G2
  property: a co-tenancy-sensitive server is caught in sweep mode and missed in
  burst mode.
- `flaky`: a planted share of requests fails with HTTP 429, to exercise the
  unscorable accounting (measured requests are never retried).

Used by tests and by `python -m rashomon_llm.mock` for manual checks.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _State:
    def __init__(self, mode: str):
        self.mode = mode
        self.lock = threading.Lock()
        self.inflight_prompts: set[str] = set()
        self.request_count = 0


def _prompt_key(body: dict) -> str:
    msgs = json.dumps(body.get("messages", []), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(msgs.encode()).hexdigest()[:12]


def _sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _completion_payload(body: dict, text: str, with_tool: bool) -> list[bytes]:
    model = body.get("model", "mock")
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": model,
            "system_fingerprint": "mock-fp-1"}
    chunks: list[bytes] = []
    first = dict(base, choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
    chunks.append(_sse_chunk(first))
    wants_tools = bool(body.get("tools"))
    if wants_tools:
        delta = {"tool_calls": [{"index": 0, "id": "call_mock", "type": "function",
                                 "function": {"name": "mock_tool", "arguments": ""}}]}
        chunks.append(_sse_chunk(dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])))
        delta2 = {"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"q": text})}}]}
        chunks.append(_sse_chunk(dict(base, choices=[{"index": 0, "delta": delta2, "finish_reason": None}])))
        last = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
    else:
        # Split the text so chunk-assembly is exercised.
        mid = max(1, len(text) // 2)
        for part in (text[:mid], text[mid:]):
            chunks.append(_sse_chunk(dict(base, choices=[{"index": 0, "delta": {"content": part},
                                                          "finish_reason": None}])))
        last = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
    chunks.append(_sse_chunk(last))
    usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
             "prompt_tokens_details": {"cached_tokens": 0}}
    chunks.append(_sse_chunk(dict(base, choices=[], usage=usage)))
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def make_handler(state: _State, chunk_delay_s: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence
            pass

        def _json_body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{"id": "mock-model",
                                                                  "object": "model"}]})
            elif self.path == "/v1/version":
                self._send_json(200, {"version": "mock-0.0"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return
            body = self._json_body()
            with state.lock:
                state.request_count += 1
                n = state.request_count
                if state.mode == "flaky" and n % 10 == 0:
                    self._send_json(429, {"error": "rate limited (planted)"})
                    return
                key = _prompt_key(body)
                # Co-tenancy signature = the set of distinct prompts in flight,
                # INCLUDING this one. In a burst wave all copies of one prompt
                # therefore share one signature (the set is just {prompt}); in a
                # sweep the companions vary from copy to copy. Self must be in
                # the set or the first burst copy would see "empty" and the last
                # would see "{prompt}" — a mock artifact, not batching physics.
                state.inflight_prompts.add(key)
                if state.mode == "cotenant":
                    signature = hashlib.sha256(
                        "|".join(sorted(state.inflight_prompts)).encode()).hexdigest()[:8]
                else:
                    signature = "fixed"
            try:
                seed_material = f"{key}:{signature}"
                text = "out-" + hashlib.sha256(seed_material.encode()).hexdigest()[:16]
                chunks = _completion_payload(body, text, with_tool=bool(body.get("tools")))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # SSE has no Content-Length: the stream ends by closing the
                # connection, which is how the client learns it is done.
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                # All but the final two chunks are served while this prompt is
                # still registered as in flight. The key is discarded BEFORE the
                # usage/[DONE] chunks are written, so by the time the client can
                # act on the completed stream, the registry no longer contains
                # this request — otherwise a sequential (K=1) next request races
                # the discard and sees a ghost co-tenant.
                for chunk in chunks[:-2]:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    time.sleep(chunk_delay_s)
                with state.lock:
                    state.inflight_prompts.discard(key)
                for chunk in chunks[-2:]:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except ConnectionError:
                # The client may close early by design (agent sets stop after the
                # first tool call closes). BrokenPipeError on POSIX,
                # ConnectionAbortedError/ConnectionResetError on Windows.
                with state.lock:
                    state.inflight_prompts.discard(key)

    return Handler


def serve(mode: str = "stable", port: int = 0, chunk_delay_s: float = 0.01
          ) -> tuple[ThreadingHTTPServer, str]:
    """Start the mock on a background thread; returns (server, base_url)."""
    if mode not in ("stable", "cotenant", "flaky"):
        raise ValueError(f"unknown mock mode {mode!r}")
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(_State(mode), chunk_delay_s))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    import sys

    srv, url = serve(mode=sys.argv[1] if len(sys.argv) > 1 else "stable")
    print(f"mock listening at {url} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
