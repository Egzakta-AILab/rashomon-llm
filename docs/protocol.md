# Measurement protocol

This document is the reference for what `rashomon-llm` measures and how. The
README is the short version; [validation.md](validation.md) records the gates
and the runs on our own hardware.

The question: how often does one OpenAI-compatible endpoint, given **identical
request parameters** (same prompt, same seed if one is sent, requested
temperature 0), return a different *outcome* under the concurrency real traffic
produces?

## Vocabulary (kept strict on purpose)

- "Identical request parameters, including a requested temperature of 0" — not
  "the same request", because the server may still batch, route or cache
  differently.
- "No divergence observed" is never "the server is deterministic".
- K=8 vs K=1 is a contrast of **client-side scheduling concurrency**, not
  "batching isolation". No cause (batching, routing, prefix cache, speculative
  decoding) is attributed without a measurement that separates it.
- v1 is a **single-turn** instrument. Tools are offered in the request schema
  but never executed: after a divergent tool result the replicas are no longer
  the same request. A multi-step agent mode needs deterministic tool fixtures
  and is future work.

## Sweep design

The default mode is **sweep**:

- The set is executed in **N sweeps** (default 8). Each prompt is sent exactly
  once per sweep, in an independently shuffled order; the scheduling seed is
  recorded separately from the request seed.
- At most **K** measured requests are in flight (default 8). Free slots are
  filled by **other prompts of the same sweep** (heterogeneous co-tenancy).
- **Two copies of the same prompt are never in flight at the same time.**
  Sweep r+1 may fill slots while sweep r drains its tail, so achieved
  concurrency stays close to K.
- No artificial stagger, no background filler by default.
- Every request is tagged with the in-flight count at dispatch; dispatch, first
  byte and end times are recorded.
- **Control arm** (`--control`): a K=1 arm, sequential, with counterbalanced
  order (K=1 first or K=8 first, recorded). The report also gives the
  **cross-arm K1×K8** disagreement, because two arms can each be internally
  stable and still disagree with each other.
- `--mode burst` (N copies of one prompt sent together) stays available as a
  labelled diagnostic — same-batch clones, not the sweep estimand — and is
  never the score. It can miss divergence that depends on co-tenants.

## Metrics

### Outcome disagreement (headline)

The headline is the **pair rate**:

    D = (1/P) * sum over prompts i of (disagreeing pairs of prompt i) / C(N, 2)

with a **cluster bootstrap** over template families (10,000 resamples, seed 0,
percentile interval; `src/rashomon_llm/stats.py`). The pair rate does not depend
on N.

- **Companion:** "prompts with disagreement among N runs" (R@N), with a
  **Wilson** interval and N in the label. Not comparable across different N.
- **Degenerate bootstrap** (every prompt agrees, e.g. 0 of 40): the report
  prints a warning and exact **Clopper-Pearson** bounds at two levels:
  - prompt-level (0/40 prompts: upper bound 8.8%) — assumes the prompts are
    independent;
  - family-level (0/5 families: upper bound 52.2%) — the conservative reading,
    since prompts of one family are close variants.

  The card shows the prompt-level bound, labelled as such. Never "0% [0, 0]"
  without the bound.
- No refusal rule: n and the interval are always printed; below 20 prompts the
  result is labelled exploratory. N=1 has no pairs, and the tool says so.

### Outcome functions

The outcome is defined by a function, never by accuracy:

- `agent40`: name of the first tool call plus normalized arguments (object keys
  sorted, arrays not sorted, numbers parsed, call ids dropped, parallel calls
  as an ordered list). "No call", `finish_reason=length` and "parse failed" are
  **separate outcome classes**. An agent request stops as soon as the first
  tool call closes; the textual curve is censored from that point.
- `math40` (and `long*` sets): the extracted final numeric answer (thousands
  separators, fractions and -0 normalized). No number found — including an
  empty answer cut off at the token cap — is the `no_answer` class.
- `--grader <script>`: the user's own function (stdin JSON -> string), run as a
  local process with a timeout and a recorded hash. `rashomon report` never
  executes a grader named in a run file: the user passes `--grader` again, and
  the recorded hash must match.

Without an outcome function the tool reports textual divergence only, and says
so.

### Textual divergence

- **Share of differing pairs:** per prompt, the share of its non-censored run
  pairs whose visible outputs differ; averaged over prompts, with the same
  family cluster bootstrap as the headline (and exact bounds when it is
  degenerate). A prompt with only censored pairs leaves the mean, and the report
  says over how many prompts the mean runs ("mean over 23 of 24 prompts"). The
  pooled share over all pairs stays in `report.json` as a secondary number.
- **Censored divergence curve (descriptive, no interval):** the pooled share of
  pairs that differ at or before position 64 / 256 / 512 / 1024.
- **Unit:** token ids when the server returns them for both responses (vLLM
  `return_token_ids` via logprobs), otherwise **characters**. The unit is
  always printed next to the threshold ("chars <= 64", "tokens <= 64"), and the
  two units are never mixed.
- Only the **visible content** is compared. Reasoning tokens that a server
  returns in a separate field are not recorded, so textual divergence says
  nothing about them.
- Pairs that are identical and both hit the token cap are censored (equal as
  observed; their continuations are unknown) and counted separately. The
  median first-divergence position is reported only among pairs that diverged.

### Unscorable accounting

- A prompt with fewer than N scorable replicas leaves the denominator.
- If more than 5% of measured requests are unscorable, no headline is printed.
- A per-prompt matrix (outcome classes, cluster sizes, pair disagreement) goes
  to the JSON and Markdown report; the card carries only the aggregates.

## Measurement hygiene

- **Zero retries on measured requests** — not even before the first byte (the
  server may already be generating). 429 / 5xx / disconnect / timeout make the
  request unscorable. A timeout that cannot be confirmed as cancelled is
  recorded as `cancel_unconfirmed`.
- Timeout clocks: connect 30 s, time to first byte 300 s, idle per chunk 120 s,
  total 1800 s. Streaming by default (TTFT is recorded).
- **Achieved in-flight concurrency goes on the card**; a warning below 0.75·K.
- `system_fingerprint`, the returned `model` and
  `usage.prompt_tokens_details.cached_tokens` are recorded per response; a
  fingerprint change mid-run is flagged (requests may have reached different
  backends). Prefix-cache hits are part of the measured production behaviour:
  recorded, not removed.
- The request profile (messages, tool definitions and their order, sampling,
  stop rules, token budget) is frozen and hashed before the first measured
  call. A field the server rejects in the unscored preflight is dropped and
  recorded as a configuration change, never silently.
- Metadata probing is a read-only allowlist of GET endpoints (`/models`,
  `/version`, llama.cpp `/props`, SGLang `/get_server_info`); `--no-probe`
  turns it off. Of the responses only allowlisted keys are kept (model id,
  owner, context length, version, slot count) — never disk paths, chat
  templates or system prompts.
- **Run-file privacy:** the run file stores the endpoint as its scheme and a
  truncated sha256 of the host. The full base URL is stored only with
  `--record-base-url`.

## The card

Model (or a display label given at report time with `--model-label`), K, N,
number of prompts, **pair rate with interval**, companion with interval, the
prompt-level exact bound when the bootstrap is degenerate, the textual curve
with its unit, the K=1 control and cross-arm result when present, achieved
concurrency, unscorable share, the date the run finished, the tool version that
measured it, and the run file's sha256.
One neutral grayscale style, no colours that suggest good or bad. Never the base
URL, headers or raw prompts. Every row fits the card width.

`rashomon verify run.json` recomputes the report from the run file and checks
the run file's sha256. The display label is not part of `report_sha256`.

## Built-in sets

- **agent40:** 40 single-turn agent tasks in 5 families, each offering 2–3
  tools in the OpenAI tools format, with a deterministic outcome function.
- **math40:** 40 numeric tasks in 5 families, produced by our own solver-backed
  generator, not taken from any public benchmark.
- Every set ships a machine-readable manifest: generator and version, seed,
  date, per-item sha256, license, the statement "not derived from any
  benchmark", the statement about LLM involvement in generation, and family ids
  (the bootstrap cluster unit). Sets are regenerated by
  `data/generators/generate_sets.py`; regeneration must not change any item.
- `scripts/check_set_disjointness.py` checks the public set hashes against a
  list of private prompt hashes, so no private prompt enters a public set.

## Comparing two endpoints

`--compare-endpoint` runs the same frozen profile against a second endpoint
(e.g. a deterministic-mode server) and `report --compare` composes a labelled
side-by-side (score and tok/s). It is not called a "cost of determinism" unless
model, hardware, cache and output length are controlled.

## Validation gates

The gates the tool had to pass before release. Results are in
[validation.md](validation.md).

| Gate | Where | Condition |
|---|---|---|
| **G1** | Offline, on the archived raw records of the originating private study | The tool's statistics code reproduces the study's published aggregates from the raw records: pair rates with their intervals and the first-divergence quantiles. |
| **G2** | Mock server in this repository | Deterministic fixtures: a stable server scores 0 with the degenerate-interval warning; a server that diverges **only when co-tenants change** is caught in sweep mode and missed in burst mode; exact realized numbers are compared, not interval coverage. Planted HTTP 429s are unscorable and never retried. |
| **G3** | A single-GPU llama.cpp server | Smoke: the run completes, achieved concurrency ≥ 0.75·K, tool-call parsing works, the card renders. The number is labelled "smoke", not a release criterion. |
| **G4** | The reference vLLM stack of the originating study, with the study's sampling profile | An investigative run against pre-registered criteria for a broken client ([g4-protocol.md](g4-protocol.md)). It gates only README sentences that mention the study; the tool may measure zero. |

## What the tool does not do

- No fixes for nondeterminism; it only measures.
- Nothing is sent anywhere; results stay local until the user shares them.
- No comparison of models by quality; only stability.
- No promises about wall time: one command to run, the duration depends on the
  hardware, model and workload. `--quick` (512 tokens, no control) is labelled
  "quick, not comparable".
