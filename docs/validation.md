# Validation status

Each gate is defined in [protocol.md](protocol.md) (section "Validation gates").
Numbers from the originating private study are deliberately absent, both here
and in the README.

| Gate | What it proves | Status | Date |
|---|---|---|---|
| G1 | The tool's statistics code reproduces the archived measurements, recomputed from the archived raw records (selection reimplemented from the documented rules) | **PASS** | 2026-10-28 |
| G2 | A co-tenancy-sensitive mock server is detected in sweep mode and missed in burst mode; a stable server scores 0 with the degenerate-interval warning; planted HTTP 429s are unscorable and never retried | **PASS** (test suite) | 2026-10-28 |
| G3 | Smoke on a real llama.cpp server: run completes, achieved concurrency ≥ 0.75·K, tool-call parsing works | **PASS** | 2026-10-28 |
| G4 | Live replay on the reference vLLM stack with the sampling profile of the archived study, against pre-registered investigation criteria | **PASS** (valid runs; measured zero with the exact bound) | 2026-10-29 |

## G4 record (protocol: docs/g4-protocol.md, frozen before the runs)

- Stack: vLLM TP2, two DGX Spark (GB10), in-house model (FP8), MTP 3 — the same
  environment as the archived study. The stack was started for G4 and stopped
  after; the previous machine state was restored (verified against pre-G4
  snapshots).
- Four runs, sequential: agent40 and math40, each at requested temperature 0
  and at the archived study's sampling profile (T=1.0, top_k 20, top_p 0.95,
  min_p 0, seed 1); sweep mode, N=8, K=8, with the K=1 control.
- Criteria check (all five pre-registered): 0 unscorable of 2240 requests;
  mean achieved in-flight 7.8–7.9 of K=8; no system_fingerprint change; 0
  tool-call extraction failures; the K=1 control never showed more
  disagreement than K=8.
- **Result: 0.0% outcome pair-disagreement in all four runs**, with the exact
  Clopper-Pearson upper bound 8.8% at the prompt level (0 of 40 prompts, N=8;
  assumes independent prompts) and 52.2% at the family level (0 of 5
  families). Cross-arm K1↔K8 outcome
  agreement: 40/40 prompts in every run. Outputs were not constant across
  prompts (40/40 distinct for agent40), and per-request sampling parameters
  reached the server.
- Interpretation: on this stack, short synthetic tasks with a fixed per-request
  seed show no observable divergence. The archived study's divergence appeared
  on long-horizon agentic traces (first divergence at a median of hundreds of
  tokens into multi-thousand-token generations). Divergence is
  workload-dependent; the honest sentence is "measure your own workload", which
  is what the tool is for. The tool's sensitivity to planted divergence is
  proven by G2 (the sweep catches what the burst misses).
- This result does not reproduce the archived study's long-task numbers and is
  not presented as doing so; per the protocol, any README sentence names the
  stack, the sampling and the interval.

## G3 record (smoke, labelled as such)

- Host: RTX 4090 llama.cpp server, in-house model (IQ4_XS GGUF).
- Protocol: agent40, sweep mode, N=4, K=8, requested temperature 0; 160 requests, 160 ok, 0 unscorable, 240 s wall.
- Achieved in-flight concurrency: max 8 (client-side; server-side batching not measured).
- Tool calls parsed on 160/160 records; the dropped field `logprobs` was
  recorded (llama.cpp rejects `logprobs` together with tools + stream — the
  preflight dropped it and the run file carries the change).
- Outcome pair-disagreement: 0.0%, degenerate bootstrap, exact Clopper-Pearson
  upper bound 8.8% — reported with the bound, not as proof of determinism.
- The G3 run file was not retained, so this record cannot be recomputed with
  `rashomon verify`; it stays here as a text record only, and no G3 card is
  published.

The archived records used for G1 are private and are not part of this
repository; neither is G1's harness, which lives in a private Egzakta
repository and prints aggregates only.

## Exploratory runs (not gates)

### 2026-11-04 — workload-length probe on H100 NVL (stack b)

Motivation: G4 measured zero divergence on the short synthetic sets while the
archived study found divergence on long agentic traces. Hypothesis: divergence
grows with generation length. Run on a dedicated single-GPU in-house model (FP8) server
(vLLM 0.30.0, TP1, MTP 3), removed after the run; nothing else on the machine
was touched.

- Set: `long24` (validation set, `scripts/gen_long_set.py`): 24 tasks
  in 6 families that force long reasoning with a numeric outcome; run file and
  set are not part of the repo.
- Protocol: sweep, N=8, K=8, max_tokens 16384, K=1 control (N=2); two sampling
  arms (requested T=0; and the archived study's sampling profile).
- Generations were genuinely long: median 943 completion tokens at T=0 (1,095
  under the study sampling), p90 at the 16,384 cap, as reported by the server in
  `usage` (which includes reasoning tokens the tool does not record).

| arm | pair-disagreement (outcome) | prompts with divergence | K1↔K8 agree |
|---|---|---|---|
| T=0 | 6.7% [0.0%, 20.0%] | 3/24 | 22/24 |
| study sampling | 7.1% [0.0%, 19.7%] | 4/24 | 23/24 |

What the numbers are made of (recomputed from the run files):

- **Concentration.** At T=0 all outcome divergence is in one of the six
  families (`long_modpow`, 3 of its 4 prompts). Under the study sampling it is
  in two families (`long_modpow`, 3 prompts; `long_lastdigits`, 1 prompt).
- **No-answer outcomes.** 20 of 192 K=8 requests at T=0 (22 of 192 under the
  study sampling) ended with `finish_reason=length` and an empty visible
  answer. They are the `no_answer` outcome class. Every divergent
  `long_modpow` prompt mixes `no_answer` with numeric answers, and at K=8 two
  of the three (T=0) also return different numeric answers across runs; so the
  outcome divergence is partly "answered vs ran out of tokens" and partly
  "different answers".
- **Textual divergence.** The unit is the character (the server returned no
  token ids), and only the visible answer is compared (median 373 characters
  at T=0, 430.5 under the study sampling); reasoning tokens are not recorded.
  At K=8 the visible text differed in 84.8% [72.4%, 97.4%] of pairs at T=0 and
  95.0% [88.5%, 100.0%] under the study sampling: the mean over 23 of the 24
  prompts (one prompt has only censored pairs), with the same family cluster
  bootstrap as the headline. The pooled shares (secondary, in `report.json`)
  are 84.2% and 94.8%.
- **K=1 control.** The visible text differed in 0.0% of pairs on both samplings
  (24 pairs each, 2 censored; exact upper bound 15.4% over 22 prompts). The
  control ran N=2, so this is a small sample, not a rate; it does not say what
  causes the K=8 divergence.

Reading: on long tasks the point estimate is nonzero on both samplings, on
short tasks (G4) it was zero; the intervals are wide (24 prompts), so this is a
directional signal that divergence is workload-length-dependent, not a tight
rate. Both numbers are reported with their intervals; neither is a claim about
another stack or another workload.
