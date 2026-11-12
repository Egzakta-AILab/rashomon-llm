# G4 protocol — pre-registered before any run

> Edited for publication: names of internal systems and people were replaced
> with neutral wording. The criteria and the run configuration are unchanged.

Written 2026-10-29, before any G4 run. G4 gates exactly one thing: whether the
README may carry a sentence linking the tool to the archived study. G4 never
gates the existence of the tool. The tool is allowed to measure zero.

## Amendment 2026-10-29 (written before any G4 run)

1. The K=1 control runs at N=4 for `math40` (wall-clock; control requests are
   long-form and sequential). The control for `agent40` keeps N=8. All other
   parameters unchanged.
2. Stack (a) is chosen: the Spark TP2 vLLM stack is restarted for the replay,
   then stopped again; the pre-G4 machine state is restored.
3. A third-party container (~10.8 GB GPU on each Spark)
   runs on both machines and is left untouched. The G4 run proceeds only if the
   TP2 stack starts and answers cleanly alongside it; otherwise the run stops
   and the conflict is reported instead of forcing anything off the GPU.
4. Sampling arm B uses the archived study's request seed (1). The scheduling
   permutation seed is the tool's own and is recorded per run.

## Stack choice (open when written; decided in amendment 2)

The archived study ran on vLLM TP2 across two DGX Spark (GB10) machines. That
stack is currently down; the production endpoint for the same model family now
runs on an H100. The two candidate G4 stacks:

- **(a) Spark TP2 replay** — a true reproduction of the archived environment.
- **(b) H100 production run** — a new measurement on the stack that is actually
  deployed; not a reproduction of the archived environment, and it will be
  described as such.

Whichever is chosen, this document and the criteria below are frozen first.

## Fixed run configuration

- Set: `agent40` and `math40` (current manifests).
- Mode: sweep (the estimand). Burst only as a labelled diagnostic, never mixed.
- N=8, K=8, `--control` (K=1 arm, counterbalanced order recorded).
- Sampling arm A: requested temperature 0 (the tool's public default).
- Sampling arm B: the archived study's sampling (temperature 1.0, top_k 20,
  top_p 0.95, min_p 0.0, seed 1). Reported separately, never mixed with A.
- `--engine` / `--hardware` recorded as user-reported; server metadata probed
  with the read-only allowlist.

## Investigation criteria (what counts as a broken client)

Any of these invalidates the run and triggers investigation, not a claim:

1. >5% of measured requests unscorable.
2. Achieved in-flight concurrency below 0.75·K on the K=8 arm.
3. A `system_fingerprint` change mid-run (requests may have hit different
   backends).
4. Tool-call extraction fails on >5% of agent40 records.
5. The K=1 control shows materially *more* disagreement than the K=8 arm
   (would point at a client-side artifact).

## What G4 may and may not conclude

- The measured pair rates, with their intervals, are publishable as
  measurements on the named stack, whatever they are — including zero (with the
  exact upper bound).
- A README sentence about the archived study is allowed only after a completed
  G4 run on stack (a) whose results are not invalidated by the criteria above,
  and it must name the stack, the sampling, and the interval in the same
  sentence.
- Overlap of a wide interval with the archived interval is not a proof of
  equivalence and will not be presented as one.
- A run on stack (b) is published as "measured on the H100 production stack",
  never as a reproduction of the archived study.
