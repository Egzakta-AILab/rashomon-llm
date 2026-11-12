# Changelog

## 0.1.0 — 2026-11-12

First public release.

- `rashomon run / report / verify`: sweep measurement (N shuffled rounds, K in
  flight, never two copies of one prompt in flight), labelled burst diagnostic,
  lower-concurrency control arm with counterbalanced order, cross-arm K1×K8
  comparison.
- Metrics: pair-disagreement rate (cluster bootstrap interval), companion
  "prompts with disagreement among N runs" (Wilson), censored divergence curve,
  per-prompt matrix, unscorable accounting (headline suppressed above 5%).
- Outcome functions: first tool call (agent), final number (math), user grader
  script; textual divergence when no outcome function applies.
- Built-in synthetic sets `agent40` and `math40` with manifests (verified on
  load).
- Neutral PNG card with run checksum (Pillow is a core dependency; the card is
  skipped cleanly if it cannot be imported); `--model-label` display label;
  `--compare-endpoint` side-by-side.
- Degenerate intervals carry exact bounds at prompt and family level.
- Textual divergence is a mean over prompts with the headline's cluster bootstrap
  interval; the pooled share and the curve are descriptive. The card carries the
  date the run finished.
- `rashomon report` never executes a grader named inside a run file; pass
  `--grader` explicitly, and it must match the recorded sha256.
- Run-file privacy: the endpoint is stored as scheme + truncated host hash
  (`--record-base-url` to opt in); probed server metadata is reduced to an
  allowlist.
- Validation: gates G1–G4 and the long-task probe are recorded in
  `docs/validation.md`.
