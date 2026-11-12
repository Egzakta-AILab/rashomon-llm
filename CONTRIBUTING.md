# Contributing

Thanks for helping. A few hard rules keep this tool trustworthy.

## Ground rules

- **No benchmark data and no client data.** Prompt sets are synthetic, generated
  by scripts in `data/generators/` with a recorded seed, and carry a manifest.
  Never add items copied from a benchmark or a customer workload.
- **No secrets, no private addresses.** `python scripts/check_secrets.py --tree`
  runs in CI on every push and scans every tracked file; private network
  addresses are errors. API keys come from environment variables only.
- **Every number gets an interval.** If you touch metrics, keep the interval
  semantics intact: cluster bootstrap for the pair rate (seed 0, 10,000
  resamples; the code in `src/rashomon_llm/stats.py` is kept unchanged from its
  source — do not restyle it),
  Wilson for the companion, and an exact bound whenever the bootstrap is
  degenerate.
- **The tool measures; it does not fix.** No "determinism repair" features.
- **No quality rankings** of models or engines.

## Development

```bash
python -m pip install -e .[dev]
python -m pytest -q
ruff check src tests data scripts
```

Python 3.11+. CI scans for secrets, runs tests on Linux and Windows
(3.11–3.13), lints `src tests data scripts`, and builds the wheel.

Security issues: see [SECURITY.md](SECURITY.md), not the public issue tracker.

## Adding a prompt set

1. Write a generator in `data/generators/` with a fixed seed.
2. Ship the manifest: generator name/version, seed, date, per-item sha256,
   license, provenance statements ("not derived from any benchmark", LLM
   involvement), and family ids (the bootstrap cluster unit).
3. Explain the outcome function for the set.

## What a good PR looks like

- One concern per PR, with tests.
- For metric changes: a synthetic fixture with a known answer.
- For protocol changes: update docs/protocol.md in the same PR.
