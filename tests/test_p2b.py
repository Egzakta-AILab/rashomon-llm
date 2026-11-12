# SPDX-License-Identifier: Apache-2.0
"""P2b tests: --compare-endpoint and the per-prompt matrix in Markdown."""
from __future__ import annotations

from rashomon_llm.cli import main
from rashomon_llm.mock import serve
from rashomon_llm.model import load_run


def test_compare_endpoint_runs_second_and_report_composes(tmp_path):
    srv1, url1 = serve("stable", chunk_delay_s=0.003)
    srv2, url2 = serve("cotenant", chunk_delay_s=0.003)
    out = tmp_path / "run.json"
    try:
        rc = main(["run", "--base-url", url1, "--model", "mock-model", "--set", "math40",
                   "--n", "2", "--k", "4", "--no-probe", "--out", str(out),
                   "--compare-endpoint", url2])
        assert rc == 0
        compare_file = tmp_path / "run.compare.json"
        assert compare_file.is_file()
        run2 = load_run(str(compare_file))
        assert len(run2["requests"]) == 80

        rc = main(["report", str(out), "--compare", str(compare_file)])
        assert rc == 0
        md = (tmp_path / "run.report.md").read_text(encoding="utf-8")
        assert "side-by-side comparison" in md
        assert "| A |" in md and "| B |" in md
        assert "not a quality ranking" in md
        # math40 has an outcome function: the per-prompt matrix must be present
        assert "per-prompt matrix" in md
        assert "| prompt | family |" in md
    finally:
        srv1.shutdown()
        srv2.shutdown()


def test_cotenant_side_shows_divergence_in_compare(tmp_path):
    """The labelled comparison must actually show the contrast: stable A vs
    co-tenancy-sensitive B."""
    srv1, url1 = serve("stable", chunk_delay_s=0.003)
    srv2, url2 = serve("cotenant", chunk_delay_s=0.003)
    out = tmp_path / "run.json"
    try:
        assert main(["run", "--base-url", url1, "--model", "mock-model", "--set", "agent40",
                     "--n", "3", "--k", "6", "--no-probe", "--out", str(out),
                     "--compare-endpoint", url2]) == 0
        assert main(["report", str(out), "--compare",
                     str(tmp_path / "run.compare.json")]) == 0
        import json
        rep = json.loads((tmp_path / "run.report.json").read_text(encoding="utf-8"))
        a = rep["headline_pair_rate"]
        b = rep["comparison"]["headline_pair_rate"]
        assert not a["suppressed"] and not b["suppressed"]
        assert a["estimate"] == 0.0          # stable endpoint: no divergence
        assert b["estimate"] > 0.0           # co-tenancy endpoint: divergence
        assert a.get("degenerate") is True   # with the exact bound, never bare zero
    finally:
        srv1.shutdown()
        srv2.shutdown()
