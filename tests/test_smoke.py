# SPDX-License-Identifier: Apache-2.0
"""P0 smoke tests: the package imports and the CLI surface parses."""
from __future__ import annotations

from rashomon_llm import __version__
from rashomon_llm.cli import build_parser, main


def test_version_is_set():
    assert __version__


def test_cli_run_parses_minimal_args():
    args = build_parser().parse_args(
        ["run", "--base-url", "http://localhost:8000/v1", "--model", "m", "--set", "agent40", "--out", "run.json"]
    )
    assert args.command == "run"
    assert args.mode == "sweep"  # sweep is the default estimand; burst is a labelled diagnostic
    assert args.n == 8 and args.k == 8
    assert args.temperature == 0.0


def test_cli_run_missing_set_exit_code(tmp_path):
    out = tmp_path / "r.json"
    rc = main(["run", "--base-url", "http://localhost:1/v1", "--model", "m",
               "--set", str(tmp_path / "no_such_set.jsonl"), "--out", str(out)])
    assert rc == 2  # a missing set file is a usage error


def test_cli_report_and_verify_parse():
    assert build_parser().parse_args(["report", "run.json"]).command == "report"
    assert build_parser().parse_args(["verify", "run.json"]).command == "verify"
