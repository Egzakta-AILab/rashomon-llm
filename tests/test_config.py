# SPDX-License-Identifier: Apache-2.0
"""Regression: the frozen config always carries a temperature key (the G4
production-sampling arms failed on a KeyError when --top-k/--top-p/--min-p were passed)."""
from __future__ import annotations

from rashomon_llm.cli import build_parser
from rashomon_llm.cli_run import _frozen_config


def _args(extra):
    return build_parser().parse_args(
        ["run", "--base-url", "http://x/v1", "--model", "m", "--set", "agent40",
         "--out", "r.json"] + extra)


def test_temperature_present_with_sampling_extras():
    cfg = _frozen_config(_args(["--temperature", "1.0", "--top-k", "20",
                                "--top-p", "0.95", "--min-p", "0.0"]))
    assert cfg["temperature"] == 1.0
    assert cfg["sampling_extra"] == {"top_k": 20, "top_p": 0.95, "min_p": 0.0}


def test_temperature_present_plain_and_production():
    assert _frozen_config(_args([]))["temperature"] == 0.0
    assert _frozen_config(_args(["--sampling", "production"]))["temperature"] is None
    # production + extras keeps extras without touching temperature
    cfg = _frozen_config(_args(["--sampling", "production", "--top-p", "0.9"]))
    assert cfg["temperature"] is None and cfg["sampling_extra"] == {"top_p": 0.9}
