# SPDX-License-Identifier: Apache-2.0
"""Release-prep tests: unit labels, card width, family-level bound, display label,
run-file privacy, version consistency, Pillow as a core dependency."""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

import rashomon_llm
from rashomon_llm import cli_report
from rashomon_llm.card import CARD_W, card_rows, render_card
from rashomon_llm.cli import main
from rashomon_llm.intervals import cluster_rate_interval
from rashomon_llm.mock import serve
from rashomon_llm.model import load_run, save_run, verify_run
from rashomon_llm.privacy import describe_base_url, public_server_meta, sanitize_fields
from rashomon_llm.report import compute_report, to_markdown

ROOT = Path(__file__).resolve().parents[1]
PIL = pytest.importorskip("PIL")


def _run(n_prompts: int = 40, n_families: int = 5, unit_tokens: bool = False,
         model: str = "served-model-name", server: dict | None = None) -> dict:
    requests = []
    for p in range(n_prompts):
        for s in range(3):
            requests.append({
                "id": f"k8:p{p}:s{s}", "prompt_id": f"p{p}", "family": f"fam{p % n_families}",
                "arm": "k8", "mode": "sweep", "sweep": s, "status": "ok",
                "inflight_at_dispatch": 7, "finish_reason": "stop",
                "text": "Answer: 42" + ("" if s == 0 else " (same)"),
                "token_ids": [1, 2, 3, s] if unit_tokens else None,
            })
    return {
        "schema": "rashomon-run/1", "tool": {"name": "rashomon-llm", "version": "0.1.0"},
        "config": {"model": model, "temperature": 0.0, "sampling": "fixed", "mode": "sweep",
                   "max_tokens": 64, "n": 3, "k": 8, "quick": False},
        "set": {"name": "math40.jsonl", "n_items": n_prompts},
        "server": server if server is not None else {"probed": False, "fields": {}},
        "requests": requests, "scheduling": {}, "run_sha256": "0" * 64,
    }


# ------------------------------------------------------------ C1: unit labels

@pytest.mark.parametrize("tokens,label", [(False, "chars <= 64"), (True, "tokens <= 64")])
def test_threshold_label_follows_unit(tokens, label):
    rep = compute_report(_run(unit_tokens=tokens))
    md = to_markdown(rep)
    rows = "\n".join(card_rows(rep))
    assert label in md and label in rows
    assert "t<=" not in md and "t<=" not in rows


# ------------------------------------------------------------ C2: card width

def test_every_card_row_fits(tmp_path):
    from PIL import Image, ImageDraw

    long_label = "an extremely long display label for a model " * 4
    rep = compute_report(_run())
    rep["display"] = {"model_label": long_label}
    drawn = render_card(rep, str(tmp_path / "c.png"))
    measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    widths = [measure.textlength(text, font=font) for text, font in drawn]
    assert widths and max(widths) <= CARD_W - 96
    assert any(text.startswith("model: an extremely long") for text, _ in drawn)


# ------------------------------------------------------------ C5: family-level bound

def test_family_level_bound_next_to_prompt_level():
    values = {f"p{i}": [0.0] for i in range(40)}
    clusters = {f"p{i}": f"fam{i % 5}" for i in range(40)}
    res = cluster_rate_interval(values, clusters=clusters)
    assert res["exact_bound"]["level"] == "prompt"
    assert res["exact_bound"]["ci_high"] == pytest.approx(0.0881, abs=5e-4)
    assert res["exact_bound_family"]["level"] == "family"
    assert res["exact_bound_family"]["n"] == 5
    assert res["exact_bound_family"]["ci_high"] == pytest.approx(0.5218, abs=5e-4)


def test_report_shows_both_bounds_card_shows_prompt_level():
    rep = compute_report(_run())
    md = to_markdown(rep)
    assert "prompt-level (0/40 prompts)" in md and "[0.0%, 8.8%]" in md
    assert "family-level bound (0/5 families): [0.0%, 52.2%]" in md
    assert "assumes independent prompts" in md
    rows = card_rows(rep)
    assert any("prompt-level" in r and "8.8%" in r for r in rows)
    assert not any("52.2%" in r for r in rows)


# ------------------------------------------------------------ A6: display label

def _report(tmp_path, monkeypatch, label=None) -> tuple[dict, list[str]]:
    run_path = tmp_path / "run.json"
    if not run_path.exists():
        save_run(_run(), str(run_path))
    monkeypatch.setattr("rashomon_llm.report.time.time", lambda: 1_800_000_000.0)
    seen: list[list[str]] = []
    real = cli_report.render_card
    monkeypatch.setattr(cli_report, "render_card",
                        lambda rep, path: (seen.append(card_rows(rep)), real(rep, path))[1])
    argv = ["report", str(run_path)] + (["--model-label", label] if label else [])
    assert main(argv) == 0
    rep = json.loads((tmp_path / "run.report.json").read_text(encoding="utf-8"))
    md = (tmp_path / "run.report.md").read_text(encoding="utf-8")
    return rep, seen[0] + [md]


def test_model_label_is_display_only(tmp_path, monkeypatch):
    plain, _ = _report(tmp_path, monkeypatch)
    labelled, shown = _report(tmp_path, monkeypatch, label="in-house model")
    assert labelled["report_sha256"] == plain["report_sha256"]
    assert labelled["display"] == {"model_label": "in-house model"}
    assert "display" not in plain
    assert any("in-house model" in s for s in shown)
    assert not any("served-model-name" in s for s in shown)
    assert main(["verify", str(tmp_path / "run.json")]) == 0


# ------------------------------------------------------------ C4: run-file privacy

def test_base_url_is_hashed_unless_opted_in():
    d = describe_base_url("http://gpu-box.example:8000/v1")
    assert d["scheme"] == "http" and len(d["host_sha256"]) == 16 and "base_url" not in d
    assert "gpu-box.example" not in json.dumps(d)
    assert describe_base_url("http://gpu-box.example:8000/v1", record_full=True)["base_url"] \
        == "http://gpu-box.example:8000/v1"
    assert describe_base_url("http://a:1/v1") != describe_base_url("http://b:1/v1")


def test_server_fields_are_reduced_to_the_allowlist():
    fields = {
        "models": {"value": {"object": "list", "data": [{
            "id": "m", "object": "model", "owned_by": "vllm", "root": "/models/secret-path",
            "max_model_len": 4096, "permission": [{"id": "x"}]}]}, "provenance": "server-reported"},
        "props": {"value": {"model_path": "/disk/model.gguf", "system_prompt": "you are",
                            "chat_template": "{{ x }}", "total_slots": 4}, "provenance": "server-reported"},
        "server_info": {"value": {"model_path": "/disk", "version": "1.2"}, "provenance": "server-reported"},
        "unknown": {"value": {"anything": 1}, "provenance": "server-reported"},
    }
    out = sanitize_fields(fields)
    blob = json.dumps(out)
    for leak in ("secret-path", "/disk", "you are", "chat_template", "permission", "anything"):
        assert leak not in blob
    assert out["models"]["value"]["data"][0] == {"id": "m", "object": "model", "owned_by": "vllm",
                                                 "max_model_len": 4096}
    assert out["props"]["value"] == {"total_slots": 4}
    assert out["server_info"]["value"] == {"version": "1.2"}


def test_old_run_file_still_verifies_and_reports_without_url(tmp_path):
    old_server = {"base_url": "http://old-host.example:8894/v1", "probed": True, "fields": {
        "models": {"value": {"data": [{"id": "m", "root": "/model"}]}, "provenance": "server-reported"}}}
    path = tmp_path / "old.json"
    save_run(_run(server=old_server), str(path))
    assert verify_run(str(path))
    rep = compute_report(load_run(str(path)))
    blob = json.dumps(rep["server"])
    assert "old-host.example" not in blob and "/model" not in blob
    assert rep["server"]["endpoint"]["scheme"] == "http"


def test_live_run_records_no_base_url_by_default(tmp_path):
    server, url = serve("stable", chunk_delay_s=0.002)
    try:
        for flag, expect_url in (([], False), (["--record-base-url"], True)):
            out = tmp_path / f"run{len(flag)}.json"
            assert main(["run", "--base-url", url, "--model", "mock-model", "--set", "math40",
                         "--n", "2", "--k", "4", "--out", str(out)] + flag) == 0
            run = load_run(str(out))
            text = out.read_text(encoding="utf-8")
            assert ("127.0.0.1" in text) is expect_url
            assert run["server"]["endpoint"]["scheme"] == "http"
            assert run["server"]["fields"]["models"]["value"]["data"][0]["id"] == "mock-model"
            assert verify_run(str(out))
    finally:
        server.shutdown()
    assert public_server_meta(None) is None


# ------------------------------------------------------------ C8 / C3: packaging

def _pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


def test_version_matches_pyproject():
    assert _pyproject()["version"] == rashomon_llm.__version__ == "0.1.0"


def test_pillow_is_a_core_dependency():
    assert any(d.lower().startswith("pillow") for d in _pyproject()["dependencies"])


def test_report_skips_card_cleanly_without_pillow(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.json"
    save_run(_run(), str(path))
    monkeypatch.setitem(sys.modules, "PIL", None)  # makes `import PIL` raise ImportError
    assert main(["report", str(path)]) == 0
    assert "card: skipped" in capsys.readouterr().out
    assert (tmp_path / "run.report.md").is_file()
    assert not (tmp_path / "run.card.png").exists()


# ------------------------------------------------------------ textual mean + run date

def test_textual_share_is_a_prompt_mean_with_interval(tmp_path):
    run = _run(n_prompts=10, n_families=5)
    for r in run["requests"]:
        if r["prompt_id"] == "p0":  # only censored pairs: identical and capped
            r["text"], r["finish_reason"] = "same", "length"
    rep = compute_report(run)
    t = rep["textual"]["k8"]
    tm = t["mean_over_prompts"]
    # each usable prompt: runs "x", "x (same)", "x (same)" -> 2 of 3 pairs differ
    assert tm["estimate"] == pytest.approx(2 / 3)
    assert tm["n_items"] == 9 and tm["n_prompts_total"] == 10
    assert tm["ci_low"] <= tm["estimate"] <= tm["ci_high"]
    assert t["pair_share"] == pytest.approx(18 / 27)  # pooled, secondary
    assert t["curve"]["descriptive"] is True
    summary = "66.7% [66.7%, 66.7%], mean over 9 of 10 prompts"
    assert summary in to_markdown(rep)
    assert any(summary in r for r in card_rows(rep))
    assert "descriptive" in to_markdown(rep)


def test_card_carries_the_run_date_not_the_render_date(tmp_path):
    run = _run()
    run["finished_at"] = 1793755965.0  # 2026-11-04 UTC
    run["tool"]["version"] = "0.1.0a0"
    rep = compute_report(run)
    drawn = render_card(rep, str(tmp_path / "c.png"))
    assert any(text == "measured 2026-11-04 · rashomon-llm v0.1.0a0" for text, _ in drawn)


# ------------------------------------------------------------ grader named in a run file

def test_report_never_executes_a_grader_named_in_the_run_file(tmp_path):
    from rashomon_llm.model import sha256_text

    marker = tmp_path / "pwned.txt"
    evil = tmp_path / "grader.py"
    evil.write_text(f"open({str(marker)!r}, 'w').write('x')\nprint('v')\n", encoding="utf-8")
    digest = sha256_text(evil.read_text(encoding="utf-8"))
    run = _run(n_prompts=4)
    run["config"]["grader"] = {"path": str(evil), "sha256": digest}
    path = tmp_path / "shared.json"
    save_run(run, str(path))
    with pytest.raises(SystemExit, match="never executed"):
        main(["report", str(path)])
    assert not marker.exists()

    other = tmp_path / "mine.py"
    other.write_text("print('v')\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match"):
        main(["report", str(path), "--grader", str(other)])
    assert not marker.exists()
