# SPDX-License-Identifier: Apache-2.0
"""Tests for the repository hygiene scripts: check_secrets and check_public_tree."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


check_secrets = _load("check_secrets")
check_public_tree = _load("check_public_tree")


def _git_repo(tmp_path: Path, files: dict[str, bytes | str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            p.write_text(content, encoding="utf-8")
        else:
            p.write_bytes(content)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return repo


# ------------------------------------------------------------ check_secrets

def test_private_address_is_an_error():
    diff = "+++ b/docs/x.md\n+host at 192.168.1.20 port 80\n"
    assert check_secrets.scan(diff) == ["docs/x.md: private network address"]
    for addr in ("10.0.0.1", "172.16.4.4", "100.64.1.1"):
        assert check_secrets.scan(f"+++ b/a.py\n+{addr}\n")
    assert check_secrets.scan("+++ b/a.py\n+8.8.8.8 and 100.200.1.1\n") == []


def test_tree_mode_scans_every_tracked_file(tmp_path, monkeypatch):
    fake_key = "sk-" + "A" * 30
    repo = _git_repo(tmp_path, {"clean.md": "nothing here\n", "deep/leak.txt": f"x = 1\nk={fake_key}\n",
                                "img.png": b"\x89PNG\r\n\x1a\n10.1.1.1"})
    monkeypatch.chdir(repo)
    found = check_secrets.scan_lines(check_secrets.tracked_lines())
    assert found == ["deep/leak.txt: OpenAI/Anthropic-style key"]
    assert check_secrets.main(["--tree"]) == 1


# ------------------------------------------------------------ check_public_tree

DENY = "# comment\nsecretname\nQ7'\nallow gen/*.py secretname\n"


def test_denylist_parse_and_word_boundaries():
    d = check_public_tree.parse_denylist(DENY)
    assert d.terms == ("secretname", "Q7'")
    assert check_public_tree.scan_text(d, "a.md", "uses SecretName here")
    assert not check_public_tree.scan_text(d, "a.md", "uses secretnames here")
    assert check_public_tree.scan_text(d, "a.md", "the Q7' gate")
    assert not check_public_tree.scan_text(d, "gen/x.py", "secretname is synthetic")
    with pytest.raises(ValueError):
        check_public_tree.parse_denylist("# only comments\n")


def test_public_tree_flags_content_and_file_names(tmp_path):
    repo = _git_repo(tmp_path, {"ok.md": "fine\n", "docs/secretname-card.md": "fine\n",
                                "b.txt": "line\nmentions secretname\n", "gen/x.py": "secretname\n"})
    deny = tmp_path / "deny.txt"
    deny.write_text(DENY, encoding="utf-8")
    hits = check_public_tree.check_tree(str(repo), check_public_tree.parse_denylist(DENY))
    assert "b.txt:2: denied term" in hits
    assert "docs/secretname-card.md: file name contains a denied term" in hits
    assert not any(h.startswith(("ok.md", "gen/")) for h in hits)
    assert check_public_tree.main(["--denylist", str(deny), "--root", str(repo)]) == 1


def test_public_tree_png_metadata_is_scanned(tmp_path):
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return len(body).to_bytes(4, "big") + kind + body + zlib.crc32(kind + body).to_bytes(4, "big")

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IDAT", b"xx secretname xx") + chunk(b"tEXt", b"Author\0secretname")
    text = check_public_tree.png_text(png)
    assert "secretname" in text and "xx" not in text


def test_public_tree_refuses_denylist_inside_repo(tmp_path):
    repo = _git_repo(tmp_path, {"a.md": "x\n"})
    inside = repo / "deny.txt"
    inside.write_text(DENY, encoding="utf-8")
    assert check_public_tree.main(["--denylist", str(inside), "--root", str(repo)]) == 2
