# SPDX-License-Identifier: Apache-2.0
"""Set integrity: manifests are verified on load; tampering is refused."""
from __future__ import annotations

import json
import os

import pytest

from rashomon_llm.sets import BUILTIN_DIR, SetError, load_set


def test_builtin_sets_verify_against_their_manifests():
    for name in ("agent40", "math40"):
        items, info = load_set(name)
        assert info["manifest_verified"] is True
        assert info["n_items"] == 40
        assert len({it.family for it in items}) == 5  # five template families each


def test_tampered_set_is_refused(tmp_path):
    items, _ = load_set("math40")
    src = tmp_path / "math40.jsonl"
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        for i, it in enumerate(items):
            obj = {"id": it.id, "family": it.family, "messages": it.messages, "tools": it.tools}
            if i == 0:
                obj["messages"] = [{"role": "user", "content": "tampered"}]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(os.path.join(BUILTIN_DIR, "math40.manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    (tmp_path / "math40.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SetError, match="does not match the manifest"):
        load_set(str(src))


def test_set_without_manifest_loads_unverified(tmp_path):
    p = tmp_path / "custom.jsonl"
    p.write_text(json.dumps({"id": "x", "messages": [{"role": "user", "content": "hi"}]}) + "\n",
                 encoding="utf-8")
    _items, info = load_set(str(p))
    assert info["manifest"] is None
    assert info["manifest_verified"] is False
