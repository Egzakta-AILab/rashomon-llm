# SPDX-License-Identifier: Apache-2.0
"""Prompt-set loading (docs/protocol.md).

A set is a JSONL file, one object per line:
  {"id": "...", "family": "<template family>", "messages": [...], "tools": [...]?}

plus an optional sibling `<name>.manifest.yaml` (generator, seed, date, per-item
sha256, license, provenance statements). Families are the bootstrap cluster unit;
a missing family defaults to the item id (one item = one cluster).

Built-in sets (agent40, math40) are generated with recorded provenance in P2;
the loader already resolves them by name from `data/sets/`.
"""
from __future__ import annotations

import hashlib
import json
import os

from .model import PromptItem

BUILTIN_DIR = os.path.join(os.path.dirname(__file__), "data", "sets")
BUILTINS = ("agent40", "math40")


class SetError(ValueError):
    pass


def load_set(name_or_path: str) -> tuple[list[PromptItem], dict]:
    path = name_or_path
    if name_or_path in BUILTINS:
        path = os.path.join(BUILTIN_DIR, f"{name_or_path}.jsonl")
    if not os.path.isfile(path):
        if name_or_path in BUILTINS:
            raise SetError(
                f"built-in set {name_or_path!r} is not generated yet (P2); pass a JSONL path instead")
        raise SetError(f"set file not found: {name_or_path}")

    items: list[PromptItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise SetError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not obj.get("id") or not obj.get("messages"):
                raise SetError(f"{path}:{lineno}: every item needs 'id' and 'messages'")
            items.append(PromptItem(
                id=str(obj["id"]),
                family=str(obj.get("family") or obj["id"]),
                messages=obj["messages"],
                tools=obj.get("tools"),
            ))
    if not items:
        raise SetError(f"{path}: empty set")
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise SetError(f"{path}: duplicate item ids")

    manifest = _load_manifest(path)
    info = {"name": os.path.basename(path), "n_items": len(items),
            "items_sha256": _items_digest(items), "manifest": manifest,
            "manifest_verified": _verify_manifest(items, manifest, path)}
    return items, info


def _item_sha(item: PromptItem) -> str:
    """The same canonical body the generators hash (id + messages + tools)."""
    body = json.dumps({"id": it_id(item), "messages": item.messages,
                       "tools": item.tools}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def it_id(item: PromptItem) -> str:
    return str(item.id)


def _verify_manifest(items: list[PromptItem], manifest: dict | None, path: str) -> bool:
    """A set whose manifest exists but does not match its items is refused:
    a silently edited set would silently change the measurement."""
    if manifest is None:
        return False
    recorded = manifest.get("items_sha256")
    if not isinstance(recorded, dict) or not recorded:
        return False
    for item in items:
        want = recorded.get(item.id)
        if want is None or want != _item_sha(item):
            raise SetError(
                f"{path}: item {item.id!r} does not match the manifest sha256 — "
                f"the set file or the manifest was modified after generation")
    if set(recorded) - {it.id for it in items}:
        raise SetError(f"{path}: manifest lists items missing from the set file")
    return True


def _load_manifest(path: str) -> dict | None:
    base, _ = os.path.splitext(path)
    for cand in (path + ".manifest.json", base + ".manifest.json"):
        if os.path.isfile(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f)
    return None


def _items_digest(items: list[PromptItem]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it.id.encode())
        h.update(b"\x00")
        h.update(json.dumps(it.messages, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\x00")
    return h.hexdigest()
