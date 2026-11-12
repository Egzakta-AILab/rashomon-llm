# SPDX-License-Identifier: Apache-2.0
"""The shareable PNG card (docs/protocol.md).

Deliberately neutral: grayscale, one font, no green/red, no verdicts. The card
carries the numbers with their intervals, the protocol identity, the achieved
concurrency, and the run file's sha256 — never the base URL, headers, or raw
prompts. Every text row fits inside the card with a 48 px margin on both sides
(`MAX_ROW_W`); a row that would not fit is drawn smaller, then shortened.
Pillow is a core dependency; if it cannot be imported, `render_card` raises
SystemExit and `rashomon report` skips the card.
"""
from __future__ import annotations

import time
from typing import Any

from .report import display_model, textual_summary, unit_label

CARD_W, CARD_H = 1200, 675
MARGIN = 48
MAX_ROW_W = CARD_W - 2 * MARGIN
_BG = (250, 250, 250)
_INK = (24, 24, 24)
_DIM = (90, 90, 90)
_LINE = (200, 200, 200)
_MIN_FONT = 14
_FONT_NAMES = ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "LiberationMono-Regular.ttf")


ROW_STEP = 36


def run_date(rep: dict) -> str:
    """The date the run finished (UTC), not the date the card was drawn."""
    ts = rep.get("run_finished_at") or rep["generated_at"]
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


def _fmt_pct(x: Any) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _font(size: int):
    from PIL import ImageFont

    for name in _FONT_NAMES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def fit_text(draw, text: str, size: int, max_w: int = MAX_ROW_W) -> tuple[str, Any]:
    """Shrink the font down to _MIN_FONT, then shorten with an ellipsis, until it fits."""
    for s in range(size, _MIN_FONT - 1, -1):
        font = _font(s)
        if draw.textlength(text, font=font) <= max_w:
            return text, font
    font = _font(_MIN_FONT)
    cut = text
    while cut and draw.textlength(cut + "…", font=font) > max_w:
        cut = cut[:-1]
    return cut + "…", font


def card_rows(rep: dict) -> list[str]:
    """The body rows of the card, in order (pure; no drawing)."""
    cfg = rep["config"]
    h = rep["headline_pair_rate"]
    if h.get("suppressed"):
        headline = f"pair rate: not printed ({h['reason']})"
    else:
        headline = f"pair-disagreement rate: {h['estimate']:.1%} [{h['ci_low']:.1%}, {h['ci_high']:.1%}]"
    rows = [
        f"model: {display_model(rep)}  (requested temperature: {cfg.get('temperature')})",
        (f"set: {rep['set']['name']} x{rep['set']['n_items']}   mode: {cfg.get('mode')}   "
         f"N={cfg.get('n')}  K={cfg.get('k')}"),
        headline,
    ]
    c = rep.get("companion")
    if c:
        rows.append(f"prompts with disagreement among {cfg.get('n')} runs: "
                    f"{c['estimate']:.1%} [{c['ci_low']:.1%}, {c['ci_high']:.1%}]")
    if h.get("degenerate"):
        eb = h["exact_bound"]
        rows.append(f"exact bound, prompt-level (Clopper-Pearson): upper {_fmt_pct(eb['ci_high'])} — "
                    f"not proof of determinism")
    x = rep.get("cross_arm_k1xk8")
    if x:
        rows.append(f"cross-arm K1xK8 disagreement: {x['rate']:.1%} of {x['pairs']} pairs")
    t8 = (rep.get("textual") or {}).get("k8")
    if t8 and t8.get("mean_over_prompts"):
        ul = unit_label(t8.get("unit") or "char")
        rows.append(f"visible text differing ({ul}): {textual_summary(t8)}")
        pts = ", ".join(f"{ul} <= {p['le']}: {_fmt_pct(p['share'])}"
                        for p in t8["curve"]["points"] if p["share"] is not None)
        if pts:
            rows.append(f"curve (descriptive): {pts}")
    ach = (rep.get("hygiene") or {}).get("achieved_concurrency") or {}
    if ach.get("max") is not None:
        rows.append(f"achieved in-flight concurrency: max {ach['max']}, mean {ach['mean']:.1f}")
    rows.append(f"requests: {rep['counts']['requests']} "
                f"({rep['counts']['unscorable']} unscorable, "
                f"{rep['counts']['unscorable_share']:.1%})")
    return rows


def render_card(rep: dict, path: str) -> list[tuple[str, Any]]:
    """Draw the card to `path`. Returns every drawn (text, font) for inspection."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise SystemExit("the card needs Pillow (a core dependency): pip install Pillow") from e

    img = Image.new("RGB", (CARD_W, CARD_H), _BG)
    d = ImageDraw.Draw(img)
    drawn: list[tuple[str, Any]] = []

    def put(y: int, text: str, size: int, fill) -> None:
        fitted, font = fit_text(d, text, size)
        d.text((MARGIN, y), fitted, fill=fill, font=font)
        drawn.append((fitted, font))

    tool = rep.get("tool") or {}
    y = 36
    put(y, "rashomon-llm", 54, _INK)
    y += 66
    put(y, "same request, concurrent load — outcome divergence with intervals", 20, _DIM)
    y += 34
    d.line((MARGIN, y, CARD_W - MARGIN, y), fill=_LINE, width=2)
    y += 22

    for row in card_rows(rep):
        put(y, row, 26, _INK)
        y += ROW_STEP

    y = CARD_H - 88
    d.line((MARGIN, y - 12, CARD_W - MARGIN, y - 12), fill=_LINE, width=1)
    put(y, f"measured {run_date(rep)} · {tool.get('name', 'rashomon-llm')} "
           f"v{tool.get('version', '?')}", 20, _DIM)
    put(y + 28, f"run sha256: {(rep.get('run_sha256') or '?')[:32]}…", 20, _DIM)

    img.save(path)
    return drawn
