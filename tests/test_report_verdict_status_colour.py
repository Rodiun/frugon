"""Fix B — semantic colour on the verdict status word.

The synthesis line from :func:`_classify_verdict` renders in its state accent
(cyan for confirmed, amber for the cautions), which makes the ONE word that
carries the verdict — "confirmed" / "borderline" / "NOT confirmed" / "not
verified" — fail to stand out.  Fix B colours JUST that word using the SAME
WIN/LOSS/TIE palette as the judge tally table, single-sourced so the terminal,
HTML and Markdown can never drift:

  * confirmed                 → tally WIN  green
  * borderline                → tally TIE  amber
  * NOT confirmed             → tally LOSS red
  * not verified / unmeasured → neutral / dim (no strong colour)

Markdown has no colour, so the status word is bolded instead.  These tests pin
the state → style mapping on each surface (terminal Rich markup, HTML span,
Markdown bold) and prove the rest of the sentence is untouched.
"""

from __future__ import annotations

import io
import re

import pytest
from rich.console import Console

import frugon.report as report
from frugon.measure import MeasureResult, Tier1Tally

# Tallies that resolve to each verdict state (mirrors the classifier thresholds).
_TALLIES: dict[str, Tier1Tally] = {
    "confirmed": Tier1Tally(candidate="gpt-4o-mini", wins=5, losses=0, ties=0),
    "borderline": Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=2, ties=3),
    "not_confirmed": Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=4, ties=1),
    "not_verified": Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=0, ties=0, errors=5),
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Rich 8-colour SGR codes the bold status word emits under force_terminal.
_GREEN_SGR = "\x1b[1;32m"  # bold green  (WIN)
_RED_SGR = "\x1b[1;31m"  # bold red    (LOSS)
_AMBER_SGR = "\x1b[1;33m"  # bold yellow (TIE)


def _result(tally: Tier1Tally) -> MeasureResult:
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4o",
        candidates=[tally.candidate],
        comparisons=[],
        tier1_tallies=[tally],
    )


# ---------------------------------------------------------------------------
# Helper-level mapping — the single source every surface styles from.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "tally_token"),
    [
        ("confirmed", "win"),
        ("borderline", "tie"),
        ("not_confirmed", "loss"),
        ("not_verified", None),
        ("unmeasured", None),
    ],
)
def test_status_tally_token_matches_palette(state: str, tally_token: str | None) -> None:
    assert report._VERDICT_STATUS_TALLY[state] == tally_token


def test_status_phrases_are_the_visible_status_words() -> None:
    assert report._VERDICT_STATUS_PHRASE["confirmed"] == "confirmed"
    assert report._VERDICT_STATUS_PHRASE["borderline"] == "borderline"
    assert report._VERDICT_STATUS_PHRASE["not_confirmed"] == "NOT confirmed"
    assert report._VERDICT_STATUS_PHRASE["not_verified"] == "not verified"
    assert report._VERDICT_STATUS_PHRASE["unmeasured"] == "unmeasured"


# ---------------------------------------------------------------------------
# Terminal (Rich markup) — bold status word in the matching colour.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "colour"),
    [("confirmed", "green"), ("borderline", "yellow"), ("not_confirmed", "red")],
)
def test_terminal_status_markup_colour(state: str, colour: str) -> None:
    phrase = report._VERDICT_STATUS_PHRASE[state]
    assert report._verdict_status_terminal_markup(state) == (
        f"[bold {colour}]{phrase}[/bold {colour}]"
    )


@pytest.mark.parametrize("state", ["not_verified", "unmeasured"])
def test_terminal_status_markup_neutral_is_bold_only(state: str) -> None:
    phrase = report._VERDICT_STATUS_PHRASE[state]
    # Neutral states carry NO semantic colour — only bold, keeping the line's own
    # accent (nothing was measured to win/lose/tie).
    assert report._verdict_status_terminal_markup(state) == f"[bold]{phrase}[/bold]"


def _render_synthesis_ansi(tally: Tier1Tally) -> str:
    """Render the terminal Tier-1 synthesis WITH colour and return the raw ANSI."""
    buf = io.StringIO()
    console = Console(
        file=buf, width=120, force_terminal=True, color_system="standard", highlight=False
    )
    orig_rprint = report.rprint
    orig_console = report._render_console
    report.rprint = lambda *a, **k: console.print(*a, **k)  # type: ignore[attr-defined]
    report._render_console = lambda: console  # type: ignore[attr-defined]
    try:
        report._render_tier1_synthesis(_result(tally))
    finally:
        report.rprint = orig_rprint  # type: ignore[attr-defined]
        report._render_console = orig_console  # type: ignore[attr-defined]
    return buf.getvalue()


@pytest.mark.parametrize(
    ("state", "sgr"),
    [("confirmed", _GREEN_SGR), ("borderline", _AMBER_SGR), ("not_confirmed", _RED_SGR)],
)
def test_terminal_render_colours_status_word(state: str, sgr: str) -> None:
    ansi = _render_synthesis_ansi(_TALLIES[state])
    plain = _ANSI_RE.sub("", ansi)
    phrase = report._VERDICT_STATUS_PHRASE[state]
    # The verdict word is present and carries the matching bold colour SGR.
    assert phrase in plain
    assert sgr in ansi, f"{state}: expected {sgr!r} before the status word"


def test_terminal_render_not_verified_status_has_no_semantic_colour() -> None:
    ansi = _render_synthesis_ansi(_TALLIES["not_verified"])
    # Neutral state: the status word is given NO semantic win/loss colour of its
    # own.  (It is bold and keeps the line's own amber accent — that bold-amber
    # is the inherited LINE colour, not a TIE choice — so only green/red, the
    # colours that would be wrong here, are asserted absent.)
    assert _GREEN_SGR not in ansi
    assert _RED_SGR not in ansi
    # The status markup itself carries no colour token — bold only.
    assert report._verdict_status_terminal_markup("not_verified") == "[bold]not verified[/bold]"


# ---------------------------------------------------------------------------
# HTML — status word wrapped in the matching .verdict-* tally class.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "cls"),
    [
        ("confirmed", "verdict-win"),
        ("borderline", "verdict-tie"),
        ("not_confirmed", "verdict-loss"),
    ],
)
@pytest.mark.parametrize("style", ["v1", "v2"])
def test_html_status_word_carries_tally_class(state: str, cls: str, style: str) -> None:
    html = report._quality_section_html(_result(_TALLIES[state]), style=style)
    phrase = report._VERDICT_STATUS_PHRASE[state]
    assert f'<span class="{cls}">{phrase}</span>' in html


@pytest.mark.parametrize("style", ["v1", "v2"])
def test_html_not_verified_status_word_is_uncoloured(style: str) -> None:
    html = report._quality_section_html(_result(_TALLIES["not_verified"]), style=style)
    # Neutral state: the status word is bare (no .verdict-* span around it).
    assert 'class="verdict-win">not verified' not in html
    assert 'class="verdict-loss">not verified' not in html
    assert 'class="verdict-tie">not verified' not in html
    assert "not verified" in html


# ---------------------------------------------------------------------------
# Markdown — no colour, so the status word is bolded.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["confirmed", "borderline", "not_confirmed", "not_verified"])
def test_markdown_status_word_is_bold(state: str) -> None:
    md = "\n".join(report._quality_section_md(_result(_TALLIES[state])))
    phrase = report._VERDICT_STATUS_PHRASE[state]
    assert f"**{phrase}**" in md
