"""Tests for the escalation ladder + small-sample nudge across all three surfaces.

When ``--judge`` returns a NOT-confirmed verdict, the synthesis escalates to the
next rung up the quality ladder (when one exists) on the terminal, the Markdown
report AND the HTML report — verbatim model name, tier label, ~NN% figure and
command on every surface.  When NO cheaper higher-tier model exists it keeps the
honest "keep these on the baseline" dead-end.  A statistically thin tally also
gets a dim "Small sample — re-run with --samples 25" nudge; a decisive one does
not.

The escalation pick (``frugon.cost.next_rung_up``) is exercised on its own in
test_escalation_ladder.py.  Here it is monkeypatched in the report module so the
RENDERING is asserted independently of the shipped pricing/quality snapshot.

All tests run fully offline — no provider call is made.
"""

from __future__ import annotations

import re
import sys

import pytest
from rich.console import Console

from frugon import report as _report
from frugon.cost import EscalationSuggestion
from frugon.measure import MeasureResult, Tier1Tally
from frugon.report import (
    _quality_section_html,
    _quality_section_md,
    render_quality_terminal,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")

_SUGGESTION = EscalationSuggestion(
    model="gpt-4o",
    tier=1,
    tier_label="Strong",
    pct_cheaper_than_baseline=42,
    command="frugon analyze --measure --candidates gpt-4o",
)


@pytest.fixture
def stub_rung(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force next_rung_up to return a fixed suggestion (deterministic rendering)."""
    monkeypatch.setattr(_report, "next_rung_up", lambda failed, baseline: _SUGGESTION)


@pytest.fixture
def stub_no_rung(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force next_rung_up to return None (the honest dead-end path)."""
    monkeypatch.setattr(_report, "next_rung_up", lambda failed, baseline: None)


def _capture(measure_result: MeasureResult) -> str:
    console = Console(width=200, force_terminal=False, no_color=True)
    with console.capture() as cap:
        rep = sys.modules[render_quality_terminal.__module__]
        original = rep.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            console.print(*args, **kwargs)

        rep.rprint = _rp  # type: ignore[assignment]
        try:
            rep.render_quality_terminal(measure_result)
        finally:
            rep.rprint = original  # type: ignore[assignment]
    return _ANSI_RE.sub("", cap.get())


def _not_confirmed_result(*, losses: int = 3, ties: int = 1, wins: int = 1,
                          samples: int = 5) -> MeasureResult:
    return MeasureResult(
        samples_requested=samples,
        samples_taken=samples,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[
            Tier1Tally(candidate="gpt-4o-mini", wins=wins, losses=losses, ties=ties)
        ],
    )


# ---------------------------------------------------------------------------
# Escalation — terminal
# ---------------------------------------------------------------------------


def test_terminal_not_confirmed_escalates_to_next_rung(stub_rung: None) -> None:
    out = _capture(_not_confirmed_result())
    assert "NOT confirmed" in out, out
    assert "try the next rung up" in out, out
    assert "gpt-4o" in out, out
    assert "Strong tier" in out, out
    assert "~42% cheaper than gpt-4-turbo" in out, out
    assert "frugon analyze --measure --candidates gpt-4o" in out, out
    # The dead-end guidance is replaced, not shown alongside the escalation.
    assert "keeping these calls on" not in out, out


def test_terminal_no_rung_keeps_dead_end_guidance(stub_no_rung: None) -> None:
    out = _capture(_not_confirmed_result())
    assert "NOT confirmed" in out, out
    assert "keeping these calls on gpt-4-turbo" in out, out
    assert "try the next rung up" not in out, out


# ---------------------------------------------------------------------------
# Escalation — Markdown
# ---------------------------------------------------------------------------


def test_md_not_confirmed_escalates_to_next_rung(stub_rung: None) -> None:
    md = "\n".join(_quality_section_md(_not_confirmed_result()))
    assert "try the next rung up" in md, md
    assert "gpt-4o (Strong tier, still ~42% cheaper than gpt-4-turbo)" in md, md
    assert "frugon analyze --measure --candidates gpt-4o" in md, md
    assert "keeping these calls on" not in md, md


def test_md_no_rung_keeps_dead_end_guidance(stub_no_rung: None) -> None:
    md = "\n".join(_quality_section_md(_not_confirmed_result()))
    assert "keeping these calls on gpt-4-turbo" in md, md
    assert "try the next rung up" not in md, md


# ---------------------------------------------------------------------------
# Escalation — HTML
# ---------------------------------------------------------------------------


def test_html_not_confirmed_escalates_to_next_rung(stub_rung: None) -> None:
    html = _quality_section_html(_not_confirmed_result(), style="v1")
    assert "try the next rung up" in html, html
    assert "gpt-4o" in html, html
    assert "Strong tier" in html, html
    assert "~42% cheaper than gpt-4-turbo" in html, html
    assert "frugon analyze --measure --candidates gpt-4o" in html, html
    assert "quality-escalation" in html, html
    assert "keeping these calls on" not in html, html


def test_html_no_rung_keeps_dead_end_guidance(stub_no_rung: None) -> None:
    html = _quality_section_html(_not_confirmed_result(), style="v1")
    assert "keeping these calls on gpt-4-turbo" in html, html
    assert "try the next rung up" not in html, html


# ---------------------------------------------------------------------------
# Cross-surface reconciliation — the ~NN% figure is identical everywhere
# ---------------------------------------------------------------------------


def test_escalation_percentage_reconciles_across_surfaces(stub_rung: None) -> None:
    result = _not_confirmed_result()
    term = _capture(result)
    md = "\n".join(_quality_section_md(result))
    html = _quality_section_html(result, style="v1")
    needle = "~42% cheaper than gpt-4-turbo"
    assert needle in term, term
    assert needle in md, md
    assert needle in html, html


# ---------------------------------------------------------------------------
# Small-sample nudge — fires when borderline-for-N, silent when decisive
# ---------------------------------------------------------------------------


def test_nudge_fires_on_borderline_small_sample(stub_no_rung: None) -> None:
    # 3/5 split (held=2, losses=3) — n<10 AND |held-losses|=1 → low-confidence.
    out = _capture(_not_confirmed_result(wins=0, ties=2, losses=3, samples=5))
    assert "Small sample (5)" in out, out
    assert "--samples 25 for a firmer read" in out, out


def test_nudge_silent_on_decisive_large_sample(stub_no_rung: None) -> None:
    # 8/10 lose (held=2, losses=8): n>=10 AND wide margin → decisive, no nudge.
    out = _capture(_not_confirmed_result(wins=1, ties=1, losses=8, samples=10))
    assert "Small sample" not in out, out


def test_nudge_silent_on_decisive_confirm(stub_no_rung: None) -> None:
    # 9/10 hold (held=9, losses=1): clean confirm, wide margin → no nudge.
    result = MeasureResult(
        samples_requested=10,
        samples_taken=10,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=1, ties=9)],
    )
    out = _capture(result)
    assert "confirmed" in out, out
    assert "Small sample" not in out, out


def test_nudge_fires_in_md_and_html(stub_no_rung: None) -> None:
    result = _not_confirmed_result(wins=0, ties=2, losses=3, samples=5)
    md = "\n".join(_quality_section_md(result))
    html = _quality_section_html(result, style="v1")
    assert "Small sample (5)" in md, md
    assert "--samples 25 for a firmer read" in md, md
    assert "Small sample (5)" in html, html
    assert "quality-nudge" in html, html


def test_nudge_not_emitted_for_unverified_tally(stub_no_rung: None) -> None:
    # Every comparison errored → no scored verdict; the nudge is about a thin-but-
    # real sample, so it must NOT fire here (the fix is retry/access, not n).
    result = MeasureResult(
        samples_requested=3,
        samples_taken=3,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[
            Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=0, ties=0, errors=3)
        ],
    )
    out = _capture(result)
    assert "not verified" in out, out
    assert "Small sample" not in out, out
