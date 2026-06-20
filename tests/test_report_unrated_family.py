"""Tests for the unrated-recommendation message family (audit findings #1, #4)
and the full-swap basis caption surfacing (audit finding #2).

Findings #1 + #4 are one message family: when an EXPLICIT ``--candidates`` model
is unrated, frugon surfaces a non-blocking quality caveat next to the
recommendation (#1) and explains a wholesale full-swap fallback / split-skip
caused by the unrated model (#4) — the same wording on every surface (terminal +
Markdown v1/v2 + HTML v1/v2).  Finding #2 ensures the full-swap upper-bound basis
caption is shown even when the split's easy-call target is the same model as the
wholesale full-swap winner.

The unrated recommended-candidate role is a priced-but-unrated sentinel installed
via the shared ``install_unrated_sentinel`` fixture (absent from every quality
table, so always "unrated"), keeping the behaviour independent of real-registry
drift.  ``gpt-4o-mini`` is Capable (rated) in the bundled quality table.
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys
from pathlib import Path
from typing import Any

import pytest

import frugon
from frugon.cost import AnalysisResult, analyze_records, iter_records
from frugon.report import (
    _md_unrated_family_lines,
    _recommended_unrated_model,
    _unrated_family_messages,
    _unrated_recommendation_caveat,
    _upper_bound_pct,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
    render_terminal,
)

_sys.path.insert(0, str(_pathlib.Path(__file__).parent))
from conftest import FRUGON_TEST_UNRATED, install_unrated_sentinel

# Resolve the bundled demo sample exactly as the CLI does (``--demo``).
assert frugon.__file__ is not None
_SAMPLE = Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"

_UNRATED = FRUGON_TEST_UNRATED  # priced-but-unrated sentinel (drift-proof)
_RATED = "gpt-4o-mini"  # Capable tier in the bundled quality table


@pytest.fixture(autouse=True)
def _sentinel_pricing(monkeypatch, tmp_path):
    install_unrated_sentinel(monkeypatch, tmp_path)
    yield
    import frugon.pricing as _p

    _p.clear_pricing_cache()


def _result(candidates: list[str] | None) -> AnalysisResult:
    """Build the real ``--demo`` AnalysisResult for *candidates* via the engine."""
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(
        records,
        candidates=candidates,
        skipped_malformed=skipped,
        split_routing=True,
    )


# ---------------------------------------------------------------------------
# Shared string-source sanity (the one wording family every surface reuses)
# ---------------------------------------------------------------------------


def test_unrated_recommendation_caveat_names_model_and_command() -> None:
    """#1 caveat names the model and the exact --measure command to verify it."""
    text = _unrated_recommendation_caveat(_UNRATED)
    assert _UNRATED in text
    assert "unrated" in text
    # The verify command is --measure --judge: only the judge produces the scored
    # verdict that verifies the model's quality.
    assert f"--measure --judge --candidates {_UNRATED}" in text
    assert "verify it before you switch" in text


# ---------------------------------------------------------------------------
# Unified split basis — an unrated candidate routes via the split (no fallback)
# ---------------------------------------------------------------------------


def test_unrated_only_candidate_routes_via_split_with_caveat() -> None:
    """Arrange: --candidates with a single UNRATED model on the demo log.
    Act: analyze.
    Assert: under the unified split basis the unrated candidate is a first-class
    routing target — a split IS formed and routed to it (no wholesale fallback) —
    while the #1 unrated-recommendation caveat still fires so quality stays
    honestly disclosed.  The obsolete "held out / wholesale fallback" path no
    longer exists.
    """
    result = _result([_UNRATED])
    # A split now EXISTS and routes the easy calls to the unrated candidate.
    assert result.split is not None
    assert result.split.candidate_model == _UNRATED
    assert result.candidate_model == _UNRATED
    # Honesty is preserved via the #1 caveat: the recommendation is unrated.
    assert _recommended_unrated_model(result) == _UNRATED


def test_unrated_only_recommendation_emits_caveat_not_fallback_terminal(
    capsys: Any,
) -> None:
    """Arrange: the unrated single-candidate result (now a split, not a fallback).
    Act: render_terminal.
    Assert: the #1 recommendation caveat renders (honest disclosure of the unrated
    routing target) and the obsolete "Routing split unavailable" fallback note
    does NOT appear.
    """
    render_terminal(_result([_UNRATED]))
    out = " ".join(capsys.readouterr().out.split())
    assert " ".join(_unrated_recommendation_caveat(_UNRATED).split()) in out
    assert "Routing split unavailable" not in out


def test_unrated_only_recommendation_emits_caveat_not_fallback_markdown(
    tmp_path: Path,
) -> None:
    """The #1 caveat renders (not the #4 fallback note) in the Markdown report."""
    out = tmp_path / "r.md"
    render_markdown(_result([_UNRATED]), out)
    text = out.read_text(encoding="utf-8")
    assert _unrated_recommendation_caveat(_UNRATED) in text
    assert "Routing split unavailable" not in text


def test_unrated_only_recommendation_emits_caveat_not_fallback_markdown_v2(
    tmp_path: Path,
) -> None:
    """The #1 caveat renders (not the #4 fallback note) in the v2 Markdown report."""
    out = tmp_path / "r.md"
    render_markdown_v2(_result([_UNRATED]), out)
    text = out.read_text(encoding="utf-8")
    assert _unrated_recommendation_caveat(_UNRATED) in text
    assert "Routing split unavailable" not in text


@pytest.mark.parametrize("renderer", [render_html, render_html_v2])
def test_unrated_only_recommendation_emits_caveat_not_fallback_html(
    renderer: Any, tmp_path: Path
) -> None:
    """The #1 caveat renders (not the #4 fallback note) in both HTML report styles."""
    out = tmp_path / "r.html"
    renderer(_result([_UNRATED]), out)
    text = out.read_text(encoding="utf-8")
    # The verify command (part of the #1 caveat) is present; the obsolete
    # wholesale-fallback note is not.
    assert f"--measure --judge --candidates {_UNRATED}" in text
    assert "Routing split unavailable" not in text


# ---------------------------------------------------------------------------
# Finding #1 — rated recommended candidate fires NO caveat
# ---------------------------------------------------------------------------


def test_rated_candidate_emits_no_unrated_caveat_terminal(capsys: Any) -> None:
    """Arrange: an explicit RATED candidate (gpt-4o-mini, Capable).
    Act: render_terminal.
    Assert: none of the unrated-family wording appears — the caveat must not fire
    when the recommendation is rated.
    """
    result = _result([_RATED])
    assert _recommended_unrated_model(result) is None
    assert _unrated_family_messages(result) == []
    render_terminal(result)
    out = capsys.readouterr().out
    assert "is unrated — its quality is unverified" not in out
    assert "Routing split unavailable" not in out


def test_rated_candidate_emits_no_unrated_caveat_markdown(tmp_path: Path) -> None:
    """The rated single-candidate Markdown report carries no unrated-family line."""
    out = tmp_path / "r.md"
    render_markdown(_result([_RATED]), out)
    text = out.read_text(encoding="utf-8")
    assert "is unrated — its quality is unverified" not in text
    assert "Routing split unavailable" not in text
    # The unrated-family MD helper is empty for a rated recommendation.
    assert _md_unrated_family_lines(_result([_RATED])) == []


# ---------------------------------------------------------------------------
# Finding #2 — full-swap basis caption shown when split-winner == wholesale-winner
# ---------------------------------------------------------------------------


def test_upper_bound_surfaced_when_split_and_wholesale_winner_match() -> None:
    """Arrange: a single RATED candidate where the split easy-call target and the
    wholesale full-swap winner are the SAME model (gpt-4o-mini).
    Act: _upper_bound_pct.
    Assert: a meaningful full-swap upper bound is returned (NOT suppressed) — the
    split routes only the easy baseline calls while the full swap moves every
    call, so the figures differ and the basis context must be shown.
    """
    result = _result([_RATED])
    assert result.split is not None
    assert result.candidate_model == result.split.candidate_model  # same winner
    upper = _upper_bound_pct(result)
    assert upper is not None
    # The full swap is materially more aggressive than the conservative split.
    assert result.split.saving_pct is not None
    assert upper > result.split.saving_pct


def test_upper_bound_caption_in_terminal_when_winners_match(capsys: Any) -> None:
    """The Upper-bound row renders for the same-winner case (basis context)."""
    render_terminal(_result([_RATED]))
    out = " ".join(capsys.readouterr().out.split())
    assert "Upper bound" in out
    assert "a full swap to" in out


def test_upper_bound_caption_in_markdown_when_winners_match(tmp_path: Path) -> None:
    """The Upper-bound caption renders in the Markdown report for same-winner."""
    out = tmp_path / "r.md"
    render_markdown(_result([_RATED]), out)
    text = out.read_text(encoding="utf-8")
    assert "Upper bound: moving every call to" in text
    assert _RATED in text


def test_upper_bound_suppressed_when_split_routes_whole_dataset() -> None:
    """The 100%-routed identical case: split saving == full-swap saving, so no
    separate upper bound is shown (no double-print of the same figure).
    """
    # A single-baseline-model log of tiny easy calls routes 100% to the candidate.
    from frugon.cost import LogRecord

    records = [
        LogRecord(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": "hi"}],
            completion_text="ok",
            prompt_tokens=5,
            completion_tokens=3,
            timestamp=None,
        )
        for _ in range(20)
    ]
    result = analyze_records(records, candidates=[_RATED], split_routing=True)
    assert result.split is not None
    # Every call routed → the split blended equals the full-swap projection.
    assert result.split.kept_count == 0
    assert _upper_bound_pct(result) is None


# ---------------------------------------------------------------------------
# Default --demo path is unaffected (rated recommendation, no family lines)
# ---------------------------------------------------------------------------


def test_default_demo_pool_emits_no_unrated_family() -> None:
    """The default (no --candidates) demo recommends a rated model, so the
    unrated-message family is silent.
    """
    result = _result(None)
    assert _unrated_family_messages(result) == []
    assert _md_unrated_family_lines(result) == []
