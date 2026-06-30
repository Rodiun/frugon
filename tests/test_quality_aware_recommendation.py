"""Quality-aware candidate recommendation (Change 1) + post-measurement promotion (Change 2).

Change 1 — eligibility for the headline recommendation is RATED (or
measured-confirmed).  On an explicit ``--candidates`` run the headline routing
target is the cheapest RATED candidate that beats baseline; an unrated candidate
that beats baseline is surfaced as ``considered`` (with its real split figure)
but held out of the recommended route, and a clear "could save ~X%, but it's
unrated — excluded until you verify it" caveat is shown.  The "within tolerance"
quality badge must NOT appear on an unrated route.  When NO rated candidate beats
baseline the headline falls back to the cheapest unrated one (with the existing
unrated recommendation caveat).

Change 2 — once ``--measure --judge`` CONFIRMS an excluded-unrated candidate AND
it saves more than the headline rated pick, a positive green ✓ promotion callout
surfaces across terminal + Markdown + HTML.  It fires ONLY on confirmed + saves-
more + was-excluded-for-being-unrated; never when it confirms-but-saves-less,
fails quality, or was already the recommendation.

These tests exercise the bundled demo log (``gpt-5.5`` baseline,
``gpt-4o`` rated candidate, ``frugon-eval-unrated-x1`` unrated) for the offline
behaviour, and construct ``MeasureResult`` stubs directly for the promotion
(fully offline — no provider call).
"""

from __future__ import annotations

import re
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from rich.console import Console

import frugon
from frugon.cost import (
    AnalysisResult,
    CandidateProjection,
    analyze_records,
    iter_records,
)
from frugon.measure import MeasureResult, Tier1Tally
from frugon.report import (
    _detect_promotion,
    _promotion_message,
    _quality_section_html,
    _quality_section_md,
    judged_models_from_measure,
    render_quality_terminal,
)

sys.path.insert(0, str(Path(__file__).parent))
from conftest import install_unrated_sentinel

assert frugon.__file__ is not None
_SAMPLE = Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


@pytest.fixture(autouse=True)
def _sentinel_pricing(monkeypatch, tmp_path):
    install_unrated_sentinel(monkeypatch, tmp_path)
    yield
    import frugon.pricing as _p

    _p.clear_pricing_cache()


def _demo_result(candidates: list[str]) -> AnalysisResult:
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(
        list(records),
        candidates=candidates,
        skipped_malformed=skipped,
        split_routing=True,
    )


def _block(result: AnalysisResult) -> dict[str, CandidateProjection]:
    return {p.model: p for p in result.candidate_projections}


# ===========================================================================
# Change 1 — rated eligibility, considered unrated, excluded caveat
# ===========================================================================


def test_explicit_candidates_headline_is_cheapest_rated() -> None:
    """gpt-4o,frugon-eval-unrated-x1 on gpt-5.5 demo: headline routes to rated gpt-4o."""
    result = _demo_result(["gpt-4o", "frugon-eval-unrated-x1"])
    assert result.candidate_model == "gpt-4o"
    assert result.split is not None
    assert result.split.candidate_model == "gpt-4o"
    assert result.candidate_is_unrated is False


def test_unrated_beating_candidate_is_considered_not_recommended() -> None:
    """frugon-eval-unrated-x1 beats baseline but is unrated → considered + excluded."""
    result = _demo_result(["gpt-4o", "frugon-eval-unrated-x1"])
    block = _block(result)
    assert block["gpt-4o"].status == "recommended"
    assert block["frugon-eval-unrated-x1"].status == "considered"
    # Held out of the recommended route for being unrated (Change 1b).
    assert result.excluded_unrated_models == ["frugon-eval-unrated-x1"]


def test_excluded_unrated_caveat_present_with_real_saving_pct() -> None:
    """The 'could save ~X%, but it's unrated — measure to unlock' caveat shows."""
    result = _demo_result(["gpt-4o", "frugon-eval-unrated-x1"])
    from frugon.report import _unrated_family_messages

    messages = [m for m, _sev in _unrated_family_messages(result)]
    caveat = next(
        (m for m in messages if "excluded from the recommended route" in m), None
    )
    assert caveat is not None, messages
    assert "frugon-eval-unrated-x1" in caveat
    # The percentage matches the unrated candidate's block saving% (30.7%).
    # Derived from _demo_result(["gpt-4o","frugon-eval-unrated-x1"]) with the
    # gpt-5.5 baseline; updated when the demo baseline was modernised (gpt-5.5
    # is pricier than the former gpt-4o baseline, raising the unrated candidate's
    # apparent block saving from 26.5% to 30.7%).
    assert "30.7%" in caveat
    # Only --judge yields the scored verdict that unlocks the model.
    assert "--measure --judge --candidates frugon-eval-unrated-x1" in caveat
    assert "unlock it as the recommendation" in caveat


def test_rated_route_keeps_within_tolerance_badge() -> None:
    """gpt-4o is rated, so its route still shows 'within tolerance' (terminal)."""
    console = Console(width=200, force_terminal=False, no_color=True)
    result = _demo_result(["gpt-4o", "frugon-eval-unrated-x1"])
    from frugon.report import _render_split_panel

    report = sys.modules[_render_split_panel.__module__]
    original = report.rprint
    captured: list[str] = []

    def _rp(*args: object, **kwargs: object) -> None:
        with console.capture() as cap:
            console.print(*args, **kwargs)
        captured.append(cap.get())

    report.rprint = _rp  # type: ignore[assignment]
    try:
        assert result.split is not None
        report._render_split_panel(result, result.split)
    finally:
        report.rprint = original  # type: ignore[assignment]
    out = _ANSI_RE.sub("", "".join(captured))
    assert "gpt-4o" in out
    assert "within tolerance" in out  # gpt-4o IS rated


def test_unrated_fallback_route_has_no_tolerance_badge() -> None:
    """Only-unrated candidate: route to it, but NO 'within tolerance' badge."""
    result = _demo_result(["frugon-eval-unrated-x1"])
    assert result.candidate_model == "frugon-eval-unrated-x1"
    assert result.candidate_is_unrated is True
    assert result.split is not None

    console = Console(width=200, force_terminal=False, no_color=True)
    from frugon.report import _render_split_panel

    report = sys.modules[_render_split_panel.__module__]
    original = report.rprint
    captured: list[str] = []

    def _rp(*args: object, **kwargs: object) -> None:
        with console.capture() as cap:
            console.print(*args, **kwargs)
        captured.append(cap.get())

    report.rprint = _rp  # type: ignore[assignment]
    try:
        report._render_split_panel(result, result.split)
    finally:
        report.rprint = original  # type: ignore[assignment]
    out = _ANSI_RE.sub("", "".join(captured))
    assert "frugon-eval-unrated-x1" in out
    # The route is unrated — no offline quality claim.
    assert "within tolerance" not in out


def test_unrated_fallback_has_recommendation_caveat_not_excluded() -> None:
    """Fallback: the #1 unrated recommendation caveat shows; no 'excluded' note."""
    result = _demo_result(["frugon-eval-unrated-x1"])
    from frugon.report import _unrated_family_messages

    messages = [m for m, _sev in _unrated_family_messages(result)]
    assert any(
        "frugon-eval-unrated-x1 is unrated" in m and "Run --measure" in m for m in messages
    ), messages
    # It IS the recommendation, so it is never "excluded from the route".
    assert not any("excluded from the recommended route" in m for m in messages)
    assert result.excluded_unrated_models == []


def test_unrated_html_route_omits_tolerance_badge() -> None:
    """HTML v1 + v2 omit the 'within tolerance' badge on an unrated route."""
    result = _demo_result(["frugon-eval-unrated-x1"])
    from frugon.report import render_html, render_html_v2

    for renderer in (render_html, render_html_v2):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.html"
            renderer(result, path)
            html = path.read_text(encoding="utf-8")
        assert "frugon-eval-unrated-x1" in html
        # The RENDERED badge is a span; "within tolerance" also appears in a CSS
        # comment, so assert on the badge markup itself, not the raw substring.
        assert '<span class="badge">within tolerance</span>' not in html, (
            renderer.__name__
        )


# ===========================================================================
# Change 2 — post-measurement promotion
# ===========================================================================


def _promo_result(
    *,
    excluded: list[str],
    headline: str = "gpt-4o",
    haiku_status: str = "considered",
    haiku_pct: Decimal = Decimal("26.5"),
    headline_pct: Decimal = Decimal("15.1"),
) -> AnalysisResult:
    return AnalysisResult(
        total_calls=56100,
        priced_calls=56100,
        unpriced_calls=0,
        total_cost=Decimal("389.88"),
        candidate_model=headline,
        candidate_is_unrated=False,
        excluded_unrated_models=excluded,
        candidate_projections=[
            CandidateProjection(
                model="gpt-4o",
                status="recommended",
                monthly_cost=Decimal("331.03"),
                saving_pct=headline_pct,
            ),
            CandidateProjection(
                model="frugon-eval-unrated-x1",
                status=haiku_status,
                monthly_cost=Decimal("286.52"),
                saving_pct=haiku_pct,
            ),
        ],
    )


def _measure(wins: int, losses: int = 0, ties: int = 0) -> MeasureResult:
    return MeasureResult(
        samples_requested=10,
        samples_taken=10,
        current_model="gpt-5.5",
        candidates=["frugon-eval-unrated-x1"],
        comparisons=[],
        tier1_tallies=[
            Tier1Tally(
                candidate="frugon-eval-unrated-x1",
                wins=wins,
                losses=losses,
                ties=ties,
                errors=0,
            )
        ],
    )


def test_promotion_detects_confirmed_cheaper_excluded_candidate() -> None:
    """Confirmed (10/10) + saves more (26.5 > 15.1) + excluded → promote."""
    result = _promo_result(excluded=["frugon-eval-unrated-x1"])
    promo = _detect_promotion(result, _measure(10))
    assert promo is not None
    assert promo.candidate == "frugon-eval-unrated-x1"
    msg = _promotion_message(promo)
    assert "held quality on your data (10/10)" in msg
    assert "26.5%" in msg
    assert "gpt-4o (15.1%)" in msg
    assert "--candidates frugon-eval-unrated-x1 to switch" in msg


def test_no_promotion_when_confirmed_but_saves_less() -> None:
    """Confirmed but saving% ≤ headline → NO promotion.

    The headline is gpt-4o at 15.1%; sentinel at 10.0% saves LESS → no promo.
    """
    result = _promo_result(
        excluded=["frugon-eval-unrated-x1"], haiku_pct=Decimal("10.0")
    )
    assert _detect_promotion(result, _measure(10)) is None


def test_no_promotion_when_not_confirmed() -> None:
    """Not confirmed (lost the majority) → NO promotion even though cheaper."""
    result = _promo_result(excluded=["frugon-eval-unrated-x1"])
    assert _detect_promotion(result, _measure(wins=2, losses=8)) is None


def test_no_promotion_when_not_excluded() -> None:
    """Candidate was the recommendation (not excluded-unrated) → NO promotion."""
    result = _promo_result(excluded=[])
    assert _detect_promotion(result, _measure(10)) is None


def test_promotion_renders_on_terminal() -> None:
    """The green ✓ promotion callout renders in the terminal quality section."""
    result = _promo_result(excluded=["frugon-eval-unrated-x1"])
    console = Console(width=200, force_terminal=False, no_color=True)
    report = sys.modules[render_quality_terminal.__module__]
    original = report.rprint

    def _rp(*args: object, **kwargs: object) -> None:
        console.print(*args, **kwargs)

    report.rprint = _rp  # type: ignore[assignment]
    with console.capture() as cap:
        try:
            report.render_quality_terminal(_measure(10), result=result)
        finally:
            report.rprint = original  # type: ignore[assignment]
    out = _ANSI_RE.sub("", cap.get())
    assert "✓" in out
    assert "held quality on your data (10/10)" in out
    assert "it's the better route" in out


def test_promotion_renders_in_markdown_and_html() -> None:
    """The promotion callout appears in the MD + HTML quality sections."""
    result = _promo_result(excluded=["frugon-eval-unrated-x1"])
    md = "\n".join(_quality_section_md(_measure(10), result=result))
    assert "✓" in md
    assert "held quality on your data (10/10)" in md

    html = _quality_section_html(_measure(10), style="v1", result=result)
    assert "quality-promotion" in html
    assert "held quality on your data (10/10)" in html


def test_promotion_suppresses_excluded_caveat_once_measured() -> None:
    """Once the unrated candidate is judged, the 'excluded — measure to unlock' caveat drops."""
    result = _promo_result(excluded=["frugon-eval-unrated-x1"])
    from frugon.report import _unrated_family_messages

    judged = judged_models_from_measure(_measure(10))
    messages = [m for m, _sev in _unrated_family_messages(result, judged)]
    assert not any("excluded from the recommended route" in m for m in messages)
