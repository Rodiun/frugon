"""Tests for quality-delta wording branches across all render surfaces.

The routing recommendation must gate the "within tolerance" risk phrase and the
"Quality is not verified — verify before you switch" caveat on whether the
candidate is a genuine quality step-DOWN (tier_drop >= 1).  When the candidate
is rated SAME or BETTER quality (tier_drop <= 0), those risk strings must not
appear; instead the report states the stronger, accurate positive claim.

When either model is UNRATED (tier_drop is None), the existing "unverified /
unrated" disclosures must still fire — this path is fully separate from the
equal-or-better path and must never be silenced by this logic.

Coverage: terminal panel, terminal footer, Markdown (v1 + v2), HTML v1, HTML v2.
Each render surface that has distinct wording for these branches is covered by
its own assertion class.
"""

from __future__ import annotations

import io
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from rich.console import Console

from frugon.cost import AnalysisResult, LogRecord
from frugon.measure import (
    Comparison,
    MeasureResult,
    SampledOutput,
    Tier1Tally,
)
from frugon.report import (
    QUALITY_NOT_VERIFIED_ACTION,
    QUALITY_NOT_VERIFIED_ASSERTION,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
    render_quality_terminal,
    render_terminal,
)
from frugon.routing import SplitRouting

# ---------------------------------------------------------------------------
# Fixtures — rated models only, explicit tier_drop values
# ---------------------------------------------------------------------------


def _split() -> SplitRouting:
    """A split routing fixture using two models that are in the quality registry."""
    return SplitRouting(
        baseline_model="gpt-4-turbo",
        candidate_model="gpt-4o-mini",
        routed_count=24,
        kept_count=3,
        routed_cost=Decimal("0.0004"),
        kept_cost=Decimal("0.0435"),
        baseline_cost=Decimal("0.0650"),
        blended_cost=Decimal("0.0439"),
        easy_threshold=Decimal("0.35"),
        monthly_baseline=Decimal("0.2788"),
        monthly_blended=Decimal("0.1881"),
    )


def _result(tier_drop: int | None, **kwargs: Any) -> AnalysisResult:
    """AnalysisResult with explicit tier_drop.

    Both baseline and candidate are real rated models so the quality-registry
    lookup does not interfere with the tier_drop we inject.  All other fields
    are the minimum required for a split rendering to be triggered.
    """
    defaults: dict[str, Any] = {
        "total_calls": 27,
        "priced_calls": 27,
        "unpriced_calls": 0,
        "total_cost": Decimal("0.0650"),
        "cost_by_model": {"gpt-4-turbo": Decimal("0.0650")},
        "calls_by_model": {"gpt-4-turbo": 27},
        "projected_cost": Decimal("0.0222"),
        "candidate_model": "gpt-4o",
        "observed_span_days": 7.0,
        "split": _split(),
        "tier_drop": tier_drop,
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _result_step_down() -> AnalysisResult:
    """Candidate is a genuine quality step-DOWN (tier_drop=1)."""
    return _result(tier_drop=1)


def _result_same_tier() -> AnalysisResult:
    """Candidate is the SAME quality tier as baseline (tier_drop=0)."""
    return _result(tier_drop=0)


def _result_better_tier() -> AnalysisResult:
    """Candidate is a HIGHER quality tier than baseline (tier_drop=-1)."""
    return _result(tier_drop=-1)


def _result_unrated() -> AnalysisResult:
    """Both tier_drop is None AND candidate is unrated — the unverified path."""
    return _result(tier_drop=None, candidate_is_unrated=True)


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------


def _render_terminal_to_str(result: AnalysisResult, **kwargs: Any) -> str:
    """Render the terminal split view to a plain-text string (no colour)."""
    report_mod = sys.modules[render_terminal.__module__]
    buf = io.StringIO()
    console = Console(
        file=buf,
        width=100,
        no_color=True,
        force_terminal=True,
        highlight=False,
        legacy_windows=False,
    )
    original_rprint = report_mod.rprint
    original_render_console = report_mod._render_console

    def _patched(*args: Any, **kw: Any) -> None:
        console.print(*args, **kw)

    report_mod.rprint = _patched  # type: ignore[attr-defined]
    report_mod._render_console = lambda: console  # type: ignore[attr-defined]
    try:
        render_terminal(result, **kwargs)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    return " ".join(buf.getvalue().split())


# ---------------------------------------------------------------------------
# Terminal surface
# ---------------------------------------------------------------------------


class TestTerminalTierDrop:
    """Terminal panel + footer quality-wording branches."""

    def test_step_down_shows_within_tolerance_in_panel(self) -> None:
        """tier_drop=1 → panel Route line shows 'within tolerance'."""
        text = _render_terminal_to_str(_result_step_down())
        assert "within tolerance" in text

    def test_step_down_shows_risk_caveat_in_footer(self) -> None:
        """tier_drop=1 → footer shows the 'Quality is not verified' risk caveat."""
        text = _render_terminal_to_str(_result_step_down())
        assert QUALITY_NOT_VERIFIED_ASSERTION in text
        assert QUALITY_NOT_VERIFIED_ACTION in text

    def test_step_down_no_same_or_better_in_panel(self) -> None:
        """tier_drop=1 → 'same or better quality' must NOT appear in the output."""
        text = _render_terminal_to_str(_result_step_down())
        assert "same or better quality" not in text

    def test_same_tier_shows_same_or_better_in_panel(self) -> None:
        """tier_drop=0 → panel Route line shows 'same or better quality'."""
        text = _render_terminal_to_str(_result_same_tier())
        assert "same or better quality" in text

    def test_same_tier_no_risk_caveat_in_footer(self) -> None:
        """tier_drop=0 → footer must NOT show the 'Quality is not verified' risk caveat."""
        text = _render_terminal_to_str(_result_same_tier())
        assert QUALITY_NOT_VERIFIED_ASSERTION not in text
        assert QUALITY_NOT_VERIFIED_ACTION not in text

    def test_same_tier_no_within_tolerance_in_panel(self) -> None:
        """tier_drop=0 → 'within tolerance' must NOT appear anywhere."""
        text = _render_terminal_to_str(_result_same_tier())
        assert "within tolerance" not in text

    def test_better_tier_shows_same_or_better_in_panel(self) -> None:
        """tier_drop=-1 (higher quality candidate) → 'same or better quality' shown."""
        text = _render_terminal_to_str(_result_better_tier())
        assert "same or better quality" in text

    def test_better_tier_no_risk_caveat_in_footer(self) -> None:
        """tier_drop=-1 → footer must NOT show the quality-risk caveat."""
        text = _render_terminal_to_str(_result_better_tier())
        assert QUALITY_NOT_VERIFIED_ASSERTION not in text
        assert QUALITY_NOT_VERIFIED_ACTION not in text

    def test_better_tier_no_within_tolerance_in_panel(self) -> None:
        """tier_drop=-1 → 'within tolerance' must NOT appear anywhere."""
        text = _render_terminal_to_str(_result_better_tier())
        assert "within tolerance" not in text

    def test_unrated_path_unaffected_no_same_or_better(self) -> None:
        """Unrated candidate (tier_drop=None) → 'same or better quality' never shown."""
        text = _render_terminal_to_str(_result_unrated())
        assert "same or better quality" not in text

    def test_unrated_path_still_shows_quality_risk(self) -> None:
        """Unrated candidate → the existing quality-risk/verify caveat still fires."""
        text = _render_terminal_to_str(_result_unrated())
        # For unrated candidates the wholesale assertion is used (no "within tolerance"
        # band reference), but the call-to-action still appears.
        assert QUALITY_NOT_VERIFIED_ACTION in text


# ---------------------------------------------------------------------------
# Markdown surface
# ---------------------------------------------------------------------------


class TestMarkdownTierDrop:
    """Markdown (v1 + v2 share one renderer) quality-wording branches."""

    def _md(self, result: AnalysisResult, tmp_path: Path, *, v2: bool = False) -> str:
        out = tmp_path / ("report_v2.md" if v2 else "report.md")
        if v2:
            render_markdown_v2(result, out)
        else:
            render_markdown(result, out)
        return out.read_text(encoding="utf-8")

    def test_md_step_down_headline_contains_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=1 → headline bold line contains 'within tolerance'."""
        text = self._md(_result_step_down(), tmp_path)
        assert "within tolerance" in text

    def test_md_step_down_tagline_contains_split_caveat(self, tmp_path: Path) -> None:
        """tier_drop=1 → tagline italic line contains the risk caveat text."""
        text = self._md(_result_step_down(), tmp_path)
        assert "Quality is not verified" in text

    def test_md_same_tier_headline_contains_same_or_better(self, tmp_path: Path) -> None:
        """tier_drop=0 → headline bold line contains 'same or better quality'."""
        text = self._md(_result_same_tier(), tmp_path)
        assert "same or better quality" in text

    def test_md_same_tier_no_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=0 → 'within tolerance' must not appear anywhere in the output."""
        text = self._md(_result_same_tier(), tmp_path)
        assert "within tolerance" not in text

    def test_md_same_tier_no_quality_not_verified(self, tmp_path: Path) -> None:
        """tier_drop=0 → 'Quality is not verified' must not appear anywhere."""
        text = self._md(_result_same_tier(), tmp_path)
        assert "Quality is not verified" not in text

    def test_md_better_tier_no_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=-1 → 'within tolerance' must not appear anywhere."""
        text = self._md(_result_better_tier(), tmp_path)
        assert "within tolerance" not in text

    def test_md_better_tier_no_quality_not_verified(self, tmp_path: Path) -> None:
        """tier_drop=-1 → 'Quality is not verified' must not appear anywhere."""
        text = self._md(_result_better_tier(), tmp_path)
        assert "Quality is not verified" not in text

    def test_md_unrated_path_unaffected_no_same_or_better(self, tmp_path: Path) -> None:
        """Unrated path (tier_drop=None) → 'same or better quality' never shown."""
        text = self._md(_result_unrated(), tmp_path)
        assert "same or better quality" not in text

    def test_md_unrated_path_still_shows_quality_risk(self, tmp_path: Path) -> None:
        """Unrated path → the quality-risk / verify caveat still appears."""
        text = self._md(_result_unrated(), tmp_path)
        assert "Quality is not verified" in text

    # v2 variant — same renderer, just sanity-check the v2 writer path

    def test_md_v2_same_tier_no_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=0 via render_markdown_v2 → no 'within tolerance'."""
        text = self._md(_result_same_tier(), tmp_path, v2=True)
        assert "within tolerance" not in text

    def test_md_v2_step_down_has_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=1 via render_markdown_v2 → 'within tolerance' present."""
        text = self._md(_result_step_down(), tmp_path, v2=True)
        assert "within tolerance" in text


# ---------------------------------------------------------------------------
# HTML v1 surface
# ---------------------------------------------------------------------------


class TestHtmlV1TierDrop:
    """HTML v1 routing-plan table and Quality card wording branches."""

    def _html(self, result: AnalysisResult, tmp_path: Path) -> str:
        out = tmp_path / "report.html"
        render_html(result, out)
        return out.read_text(encoding="utf-8")

    def test_html_v1_step_down_shows_within_tolerance_badge(self, tmp_path: Path) -> None:
        """tier_drop=1 → routing-plan routed row carries the 'within tolerance' badge."""
        html = self._html(_result_step_down(), tmp_path)
        assert "within tolerance" in html

    def test_html_v1_step_down_caveat_in_quality_card(self, tmp_path: Path) -> None:
        """tier_drop=1 → Quality card shows 'Quality is not verified'."""
        html = self._html(_result_step_down(), tmp_path)
        assert "Quality is not verified" in html

    def test_html_v1_same_tier_shows_same_or_better_badge(self, tmp_path: Path) -> None:
        """tier_drop=0 → routing-plan routed row carries 'same or better quality' badge."""
        html = self._html(_result_same_tier(), tmp_path)
        assert "same or better quality" in html

    def test_html_v1_same_tier_no_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=0 → 'within tolerance' must not appear anywhere."""
        html = self._html(_result_same_tier(), tmp_path)
        assert "within tolerance" not in html

    def test_html_v1_same_tier_no_quality_not_verified(self, tmp_path: Path) -> None:
        """tier_drop=0 → 'Quality is not verified' must not appear in the HTML output."""
        html = self._html(_result_same_tier(), tmp_path)
        assert "Quality is not verified" not in html

    def test_html_v1_better_tier_no_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=-1 → no 'within tolerance' anywhere in the output."""
        html = self._html(_result_better_tier(), tmp_path)
        assert "within tolerance" not in html

    def test_html_v1_unrated_no_same_or_better(self, tmp_path: Path) -> None:
        """Unrated (tier_drop=None) → 'same or better quality' never appears."""
        html = self._html(_result_unrated(), tmp_path)
        assert "same or better quality" not in html

    def test_html_v1_unrated_quality_risk_present(self, tmp_path: Path) -> None:
        """Unrated (tier_drop=None) → quality-risk caveat still fires."""
        html = self._html(_result_unrated(), tmp_path)
        assert "Quality is not verified" in html


# ---------------------------------------------------------------------------
# HTML v2 surface
# ---------------------------------------------------------------------------


class TestHtmlV2TierDrop:
    """HTML v2 hero lede, routing-plan badge, and footer fineprint wording."""

    def _html(self, result: AnalysisResult, tmp_path: Path) -> str:
        out = tmp_path / "report_v2.html"
        render_html_v2(result, out)
        return out.read_text(encoding="utf-8")

    def test_html_v2_step_down_hero_lede_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=1 → v2 hero lede contains 'within tolerance'."""
        html = self._html(_result_step_down(), tmp_path)
        assert "within tolerance" in html

    def test_html_v2_step_down_fineprint_not_verified(self, tmp_path: Path) -> None:
        """tier_drop=1 → v2 footer fineprint contains 'Quality is not verified'."""
        html = self._html(_result_step_down(), tmp_path)
        assert "Quality is not verified" in html

    def test_html_v2_same_tier_hero_lede_same_or_better(self, tmp_path: Path) -> None:
        """tier_drop=0 → v2 hero lede contains 'same or better quality'."""
        html = self._html(_result_same_tier(), tmp_path)
        assert "same or better quality" in html

    def test_html_v2_same_tier_no_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=0 → 'within tolerance' must not appear anywhere in v2 output."""
        html = self._html(_result_same_tier(), tmp_path)
        assert "within tolerance" not in html

    def test_html_v2_same_tier_no_quality_not_verified(self, tmp_path: Path) -> None:
        """tier_drop=0 → 'Quality is not verified' must not appear in v2 output."""
        html = self._html(_result_same_tier(), tmp_path)
        assert "Quality is not verified" not in html

    def test_html_v2_better_tier_same_or_better_badge(self, tmp_path: Path) -> None:
        """tier_drop=-1 → v2 routed row badge shows 'same or better quality'."""
        html = self._html(_result_better_tier(), tmp_path)
        assert "same or better quality" in html

    def test_html_v2_better_tier_no_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=-1 → no 'within tolerance' in v2 output."""
        html = self._html(_result_better_tier(), tmp_path)
        assert "within tolerance" not in html

    def test_html_v2_unrated_no_same_or_better(self, tmp_path: Path) -> None:
        """Unrated (tier_drop=None) → 'same or better quality' never appears in v2."""
        html = self._html(_result_unrated(), tmp_path)
        assert "same or better quality" not in html

    def test_html_v2_unrated_quality_risk_present(self, tmp_path: Path) -> None:
        """Unrated (tier_drop=None) → quality-risk caveat still fires in v2."""
        html = self._html(_result_unrated(), tmp_path)
        assert "Quality is not verified" in html


# ---------------------------------------------------------------------------
# _is_equal_or_better_quality helper unit tests
# ---------------------------------------------------------------------------


class TestIsEqualOrBetterQualityHelper:
    """Direct unit tests for the _is_equal_or_better_quality predicate."""

    def _subject(self, tier_drop: int | None) -> bool:
        from frugon.report import _is_equal_or_better_quality  # type: ignore[attr-defined]

        return _is_equal_or_better_quality(_result(tier_drop=tier_drop))

    def test_tier_drop_none_returns_false(self) -> None:
        assert self._subject(None) is False

    def test_tier_drop_negative_one_returns_true(self) -> None:
        assert self._subject(-1) is True

    def test_tier_drop_zero_returns_true(self) -> None:
        assert self._subject(0) is True

    def test_tier_drop_one_returns_false(self) -> None:
        assert self._subject(1) is False

    def test_tier_drop_two_returns_false(self) -> None:
        assert self._subject(2) is False


# ---------------------------------------------------------------------------
# _shown_quality_phrase helper unit tests
# ---------------------------------------------------------------------------


class TestShownQualityPhraseHelper:
    """Direct unit tests for the _shown_quality_phrase helper.

    The helper is the single source of truth for which back-reference phrase
    the --measure/--judge synthesis lines should echo, keeping them consistent
    with whatever the offline panel already showed the user (§6 honesty
    invariant).
    """

    def _subject(self, tier_drop: int | None) -> str:
        from frugon.report import _shown_quality_phrase  # type: ignore[attr-defined]

        return _shown_quality_phrase(_result(tier_drop=tier_drop))

    def test_tier_drop_zero_returns_same_or_better(self) -> None:
        """Same quality tier → phrase tracks the panel ('same or better quality')."""
        assert self._subject(0) == "same or better quality"

    def test_tier_drop_negative_one_returns_same_or_better(self) -> None:
        """Better quality tier → phrase tracks the panel ('same or better quality')."""
        assert self._subject(-1) == "same or better quality"

    def test_tier_drop_one_returns_within_tolerance(self) -> None:
        """Genuine step-down → phrase tracks the panel ('within tolerance')."""
        assert self._subject(1) == "within tolerance"

    def test_tier_drop_none_returns_within_tolerance(self) -> None:
        """Unrated path (tier_drop=None) → phrase tracks the panel ('within tolerance')."""
        assert self._subject(None) == "within tolerance"

    def test_none_result_returns_within_tolerance(self) -> None:
        """No AnalysisResult at all → safe fallback is 'within tolerance'."""
        from frugon.report import _shown_quality_phrase  # type: ignore[attr-defined]

        assert _shown_quality_phrase(None) == "within tolerance"


# ---------------------------------------------------------------------------
# Shared helpers for --measure / --judge synthesis surface tests
# ---------------------------------------------------------------------------


def _log_record() -> LogRecord:
    return LogRecord(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": "Classify this ticket"}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _sampled_output(content: str = "ok", *, error: str | None = None) -> SampledOutput:
    return SampledOutput(model="gpt-4-turbo", content=content, error=error)


def _comparison() -> Comparison:
    return Comparison(
        record=_log_record(),
        current_output=_sampled_output("baseline answer"),
        candidate_outputs=[_sampled_output("candidate answer")],
    )


def _confirmed_measure_result(candidate: str = "gpt-4o-mini") -> MeasureResult:
    """A Tier-1 MeasureResult where the single candidate confirmed quality."""
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4-turbo",
        candidates=[candidate],
        comparisons=[_comparison()],
        tier1_tallies=[Tier1Tally(candidate=candidate, wins=4, losses=0, ties=1)],
    )


def _tier0_measure_result(candidate: str = "gpt-4o-mini") -> MeasureResult:
    """A Tier-0 MeasureResult (no judge tallies) for the framing-line tests."""
    return MeasureResult(
        samples_requested=3,
        samples_taken=3,
        current_model="gpt-4-turbo",
        candidates=[candidate],
        comparisons=[_comparison()],
        tier1_tallies=None,
    )


def _capture_quality_terminal(
    measure_result: MeasureResult,
    analysis_result: AnalysisResult | None = None,
) -> str:
    """Render render_quality_terminal to a plain-text string (no colour)."""
    report_mod = sys.modules[render_quality_terminal.__module__]
    buf = io.StringIO()
    console = Console(
        file=buf,
        width=120,
        no_color=True,
        force_terminal=True,
        highlight=False,
        legacy_windows=False,
    )
    original_rprint = report_mod.rprint

    def _patched(*args: Any, **kw: Any) -> None:
        console.print(*args, **kw)

    report_mod.rprint = _patched  # type: ignore[attr-defined]
    try:
        render_quality_terminal(measure_result, result=analysis_result)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
    return " ".join(buf.getvalue().split())


# ---------------------------------------------------------------------------
# --judge (Tier-1) synthesis back-reference: terminal
# ---------------------------------------------------------------------------


class TestJudgeSynthesisTerminalBackReference:
    """Tier-1 synthesis lines in the terminal must echo the phrase the panel showed.

    When tier_drop <= 0 the panel said "same or better quality"; the confirmed
    verdict synthesis line must back-reference that exact phrase, not the
    step-down phrase "within tolerance" the user never saw.

    When tier_drop >= 1 the panel said "within tolerance"; the confirmed verdict
    must echo that phrase.
    """

    def test_judge_same_tier_confirmed_says_same_or_better(self) -> None:
        """tier_drop=0 → confirmed synthesis says 'same or better quality', not 'within tolerance'."""
        # Arrange
        mr = _confirmed_measure_result()
        ar = _result_same_tier()
        # Act
        out = _capture_quality_terminal(mr, ar)
        # Assert
        assert "same or better quality" in out, out
        assert "within tolerance" not in out, out

    def test_judge_better_tier_confirmed_says_same_or_better(self) -> None:
        """tier_drop=-1 → confirmed synthesis says 'same or better quality'."""
        mr = _confirmed_measure_result()
        ar = _result_better_tier()
        out = _capture_quality_terminal(mr, ar)
        assert "same or better quality" in out, out
        assert "within tolerance" not in out, out

    def test_judge_step_down_confirmed_says_within_tolerance(self) -> None:
        """tier_drop=1 → confirmed synthesis still says 'within tolerance'."""
        mr = _confirmed_measure_result()
        ar = _result_step_down()
        out = _capture_quality_terminal(mr, ar)
        assert "within tolerance" in out, out
        assert "same or better quality" not in out, out

    def test_judge_no_result_falls_back_to_within_tolerance(self) -> None:
        """No AnalysisResult supplied → safe fallback is 'within tolerance'."""
        mr = _confirmed_measure_result()
        out = _capture_quality_terminal(mr, None)
        assert "within tolerance" in out, out

    def test_judge_confirmed_still_says_confirmed(self) -> None:
        """The 'confirmed' status word must remain present regardless of tier_drop."""
        mr = _confirmed_measure_result()
        ar = _result_same_tier()
        out = _capture_quality_terminal(mr, ar)
        assert "confirmed" in out, out


# ---------------------------------------------------------------------------
# --judge (Tier-1) synthesis back-reference: Markdown / HTML reports
# ---------------------------------------------------------------------------


class TestJudgeSynthesisReportBackReference:
    """Tier-1 synthesis lines in the MD and HTML reports must echo the correct phrase.

    The MD/HTML surfaces call _classify_verdict directly (they don't go through
    _render_tier1_synthesis), so each needs its own assertion.  These tests
    exercise the render_markdown / render_html paths with a MeasureResult
    attached.
    """

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _md_with_measure(
        self,
        analysis_result: AnalysisResult,
        measure_result: MeasureResult,
        tmp_path: Path,
        *,
        v2: bool = False,
    ) -> str:
        """Render to Markdown and return full text.

        The public render_markdown / render_markdown_v2 functions accept
        *measure_result* as a separate keyword argument; it is NOT a field
        on AnalysisResult.  Pass it directly.
        """
        out = tmp_path / ("report_v2.md" if v2 else "report.md")
        if v2:
            render_markdown_v2(analysis_result, out, measure_result=measure_result)
        else:
            render_markdown(analysis_result, out, measure_result=measure_result)
        return out.read_text(encoding="utf-8")

    def test_md_judge_same_tier_confirmed_says_same_or_better(
        self, tmp_path: Path
    ) -> None:
        """tier_drop=0 → MD synthesis says 'same or better quality', not 'within tolerance'."""
        ar = _result_same_tier()
        mr = _confirmed_measure_result()
        text = self._md_with_measure(ar, mr, tmp_path)
        assert "same or better quality" in text, text[:2000]
        assert "within tolerance" not in text, text[:2000]

    def test_md_judge_better_tier_confirmed_says_same_or_better(
        self, tmp_path: Path
    ) -> None:
        """tier_drop=-1 → MD synthesis says 'same or better quality'."""
        ar = _result_better_tier()
        mr = _confirmed_measure_result()
        text = self._md_with_measure(ar, mr, tmp_path)
        assert "same or better quality" in text, text[:2000]
        assert "within tolerance" not in text, text[:2000]

    def test_md_judge_step_down_confirmed_says_within_tolerance(
        self, tmp_path: Path
    ) -> None:
        """tier_drop=1 → MD synthesis says 'within tolerance'."""
        ar = _result_step_down()
        mr = _confirmed_measure_result()
        text = self._md_with_measure(ar, mr, tmp_path)
        assert "within tolerance" in text, text[:2000]

    def test_md_v2_judge_same_tier_no_within_tolerance(self, tmp_path: Path) -> None:
        """tier_drop=0 via render_markdown_v2 → 'within tolerance' absent from synthesis."""
        ar = _result_same_tier()
        mr = _confirmed_measure_result()
        text = self._md_with_measure(ar, mr, tmp_path, v2=True)
        assert "within tolerance" not in text, text[:2000]

    # ------------------------------------------------------------------
    # HTML v1
    # ------------------------------------------------------------------

    def _html_with_measure(
        self,
        analysis_result: AnalysisResult,
        measure_result: MeasureResult,
        tmp_path: Path,
        *,
        v2: bool = False,
    ) -> str:
        """Render to HTML and return full text.

        The public render_html / render_html_v2 functions accept *measure_result*
        as a separate keyword argument; it is NOT a field on AnalysisResult.
        """
        out = tmp_path / ("report_v2.html" if v2 else "report.html")
        if v2:
            render_html_v2(analysis_result, out, measure_result=measure_result)
        else:
            render_html(analysis_result, out, measure_result=measure_result)
        return out.read_text(encoding="utf-8")

    def test_html_v1_judge_same_tier_confirmed_says_same_or_better(
        self, tmp_path: Path
    ) -> None:
        """tier_drop=0 → HTML v1 synthesis contains 'same or better quality', not 'within tolerance'."""
        ar = _result_same_tier()
        mr = _confirmed_measure_result()
        html = self._html_with_measure(ar, mr, tmp_path)
        assert "same or better quality" in html, html[:2000]
        assert "within tolerance" not in html, html[:2000]

    def test_html_v1_judge_step_down_confirmed_says_within_tolerance(
        self, tmp_path: Path
    ) -> None:
        """tier_drop=1 → HTML v1 synthesis still says 'within tolerance'."""
        ar = _result_step_down()
        mr = _confirmed_measure_result()
        html = self._html_with_measure(ar, mr, tmp_path)
        assert "within tolerance" in html, html[:2000]

    def test_html_v2_judge_same_tier_no_within_tolerance(
        self, tmp_path: Path
    ) -> None:
        """tier_drop=0 → HTML v2 synthesis contains 'same or better quality', absent 'within tolerance'."""
        ar = _result_same_tier()
        mr = _confirmed_measure_result()
        html = self._html_with_measure(ar, mr, tmp_path, v2=True)
        assert "same or better quality" in html, html[:2000]
        assert "within tolerance" not in html, html[:2000]


# ---------------------------------------------------------------------------
# --measure (Tier-0) framing back-reference: terminal
# ---------------------------------------------------------------------------


class TestMeasureTier0FramingBackReference:
    """The Tier-0 (--measure, no --judge) framing line echoes the correct phrase.

    The framing line is the single back-reference the user sees when they ran
    --measure but not --judge.  It must echo the phrase the routing panel
    showed — "same or better quality" when tier_drop <= 0, "within tolerance"
    for a step-down or unrated result.
    """

    def test_tier0_same_tier_framing_says_same_or_better(self) -> None:
        """tier_drop=0 → Tier-0 framing says 'same or better quality', not 'within tolerance'."""
        mr = _tier0_measure_result()
        ar = _result_same_tier()
        out = _capture_quality_terminal(mr, ar)
        assert "same or better quality" in out, out
        assert "within tolerance" not in out, out

    def test_tier0_better_tier_framing_says_same_or_better(self) -> None:
        """tier_drop=-1 → Tier-0 framing says 'same or better quality'."""
        mr = _tier0_measure_result()
        ar = _result_better_tier()
        out = _capture_quality_terminal(mr, ar)
        assert "same or better quality" in out, out
        assert "within tolerance" not in out, out

    def test_tier0_step_down_framing_says_within_tolerance(self) -> None:
        """tier_drop=1 → Tier-0 framing says 'within tolerance'."""
        mr = _tier0_measure_result()
        ar = _result_step_down()
        out = _capture_quality_terminal(mr, ar)
        assert "within tolerance" in out, out
        assert "same or better quality" not in out, out

    def test_tier0_no_result_falls_back_to_within_tolerance(self) -> None:
        """No AnalysisResult supplied → framing safely falls back to 'within tolerance'."""
        mr = _tier0_measure_result()
        out = _capture_quality_terminal(mr, None)
        assert "within tolerance" in out, out

    def test_tier0_framing_always_mentions_judge(self) -> None:
        """Tier-0 framing always tells the user to run --judge regardless of tier_drop."""
        mr = _tier0_measure_result()
        ar = _result_same_tier()
        out = _capture_quality_terminal(mr, ar)
        assert "--judge" in out, out
