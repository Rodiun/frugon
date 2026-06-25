"""Tests for frugon.report — terminal, HTML, and Markdown surfaces."""

from __future__ import annotations

import importlib
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from frugon.cost import AnalysisResult
from frugon.report import QUALITY_CAVEAT, render_html, render_markdown, render_terminal


def _result_with_candidate(**kwargs: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "total_calls": 1,
        "priced_calls": 1,
        "unpriced_calls": 0,
        "total_cost": Decimal("1.00"),
        "cost_by_model": {"gpt-4-turbo": Decimal("1.00")},
        "calls_by_model": {"gpt-4-turbo": 1},
        "projected_cost": Decimal("0.50"),
        "candidate_model": "gpt-4o",
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _result_no_candidate(**kwargs: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "total_calls": 1,
        "priced_calls": 1,
        "unpriced_calls": 0,
        "total_cost": Decimal("1.00"),
        "cost_by_model": {"gpt-4-turbo": Decimal("1.00")},
        "calls_by_model": {"gpt-4-turbo": 1},
        "projected_cost": Decimal("0"),
        "candidate_model": None,
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


# ---------------------------------------------------------------------------
# Terminal surface — quality caveat
# ---------------------------------------------------------------------------


def test_terminal_report_wholesale_quality_caveat_present(capsys: Any) -> None:
    """Arrange: a wholesale result with a routing recommendation.
    Act: render_terminal.
    Assert: the redesigned footer's amber quality caveat (assertion + action) shows.

    The wholesale headline now shares the split design language: the quality
    caveat lives in the footer as two amber lines (assertion + "run --measure …"),
    not the legacy single QUALITY_CAVEAT sentence.
    """
    from frugon.report import (
        QUALITY_NOT_VERIFIED_ACTION,
        QUALITY_NOT_VERIFIED_ASSERTION_WHOLESALE,
    )

    render_terminal(_result_with_candidate())

    out = " ".join(capsys.readouterr().out.split())
    assert " ".join(QUALITY_NOT_VERIFIED_ASSERTION_WHOLESALE.split()) in out
    assert " ".join(QUALITY_NOT_VERIFIED_ACTION.split()) in out


# ---------------------------------------------------------------------------
# Terminal surface — suppresses recommendation when no candidate (P3)
# ---------------------------------------------------------------------------


def test_terminal_report_suppresses_recommendation_when_no_candidate(
    monkeypatch: Any,
) -> None:
    """Arrange: result where no candidate is cheaper.
    Act: render_terminal.
    Assert: no 'Recommendation' line, no 'Save N%' hero, QUALITY_CAVEAT absent.
    """
    printed: list[object] = []
    monkeypatch.setattr("frugon.report.rprint", printed.append)

    render_terminal(_result_no_candidate())

    combined = " ".join(str(item) for item in printed)
    assert QUALITY_CAVEAT not in combined
    assert "Recommendation" not in combined
    # 100% saving must not appear (projected=0 means saving_pct=100 internally)
    assert "100%" not in combined


# ---------------------------------------------------------------------------
# HTML surface — quality caveat (P2)
# ---------------------------------------------------------------------------


def test_html_report_uses_quality_caveat_constant(tmp_path: Path) -> None:
    """Arrange: result with a routing recommendation.
    Act: render_html.
    Assert: QUALITY_CAVEAT text appears in the HTML file.
    """
    out = tmp_path / "report.html"
    render_html(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert QUALITY_CAVEAT in html


def test_html_report_suppresses_recommendation_when_no_candidate(
    tmp_path: Path,
) -> None:
    """Arrange: candidate_model is None.
    Act: render_html.
    Assert: QUALITY_CAVEAT absent; no 'Recommendation' hero present.
    """
    out = tmp_path / "report.html"
    render_html(_result_no_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert QUALITY_CAVEAT not in html
    assert "Recommendation" not in html


# ---------------------------------------------------------------------------
# Cross-platform line endings — report files must be LF-only on every OS
# ---------------------------------------------------------------------------


def test_html_report_written_with_lf_only_line_endings(tmp_path: Path) -> None:
    """Arrange: result with a routing recommendation.
    Act: render_html to a file.
    Assert: the on-disk bytes contain no carriage returns — the report is
            byte-identical across Linux, macOS and Windows.

    Without newline="\\n" on write_text, Windows text mode would translate
    every "\\n" to "\\r\\n", diverging from the *nix output.
    """
    out = tmp_path / "report.html"
    render_html(_result_with_candidate(), out)
    raw = out.read_bytes()
    assert b"\r" not in raw, (
        "HTML report must use LF-only line endings on every OS; "
        "found a carriage return in the written bytes"
    )


def test_markdown_report_written_with_lf_only_line_endings(tmp_path: Path) -> None:
    """Arrange: result with a routing recommendation.
    Act: render_markdown to a file.
    Assert: the on-disk bytes contain no carriage returns — the report is
            byte-identical across Linux, macOS and Windows.
    """
    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    raw = out.read_bytes()
    assert b"\r" not in raw, (
        "Markdown report must use LF-only line endings on every OS; "
        "found a carriage return in the written bytes"
    )


# ---------------------------------------------------------------------------
# Markdown surface — quality caveat (P2)
# ---------------------------------------------------------------------------


def test_markdown_report_uses_quality_caveat_constant(tmp_path: Path) -> None:
    """Arrange: result with a routing recommendation.
    Act: render_markdown.
    Assert: QUALITY_CAVEAT text appears in the Markdown file.
    """
    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert QUALITY_CAVEAT in md


def test_markdown_report_suppresses_recommendation_when_no_candidate(
    tmp_path: Path,
) -> None:
    """Arrange: candidate_model is None.
    Act: render_markdown.
    Assert: QUALITY_CAVEAT absent; no 'Recommendation' line.
    """
    out = tmp_path / "report.md"
    render_markdown(_result_no_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert QUALITY_CAVEAT not in md
    assert "Recommendation" not in md


# ---------------------------------------------------------------------------
# Import isolation -- Tier-0 default path must not require [measure] extra.
# ---------------------------------------------------------------------------


def test_report_importable_when_measure_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block frugon.measure in sys.modules and verify report still works.

    Privacy/scope invariant: frugon analyze (no --measure) must never
    require litellm or any optional dependency.
    """
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "frugon.measure", None)
    if "frugon.report" in _sys.modules:
        monkeypatch.delitem(_sys.modules, "frugon.report")

    report_mod = importlib.import_module("frugon.report")

    result = AnalysisResult(
        total_calls=1,
        priced_calls=1,
        unpriced_calls=0,
        total_cost=Decimal("0.0020"),
        cost_by_model={"gpt-4o": Decimal("0.0020")},
        calls_by_model={"gpt-4o": 1},
    )

    report_mod.render_terminal(result)

# ---------------------------------------------------------------------------
# FUNNEL_LINE constant -- structural invariant
# ---------------------------------------------------------------------------


def test_funnel_line_is_single_line() -> None:
    """Arrange/Act: import the FUNNEL_LINE constant.
    Assert: it contains no embedded newline characters.
    """
    from frugon.report import FUNNEL_LINE

    assert "\n" not in FUNNEL_LINE


# ---------------------------------------------------------------------------
# Terminal surface -- funnel line
# ---------------------------------------------------------------------------


def test_terminal_report_upsell_present_when_candidate(capsys: Any) -> None:
    """Arrange: a wholesale result with a routing recommendation.
    Act: render_terminal.
    Assert: the footer upsell line + URL are shown (normalised across Rich wrapping).

    The redesigned wholesale footer carries one upsell line ("Route every call
    automatically and hold the saving:  <url>"), the split design language's
    single funnel pointer — not the legacy FUNNEL_LINE sentence.
    """
    from frugon.report import FUNNEL_URL

    render_terminal(_result_with_candidate())

    # Rich may word-wrap long lines; normalise newlines+indent to spaces before matching.
    out = " ".join(capsys.readouterr().out.split())
    assert "Route every call automatically and hold the saving" in out
    assert FUNNEL_URL in out


def test_terminal_report_upsell_absent_when_no_candidate(capsys: Any) -> None:
    """Arrange: result with no candidate model.
    Act: render_terminal.
    Assert: no footer upsell — nothing to switch to means nothing to upsell.
    """
    from frugon.report import FUNNEL_URL

    render_terminal(_result_no_candidate())

    out = " ".join(capsys.readouterr().out.split())
    assert "Route every call automatically and hold the saving" not in out
    assert FUNNEL_URL not in out


def test_terminal_report_upsell_absent_when_suppress_caveat(capsys: Any) -> None:
    """Arrange: result with candidate; suppress_caveat=True (--measure path).
    Act: render_terminal with suppress_caveat=True.
    Assert: the whole footer (caveat + upsell) is suppressed for the measure path.
    """
    from frugon.report import FUNNEL_URL

    render_terminal(_result_with_candidate(), suppress_caveat=True)

    out = " ".join(capsys.readouterr().out.split())
    assert "Route every call automatically and hold the saving" not in out
    assert FUNNEL_URL not in out

def test_markdown_report_funnel_line_present_when_candidate(tmp_path: Any) -> None:
    """Arrange: result with a routing recommendation.
    Act: render_markdown.
    Assert: FUNNEL_LINE text and the markdown hyperlink to frugon.rodiun.io appear.
    """
    from frugon.report import FUNNEL_LINE

    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")

    assert FUNNEL_LINE in md
    assert "https://frugon.rodiun.io" in md


def test_markdown_report_funnel_line_absent_when_no_candidate(tmp_path: Any) -> None:
    """Arrange: result with no candidate model.
    Act: render_markdown.
    Assert: FUNNEL_LINE is absent from the file.
    """
    from frugon.report import FUNNEL_LINE

    out = tmp_path / "report.md"
    render_markdown(_result_no_candidate(), out)
    md = out.read_text(encoding="utf-8")

    assert FUNNEL_LINE not in md


# ---------------------------------------------------------------------------
# HTML surface -- funnel line
# ---------------------------------------------------------------------------


def test_html_report_funnel_line_present_when_candidate(tmp_path: Any) -> None:
    """Arrange: result with a routing recommendation.
    Act: render_html.
    Assert: FUNNEL_LINE text AND the href to frugon.rodiun.io both present.
    """
    from frugon.report import FUNNEL_LINE

    out = tmp_path / "report.html"
    render_html(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")

    assert FUNNEL_LINE in html
    assert 'href="https://frugon.rodiun.io"' in html


def test_html_report_funnel_line_absent_when_no_candidate(tmp_path: Any) -> None:
    """Arrange: result with no candidate model.
    Act: render_html.
    Assert: FUNNEL_LINE is absent from the file.
    """
    from frugon.report import FUNNEL_LINE

    out = tmp_path / "report.html"
    render_html(_result_no_candidate(), out)
    html = out.read_text(encoding="utf-8")

    assert FUNNEL_LINE not in html


# ---------------------------------------------------------------------------
# Fix #7 — no doubled parentheses in cost period label
# ---------------------------------------------------------------------------


def test_markdown_report_no_doubled_parentheses_in_cost_period(tmp_path: Path) -> None:
    """Arrange: result with window_days and monthly_cost set.
    Act: render_markdown.
    Assert: the doubled-paren form '(monthly (projected))' is absent;
            the clean single-paren form '(monthly projection)' is present
            in the monthly_cost row.
    """
    from decimal import Decimal

    out = tmp_path / "report.md"
    render_markdown(
        _result_with_candidate(window_days=30, monthly_cost=Decimal("1.00")),
        out,
    )
    md = out.read_text(encoding="utf-8")

    assert "(monthly (projected))" not in md
    assert "(monthly projection)" in md


def test_markdown_report_no_doubled_parentheses_with_span_days(tmp_path: Path) -> None:
    """Arrange: result with observed_span_days and monthly_cost set.
    Act: render_markdown.
    Assert: the doubled-paren form is absent; clean form is present.
    """
    from decimal import Decimal

    out = tmp_path / "report.md"
    render_markdown(
        _result_with_candidate(observed_span_days=14.5, monthly_cost=Decimal("2.07")),
        out,
    )
    md = out.read_text(encoding="utf-8")

    assert "(monthly (projected))" not in md
    assert "(monthly projection)" in md


def test_html_report_no_doubled_parentheses_in_cost_period(tmp_path: Path) -> None:
    """Arrange: result with window_days set.
    Act: render_html.
    Assert: the doubled-paren form '(monthly (projected))' is absent;
            the phrase 'monthly projection' appears in the HTML (case-insensitive).
    """
    out = tmp_path / "report.html"
    render_html(_result_with_candidate(window_days=30), out)
    html = out.read_text(encoding="utf-8")

    assert "(monthly (projected))" not in html
    assert "monthly projection" in html.lower()


# ---------------------------------------------------------------------------
# Fix #6 — swap line names both baseline and candidate
# ---------------------------------------------------------------------------


def test_markdown_report_swap_line_names_baseline_and_candidate(tmp_path: Path) -> None:
    """Arrange: result with a dominant baseline model and a candidate.
    Act: render_markdown.
    Assert: the output contains both the baseline model name and the candidate
            model name on the recommended-swap line (e.g. 'gpt-4-turbo → gpt-4o').
    """
    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")

    # _result_with_candidate uses cost_by_model={"gpt-4-turbo": 1.00}, candidate="gpt-4o"
    assert "gpt-4-turbo" in md
    assert "gpt-4o" in md
    # The swap arrow must appear on the same line
    swap_line = next(
        (line for line in md.splitlines() if "→" in line or "->" in line), None
    )
    assert swap_line is not None, "Expected a swap line containing '→' or '->'"
    assert "gpt-4-turbo" in swap_line
    assert "gpt-4o" in swap_line


def test_markdown_report_no_bare_candidate_model_label(tmp_path: Path) -> None:
    """Arrange: result with candidate.
    Act: render_markdown.
    Assert: the old 'Candidate model:' label is gone; swap label replaces it.
    """
    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")

    assert "Candidate model:" not in md
    assert "Recommended swap:" in md


def test_html_report_swap_label_names_baseline_and_candidate(tmp_path: Path) -> None:
    """Arrange: result with a dominant baseline model and a candidate.
    Act: render_html.
    Assert: the HTML contains both the baseline and candidate model names
            in the context of a 'Recommended swap' label.
    """
    out = tmp_path / "report.html"
    render_html(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")

    assert "Recommended swap" in html
    assert "gpt-4-turbo" in html
    assert "gpt-4o" in html
    # Arrow (HTML-escaped or literal) must appear between them
    assert "→" in html or "-&gt;" in html or "->" in html


def test_html_report_no_bare_candidate_model_label(tmp_path: Path) -> None:
    """Arrange: result with candidate.
    Act: render_html.
    Assert: the old 'Candidate model' stat-label is gone; 'Recommended swap' replaces it.
    """
    out = tmp_path / "report.html"
    render_html(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")

    assert "Candidate model" not in html
    assert "Recommended swap" in html


# ---------------------------------------------------------------------------
# brand-mark — the wordmark dot CSS class
# ---------------------------------------------------------------------------


def test_html_uses_brand_mark_class(tmp_path: Path) -> None:
    """Arrange: normal analysis result.
    Act: render_html.
    Assert: the 'brand-mark' CSS class is present in the rendered wordmark.
    """
    out = tmp_path / "report.html"
    render_html(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")

    assert "brand-mark" in html


def test_css_uses_brand_mark_selector(tmp_path: Path) -> None:
    """Assert: the CSS block defines the .brand-mark selector."""
    out = tmp_path / "report.html"
    render_html(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")

    assert ".brand-mark" in html


# ---------------------------------------------------------------------------
# P0 — monthly rows shown in all renderers
# ---------------------------------------------------------------------------


def test_terminal_monthly_cadence_shown_when_window_set(
    capsys: Any,
) -> None:
    """Arrange: a wholesale result with a monthly projection (window_days=7 path).
    Act: render_terminal.
    Assert: the redesigned panel shows the monthly cadence as a '/ mo' suffix.

    The redesigned panel carries the monthly cadence on the spend figures via a
    '/ mo' suffix (shown only when a projection is available) rather than a
    separate "monthly cost" label row.
    """
    from decimal import Decimal

    render_terminal(
        _result_with_candidate(
            window_days=7,
            monthly_cost=Decimal("4.29"),
            monthly_projected=Decimal("2.15"),
        )
    )

    out = " ".join(capsys.readouterr().out.split())
    assert "/ mo" in out


def test_markdown_monthly_cost_row_present_when_set(tmp_path: Path) -> None:
    """Arrange: result with monthly_cost set.
    Act: render_markdown.
    Assert: markdown contains a 'Monthly cost' line.
    """
    from decimal import Decimal

    out = tmp_path / "report.md"
    render_markdown(
        _result_with_candidate(window_days=7, monthly_cost=Decimal("4.29")),
        out,
    )
    md = out.read_text(encoding="utf-8")

    assert "Monthly cost" in md


def test_html_monthly_cost_stat_present_when_set(tmp_path: Path) -> None:
    """Arrange: result with monthly_cost set.
    Act: render_html.
    Assert: HTML contains 'Monthly cost' stat label.
    """
    from decimal import Decimal

    out = tmp_path / "report.html"
    render_html(
        _result_with_candidate(window_days=7, monthly_cost=Decimal("4.29")),
        out,
    )
    html = out.read_text(encoding="utf-8")

    assert "Monthly cost" in html


def test_markdown_monthly_cost_absent_when_not_set(tmp_path: Path) -> None:
    """Arrange: result without monthly_cost (no window, no span).
    Act: render_markdown.
    Assert: 'Monthly cost' line is absent.
    """
    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")

    assert "Monthly cost" not in md


def test_total_cost_labeled_observed_in_markdown(tmp_path: Path) -> None:
    """Assert: the current cost row uses 'across analyzed calls' label, not
    'monthly projection' — total_cost is always the raw observed total.
    """
    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")

    assert "across analyzed calls" in md


# ---------------------------------------------------------------------------
# P2 — CC-BY attribution in report footer
# ---------------------------------------------------------------------------


def test_attribution_present_in_html_when_candidate_and_attribution_available(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Arrange: result with a candidate; mock get_attribution to return a string.
    Act: render_html.
    Assert: attribution text appears in the HTML quality card.
    """
    _SENTINEL = "QualityTiersSentinel-XYZ"
    monkeypatch.setitem(render_html.__globals__, "_get_attribution", lambda: _SENTINEL)
    out = tmp_path / "report.html"
    render_html(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")

    assert _SENTINEL in html


def test_attribution_absent_in_html_when_no_candidate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Arrange: result with no candidate; attribution returns a value.
    Act: render_html.
    Assert: attribution is absent (quality card not shown without candidate).
    """
    _SENTINEL = "QualityTiersSentinel-XYZ"
    monkeypatch.setitem(render_html.__globals__, "_get_attribution", lambda: _SENTINEL)
    out = tmp_path / "report.html"
    render_html(_result_no_candidate(), out)
    html = out.read_text(encoding="utf-8")

    assert _SENTINEL not in html


def test_attribution_present_in_markdown_when_candidate_and_attribution_available(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Arrange: result with a candidate; mock get_attribution.
    Act: render_markdown.
    Assert: attribution line appears in the markdown output.
    """
    _SENTINEL = "QualityTiersSentinel-XYZ"
    monkeypatch.setitem(render_markdown.__globals__, "_get_attribution", lambda: _SENTINEL)
    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")

    assert _SENTINEL in md


def test_attribution_absent_in_markdown_when_get_attribution_returns_none(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Arrange: get_attribution returns None (quality file absent or no field).
    Act: render_markdown.
    Assert: no attribution line appears.

    Note: patches render_markdown.__globals__ directly so the correct module's
    namespace is reached regardless of re-import side effects from other tests.
    """
    monkeypatch.setitem(render_markdown.__globals__, "_get_attribution", lambda: None)
    out = tmp_path / "report.md"
    render_markdown(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")

    assert "_Source:" not in md


# ---------------------------------------------------------------------------
# P2 — _fmt_usd adaptive precision
# ---------------------------------------------------------------------------


def test_fmt_usd_sub_0001_uses_6_decimals() -> None:
    """Arrange: amount smaller than $0.0001 (e.g. $0.000001).
    Assert: _fmt_usd uses 6 decimal places so it doesn't show as $0.0000.
    """
    from decimal import Decimal

    from frugon.report import _fmt_usd

    result = _fmt_usd(Decimal("0.000001"))
    assert result == "$0.000001"
    assert result != "$0.0000"  # must not be truncated to zero


def test_fmt_usd_sub_cent_uses_4_decimals() -> None:
    """Arrange: amount in [$0.0001, $0.01) — sub-cent but not sub-$0.0001.
    Assert: _fmt_usd uses 4 decimal places so the value is not silently rounded
    to $0.00 (which would be a 100% error for a $0.005 amount).
    """
    from decimal import Decimal

    from frugon.report import _fmt_usd

    assert _fmt_usd(Decimal("0.0025")) == "$0.0025"
    assert _fmt_usd(Decimal("0.0050")) == "$0.0050"
    assert _fmt_usd(Decimal("0.0099")) == "$0.0099"


def test_fmt_usd_normal_amount_uses_2_decimals() -> None:
    """Arrange: amount >= $0.01.
    Assert: _fmt_usd uses 2 decimal places with ROUND_HALF_UP.
    """
    from decimal import Decimal

    from frugon.report import _fmt_usd

    assert _fmt_usd(Decimal("1.50")) == "$1.50"
    assert _fmt_usd(Decimal("389.8849")) == "$389.88"
    assert _fmt_usd(Decimal("389.885")) == "$389.89"  # half-up rounds away from zero
    assert _fmt_usd(Decimal("0.01")) == "$0.01"
    assert _fmt_usd(Decimal("100.00")) == "$100.00"


def test_fmt_usd_zero_uses_2_decimals() -> None:
    """Assert: zero amount uses 2 decimal places ($0.00)."""
    from decimal import Decimal

    from frugon.report import _fmt_usd

    assert _fmt_usd(Decimal("0")) == "$0.00"


def test_html_report_uses_fmt_usd_for_sub_cent_cost(tmp_path: Path) -> None:
    """Arrange: a result whose cost is sub-$0.0001.
    Act: render_html.
    Assert: the HTML shows 6 decimal places, not truncated $0.0000.
    """
    from decimal import Decimal

    out = tmp_path / "report.html"
    result = _result_no_candidate(
        total_cost=Decimal("0.000005"),
        cost_by_model={"gpt-4o-mini": Decimal("0.000005")},
    )
    render_html(result, out)
    html = out.read_text(encoding="utf-8")

    assert "$0.000005" in html
    assert "$0.0000" not in html or "$0.000005" in html
