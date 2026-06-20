"""Tests for the Quality-tier disclosure in the split report (all surfaces).

The split report shows the published LMArena quality CLASS the routing moves
between — ``baseline: <tier>  ->  candidate: <tier>  (LMArena)`` — so a reader
sees the quality class the routing trades between, complementing "within
tolerance" (the offline heuristic) and ``--measure`` (the measured verdict).

These tests pin three things:

  * the line is PRESENT on every surface (terminal, HTML v1/v2, Markdown);
  * the labels are CORRECT — each side equals the shared :func:`_tier_label`
    for that model, so the four surfaces can never drift from each other or
    from the live quality table; and
  * an UNRATED model renders ``unrated`` rather than being dropped (the gap is
    marked, not hidden).

A separate test asserts no pure-white text colour survives in the report CSS:
the report's "white" body text must use the landing page's softer off-white
(#F7F7F7), never #fff / #ffffff / pure ``white`` / rgb(255,255,255).
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from frugon.cost import AnalysisResult
from frugon.report import (
    _HTML_CSS,
    _HTML_CSS_V2,
    _QUALITY_HTML_CSS,
    _tier_label,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
    render_terminal,
)
from frugon.routing import SplitRouting

# An obviously-unknown model name guaranteed to be absent from the LMArena
# quality table, so its tier is always UNRATED regardless of table sync state.
_UNRATED_MODEL = "frobnicator-x-9000-never-rated"


def _split(**kw: Any) -> SplitRouting:
    defaults: dict[str, Any] = {
        "baseline_model": "gpt-4-turbo",
        "candidate_model": "gpt-4o-mini",
        "routed_count": 24,
        "kept_count": 3,
        "routed_cost": Decimal("0.0004"),
        "kept_cost": Decimal("0.0435"),
        "baseline_cost": Decimal("0.0650"),
        "blended_cost": Decimal("0.0439"),
        "easy_threshold": Decimal("0.35"),
        "monthly_baseline": Decimal("0.2788"),
        "monthly_blended": Decimal("0.1881"),
    }
    defaults.update(kw)
    return SplitRouting(**defaults)


def _result(split: SplitRouting, **kw: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "total_calls": 37,
        "priced_calls": 37,
        "unpriced_calls": 0,
        "total_cost": Decimal("0.0676"),
        "cost_by_model": {
            split.baseline_model: Decimal("0.0650"),
            "gpt-4o": Decimal("0.0026"),
        },
        "calls_by_model": {split.baseline_model: 27, "gpt-4o": 10},
        "projected_cost": Decimal("0.0222"),
        "candidate_model": "gpt-4o",
        "observed_span_days": 7.0,
        "split": split,
    }
    defaults.update(kw)
    return AnalysisResult(**defaults)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _render_terminal_text(result: AnalysisResult) -> str:
    """Render the terminal split view to width-stable, ANSI-stripped text.

    ``force_terminal=True`` makes Rich emit style escapes even under ``no_color``;
    strip them so substring assertions see the plain text a screenshot would show
    (the same approach the parity tests use).
    """
    import io
    import sys

    from rich.console import Console

    report_mod = sys.modules[render_terminal.__module__]
    buf = io.StringIO()
    console = Console(file=buf, width=100, no_color=True, force_terminal=True, highlight=False)
    original_rprint = report_mod.rprint
    original_render_console = report_mod._render_console
    report_mod.rprint = lambda *a, **k: console.print(*a, **k)  # type: ignore[attr-defined]
    report_mod._render_console = lambda: console  # type: ignore[attr-defined]
    try:
        render_terminal(result)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    return _ANSI_RE.sub("", buf.getvalue())


# ---------------------------------------------------------------------------
# Presence + correct labels on every surface
# ---------------------------------------------------------------------------


class TestQualityTierLine:
    def test_terminal_quality_tier_line_present_with_correct_labels(self) -> None:
        split = _split()
        out = " ".join(_render_terminal_text(_result(split)).split())
        base_label = _tier_label(split.baseline_model)
        cand_label = _tier_label(split.candidate_model)
        assert "Quality tier" in out
        assert f"{split.baseline_model}: {base_label}" in out
        assert f"{split.candidate_model}: {cand_label}" in out
        assert "(LMArena)" in out

    def test_markdown_v1_quality_tier_line_present_with_correct_labels(
        self, tmp_path: Path
    ) -> None:
        split = _split()
        out_path = tmp_path / "r.md"
        render_markdown(_result(split), out_path)
        text = out_path.read_text(encoding="utf-8")
        base_label = _tier_label(split.baseline_model)
        cand_label = _tier_label(split.candidate_model)
        assert "**Quality tier:**" in text
        assert (
            f"`{split.baseline_model}` {base_label} → "
            f"`{split.candidate_model}` {cand_label} (LMArena)"
        ) in text

    def test_markdown_v2_quality_tier_line_present(self, tmp_path: Path) -> None:
        split = _split()
        out_path = tmp_path / "r.md"
        render_markdown_v2(_result(split), out_path)
        text = out_path.read_text(encoding="utf-8")
        assert "**Quality tier:**" in text
        assert "(LMArena)" in text

    def test_html_v1_quality_tier_line_present_with_correct_labels(
        self, tmp_path: Path
    ) -> None:
        split = _split()
        out_path = tmp_path / "r.html"
        render_html(_result(split), out_path)
        html = out_path.read_text(encoding="utf-8")
        base_label = _tier_label(split.baseline_model)
        cand_label = _tier_label(split.candidate_model)
        assert "Quality tier" in html
        # Model names ride the cyan .model-name class; labels follow each one.
        assert f'<span class="model-name">{split.baseline_model}</span>: {base_label}' in html
        assert f'<span class="model-name">{split.candidate_model}</span>: {cand_label}' in html
        assert "(LMArena)" in html

    def test_html_v2_quality_tier_line_present_with_correct_labels(
        self, tmp_path: Path
    ) -> None:
        split = _split()
        out_path = tmp_path / "r.html"
        render_html_v2(_result(split), out_path)
        html = out_path.read_text(encoding="utf-8")
        base_label = _tier_label(split.baseline_model)
        cand_label = _tier_label(split.candidate_model)
        assert "Quality tier" in html
        # Model names ride the cyan .route-to class on the v2 surface.
        assert f'<span class="route-to">{split.baseline_model}</span>: {base_label}' in html
        assert f'<span class="route-to">{split.candidate_model}</span>: {cand_label}' in html
        assert "(LMArena)" in html

    def test_terminal_order_upper_bound_then_quality_tier_then_prices(self) -> None:
        """Terminal: Upper bound → Quality tier → Prices, in that order.

        The Quality-tier benchmark comparison must sit BELOW the Upper-bound swap
        context and ABOVE the Prices freshness line — the decision context grouped
        together, the freshness metadata after it.
        """
        split = _split()
        result = _result(
            split,
            pricing_json_last_synced="2026-06-01",
            quality_json_last_synced="2026-06-01",
        )
        out = _render_terminal_text(result)
        i_upper = out.index("Upper bound")
        i_tier = out.index("Quality tier")
        i_prices = out.index("Prices")
        assert i_upper < i_tier < i_prices, (
            f"expected Upper bound < Quality tier < Prices; "
            f"got {i_upper}, {i_tier}, {i_prices}"
        )

    def test_markdown_v1_order_quality_tier_above_prices(self, tmp_path: Path) -> None:
        """Markdown v1 Details: Quality tier above the Pricing last-synced line."""
        split = _split()
        out_path = tmp_path / "r.md"
        render_markdown(
            _result(
                split,
                pricing_json_last_synced="2026-06-01",
                quality_json_last_synced="2026-06-01",
            ),
            out_path,
        )
        text = out_path.read_text(encoding="utf-8")
        assert text.index("**Quality tier:**") < text.index("Pricing last synced")

    def test_html_v1_order_quality_tier_above_prices(self, tmp_path: Path) -> None:
        """HTML v1: the Quality-tier note precedes the Pricing-last-synced stat."""
        split = _split()
        out_path = tmp_path / "r.html"
        render_html(
            _result(
                split,
                pricing_json_last_synced="2026-06-01",
                quality_json_last_synced="2026-06-01",
            ),
            out_path,
        )
        html = out_path.read_text(encoding="utf-8")
        assert html.index("Quality tier") < html.index("Pricing last synced")

    def test_html_v2_order_quality_tier_above_prices(self, tmp_path: Path) -> None:
        """HTML v2: the Quality-tier meta-line precedes the pricing-synced meta."""
        split = _split()
        out_path = tmp_path / "r.html"
        render_html_v2(
            _result(
                split,
                pricing_json_last_synced="2026-06-01",
                quality_json_last_synced="2026-06-01",
            ),
            out_path,
        )
        html = out_path.read_text(encoding="utf-8")
        assert html.index("Quality tier") < html.index("pricing synced")

    def test_tiers_reconcile_across_all_surfaces(self, tmp_path: Path) -> None:
        """The baseline + candidate tier labels are IDENTICAL on every surface."""
        split = _split()
        result = _result(split)
        base_label = _tier_label(split.baseline_model)
        cand_label = _tier_label(split.candidate_model)

        terminal = _render_terminal_text(result)
        md1 = tmp_path / "a.md"
        md2 = tmp_path / "b.md"
        h1 = tmp_path / "a.html"
        h2 = tmp_path / "b.html"
        render_markdown(result, md1)
        render_markdown_v2(result, md2)
        render_html(result, h1)
        render_html_v2(result, h2)

        for surface in (
            terminal,
            md1.read_text(encoding="utf-8"),
            md2.read_text(encoding="utf-8"),
            h1.read_text(encoding="utf-8"),
            h2.read_text(encoding="utf-8"),
        ):
            # Every surface carries the same "Quality tier" disclosure with both
            # model names and BOTH tier labels resolved from the shared helper —
            # so no surface can show a different class than another.
            assert "Quality tier" in surface
            assert split.baseline_model in surface
            assert split.candidate_model in surface
            assert base_label in surface
            assert cand_label in surface


# ---------------------------------------------------------------------------
# Unrated handling — the gap is marked, never dropped
# ---------------------------------------------------------------------------


class TestQualityTierUnrated:
    def test_tier_label_unrated_model_reads_unrated(self) -> None:
        assert _tier_label(_UNRATED_MODEL) == "unrated"

    def test_terminal_unrated_candidate_reads_unrated(self) -> None:
        split = _split(candidate_model=_UNRATED_MODEL)
        out = " ".join(_render_terminal_text(_result(split)).split())
        assert f"{_UNRATED_MODEL}: unrated" in out

    def test_markdown_unrated_candidate_reads_unrated(self, tmp_path: Path) -> None:
        split = _split(candidate_model=_UNRATED_MODEL)
        out_path = tmp_path / "r.md"
        render_markdown(_result(split), out_path)
        text = out_path.read_text(encoding="utf-8")
        assert f"`{_UNRATED_MODEL}` unrated" in text

    def test_html_v2_unrated_candidate_reads_unrated(self, tmp_path: Path) -> None:
        split = _split(candidate_model=_UNRATED_MODEL)
        out_path = tmp_path / "r.html"
        render_html_v2(_result(split), out_path)
        html = out_path.read_text(encoding="utf-8")
        assert f'<span class="route-to">{_UNRATED_MODEL}</span>: unrated' in html


# ---------------------------------------------------------------------------
# Colour discipline — no pure-white text in the report CSS
# ---------------------------------------------------------------------------


# Matches a pure-white TEXT colour: hex #fff / #ffffff, the keyword ``white``
# (but NOT the ``white-space`` layout property), or rgb(255,255,255).  Hunted in
# the report CSS so the report reads with the landing page's softer off-white
# (#F7F7F7) rather than harsh pure white.
_PURE_WHITE_TEXT = re.compile(
    r"#fff\b|#ffffff\b|\bwhite\b(?!-space)|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\)",
    re.IGNORECASE,
)


class TestReportCssNoPureWhiteText:
    def test_html_v1_css_has_no_pure_white_text(self) -> None:
        assert _PURE_WHITE_TEXT.search(_HTML_CSS) is None
        # The off-white the landing page uses is the report's primary ink.
        assert "--ink:#F7F7F7" in _HTML_CSS

    def test_html_v2_css_has_no_pure_white_text(self) -> None:
        assert _PURE_WHITE_TEXT.search(_HTML_CSS_V2) is None
        assert "--ink:#F7F7F7" in _HTML_CSS_V2
        # Muted/dim tones align to the landing page values too.
        assert "--ink-mute:#A1A1AA" in _HTML_CSS_V2
        assert "--ink-dim:#6B6B72" in _HTML_CSS_V2

    def test_quality_section_css_has_no_pure_white_text(self) -> None:
        assert _PURE_WHITE_TEXT.search(_QUALITY_HTML_CSS) is None

    def test_rendered_reports_have_no_pure_white_text(self, tmp_path: Path) -> None:
        """End-to-end: neither rendered HTML variant emits a pure-white text colour."""
        result = _result(_split())
        h1 = tmp_path / "a.html"
        h2 = tmp_path / "b.html"
        render_html(result, h1)
        render_html_v2(result, h2)
        for html in (h1.read_text(encoding="utf-8"), h2.read_text(encoding="utf-8")):
            assert _PURE_WHITE_TEXT.search(html) is None
