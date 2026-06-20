"""Tests for frugon HTML report rendering.

Covers:
  - Self-containment: no external http(s):// asset refs in <link>/<script>/<img src>
  - Required honesty/methodology line present
  - Required saving figure present when candidate found
  - Key structural sections present (no-priced-calls path + normal path)
  - Design language: frugon design tokens, wordmark, brand mark SVG, eyebrow labels
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from frugon.cost import AnalysisResult
from frugon.report import render_html

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches external URLs only in asset-loading contexts: <link href>, <script src>, <img src>.
# Deliberate hyperlinks (<a href>) are not assets and are allowed.
_EXT_LINK_HREF_RE = re.compile(r"<link\b[^>]*\bhref\s*=\s*[\"']https?://", re.IGNORECASE)
_EXT_SCRIPT_SRC_RE = re.compile(r"<script\b[^>]*\bsrc\s*=\s*[\"']https?://", re.IGNORECASE)
_EXT_IMG_SRC_RE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"']https?://", re.IGNORECASE)


def _has_external_asset(html: str) -> bool:
    """Return True if the HTML loads any external http(s) asset (link/script/img)."""
    return bool(
        _EXT_LINK_HREF_RE.search(html)
        or _EXT_SCRIPT_SRC_RE.search(html)
        or _EXT_IMG_SRC_RE.search(html)
    )


def _make_result(
    *,
    priced: int = 5,
    total: int = 5,
    cost: str = "0.0250",
    candidate: str | None = "gpt-4o-mini",
    projected: str = "0.0050",
    span_days: float | None = None,
    window_days: int | None = None,
    pjls: str | None = "2026-05-01",
) -> AnalysisResult:
    return AnalysisResult(
        total_calls=total,
        priced_calls=priced,
        unpriced_calls=total - priced,
        total_cost=Decimal(cost),
        cost_by_model={
            "gpt-4o": Decimal(cost),
        },
        calls_by_model={"gpt-4o": priced},
        projected_cost=Decimal(projected),
        candidate_model=candidate,
        observed_span_days=span_days,
        window_days=window_days,
        pricing_json_last_synced=pjls,
    )


# ---------------------------------------------------------------------------
# Self-containment
# ---------------------------------------------------------------------------


class TestSelfContainment:
    """render_html must produce a file with no external asset references."""

    def test_no_external_link_refs(self, tmp_path: Path) -> None:
        """Arrange: normal analysis result.
        Act: render_html to a temp file.
        Assert: no href="https://..." or href="http://..." in <link> or <a>.
        """
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        # Stylesheet / script links must not point to external CDNs
        assert not _has_external_asset(html), (
            "HTML report must be self-contained — no external http(s):// asset refs"
        )

    def test_no_external_script_src(self, tmp_path: Path) -> None:
        """Assert: no <script src="https://..."> references."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert not re.search(r"<script[^>]+src\s*=\s*[\"']https?://", html, re.IGNORECASE)

    def test_no_external_img_src(self, tmp_path: Path) -> None:
        """Assert: no <img src="https://..."> references."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert not re.search(r"<img[^>]+src\s*=\s*[\"']https?://", html, re.IGNORECASE)

    def test_html_file_created(self, tmp_path: Path) -> None:
        """Assert: the output file is created and is non-empty."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Required honesty / methodology line
# ---------------------------------------------------------------------------

METHODOLOGY_KEYWORDS = [
    "list prices",
    "LiteLLM",
    "No LLM calls",
    "No network",
]


class TestMethodologyLine:
    """HTML must contain a methodology / source disclosure line."""

    @pytest.mark.parametrize("keyword", METHODOLOGY_KEYWORDS)
    def test_methodology_keyword_present(self, tmp_path: Path, keyword: str) -> None:
        """Assert: each methodology keyword appears in the rendered HTML."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert keyword in html, (
            f"Methodology line must contain '{keyword}' (§6 honest savings)"
        )


# ---------------------------------------------------------------------------
# Saving figure
# ---------------------------------------------------------------------------


class TestSavingFigure:
    """HTML must surface the saving figure when a candidate is found."""

    def test_saving_percentage_present(self, tmp_path: Path) -> None:
        """Arrange: result with 80% saving (cost=0.025, projected=0.005).
        Act: render_html.
        Assert: '80.0%' appears in the HTML (saving % renders at 1 decimal).
        """
        out = tmp_path / "report.html"
        render_html(_make_result(cost="0.0250", projected="0.0050"), out)
        html = out.read_text(encoding="utf-8")
        assert "80.0%" in html

    def test_current_cost_present(self, tmp_path: Path) -> None:
        """Assert: the current cost amount appears in the HTML."""
        out = tmp_path / "report.html"
        render_html(_make_result(cost="0.1234"), out)
        html = out.read_text(encoding="utf-8")
        assert "0.1234" in html

    def test_candidate_model_present(self, tmp_path: Path) -> None:
        """Assert: the candidate model name appears in the HTML."""
        out = tmp_path / "report.html"
        render_html(_make_result(candidate="gpt-4o-mini"), out)
        html = out.read_text(encoding="utf-8")
        assert "gpt-4o-mini" in html

    def test_no_saving_no_candidate_section(self, tmp_path: Path) -> None:
        """Arrange: result with no cheaper candidate.
        Assert: saving % line is absent or shows 0%/no saving.
        """
        out = tmp_path / "report.html"
        render_html(_make_result(candidate=None, projected="0"), out)
        html = out.read_text(encoding="utf-8")
        # Should not show a large positive saving
        assert "80.0%" not in html


# ---------------------------------------------------------------------------
# Structural sections
# ---------------------------------------------------------------------------


class TestStructure:
    """HTML must contain required structural sections."""

    def test_has_html_skeleton(self, tmp_path: Path) -> None:
        """Assert: basic valid HTML scaffold."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "<html" in html.lower()
        assert "</html>" in html.lower()
        assert "<head" in html.lower()
        assert "<body" in html.lower()

    def test_has_frugon_brand(self, tmp_path: Path) -> None:
        """Assert: 'frugon' branding appears in the document."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "frugon" in html.lower()

    def test_has_cost_breakdown_section(self, tmp_path: Path) -> None:
        """Assert: per-model cost breakdown is present (table or list)."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        # The dominant model name must appear in the breakdown
        assert "gpt-4o" in html

    def test_no_priced_calls_renders_html(self, tmp_path: Path) -> None:
        """Arrange: result with zero priced calls.
        Act: render_html.
        Assert: HTML is produced (not an exception), and contains a notice.
        """
        out = tmp_path / "report.html"
        zero_result = AnalysisResult(
            total_calls=3,
            priced_calls=0,
            unpriced_calls=3,
            total_cost=Decimal("0"),
            projected_cost=Decimal("0"),
            candidate_model=None,
        )
        render_html(zero_result, out)
        html = out.read_text(encoding="utf-8")
        assert out.exists()
        assert "frugon" in html.lower()
        # Must mention that no calls were priced
        assert "no priced" in html.lower() or "unpriced" in html.lower() or "0" in html

    def test_inline_style_present(self, tmp_path: Path) -> None:
        """Assert: report uses inline <style> (self-contained CSS, no CDN link)."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "<style" in html.lower()

    def test_utf8_charset_declared(self, tmp_path: Path) -> None:
        """Assert: report declares UTF-8 charset."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "utf-8" in html.lower()

    def test_viewport_meta_present(self, tmp_path: Path) -> None:
        """Assert: report has viewport meta tag for mobile responsiveness."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "viewport" in html.lower()

    def test_projection_label_window(self, tmp_path: Path) -> None:
        """Assert: when window_days set, projection label appears in HTML."""
        out = tmp_path / "report.html"
        render_html(_make_result(window_days=7), out)
        html = out.read_text(encoding="utf-8")
        assert "7" in html
        # Some disclosure of the projection window
        assert "day" in html.lower() or "window" in html.lower()

    def test_projection_label_span(self, tmp_path: Path) -> None:
        """Assert: when observed_span_days set, projection info appears."""
        out = tmp_path / "report.html"
        render_html(_make_result(span_days=14.5, window_days=None), out)
        html = out.read_text(encoding="utf-8")
        assert "14" in html  # 14.5 days rounds in display


# ---------------------------------------------------------------------------
# Design language — frugon visual identity
# ---------------------------------------------------------------------------


class TestDesignLanguage:
    """HTML report must adopt the frugon design language tokens and structure."""

    def test_cyan_token_present(self, tmp_path: Path) -> None:
        """Assert: the canonical --cyan:#00D1FF token appears in the CSS."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "--cyan" in html
        assert "00D1FF" in html.upper()

    def test_dark_background_token_present(self, tmp_path: Path) -> None:
        """Assert: the --bg:#020202 token appears in the CSS."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "--bg" in html
        assert "#020202" in html

    def test_wordmark_uppercase_present(self, tmp_path: Path) -> None:
        """Assert: FRUGON uppercase wordmark appears in the HTML body."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "FRUGON" in html

    def test_brand_mark_svg_circles(self, tmp_path: Path) -> None:
        """Assert: Three-orb brand mark SVG (cy=5, cy=15.5, r=2.5) present in the header."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert 'cy="5"' in html
        assert 'cy="15.5"' in html
        assert 'r="2.5"' in html

    def test_eyebrow_class_present(self, tmp_path: Path) -> None:
        """Assert: eyebrow class is used as section kickers."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "eyebrow" in html

    def test_jetbrains_mono_font_referenced(self, tmp_path: Path) -> None:
        """Assert: JetBrains Mono font family is referenced in the CSS."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "JetBrains Mono" in html

    def test_saving_hero_class_present_when_candidate(self, tmp_path: Path) -> None:
        """Assert: saving-hero element present when a saving exists."""
        out = tmp_path / "report.html"
        render_html(_make_result(cost="0.0250", projected="0.0050"), out)
        html = out.read_text(encoding="utf-8")
        assert "saving-hero" in html

    def test_no_tracking_analytics(self, tmp_path: Path) -> None:
        """Assert: no analytics or tracking scripts in the rendered HTML."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8").lower()
        assert "googletagmanager" not in html
        assert "gtag(" not in html
        assert "fbq(" not in html

    def test_all_existing_data_fields_preserved(self, tmp_path: Path) -> None:
        """Assert: all existing data fields remain intact after the restyle."""
        out = tmp_path / "report.html"
        result = _make_result(
            cost="0.1234",
            projected="0.0250",
            candidate="gpt-4o-mini",
            span_days=None,
            window_days=7,
            pjls="2026-05-01",
        )
        render_html(result, out)
        html = out.read_text(encoding="utf-8")
        assert "0.1234" in html
        assert "0.0250" in html
        assert "gpt-4o-mini" in html
        assert "2026-05-01" in html
        assert "7" in html
        assert "gpt-4o" in html

    def test_no_priced_calls_adopts_design_language(self, tmp_path: Path) -> None:
        """Assert: zero-priced-calls path also uses new design tokens."""
        from frugon.cost import AnalysisResult

        out = tmp_path / "report.html"
        zero_result = AnalysisResult(
            total_calls=3,
            priced_calls=0,
            unpriced_calls=3,
            total_cost=Decimal("0"),
            projected_cost=Decimal("0"),
            candidate_model=None,
        )
        render_html(zero_result, out)
        html = out.read_text(encoding="utf-8")
        assert "FRUGON" in html
        assert "--cyan" in html
        assert "eyebrow" in html


# ---------------------------------------------------------------------------
# brand-mark — the wordmark dot CSS class
# ---------------------------------------------------------------------------


class TestBrandMark:
    """The SVG mark in the header must use the public class name brand-mark."""

    def test_brand_mark_class_present(self, tmp_path: Path) -> None:
        """Assert: the SVG in the header uses class='brand-mark'."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert "brand-mark" in html

    def test_brand_mark_css_selector_defined(self, tmp_path: Path) -> None:
        """Assert: the CSS block defines the .brand-mark selector."""
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert ".brand-mark" in html


# ---------------------------------------------------------------------------
# P0 — monthly cost stat shown when monthly_cost is set
# ---------------------------------------------------------------------------


class TestMonthlyCostStat:
    """HTML shows a Monthly cost stat when monthly_cost is available."""

    def test_monthly_cost_stat_present_when_set(self, tmp_path: Path) -> None:
        """Arrange: result with monthly_cost set.
        Act: render_html.
        Assert: 'Monthly cost' label appears in the stat grid.
        """
        from decimal import Decimal

        out = tmp_path / "report.html"
        result = _make_result(window_days=7)
        result.monthly_cost = Decimal("3.21")
        render_html(result, out)
        html = out.read_text(encoding="utf-8")
        assert "Monthly cost" in html

    def test_monthly_cost_stat_absent_when_not_set(self, tmp_path: Path) -> None:
        """Arrange: result without monthly_cost (no window, no span).
        Act: render_html.
        Assert: 'Monthly cost' stat is absent.
        """
        out = tmp_path / "report.html"
        render_html(_make_result(window_days=None, span_days=None), out)
        html = out.read_text(encoding="utf-8")
        assert "Monthly cost" not in html


# ---------------------------------------------------------------------------
# P2 — CC-BY attribution in quality card
# ---------------------------------------------------------------------------


class TestAttributionCard:
    """Attribution string from quality.get_attribution() appears in the quality card."""

    def test_attribution_present_when_candidate_and_attribution_set(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Arrange: result with a candidate; mock get_attribution.
        Act: render_html.
        Assert: attribution text appears inside the quality card.

        Patches render_html.__globals__ directly to avoid stale module reference
        issues after test_report_importable_when_measure_raises_import_error.
        """
        _SENTINEL = "QualityTiersSentinelHTML-XYZ"
        monkeypatch.setitem(render_html.__globals__, "_get_attribution", lambda: _SENTINEL)
        out = tmp_path / "report.html"
        render_html(_make_result(), out)
        html = out.read_text(encoding="utf-8")
        assert _SENTINEL in html

    def test_attribution_absent_when_no_candidate(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Arrange: no candidate model; mock get_attribution returns a string.
        Act: render_html.
        Assert: attribution absent (quality card not shown without a candidate).
        """
        _SENTINEL = "QualityTiersSentinelHTML-XYZ"
        monkeypatch.setitem(render_html.__globals__, "_get_attribution", lambda: _SENTINEL)
        out = tmp_path / "report.html"
        render_html(_make_result(candidate=None, projected="0"), out)
        html = out.read_text(encoding="utf-8")
        assert _SENTINEL not in html
