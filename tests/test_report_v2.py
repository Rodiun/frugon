"""Tests for the v2 (refined) HTML and Markdown report renderers.

Covers the v2 design surface while enforcing the same honesty/privacy
invariants as v1:
  - the saving figure, projection label, quality caveat, and privacy
    footer are present;
  - the HTML is self-contained (no external CSS/JS/font/img assets — the
    only allowed http(s) URL is the frugon.rodiun.io funnel link);
  - the Markdown leads with a bottom line, has a cost-by-model table with a
    total row, a caveat callout, and a methodology/privacy footer;
  - both renderers degrade gracefully for the no-candidate and
    no-priced-calls edge states.

v1 renderers are deliberately untouched and are covered by test_report.py
and test_report_html.py.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from frugon.cost import AnalysisResult
from frugon.report import (
    FUNNEL_LINE,
    QUALITY_CAVEAT,
    render_html_v2,
    render_markdown_v2,
)
from frugon.routing import SplitRouting

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _result_with_candidate(**kwargs: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "total_calls": 35,
        "priced_calls": 35,
        "unpriced_calls": 0,
        "total_cost": Decimal("0.0244"),
        "cost_by_model": {
            "gpt-4-turbo": Decimal("0.0226"),
            "gpt-4o": Decimal("0.0018"),
        },
        "calls_by_model": {"gpt-4-turbo": 25, "gpt-4o": 10},
        "projected_cost": Decimal("0.0083"),
        "candidate_model": "gpt-4o",
        "window_days": 7,
        "monthly_cost": Decimal("0.1042"),
        "monthly_projected": Decimal("0.0355"),
        "pricing_json_last_synced": "2026-06-04",
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _split_routing(**kwargs: Any) -> SplitRouting:
    defaults: dict[str, Any] = {
        "baseline_model": "gpt-4-turbo",
        "candidate_model": "gpt-4o-mini",
        "routed_count": 24,
        "kept_count": 3,
        "routed_cost": Decimal("12.3400"),
        "kept_cost": Decimal("489.2802"),
        "baseline_cost": Decimal("750.0000"),
        "blended_cost": Decimal("501.6202"),
        "easy_threshold": Decimal("0.35"),
    }
    defaults.update(kwargs)
    return SplitRouting(**defaults)


def _result_with_split(**kwargs: Any) -> AnalysisResult:
    """A split-routing result — the headline that renders the routing-plan table."""
    split = kwargs.pop("split", None) or _split_routing(
        candidate_model=kwargs.get("candidate_model", "gpt-4o-mini")
    )
    defaults: dict[str, Any] = {
        "total_calls": 37,
        "priced_calls": 37,
        "unpriced_calls": 0,
        "total_cost": Decimal("800.0000"),
        "cost_by_model": {
            "gpt-4-turbo": Decimal("750.0000"),
            "gpt-4o": Decimal("50.0000"),
        },
        "calls_by_model": {"gpt-4-turbo": 27, "gpt-4o": 10},
        "projected_cost": Decimal("330.0000"),
        "candidate_model": "gpt-4o",
        "observed_span_days": 7.0,
        "split": split,
    }
    defaults.update({k: v for k, v in kwargs.items() if k != "candidate_model"})
    return AnalysisResult(**defaults)


def _result_no_candidate(**kwargs: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "total_calls": 5,
        "priced_calls": 5,
        "unpriced_calls": 0,
        "total_cost": Decimal("1.00"),
        "cost_by_model": {"gpt-4o-mini": Decimal("1.00")},
        "calls_by_model": {"gpt-4o-mini": 5},
        "projected_cost": Decimal("0"),
        "candidate_model": None,
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _result_no_priced() -> AnalysisResult:
    return AnalysisResult(
        total_calls=3,
        priced_calls=0,
        unpriced_calls=3,
        total_cost=Decimal("0"),
        projected_cost=Decimal("0"),
        candidate_model=None,
    )


# Asset-loading external URL detectors (deliberate <a href> hyperlinks are allowed).
_EXT_LINK_HREF_RE = re.compile(r"<link\b[^>]*\bhref\s*=\s*[\"']https?://", re.IGNORECASE)
_EXT_SCRIPT_SRC_RE = re.compile(r"<script\b[^>]*\bsrc\s*=\s*[\"']https?://", re.IGNORECASE)
_EXT_IMG_SRC_RE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"']https?://", re.IGNORECASE)


def _has_external_asset(html: str) -> bool:
    return bool(
        _EXT_LINK_HREF_RE.search(html)
        or _EXT_SCRIPT_SRC_RE.search(html)
        or _EXT_IMG_SRC_RE.search(html)
    )


# ===========================================================================
# HTML v2 — honesty invariants
# ===========================================================================


def test_html_v2_contains_saving_figure_when_candidate(tmp_path: Path) -> None:
    """Arrange: result with a ~66% saving.
    Act: render_html_v2.
    Assert: the saving percentage appears in the HTML.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "66.0%" in html


def test_html_v2_contains_projection_label_when_candidate(tmp_path: Path) -> None:
    """Assert: the projection disclosure ('7-day window') is present."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "7-day window" in html


def test_html_v2_contains_quality_caveat_when_candidate(tmp_path: Path) -> None:
    """Assert: the canonical quality caveat appears in the HTML.

    The honesty invariant requires the full caveat text to be present and
    legible. In the v2 fine-print the ``--measure`` CLI flag is wrapped in a
    <code> span (so it never wraps mid-token), so we assert the caveat with
    that single, expected substitution applied.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    expected_caveat = QUALITY_CAVEAT.replace("--measure", "<code>--measure</code>")
    assert expected_caveat in html
    # The flag is never emitted as a bare, breakable token in the fine print.
    assert "<code>--measure</code>" in html


def test_html_v2_contains_privacy_footer(tmp_path: Path) -> None:
    """Assert: the privacy line is present (no data leaves the machine)."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "No LLM calls. No network. No data leaves your machine." in html


def test_html_v2_contains_methodology_keywords(tmp_path: Path) -> None:
    """Assert: methodology disclosure keywords carry over from v1."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    for keyword in ("list prices", "LiteLLM", "No LLM calls", "No network"):
        assert keyword in html


def test_html_v2_before_after_frame_present(tmp_path: Path) -> None:
    """Assert: the before -> after monthly figures are both shown."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "$0.1042" in html  # current monthly
    assert "$0.0355" in html  # projected monthly


# ===========================================================================
# HTML v2 — self-containment / privacy network invariant
# ===========================================================================


def test_html_v2_is_self_contained_no_external_assets(tmp_path: Path) -> None:
    """Assert: no external CSS/JS/img assets are loaded."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert not _has_external_asset(html)


def test_html_v2_has_no_script_tags(tmp_path: Path) -> None:
    """Assert: no <script> element at all (no JS, no analytics)."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "<script" not in html.lower()


def test_html_v2_only_external_url_is_funnel_link(tmp_path: Path) -> None:
    """Assert: the ONLY https:// URLs in the document point at frugon.rodiun.io.

    The svg xmlns (a non-network identifier) is the only http:// token allowed.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")

    https_urls = re.findall(r"https://[^\s\"'<>]+", html)
    assert https_urls, "expected the funnel link to be present"
    for url in https_urls:
        assert "frugon.rodiun.io" in url, f"unexpected external URL: {url}"

    # The only http:// reference may be the SVG namespace identifier.
    http_urls = re.findall(r"http://[^\s\"'<>]+", html)
    for url in http_urls:
        assert url.startswith("http://www.w3.org/2000/svg"), (
            f"unexpected http:// reference: {url}"
        )


def test_html_v2_no_cdn_or_font_urls(tmp_path: Path) -> None:
    """Assert: no CDN host and no @font-face/web-font URL references."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8").lower()
    assert "cdn" not in html
    assert "fonts.googleapis" not in html
    assert "@font-face" not in html
    assert "@import" not in html


def test_html_v2_inline_style_block_present(tmp_path: Path) -> None:
    """Assert: CSS is inlined in a <style> block."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "<style>" in html


def test_html_v2_no_tracking_analytics(tmp_path: Path) -> None:
    """Assert: no analytics/tracking snippets are present."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8").lower()
    assert "googletagmanager" not in html
    assert "gtag(" not in html
    assert "fbq(" not in html


# ===========================================================================
# HTML v2 — funnel link
# ===========================================================================


def test_html_v2_funnel_link_present_when_candidate(tmp_path: Path) -> None:
    """Assert: the funnel anchor to frugon.rodiun.io is present."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert 'href="https://frugon.rodiun.io"' in html


def test_html_v2_funnel_link_absent_when_no_candidate(tmp_path: Path) -> None:
    """Assert: no funnel link when there is no recommendation."""
    out = tmp_path / "report.html"
    render_html_v2(_result_no_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "frugon.rodiun.io" not in html


# ===========================================================================
# HTML v2 — edge states
# ===========================================================================


def test_html_v2_no_candidate_suppresses_saving_and_caveat(tmp_path: Path) -> None:
    """Assert: no saving figure and no quality caveat when no candidate."""
    out = tmp_path / "report.html"
    render_html_v2(_result_no_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert QUALITY_CAVEAT not in html
    assert "No cheaper swap clears the quality bar." in html
    # Privacy footer must still be present in the no-candidate state.
    assert "No data leaves your machine." in html


def test_html_v2_no_priced_calls_renders_notice(tmp_path: Path) -> None:
    """Assert: zero-priced-calls path produces a notice + privacy footer."""
    out = tmp_path / "report.html"
    render_html_v2(_result_no_priced(), out)
    html = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "No priced calls" in html
    assert "No data leaves your machine." in html
    assert not _has_external_asset(html)


def test_html_v2_no_priced_calls_has_no_saving_figure(tmp_path: Path) -> None:
    """Assert: the no-priced-calls state shows no saving percentage hero."""
    out = tmp_path / "report.html"
    render_html_v2(_result_no_priced(), out)
    html = out.read_text(encoding="utf-8")
    # The .hero-figure CSS rule is always defined in <style>; assert no
    # hero-figure *element* is emitted in the no-priced body.
    assert 'class="hero-figure"' not in html


# ===========================================================================
# HTML v2 — design surface (table + bars + swap pill)
# ===========================================================================


def test_html_v2_cost_table_has_proportion_bars(tmp_path: Path) -> None:
    """Assert: each model row carries a pure-CSS proportion bar."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert 'class="bar"' in html
    # Dominant model (0.0226 / 0.0244) is ~92.6% wide.
    assert "width:92.6%" in html


def test_html_v2_cost_table_has_total_row(tmp_path: Path) -> None:
    """Assert: the cost table carries a total row summing to 100%."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert 'class="total"' in html
    assert "100.0%" in html


def test_html_v2_swap_pill_names_baseline_and_candidate(tmp_path: Path) -> None:
    """Assert: the swap pill names both baseline and candidate models."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "Recommended swap" in html
    assert "gpt-4-turbo" in html
    assert "gpt-4o" in html
    assert 'class="pill"' in html


def test_html_v2_uses_green_for_saving(tmp_path: Path) -> None:
    """Assert: a GREEN token is reserved for the saving (matches landing page)."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "--green" in html
    # Hero figure must be coloured green.
    assert "var(--green)" in html


def test_html_v2_uses_two_column_fold_layout(tmp_path: Path) -> None:
    """Assert: the above-the-fold content is laid out in two balanced rails.

    The rails are deliberately composed so the two tallest blocks (the saving
    hero and the comparison matrix) are split across columns, eliminating the
    dead bottom-right quadrant of the earlier layout:

    - Left rail : saving hero + before/after, then the cost-by-model table.
    - Right rail: the "What we found" matrix + "You save" delta, then the
      recommended swap.

    Both rails live inside a ``.fold`` grid that collapses to one column on
    narrow viewports.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert 'class="fold"' in html
    # Two column wrappers inside the fold grid.
    assert html.count('class="col"') == 2
    # Isolate the fold, then split it into the left and right rail substrings
    # at the two column wrappers so rail membership can be asserted directly.
    fold = html.split('class="fold"', 1)[1].split('class="below"', 1)[0]
    col_parts = fold.split('class="col"')
    assert len(col_parts) == 3  # preamble + left rail + right rail
    left_rail, right_rail = col_parts[1], col_parts[2]
    # Left rail: the saving hero leads, the cost-by-model table follows beneath.
    assert 'class="hero-figure"' in left_rail
    assert 'class="tbl"' in left_rail
    assert left_rail.find('class="hero-figure"') < left_rail.find('class="tbl"')
    # Right rail: the comparison matrix leads, the recommended swap follows.
    assert 'class="matrix"' in right_rail
    assert "Recommended swap" in right_rail
    assert right_rail.find('class="matrix"') < right_rail.find("Recommended swap")
    # The matrix is in the right rail, NOT the left; the table is in the left,
    # NOT the right — guards against an accidental rail swap regressing balance.
    assert 'class="matrix"' not in left_rail
    assert 'class="tbl"' not in right_rail


def test_html_v2_wide_container_and_responsive_collapse(tmp_path: Path) -> None:
    """Assert: the container is widened (~1080px) and collapses on narrow screens."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "max-width:1080px" in html
    # Responsive single-column fallback under ~720px.
    assert "max-width:720px" in html
    assert "grid-template-columns:1fr" in html


def test_html_v2_methodology_and_footer_below_fold(tmp_path: Path) -> None:
    """Assert: methodology + privacy footer sit below the fold (full width)."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    # Methodology section is marked as below-the-fold full-width.
    assert 'class="below"' in html
    # The footer comes after the fold grid in document order.
    assert html.index('class="fold"') < html.index('class="foot"')


def test_html_v2_brand_mark_and_wordmark_present(tmp_path: Path) -> None:
    """Assert: the three-orb brand mark + FRUGON wordmark are present."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "FRUGON" in html
    assert "brand-mark" in html
    assert 'cy="5"' in html


def test_html_v2_sub_cent_cost_not_truncated(tmp_path: Path) -> None:
    """Assert: sub-$0.0001 costs render at 6dp, not $0.0000."""
    out = tmp_path / "report.html"
    result = _result_no_candidate(
        total_cost=Decimal("0.000005"),
        cost_by_model={"gpt-4o-mini": Decimal("0.000005")},
    )
    render_html_v2(result, out)
    html = out.read_text(encoding="utf-8")
    assert "$0.000005" in html


# ===========================================================================
# HTML v2 — "What we found" Current -> After-swap comparison matrix
# ===========================================================================


def test_html_v2_what_we_found_is_comparison_matrix(tmp_path: Path) -> None:
    """Assert: 'What we found' is a Current vs After-swap matrix, not flat tiles.

    The data is a 2x2 (Current / After swap) x (This sample / Monthly). The
    redesign renders a real matrix; the old equal-weight ``.stat`` tiles are gone.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert "What we found" in html
    assert 'class="matrix"' in html
    # Both rows are labelled and read as Current -> After swap.
    assert "Current" in html
    assert "After recommended swap" in html
    # Column headers expose the two axes.
    assert "This sample" in html
    assert "Monthly projection" in html
    # The retired flat tile grid is gone.
    assert 'class="stats"' not in html
    assert 'class="stat-value' not in html


def test_html_v2_after_swap_row_uses_green(tmp_path: Path) -> None:
    """Assert: the after-swap row carries the positive green treatment."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    # The after-swap row is class-tagged so the CSS paints its figures green.
    assert 'class="after"' in html
    # Both after-swap figures (sample + monthly) are present.
    assert "$0.0083" in html  # projected sample cost
    assert "$0.0355" in html  # projected monthly cost


def test_html_v2_surfaces_monthly_saving_delta(tmp_path: Path) -> None:
    """Assert: the concrete monthly saving delta grounds the hero percentage.

    monthly_cost 0.1042 - monthly_projected 0.0355 = 0.0687 saved /mo (~66%).
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert 'class="delta"' in html
    assert "$0.0687/mo" in html
    assert "You save" in html


def test_html_v2_metadata_demoted_below_comparison(tmp_path: Path) -> None:
    """Assert: calls-analyzed + pricing-synced are a quiet meta line, not tiles."""
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert 'class="meta-line"' in html
    assert "calls priced" in html
    assert "pricing synced 2026-06-04" in html


def test_html_v2_caveat_is_official_fineprint_in_footer(tmp_path: Path) -> None:
    """Assert: the quality caveat is footnote-styled fine print in the footer.

    The caveat must remain present and legible (honesty invariant) but is now
    a dagger-marked disclosure block, not inline body-weight prose in the hero.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    # Fine-print block exists and carries the full caveat (with the
    # ``--measure`` flag wrapped in <code> so it cannot wrap mid-token).
    assert 'class="fineprint"' in html
    expected_caveat = QUALITY_CAVEAT.replace("--measure", "<code>--measure</code>")
    assert expected_caveat in html
    # A dagger marker links the after-swap figures to the disclosure.
    assert "dagger" in html
    assert "&dagger;" in html
    # The fine print sits in the footer region (after the fold grid).
    assert html.index('class="fold"') < html.index('class="fineprint"')


def test_html_v2_fineprint_meets_min_font_size(tmp_path: Path) -> None:
    """Assert: fine print is muted but >=12px (brand minimum) and AA-legible.

    The fine-print body is 13px when wrapped (narrow/mobile) and is reduced to
    12.5px only in the one-line desktop treatment — both clear the 12px brand
    floor. Assert the base rule exists and that no px font-size anywhere in the
    stylesheet (integer *or* fractional) drops below 12px.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert ".fineprint .body{" in html
    assert "font-size:13px" in html
    # No font sizes below the 12px brand floor anywhere in the stylesheet —
    # match fractional px too (e.g. 12.5px) so a sub-12 fractional cannot slip
    # past an integer-only pattern.
    px_sizes = re.findall(r"font-size:(\d+(?:\.\d+)?)px", html)
    assert px_sizes, "expected px font sizes in the stylesheet"
    assert all(float(px) >= 12 for px in px_sizes), f"sub-12px font found: {px_sizes}"


def test_html_v2_fineprint_always_wraps_inside_container(tmp_path: Path) -> None:
    """Assert: the fine print wraps inside the container at EVERY viewport width.

    Regression guard for the split-report overflow defect: an earlier desktop
    treatment forced ``white-space:nowrap`` on ``.fineprint .body`` above an
    880px media query to hold the caveat on one tidy line. That width was sized
    for the shorter single-candidate caveat; the longer split caveat ("…run
    --measure to sample real outputs before you switch.") overran the container
    right edge at desktop width. The correct law is: the fine print ALWAYS wraps
    — never nowrap — so it can never force horizontal overflow at any width for
    either caveat. The only nowrap permitted inside the fine print is on the
    ``<code>--measure</code>`` token (so the flag never splits mid-word).
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    # The base fine-print body rule must NOT force nowrap.
    base_rule = re.search(r"\.fineprint \.body\{[^}]*\}", html)
    assert base_rule is not None
    assert "nowrap" not in base_rule.group(0)
    # It must shrink to the available track and break long content, so it can
    # never dictate an over-wide min-size inside the flex footer.
    assert "min-width:0" in base_rule.group(0)
    assert "overflow-wrap:break-word" in base_rule.group(0)
    # No desktop (min-width) media query may reintroduce nowrap on the body —
    # that was the exact rule that produced the overflow.
    assert (
        re.search(
            r"@media \(min-width:\d+px\)\{\s*\.fineprint \.body\{[^}]*nowrap[^}]*\}",
            html,
        )
        is None
    ), "a desktop media query must not force the fine print onto one line"
    # The --measure flag stays non-breaking (its own nowrap on the code token).
    assert ".fineprint .body code{font-size:0.92em;white-space:nowrap}" in html


def test_html_v2_matrix_reflows_on_narrow_without_clipping(tmp_path: Path) -> None:
    """Assert: the comparison matrix reflows on narrow viewports, never clips.

    The matrix container carries ``overflow:hidden`` (its rounded corners), so a
    3-column money table that exceeds the column width does not scroll — it gets
    sheared off, hiding the rightmost (Monthly projection) figure on a 320-375px
    phone. The fix stacks the matrix below a mobile breakpoint into one card per
    scenario, with the column meaning carried inline by each value cell's
    ``data-label``. This test guards three invariants of that fix:

      1. a mobile media query stacks the matrix table;
      2. no *fixed-px* width on the matrix or its value cells can defeat reflow
         (an em-based, flex-shrinkable label-column floor is permitted — see 2b);
      2b. the stacked value rows pair label->value tightly (NOT a full-width
         ``justify-content:space-between`` that floats the figure to the card
         edge), via a shrinkable em-width label column;
      3. every value cell carries a data-label so the stacked figures stay
         labelled (otherwise the reflowed report would read as bare numbers).
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")

    # 1. A narrow-viewport media query that stacks the matrix table to blocks.
    mobile_mq = re.search(
        r"@media \(max-width:(\d+)px\)\{(.*?\.matrix[^@]*?display:block.*?)\}\s*"
        r"/\* Saving delta",
        html,
        re.DOTALL,
    )
    assert mobile_mq is not None, "expected a max-width matrix reflow media query"
    # The breakpoint must cover the largest narrow target (414px) so phones in
    # the 320-414 band all reflow; <=600 would be too aggressive, >=414 is safe.
    assert int(mobile_mq.group(1)) >= 414, mobile_mq.group(1)

    # 2. No fixed-px width in the matrix *declaration blocks* that could force
    #    the table wider than its column and re-introduce the clip. We strip CSS
    #    comments first (their prose can contain "640px"), then inspect only the
    #    declaration body of each `.matrix...{ ... }` rule — never the selector
    #    or any media-query breakpoint, which legitimately carry pixel values.
    css = html.split("<style>", 1)[1].split("</style>", 1)[0]
    css_no_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    matrix_bodies = re.findall(r"\.matrix[^{}@]*\{([^}]*)\}", css_no_comments)
    assert matrix_bodies, "expected .matrix declaration blocks in the stylesheet"
    for body in matrix_bodies:
        # The visually-hidden header (position:absolute + clip) legitimately uses
        # a 1px box to retire the thead off-screen when stacked; it is not a
        # layout width and cannot affect reflow. Skip only that pattern.
        if "position:absolute" in body and "clip" in body:
            continue
        # A fixed *pixel* min-width or width on a matrix value/table element
        # would prevent the column from shrinking and could re-introduce the
        # clip. An em-based min-width on the stacked label column is the
        # deliberate tight-pairing mechanism (2b) and is shrinkable via
        # flex:0 1 auto, so it is allowed; only px floors are forbidden.
        assert not re.search(
            r"\bmin-width:\d+px", body
        ), f"fixed-px min-width on matrix defeats reflow: {body}"
        assert not re.search(r"\bwidth:\d+px", body), f"fixed px width: {body}"

    # 2b. The stacked value rows must pair label->value tightly. The original
    #     reflow used justify-content:space-between, which on a wide narrow card
    #     (~480-640px) floated the figure to the far edge and opened a large void
    #     reading as a layout accident. The fix packs label and value adjacent
    #     (flex-start) with an em-based, shrinkable label-column floor so the two
    #     figures align into a tidy column. Guard both halves of that fix.
    fig_cell_body = next(
        (b for b in matrix_bodies
         if "display:flex" in b and "align-items:baseline" in b
         and "content:attr(data-label)" not in b),
        None,
    )
    assert fig_cell_body is not None, "expected the stacked .matrix value-cell flex rule"
    assert "justify-content:space-between" not in fig_cell_body, (
        "value cell must not space-between (re-opens the label->value void): "
        f"{fig_cell_body}"
    )
    # The label->value gap is a viewport-responsive clamp() that grows the
    # breathing room on wider phones while staying overflow-safe at the 320px
    # floor. Guard that the gap is a clamp (not a flat small fixed value) and
    # that its floor and cap stay within the comfortable, 320-safe band:
    # floor in [16px, 20px], cap in [24px, 32px].
    gap_clamp = re.search(
        r"gap:clamp\((\d+)px,\s*[\d.]+vw,\s*(\d+)px\)", fig_cell_body
    )
    assert gap_clamp is not None, (
        "label->value gap must be a viewport-responsive clamp(): "
        f"{fig_cell_body}"
    )
    gap_floor, gap_cap = int(gap_clamp.group(1)), int(gap_clamp.group(2))
    assert 16 <= gap_floor <= 20, f"gap floor out of 320-safe band: {gap_floor}px"
    assert 24 <= gap_cap <= 32, f"gap cap out of comfortable band: {gap_cap}px"
    # The label column (::before) must carry a shrinkable em-width floor that
    # holds the figure next to its label rather than at the card edge.
    label_before = re.search(
        r"\.matrix td\.fig::before[^{}@]*\{([^}]*)\}", css_no_comments
    )
    assert label_before is not None, "expected the stacked label ::before rule"
    assert re.search(r"min-width:[\d.]+em", label_before.group(1)), (
        f"label column needs an em-width floor for the tight pairing: {label_before.group(1)}"
    )

    # 3. Every matrix value cell carries a data-label (the column header) so the
    #    stacked card still reads "This sample  $x" / "Monthly projection  $y".
    value_cells = re.findall(r"<td class=\"(?:fig tnum|empty)\"[^>]*>", html)
    assert value_cells, "expected matrix value cells in the report"
    for cell in value_cells:
        assert "data-label=" in cell, f"value cell missing data-label: {cell}"

    # 4. The cost-by-model table is the report's other narrow-width pinch point:
    #    its four columns plus the desktop 14px cell gutters have a min-content
    #    width that exceeds a 320px phone, forcing a horizontal scroll. A mobile
    #    media query must trim the .tbl cell gutters so it reflows within 320px.
    tbl_relief = re.search(
        r"@media \(max-width:\d+px\)\{\s*\.tbl th,\.tbl td\{([^}]*)\}", css_no_comments
    )
    assert tbl_relief is not None, "expected a narrow-width .tbl gutter-relief rule"
    relieved = tbl_relief.group(1)
    assert "padding-left" in relieved, relieved
    assert "padding-right" in relieved, relieved
    # The relieved gutters must be smaller than the 14px desktop gutters.
    tbl_px = [float(p) for p in re.findall(r"padding-(?:left|right):(\d+)px", relieved)]
    assert tbl_px, "expected pixel gutters in the .tbl relief rule"
    assert all(p < 14 for p in tbl_px), tbl_px


def test_html_v2_no_candidate_omits_after_swap_and_delta(tmp_path: Path) -> None:
    """Assert: no after-swap row, no delta, no fine print without a candidate."""
    out = tmp_path / "report.html"
    render_html_v2(_result_no_candidate(), out)
    html = out.read_text(encoding="utf-8")
    # Comparison still renders the Current row...
    assert 'class="matrix"' in html
    assert "Current" in html
    # ...but no after-swap row, delta, or fine print. Assert against the body
    # markup (the after row carries class="after"); the phrase itself appears
    # in a CSS comment, so test the emitted element, not the loose string.
    assert 'class="after"' not in html
    assert 'class="delta"' not in html
    assert 'class="fineprint"' not in html
    assert QUALITY_CAVEAT not in html


# ===========================================================================
# Markdown v2 — structure + honesty invariants
# ===========================================================================


def test_markdown_v2_leads_with_bottom_line(tmp_path: Path) -> None:
    """Assert: the report leads with a 'Bottom line' headline section."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert "## Bottom line" in md
    # The bottom line names the saving and a before -> after.
    bottom = md.split("## Bottom line", 1)[1].split("##", 1)[0]
    assert "66.0%" in bottom
    assert "→" in bottom


def test_markdown_v2_contains_quality_caveat(tmp_path: Path) -> None:
    """Assert: the canonical quality caveat literal appears."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert QUALITY_CAVEAT in md


def test_markdown_v2_caveat_rendered_as_callout(tmp_path: Path) -> None:
    """Assert: the caveat is surfaced as a dagger-marked blockquote callout.

    `_result_with_candidate()` has no split → renders via the v2 wholesale path,
    which DOES emit `† ` body markers on the "After recommended swap" cells,
    so the callout's dagger is a valid footnote leg (kept by the C-1 sweep —
    only the split-MD callout lost its dagger, where no body marker exists).
    """
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert "> **† Before you switch:**" in md


def test_markdown_v2_cost_table_with_total_row(tmp_path: Path) -> None:
    """Assert: a cost-by-model table with a bold total row is present."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert "## Cost by model" in md
    assert "| Model | Calls | Cost | % of total |" in md
    assert "| **Total** |" in md
    # Total calls = 25 + 10 = 35.
    assert "**35**" in md


def test_markdown_v2_what_we_found_is_comparison_table(tmp_path: Path) -> None:
    """Assert: 'What we found' is a Current vs After-swap table with both axes."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    section = md.split("## What we found", 1)[1].split("##", 1)[0]
    assert "This sample" in section
    assert "Monthly projection" in section
    assert "**Current**" in section
    assert "**After recommended swap**" in section
    # The after-swap figures carry the dagger that links to the fine print.
    assert "†" in section


def test_markdown_v2_surfaces_saving_delta(tmp_path: Path) -> None:
    """Assert: the concrete monthly saving delta is stated (0.0687/mo, ~66%)."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert "$0.0687/mo" in md
    assert "You save" in md
    assert "−66.0%" in md


def test_markdown_v2_metadata_demoted(tmp_path: Path) -> None:
    """Assert: calls/pricing metadata is a demoted italic meta line."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert "_35 calls priced · pricing synced 2026-06-04_" in md


def test_markdown_v2_no_candidate_omits_after_swap_row(tmp_path: Path) -> None:
    """Assert: the comparison shows Current only when there is no candidate."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_no_candidate(), out)
    md = out.read_text(encoding="utf-8")
    section = md.split("## What we found", 1)[1].split("##", 1)[0]
    assert "**Current**" in section
    assert "After recommended swap" not in section
    assert "You save" not in md


def test_markdown_v2_methodology_and_privacy_footer(tmp_path: Path) -> None:
    """Assert: the methodology + privacy footer line is present."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert "No LLM calls. No network. No data leaves your machine." in md
    assert "0 LLM calls made for this analysis" in md


def test_markdown_v2_funnel_line_present_when_candidate(tmp_path: Path) -> None:
    """Assert: the funnel line + link are present when a candidate exists."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert FUNNEL_LINE in md
    assert "https://frugon.rodiun.io" in md


def test_markdown_v2_swap_names_baseline_and_candidate(tmp_path: Path) -> None:
    """Assert: the recommended-swap line names both models with an arrow."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert "## Recommended swap" in md
    # Isolate the recommended-swap section, then find the arrow line within it.
    swap_section = md.split("## Recommended swap", 1)[1].split("\n##", 1)[0]
    swap_line = next(line for line in swap_section.splitlines() if "→" in line)
    assert "gpt-4-turbo" in swap_line
    assert "gpt-4o" in swap_line


# ===========================================================================
# Markdown v2 — edge states
# ===========================================================================


def test_markdown_v2_no_candidate_suppresses_caveat_and_funnel(tmp_path: Path) -> None:
    """Assert: no quality caveat / funnel when there is no candidate."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_no_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert QUALITY_CAVEAT not in md
    assert FUNNEL_LINE not in md
    assert "No cheaper swap clears the quality bar." in md
    # Footer still present.
    assert "No data leaves your machine." in md


def test_markdown_v2_no_candidate_still_has_cost_table(tmp_path: Path) -> None:
    """Assert: the cost table renders even without a recommendation."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_no_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert "## Cost by model" in md
    assert "| **Total** |" in md


def test_markdown_v2_no_priced_calls_renders_notice(tmp_path: Path) -> None:
    """Assert: zero-priced-calls path produces a notice + privacy footer."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_no_priced(), out)
    md = out.read_text(encoding="utf-8")
    assert "No priced calls found." in md
    assert "No data leaves your machine." in md
    # No saving / swap content in this state.
    assert "Recommended swap" not in md


def test_markdown_v2_no_external_http_except_funnel(tmp_path: Path) -> None:
    """Assert: the only http(s) URL in the Markdown is the funnel link."""
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    urls = re.findall(r"https?://[^\s\)\]]+", md)
    assert urls
    for url in urls:
        assert "frugon.rodiun.io" in url, f"unexpected URL in markdown: {url}"


# ===========================================================================
# Attribution carry-over (honesty)
# ===========================================================================


def test_html_v2_attribution_present_when_candidate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Assert: CC-BY attribution from get_attribution() appears with a candidate."""
    sentinel = "QualityTiersSentinelV2-XYZ"
    monkeypatch.setitem(render_html_v2.__globals__, "_get_attribution", lambda: sentinel)
    out = tmp_path / "report.html"
    render_html_v2(_result_with_candidate(), out)
    html = out.read_text(encoding="utf-8")
    assert sentinel in html


def test_markdown_v2_attribution_present_when_candidate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Assert: CC-BY attribution appears in the Markdown caveat callout."""
    sentinel = "QualityTiersSentinelV2MD-XYZ"
    monkeypatch.setitem(
        render_markdown_v2.__globals__, "_get_attribution", lambda: sentinel
    )
    out = tmp_path / "report.md"
    render_markdown_v2(_result_with_candidate(), out)
    md = out.read_text(encoding="utf-8")
    assert sentinel in md


def test_html_v2_routing_plan_five_columns_no_collision(tmp_path: Path) -> None:
    """Assert: the v2 routing plan renders FIVE distinct columns and the model
    name never collides with the status badge.

    A reported overflow: "Keep · already o10000 gpt-4o" — the long bucket label
    ran into the Calls number and the model + badge were crammed in one cell. The
    fix gives each datum its own column: Bucket | Calls | Model | Status | Cost.
    The model name and the badge now live in separate <td>s, so they are
    structurally incapable of overlapping at any width.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_split(), out)
    html = out.read_text(encoding="utf-8")

    # Five header columns, in order — scoped to the routing-plan table.
    plan = re.search(r'<table class="tbl tbl-plan">.*?</table>', html, re.S)
    assert plan is not None, "the routing-plan table must render"
    head = re.search(r"<thead>.*?</thead>", plan.group(0), re.S)
    assert head is not None
    head_html = head.group(0)
    for col in ("c-bucket", "c-calls", "c-model", "c-status", "c-cost"):
        assert col in head_html, f"the {col} column header must exist"

    # Model and Status are SEPARATE cells — the model name carries no badge,
    # and the badge sits in its own c-status cell.
    assert '<td class="c-model"><span class="route-to">' in html
    assert '<td class="c-status"><span class="badge">within tolerance</span></td>' in html
    # The old crammed route-cell (model + pill in one cell) is gone.
    assert "route-cell" not in html


def test_html_v2_routing_plan_contains_within_viewport(tmp_path: Path) -> None:
    """Assert: the routing-plan table is contained at desktop AND on mobile.

    Desktop: the six columns (Bucket | Calls | % calls | Model | Status | Cost)
    size to their content under ``table-layout:auto`` and pack left with even
    spacing — so there is no MODEL/STATUS dead band — while the ``max-width:100%``
    clamp guarantees the table can never exceed its (full-width) .plan-full
    section regardless of content. Mobile (<=640px): the share bar shrinks and the
    cost figure is allowed to wrap, so the six columns reflow inside a narrow
    phone with no horizontal scroll.
    """
    out = tmp_path / "report.html"
    render_html_v2(_result_with_split(), out)
    html = out.read_text(encoding="utf-8")

    # Desktop containment: AUTO layout (removes the dead band) + a max-width clamp
    # that is a STRONGER containment guarantee than the old per-track rem budget —
    # the table can never exceed 100% of its container, whatever the content.
    assert ".tbl-plan{table-layout:auto;width:100%;max-width:100%;border-collapse:collapse}" in html
    assert ".wrap{max-width:1080px;margin:0 auto}" in html
    # The SHARE column is present and reconciles to 100% on the Blended total row.
    assert '<th class="c-share">% calls</th>' in html
    assert '<td class="num r tnum c-share">100.0%</td>' in html

    # Mobile containment law must be scoped to the <=640px media query.
    mobile_idx = html.find("@media (max-width:640px)")
    assert mobile_idx != -1, "the <=640px mobile media query must exist"
    mobile_css = html[mobile_idx:]
    # The full-precision cost may wrap on mobile so its min-content is one segment.
    assert re.search(r"\.tbl-plan \.c-cost\{width:1%;white-space:normal\}", mobile_css), (
        "on mobile the cost cell must shrink and be allowed to wrap"
    )
    # The share bar narrows on a phone so its column never crowds the MODEL name.
    assert re.search(r"\.tbl-plan \.share-bar\{width:2\.2rem\}", mobile_css), (
        "on mobile the share bar must shrink so the six columns reflow"
    )


