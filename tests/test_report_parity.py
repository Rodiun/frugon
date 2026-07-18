"""Report parity with the terminal renderer (quality-synced, staleness, window
caution, log span, upper bound) + the Item-7 ordering swap.

The reports are the FULL view (no verbose mode), so every freshness / caution /
log-span / upper-bound disclosure the terminal surfaces under --verbose (or in
its Accounting block) must appear in every report variant too.  These tests pin
that parity across all four renderers (Markdown v1/v2, HTML v1/v2) on BOTH the
split-headline and the wholesale/non-split paths, and assert the Item-7 ordering
(decision/upper-bound info before freshness metadata) in the terminal and the
reports.

Numbers are NOT recomputed here — the fixtures reuse the same fields the
terminal reads, and the assertions pin presentation only.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from frugon.cost import AnalysisResult
from frugon.report import (
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
)
from frugon.routing import SplitRouting

# --- Old / stale / fresh dates ---------------------------------------------
# Far enough in the past to be unambiguously stale at BOTH thresholds (30 / 60).
_OLD_PRICING = "2025-01-01"  # > 30 days before any plausible test run date
_OLD_QUALITY = "2025-01-01"  # > 60 days before any plausible test run date
_FRESH = None  # quality date absent -> Quality disclosure omitted


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


def _split_result(**kw: Any) -> AnalysisResult:
    """A split-headline result with a distinct wholesale upper-bound candidate."""
    defaults: dict[str, Any] = {
        "total_calls": 37,
        "priced_calls": 37,
        "unpriced_calls": 0,
        "total_cost": Decimal("0.0676"),
        "cost_by_model": {"gpt-4-turbo": Decimal("0.0650"), "gpt-4o": Decimal("0.0026")},
        "calls_by_model": {"gpt-4-turbo": 27, "gpt-4o": 10},
        "projected_cost": Decimal("0.0222"),  # full swap to gpt-4o -> ~67%
        "candidate_model": "gpt-4o",  # distinct from split candidate -> upper bound fires
        "observed_span_days": 7.0,
        "observed_span_start": "2026-05-04",
        "observed_span_end": "2026-05-11",
        "pricing_json_last_synced": "2026-06-04",
        "quality_json_last_synced": "2026-06-04",
        "split": _split(),
    }
    defaults.update(kw)
    return AnalysisResult(**defaults)


def _wholesale_result(**kw: Any) -> AnalysisResult:
    """A non-split (wholesale headline) result — no split routing."""
    defaults: dict[str, Any] = {
        "total_calls": 20,
        "priced_calls": 20,
        "unpriced_calls": 0,
        "total_cost": Decimal("0.0500"),
        "cost_by_model": {"gpt-4-turbo": Decimal("0.0500")},
        "calls_by_model": {"gpt-4-turbo": 20},
        "projected_cost": Decimal("0.0200"),
        "candidate_model": "gpt-4o-mini",
        "monthly_cost": Decimal("0.1500"),
        "monthly_projected": Decimal("0.0600"),
        "observed_span_days": 10.0,
        "observed_span_start": "2026-05-01",
        "observed_span_end": "2026-05-11",
        "pricing_json_last_synced": "2026-06-04",
        "quality_json_last_synced": "2026-06-04",
        "split": None,
    }
    defaults.update(kw)
    return AnalysisResult(**defaults)


def _md(renderer: Any, result: AnalysisResult, tmp_path: Path) -> str:
    out = tmp_path / "r.md"
    renderer(result, out)
    return out.read_text(encoding="utf-8")


def _html(renderer: Any, result: AnalysisResult, tmp_path: Path) -> str:
    out = tmp_path / "r.html"
    renderer(result, out)
    return out.read_text(encoding="utf-8")


def _term(result: AnalysisResult, **kw: Any) -> str:
    """Render the terminal view to colour-free plain text (re-uses split helper)."""
    from tests.test_report_split import _render_to_text

    return _render_to_text(result, **kw)


# ===========================================================================
# Item 1 — quality synced date (beside pricing synced) in every report
# ===========================================================================


class TestQualitySyncedParity:
    def test_md_v1_split_shows_quality_synced(self, tmp_path: Path) -> None:
        text = _md(render_markdown, _split_result(), tmp_path)
        assert "Pricing last synced:** 2026-06-04" in text
        assert "Quality last synced:** 2026-06-04" in text

    def test_md_v2_wholesale_shows_quality_synced(self, tmp_path: Path) -> None:
        text = _md(render_markdown_v2, _wholesale_result(), tmp_path)
        assert "pricing synced 2026-06-04" in text
        assert "quality synced 2026-06-04" in text

    def test_html_v1_split_shows_quality_synced(self, tmp_path: Path) -> None:
        text = _html(render_html, _split_result(), tmp_path)
        assert "Pricing last synced" in text
        assert "Quality last synced" in text
        assert "2026-06-04" in text

    def test_html_v2_split_shows_quality_synced(self, tmp_path: Path) -> None:
        text = _html(render_html_v2, _split_result(), tmp_path)
        assert "pricing synced 2026-06-04" in text
        assert "quality synced 2026-06-04" in text

    def test_html_v2_wholesale_shows_quality_synced(self, tmp_path: Path) -> None:
        text = _html(render_html_v2, _wholesale_result(), tmp_path)
        assert "pricing synced 2026-06-04" in text
        assert "quality synced 2026-06-04" in text

    def test_quality_synced_omitted_when_absent(self, tmp_path: Path) -> None:
        text = _md(render_markdown, _split_result(quality_json_last_synced=None), tmp_path)
        assert "Pricing last synced" in text
        assert "Quality last synced" not in text


# ===========================================================================
# Item 2 — staleness warnings flagged (not just the bare date)
# ===========================================================================


class TestStalenessParity:
    def test_md_split_stale_pricing_and_quality(self, tmp_path: Path) -> None:
        text = _md(
            render_markdown,
            _split_result(
                pricing_json_last_synced=_OLD_PRICING,
                quality_json_last_synced=_OLD_QUALITY,
            ),
            tmp_path,
        )
        assert "days old — refresh with `frugon pricing update`." in text
        assert "days old — refresh with `frugon quality update`." in text

    def test_md_v2_wholesale_stale_callouts(self, tmp_path: Path) -> None:
        text = _md(
            render_markdown_v2,
            _wholesale_result(
                pricing_json_last_synced=_OLD_PRICING,
                quality_json_last_synced=_OLD_QUALITY,
            ),
            tmp_path,
        )
        assert "Pricing table is" in text
        assert "frugon pricing update" in text
        assert "Quality table is" in text
        assert "frugon quality update" in text

    def test_html_v1_stale_amber_caution(self, tmp_path: Path) -> None:
        text = _html(
            render_html,
            _split_result(
                pricing_json_last_synced=_OLD_PRICING,
                quality_json_last_synced=_OLD_QUALITY,
            ),
            tmp_path,
        )
        assert 'class="caution"' in text
        assert "days old — refresh with" in text
        assert "<code>frugon pricing update</code>" in text
        assert "<code>frugon quality update</code>" in text

    def test_html_v2_stale_amber_caution_both_tables(self, tmp_path: Path) -> None:
        text = _html(
            render_html_v2,
            _split_result(
                pricing_json_last_synced=_OLD_PRICING,
                quality_json_last_synced=_OLD_QUALITY,
            ),
            tmp_path,
        )
        # Both tables stale -> two amber cautions with their own refresh commands.
        assert text.count('class="caution"') >= 2
        assert "<code>frugon pricing update</code>" in text
        assert "<code>frugon quality update</code>" in text

    def test_fresh_dates_carry_no_caution(self, tmp_path: Path) -> None:
        """A recent sync renders no staleness caution.

        Clock-relative by construction (mirrors test_report_split.py's
        test_split_fresh_dates_carry_no_amber_annotation and
        test_report_wholesale.py's test_fresh_rows_carry_no_annotation): the
        fixture's hardcoded "2026-06-04" default eventually ages past the
        30/60-day staleness windows and this test would start asserting the
        OPPOSITE of what "fresh" means. Overriding both dates to 5 days before
        whatever "today" actually is asserts the boundary behaviour (fresh ->
        no caution) forever, not a calendar moment.
        """
        from datetime import date, timedelta

        recent = (date.today() - timedelta(days=5)).isoformat()
        text = _html(
            render_html,
            _split_result(
                pricing_json_last_synced=recent,
                quality_json_last_synced=recent,
            ),
            tmp_path,
        )
        assert "days old" not in text
        assert "refresh with" not in text


# ===========================================================================
# Item 3 — --window vs observed-span caution
# ===========================================================================


class TestWindowCautionParity:
    def test_md_split_window_caution(self, tmp_path: Path) -> None:
        text = _md(render_markdown, _split_result(window_days=30), tmp_path)
        assert "`--window 30` overrides your log's actual ~7-day span" in text
        assert "Drop `--window` to project from the real span." in text

    def test_md_v2_wholesale_window_caution(self, tmp_path: Path) -> None:
        text = _md(render_markdown_v2, _wholesale_result(window_days=60), tmp_path)
        assert "`--window 60` overrides your log's actual ~10-day span" in text

    def test_html_v1_window_caution_amber(self, tmp_path: Path) -> None:
        text = _html(render_html, _split_result(window_days=30), tmp_path)
        assert 'class="caution"' in text
        assert "<code>--window 30</code> overrides your log's actual ~7-day span" in text

    def test_html_v2_window_caution_amber(self, tmp_path: Path) -> None:
        text = _html(render_html_v2, _split_result(window_days=30), tmp_path)
        assert "<code>--window 30</code> overrides your log's actual ~7-day span" in text

    def test_window_caution_absent_on_match(self, tmp_path: Path) -> None:
        text = _md(render_markdown, _split_result(window_days=7), tmp_path)
        assert "overrides your log's actual" not in text


# ===========================================================================
# Item 4 — log span (earliest -> latest + days) always in reports
# ===========================================================================


class TestLogSpanParity:
    def test_md_split_log_span(self, tmp_path: Path) -> None:
        text = _md(render_markdown, _split_result(), tmp_path)
        assert "**Log span:** 2026-05-04 → 2026-05-11 (7.0 days)" in text

    def test_md_v2_wholesale_log_span(self, tmp_path: Path) -> None:
        text = _md(render_markdown_v2, _wholesale_result(), tmp_path)
        assert "**Log span:** 2026-05-01 → 2026-05-11 (10.0 days)" in text

    def test_html_v1_log_span(self, tmp_path: Path) -> None:
        text = _html(render_html, _split_result(), tmp_path)
        assert "Log span: 2026-05-04 &rarr; 2026-05-11 (7.0 days)" in text

    def test_html_v2_log_span(self, tmp_path: Path) -> None:
        text = _html(render_html_v2, _split_result(), tmp_path)
        assert "Log span: 2026-05-04 &rarr; 2026-05-11 (7.0 days)" in text

    def test_log_span_omitted_when_bounds_absent(self, tmp_path: Path) -> None:
        text = _md(
            render_markdown,
            _split_result(observed_span_start=None, observed_span_end=None),
            tmp_path,
        )
        assert "Log span" not in text


# ===========================================================================
# Item 5 — no unverifiable "verified" quality claim in any report
# ===========================================================================


class TestNoVerifiedOverclaim:
    def test_reports_never_imply_verified_quality(self, tmp_path: Path) -> None:
        for renderer in (render_markdown, render_markdown_v2):
            text = _md(renderer, _split_result(), tmp_path)
            assert "Quality is not verified" in text
            # No wording that implies a verified/confirmed quality RESULT.
            assert not re.search(r"quality (is )?(verified|confirmed)\b", text, re.I)
        for renderer in (render_html, render_html_v2):
            text = _html(renderer, _split_result(), tmp_path)
            assert "not verified" in text
            assert not re.search(r"quality (is )?(verified|confirmed)\b", text, re.I)


# ===========================================================================
# Item 6 — wholesale upper-bound parity across all four report variants
# ===========================================================================


class TestUpperBoundParity:
    def test_md_v1_upper_bound(self, tmp_path: Path) -> None:
        text = _md(render_markdown, _split_result(), tmp_path)
        assert "Upper bound: moving every call to `gpt-4o` saves" in text
        assert "the conservative" in text

    def test_md_v2_upper_bound(self, tmp_path: Path) -> None:
        text = _md(render_markdown_v2, _split_result(), tmp_path)
        assert "Upper bound: moving every call to `gpt-4o` saves" in text

    def test_html_v1_upper_bound(self, tmp_path: Path) -> None:
        text = _html(render_html, _split_result(), tmp_path)
        assert "Upper bound: moving every call to" in text
        assert "gpt-4o" in text
        assert "the conservative" in text

    def test_html_v2_upper_bound(self, tmp_path: Path) -> None:
        text = _html(render_html_v2, _split_result(), tmp_path)
        assert "Upper bound: moving every call to" in text
        assert "the conservative" in text

    def test_upper_bound_present_when_winner_matches_split(self, tmp_path: Path) -> None:
        # Audit finding #2: the full-swap basis caption is surfaced even when the
        # wholesale winner is the SAME model as the split's easy-call target.  The
        # split moves only the easy baseline calls (~31% here) while the full swap
        # moves every call (~35%), so the figures differ and the user must be able
        # to tell the larger figure is the aggressive full-swap basis.
        result = _split_result(
            candidate_model="gpt-4o-mini", projected_cost=Decimal("0.0439")
        )
        text = _md(render_markdown, result, tmp_path)
        assert "Upper bound: moving every call to `gpt-4o-mini` saves" in text
        assert "the conservative" in text

    def test_upper_bound_absent_when_full_swap_matches_split(self, tmp_path: Path) -> None:
        # The honest non-redundant case: when the full swap saves no more than the
        # split (e.g. the split already routes the whole dataset), no separate
        # upper bound is printed — there is no distinct aggressive basis to show.
        result = _split_result(
            candidate_model="gpt-4o", projected_cost=Decimal("0.0650")
        )
        text = _md(render_markdown, result, tmp_path)
        assert "Upper bound" not in text

    def test_upper_bound_reconciles_terminal_and_reports(self, tmp_path: Path) -> None:
        """The upper-bound % is computed from the same helpers, so the terminal
        verbose note and the MD report quote the same figure."""
        result = _split_result()
        term = re.sub(r"\x1b\[[0-9;]*m", "", _term(result, verbose=True))
        md = _md(render_markdown, result, tmp_path)
        # Extract "saves ~NN%" from each and compare.
        term_pct = re.search(r"moving every call to gpt-4o saves ~(\d+\.\d+)%", term)
        md_pct = re.search(r"moving every call to `gpt-4o` saves ~(\d+\.\d+)%", md)
        assert term_pct is not None, term
        assert md_pct is not None, md
        assert term_pct.group(1) == md_pct.group(1)


# ===========================================================================
# Item 7 — ordering: decision (upper bound) BEFORE freshness metadata
# ===========================================================================


class TestItem7Ordering:
    def test_terminal_accounting_upper_bound_before_prices(self) -> None:
        """Accounting block order: Upper bound -> Quality tier -> Prices -> Quality.

        The Quality-tier benchmark comparison now rides directly under the
        Upper-bound swap context (the decision area) and ABOVE the Prices/Quality
        freshness lines, so the decision context groups together above the
        last-synced metadata.
        """
        text = _term(_split_result())
        flat = re.sub(r"\x1b\[[0-9;]*m", "", text)
        assert "Upper bound" in flat
        assert "Quality tier" in flat
        assert "Prices" in flat
        # The "Quality" freshness row label is disambiguated from "Quality tier"
        # by its trailing "synced" body so the index search finds the right row.
        assert "Quality      synced" in flat or "Quality synced" in flat
        i_quality_synced = flat.index("Quality", flat.index("Prices"))
        assert flat.index("Upper bound") < flat.index("Quality tier"), flat
        assert flat.index("Quality tier") < flat.index("Prices"), flat
        assert flat.index("Prices") < i_quality_synced, flat

    def test_terminal_verbose_notes_upper_bound_before_log_span(self) -> None:
        """Verbose Notes: Upper bound -> Log span -> Method."""
        text = _term(_split_result(), verbose=True)
        flat = re.sub(r"\x1b\[[0-9;]*m", "", text)
        # Restrict to the Notes section so the Accounting Upper-bound row does not
        # confuse the index search.
        notes = flat[flat.index("Notes"):]
        assert notes.index("Upper bound") < notes.index("Log span"), notes
        assert notes.index("Log span") < notes.index("Method"), notes

    def test_md_split_upper_bound_before_details(self, tmp_path: Path) -> None:
        """Decision (Upper bound) precedes the freshness Details block."""
        text = _md(render_markdown, _split_result(), tmp_path)
        assert text.index("Upper bound") < text.index("## Details"), text
        # Within Details: Quality tier leads (grouped with the decision context),
        # then the synced freshness dates, then the Log span freshness metadata.
        details = text[text.index("## Details"):]
        assert details.index("**Quality tier:**") < details.index("Pricing last synced")
        assert details.index("Pricing last synced") < details.index("Log span")

    def test_html_v1_upper_bound_before_freshness_stats(self, tmp_path: Path) -> None:
        text = _html(render_html, _split_result(), tmp_path)
        assert text.index("Upper bound: moving every call to") < text.index(
            "Pricing last synced"
        ), text


# ===========================================================================
# --demo stdout determinism (Item 7 reorder must stay byte-stable per run)
# ===========================================================================


class TestDemoStdoutDeterminism:
    def test_two_demo_renders_are_byte_identical(self) -> None:
        """The reordered default --demo view is deterministic: two renders of the
        same result produce byte-identical text."""
        result = _split_result()
        first = _term(result)
        second = _term(result)
        assert first == second
