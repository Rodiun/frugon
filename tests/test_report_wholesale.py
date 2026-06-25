"""Tests for the wholesale single-model-swap terminal report.

The wholesale headline (``--wholesale`` / when no split is available) now shares
the split design language exactly: ONE rounded cyan-bordered panel carries the
summary, the plain-English full-swap decision, and the SAVING hero; beneath it
sit muted accounting lines that reconcile every analyzed call, and a quiet footer
(quality caveat / privacy / one upsell).  These tests pin that layout, the
full-dataset reconciliation (Current - New == SAVING, accounting sums to
analyzed), and the honest no-candidate fallback.
"""

from __future__ import annotations

import io
import sys
from decimal import Decimal
from typing import Any

from rich.console import Console

from frugon.cost import AnalysisResult
from frugon.report import (
    FUNNEL_URL,
    QUALITY_NOT_VERIFIED_ACTION,
    QUALITY_NOT_VERIFIED_ASSERTION_WHOLESALE,
    SAVING_GREEN,
    render_terminal,
)


def _result_wholesale(**kwargs: Any) -> AnalysisResult:
    """A two-model wholesale fixture mirroring the bundled demo shape.

    46,100 baseline (gpt-4-turbo) + 10,000 already-on-candidate (gpt-4o) calls;
    the full swap moves every call to gpt-4o.  ``total_cost`` is the sum of the
    cost-by-model rows; ``projected_cost`` is the full-dataset cost of moving
    every call to the candidate (the calls already on it carry through).
    """
    defaults: dict[str, Any] = {
        "total_calls": 56_100,
        "priced_calls": 56_100,
        "unpriced_calls": 0,
        "total_cost": Decimal("800.00"),
        "cost_by_model": {
            "gpt-4-turbo": Decimal("777.00"),
            "gpt-4o": Decimal("23.00"),
        },
        "calls_by_model": {"gpt-4-turbo": 46_100, "gpt-4o": 10_000},
        "projected_cost": Decimal("240.00"),
        "candidate_model": "gpt-4o",
        "observed_span_days": 30.0,
        "monthly_cost": Decimal("800.00"),
        "monthly_projected": Decimal("240.00"),
        "pricing_json_last_synced": "2026-06-04",
        "quality_json_last_synced": "2026-06-04",
        "split": None,
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _result_no_candidate(**kwargs: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "total_calls": 100,
        "priced_calls": 100,
        "unpriced_calls": 0,
        "total_cost": Decimal("12.00"),
        "cost_by_model": {"gpt-4o-mini": Decimal("12.00")},
        "calls_by_model": {"gpt-4o-mini": 100},
        "projected_cost": Decimal("0"),
        "candidate_model": None,
        "split": None,
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _render_to_text(result: AnalysisResult, **kwargs: Any) -> str:
    """Render the wholesale terminal view to plain text via a fixed-width console.

    Mirrors the split test harness: an 88-column, colour-free console so the
    layout (one bordered panel + muted lines) renders deterministically and the
    responsive hanging-indent lines measure their wrap width from the SAME console
    they are printed to.
    """
    report_mod = sys.modules[render_terminal.__module__]

    buf = io.StringIO()
    console = Console(file=buf, width=88, no_color=True, force_terminal=True, highlight=False)
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
    return buf.getvalue()


class TestWholesaleWindowCaution:
    """The wholesale Accounting block warns when --window contradicts the span.

    Shares the same shared freshness helper as the split path, so the caution is
    consistent across both surfaces.  The wholesale fixture's observed span is 30
    days, so window=7 (ratio ~4.3) fires while window=30 (an exact match) and
    no-window do not.
    """

    def test_window_caution_renders_on_mismatch(self) -> None:
        """window=7 vs a 30-day span → the amber Window caution row appears."""
        import re

        text = _render_to_text(_result_wholesale(window_days=7))
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert "Window" in flat
        assert "--window 7 overrides your log's actual ~30-day span" in flat
        assert "Drop --window to project from the real span." in flat

    def test_window_caution_absent_when_window_matches_span(self) -> None:
        """window=30 ≈ the 30-day span → no caution."""
        import re

        text = _render_to_text(_result_wholesale(window_days=30))
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert "overrides your log's actual" not in flat

    def test_window_caution_absent_when_no_window_flag(self) -> None:
        """No --window (window_days None) → no caution even with a known span."""
        import re

        text = _render_to_text(_result_wholesale(window_days=None))
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert "overrides your log's actual" not in flat


class TestWholesalePanel:
    """The bordered decision panel — summary, the full-swap line, the SAVING hero."""

    def test_panel_title_and_summary(self, capsys: Any) -> None:
        """Same panel title + summary labels as the split view."""
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "frugon · cost analysis" in out
        assert "Analyzed" in out
        assert "56,100 calls" in out  # thousands separators
        # The 80-col panel wraps this phrase across a border (a border char can
        # land between "current" and "model"), so assert on its wrap-robust
        # components rather than the contiguous string.
        assert "baseline" in out
        assert "gpt-4-turbo" in out
        assert "your current" in out
        assert "model" in out
        assert "Current spend" in out

    def test_panel_shows_full_swap_line(self, capsys: Any) -> None:
        """A single 'Swap every call → candidate (full swap)' decision line."""
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "Swap" in out
        assert "every call" in out
        assert "→ gpt-4o" in out
        assert "(full swap)" in out

    def test_panel_names_already_on_target_calls(self, capsys: Any) -> None:
        """Calls already on the candidate are named, not silently dropped."""
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "10,000 already on gpt-4o" in out
        assert "already on target — no change" in out

    def test_panel_hero_is_the_saving(self, capsys: Any) -> None:
        """The hero is the SAVING line — money + percent, both on the total basis."""
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "New spend" in out
        assert "SAVING" in out
        # Current 800 / mo, New 240 / mo -> SAVING 560 / mo, 70% lower.
        # Money renders at full _fmt_usd precision on every surface (terminal
        # matches the reports); the saving percent carries one decimal.
        assert "$560.00 / mo" in out
        assert "70.0% lower" in out

    def test_saving_uses_emerald(self) -> None:
        """The SAVING hero is rendered in the canonical emerald (money == green).

        The panel body is one styled Text; the SAVING figure carries a span
        styled ``bold #10B981``.  Assert such a span exists over the SAVING run.
        """
        from rich.panel import Panel
        from rich.text import Text

        report_mod = sys.modules[render_terminal.__module__]
        captured: list[Text] = []

        def _capture(*args: Any, **kw: Any) -> None:
            for a in args:
                body = a.renderable if isinstance(a, Panel) else a
                if isinstance(body, Text):
                    captured.append(body)

        original = report_mod.rprint
        report_mod.rprint = _capture  # type: ignore[attr-defined]
        try:
            render_terminal(_result_wholesale())
        finally:
            report_mod.rprint = original  # type: ignore[attr-defined]

        panel_bodies = [t for t in captured if "SAVING" in t.plain]
        assert panel_bodies, "expected a panel body containing the SAVING hero"
        body = panel_bodies[0]
        # Find the span covering the literal "SAVING" and assert its style is emerald.
        saving_at = body.plain.index("SAVING")
        emerald_spans = [
            s
            for s in body.spans
            if s.start <= saving_at < s.end and SAVING_GREEN in str(s.style)
        ]
        assert emerald_spans, "SAVING hero must be styled with the emerald saving green"

    def test_money_figures_are_2dp(self, capsys: Any) -> None:
        """Money displays at 2 dp on every surface — '$800.00 / mo'.

        Since v0.1.3, the terminal panel renders whole-dollar amounts at 2 dp
        (not 4 dp), matching the Markdown/HTML reports exactly.
        """
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "$800.00 / mo" in out
        # The old 4-dp format is gone.
        assert "$800.0000 / mo" not in out


class TestWholesaleAccounting:
    """Muted reconciliation — every analyzed call accounted for."""

    def test_accounting_reconciles_to_analyzed(self, capsys: Any) -> None:
        """swapped + already-on-candidate == analyzed, derived (not hardcoded)."""
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        # 46,100 swapped (gpt-4-turbo) + 10,000 already on gpt-4o = 56,100 analyzed
        assert "46,100 swapped (gpt-4-turbo)" in out
        assert "10,000 already on gpt-4o" in out
        assert "56,100 analyzed" in out

    def test_accounting_discloses_prices_synced(self, capsys: Any) -> None:
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "synced 2026-06-04" in out

    def test_accounting_discloses_prices_and_quality_synced(self, capsys: Any) -> None:
        """The wholesale path shows BOTH freshness rows (Prices + Quality)."""
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "Prices synced 2026-06-04" in out
        assert "Quality synced 2026-06-04" in out

    def test_quality_row_omitted_when_date_absent(self, capsys: Any) -> None:
        render_terminal(_result_wholesale(quality_json_last_synced=None))
        out = " ".join(capsys.readouterr().out.split())
        assert "Prices synced 2026-06-04" in out
        assert "Quality synced" not in out

    def test_stale_rows_annotate_with_refresh_commands(self) -> None:
        """Old sync dates annotate BOTH wholesale rows amber with the right commands."""
        import re
        from datetime import date, timedelta

        old = (date.today() - timedelta(days=70)).isoformat()
        text = _render_to_text(
            _result_wholesale(
                pricing_json_last_synced=old,
                quality_json_last_synced=old,
            )
        )
        # Strip interleaved SGR codes (dim date / amber annotation / cyan command).
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert f"Prices synced {old} — ⚠ 70 days old; refresh with frugon pricing update" in flat
        assert f"Quality synced {old} — ⚠ 70 days old; refresh with frugon quality update" in flat

    def test_fresh_rows_carry_no_annotation(self) -> None:
        from datetime import date, timedelta

        recent = (date.today() - timedelta(days=5)).isoformat()
        text = _render_to_text(
            _result_wholesale(
                pricing_json_last_synced=recent,
                quality_json_last_synced=recent,
            )
        )
        assert f"synced {recent}" in text
        assert "days old" not in text
        assert "refresh with" not in text


class TestWholesaleFooter:
    """The quiet footer — quality caveat (amber), privacy, one upsell."""

    def test_footer_quality_caveat(self, capsys: Any) -> None:
        """Wholesale caveat names the full-swap quality change (no 'tolerance' band)."""
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert " ".join(QUALITY_NOT_VERIFIED_ASSERTION_WHOLESALE.split()) in out
        assert " ".join(QUALITY_NOT_VERIFIED_ACTION.split()) in out

    def test_footer_upsell(self, capsys: Any) -> None:
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "Route every call automatically and hold the saving" in out
        assert FUNNEL_URL in out

    def test_unrated_tier_note_folds_into_footer(self, capsys: Any) -> None:
        """The 'no known quality tier' caution is a footer line, not mid-output."""
        result = _result_wholesale(baseline_is_unrated=True)
        text = _render_to_text(result)
        # The tier note appears AFTER the caveat assertion (footer area), never
        # floating before the panel/accounting.
        caveat_idx = text.index("a full swap can change output quality")
        tier_idx = text.index("gpt-4-turbo has no known quality tier")
        assert tier_idx > caveat_idx

    def test_footer_suppressed_under_measure(self, capsys: Any) -> None:
        """suppress_caveat=True (the --measure path) drops the whole footer."""
        render_terminal(_result_wholesale(), suppress_caveat=True)
        out = " ".join(capsys.readouterr().out.split())
        assert "a full swap can change output quality" not in out
        assert "Route every call automatically" not in out


class TestWholesaleVerbose:
    """--verbose appends the cost table + a Notes block (split pointer, method)."""

    def test_verbose_shows_cost_table_and_notes(self, capsys: Any) -> None:
        render_terminal(_result_wholesale(used_default_pool=True, candidate_pool_size=5), verbose=True)
        out = capsys.readouterr().out
        collapsed = " ".join(out.split())
        assert "Cost by model" in collapsed
        assert "Notes" in collapsed
        # Wholesale IS the upper bound, so there is NO 'Upper bound' note; instead
        # it points BACK to the conservative split (the default view).
        assert "Upper bound" not in collapsed
        assert "the default view (drop --wholesale)" in collapsed
        assert "Method" in collapsed

    def test_verbose_omits_notes_in_default_view(self, capsys: Any) -> None:
        render_terminal(_result_wholesale())
        out = " ".join(capsys.readouterr().out.split())
        assert "Cost by model" not in out
        assert "Notes" not in out


class TestWholesaleNoCandidate:
    """Honest fallback when no cheaper candidate exists — no phantom saving."""

    def test_no_candidate_states_so_and_omits_saving(self, capsys: Any) -> None:
        render_terminal(_result_no_candidate())
        out = " ".join(capsys.readouterr().out.split())
        assert "no cheaper candidate found" in out
        # No phantom 100% saving from a None candidate's zero projected_cost.
        assert "SAVING" not in out
        assert "100% lower" not in out
        assert "New spend" not in out

    def test_no_candidate_suppresses_footer(self, capsys: Any) -> None:
        render_terminal(_result_no_candidate())
        out = " ".join(capsys.readouterr().out.split())
        assert "a full swap can change output quality" not in out
        assert FUNNEL_URL not in out


def test_reconciliation_current_minus_new_equals_saving() -> None:
    """Current - New == SAVING, and accounting sums to analyzed (the invariant)."""
    from frugon.report import _wholesale_accounting, _wholesale_current_and_new

    result = _result_wholesale()
    current, new, projected = _wholesale_current_and_new(result)
    assert projected is True
    assert current - new == Decimal("560.00")  # 800 - 240

    swapped, already, models = _wholesale_accounting(result)
    assert swapped + already == result.priced_calls  # 46,100 + 10,000 == 56,100
    assert swapped == 46_100
    assert already == 10_000
    assert models == ["gpt-4-turbo"]  # the model that moves
