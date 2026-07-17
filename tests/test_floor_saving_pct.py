"""Golden-vector tests for the floor-never-round-up saving-percent display rule.

Roadmap promise: "Saving-percent display — floor, never round up." Displayed
saving percentages must never read higher than the exact computed value (a
49.96% saving must print "49.9%", never "50.0%"). This is display-only: the
underlying dollar math and the ROUND_HALF_UP dollar-quantization rule are
unchanged; only the saving-PERCENT formatting layer
(:func:`frugon.cost._display_pct`, consumed by
:func:`frugon.report._reconciled_delta_pct` and every renderer that shows
``_split_report_figures(...).total_pct``) truncates toward zero instead of
rounding half up. Zero and negative (a cost increase, not a saving) values
keep ROUND_HALF_UP — flooring exists only to stop overstating a saving, so it
must not also shrink a displayed increase.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from frugon.cost import AnalysisResult, _display_pct
from frugon.report import (
    _fmt_candidate_saving,
    _reconciled_delta_pct,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
    render_terminal,
)
from frugon.routing import SplitRouting

# ---------------------------------------------------------------------------
# Unit: frugon.cost._display_pct
# ---------------------------------------------------------------------------


class TestDisplayPctFloors:
    def test_boundary_case_floors_not_rounds_up(self) -> None:
        """49.96% must print 49.9%, never nudge up to 50.0% (the exact defect)."""
        assert _display_pct(Decimal("49.96")) == Decimal("49.9")

    def test_exact_value_stays_unchanged(self) -> None:
        assert _display_pct(Decimal("50.0")) == Decimal("50.0")
        assert _display_pct(Decimal("50.00")) == Decimal("50.0")

    def test_sub_tenth_percent_floors_to_zero(self) -> None:
        """0.06% would round UP to 0.1% under ROUND_HALF_UP; flooring keeps it 0.0%."""
        assert _display_pct(Decimal("0.06")) == Decimal("0.0")

    def test_zero_saving_unchanged(self) -> None:
        assert _display_pct(Decimal("0")) == Decimal("0.0")

    def test_negative_saving_path_unchanged_round_half_up(self) -> None:
        """A cost INCREASE (negative pct) keeps ROUND_HALF_UP: -4.96 -> -5.0.

        Flooring (truncating toward zero) a negative value would make the
        increase look SMALLER (-4.96 -> -4.9), which is the opposite of what
        this fix is for. Negative paths must be provably untouched.
        """
        assert _display_pct(Decimal("-4.96")) == Decimal("-5.0")

    @pytest.mark.parametrize(
        "raw",
        [
            Decimal("0.04"),
            Decimal("12.34"),
            Decimal("37.96"),
            Decimal("49.96"),
            Decimal("49.999"),
            Decimal("99.999"),
        ],
    )
    def test_floored_result_never_exceeds_exact_value(self, raw: Decimal) -> None:
        """The reconciling-invariant guarantee: a floored positive percent is
        always <= the exact ratio it was derived from."""
        assert _display_pct(raw) <= raw


# ---------------------------------------------------------------------------
# Unit: frugon.report._reconciled_delta_pct
# ---------------------------------------------------------------------------


class TestReconciledDeltaPctFloors:
    def test_boundary_case_floors(self) -> None:
        # saved = 1000.00 - 500.40 = 499.60; pct = 49.96% exactly.
        cur_q, proj_q, pct = _reconciled_delta_pct(
            Decimal("1000.00"), Decimal("500.40")
        )
        assert cur_q == Decimal("1000.00")
        assert proj_q == Decimal("500.40")
        assert pct == Decimal("49.9")

    def test_exact_50_percent_stays(self) -> None:
        _, _, pct = _reconciled_delta_pct(Decimal("1000.00"), Decimal("500.00"))
        assert pct == Decimal("50.0")

    def test_sub_tenth_percent_floors_to_zero(self) -> None:
        _, _, pct = _reconciled_delta_pct(Decimal("100.00"), Decimal("99.94"))
        assert pct == Decimal("0.0")

    def test_negative_saving_unchanged_round_half_up(self) -> None:
        """A cost increase (current < projected) keeps ROUND_HALF_UP."""
        _, _, pct = _reconciled_delta_pct(Decimal("100.00"), Decimal("104.96"))
        assert pct == Decimal("-5.0")

    def test_zero_current_returns_zero(self) -> None:
        _, _, pct = _reconciled_delta_pct(Decimal("0"), Decimal("0"))
        assert pct == Decimal("0.0")


# ---------------------------------------------------------------------------
# Unit: frugon.report._fmt_candidate_saving (per-candidate "X.X% lower/higher")
# ---------------------------------------------------------------------------


class TestFmtCandidateSavingFloors:
    def test_lower_boundary_case_floors(self) -> None:
        assert _fmt_candidate_saving(Decimal("49.96")) == "49.9% lower"

    def test_higher_negative_case_unchanged_round_half_up(self) -> None:
        """A candidate that costs MORE (negative pct_val) keeps ROUND_HALF_UP:
        -12.96 -> "13.0% higher", not truncated to "12.9% higher"."""
        assert _fmt_candidate_saving(Decimal("-12.96")) == "13.0% higher"


# ---------------------------------------------------------------------------
# Integration: every report surface shows the floored figure
# ---------------------------------------------------------------------------

# Fixture engineered so the full-dataset split saving is exactly 49.96%:
# current = total_cost = $1000.00 (all cost on the baseline model, no
# already-cheap bucket); blended = $500.40; saved = $499.60 -> 49.96%.
# ROUND_HALF_UP would print "50.0%"; floor must print "49.9%".
# projected_cost == blended_cost so the wholesale "Upper bound" note (which
# quotes a separately-computed, non-reconciled percent) does not also fire
# and confuse the assertions below.


def _split() -> SplitRouting:
    return SplitRouting(
        baseline_model="gpt-4-turbo",
        candidate_model="gpt-4o-mini",
        routed_count=70,
        kept_count=30,
        routed_cost=Decimal("350.00"),
        kept_cost=Decimal("150.40"),
        baseline_cost=Decimal("1000.00"),
        blended_cost=Decimal("500.40"),
        easy_threshold=Decimal("0.35"),
    )


def _result(**kwargs: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "total_calls": 100,
        "priced_calls": 100,
        "unpriced_calls": 0,
        "total_cost": Decimal("1000.00"),
        "cost_by_model": {"gpt-4-turbo": Decimal("1000.00")},
        "calls_by_model": {"gpt-4-turbo": 100},
        "projected_cost": Decimal("500.40"),
        "candidate_model": "gpt-4o-mini",
        "observed_span_days": 7.0,
        "split": _split(),
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _render_terminal_text(result: AnalysisResult) -> str:
    """Render the terminal split view to plain text via a fixed-width console."""
    import io
    import sys

    from rich.console import Console

    report_mod = sys.modules[render_terminal.__module__]
    buf = io.StringIO()
    console = Console(
        file=buf,
        width=88,
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
        render_terminal(result, verbose=True)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    return buf.getvalue()


class TestFloorSavingPctAcrossSurfaces:
    """Every report surface (terminal, md v1/v2, html v1/v2) shows the FLOORED
    49.9% figure for the 49.96%-exact fixture — never the rounded 50.0%."""

    def test_terminal(self) -> None:
        text = _render_terminal_text(_result())
        assert "49.9%" in text
        assert "50.0%" not in text

    def test_markdown_v1(self, tmp_path: Path) -> None:
        out = tmp_path / "r.md"
        render_markdown(_result(), out)
        text = out.read_text(encoding="utf-8")
        assert "49.9%" in text
        assert "50.0%" not in text

    def test_markdown_v2(self, tmp_path: Path) -> None:
        out = tmp_path / "r.md"
        render_markdown_v2(_result(), out)
        text = out.read_text(encoding="utf-8")
        assert "49.9%" in text
        assert "50.0%" not in text

    def test_html_v1(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html(_result(), out)
        text = out.read_text(encoding="utf-8")
        assert "49.9%" in text
        assert "50.0%" not in text

    def test_html_v2(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html_v2(_result(), out)
        text = out.read_text(encoding="utf-8")
        assert "49.9%" in text
        assert "50.0%" not in text
