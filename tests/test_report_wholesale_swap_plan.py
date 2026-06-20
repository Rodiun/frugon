"""Tests for the wholesale swap-plan table in HTML v1, HTML v2, and Markdown reports.

The wholesale path (``--wholesale`` / when no split is available) now renders a
'Swap plan' table in every report surface — HTML v1, HTML v2, Markdown v1, and
Markdown v2 — mirroring the split 'Routing plan' section as visual siblings.

Fixture: two-model result with gpt-4-turbo (46,100 calls, $777.00) and gpt-4o
(10,000 calls, $23.00), candidate=gpt-4o, total 56,100 calls, $800.00 total cost.
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _result_wholesale(**kwargs: Any) -> AnalysisResult:
    """Two-model wholesale fixture: gpt-4-turbo baseline + gpt-4o candidate."""
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


def _split() -> SplitRouting:
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


def _result_with_split(**kwargs: Any) -> AnalysisResult:
    """Split-routing result — wholesale code paths are skipped for this."""
    defaults: dict[str, Any] = {
        "total_calls": 37,
        "priced_calls": 37,
        "unpriced_calls": 0,
        "total_cost": Decimal("0.0676"),
        "cost_by_model": {"gpt-4-turbo": Decimal("0.0650"), "gpt-4o": Decimal("0.0026")},
        "calls_by_model": {"gpt-4-turbo": 27, "gpt-4o": 10},
        "projected_cost": Decimal("0.0222"),
        "candidate_model": "gpt-4o",
        "observed_span_days": 7.0,
        "split": _split(),
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


# ---------------------------------------------------------------------------
# HTML v1 — render_html
# ---------------------------------------------------------------------------


class TestHtmlV1WholesaleSwapPlan:
    """The v1 HTML swap-plan card is present for wholesale runs."""

    def test_swap_plan_section_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v1.html"
        render_html(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "Swap plan" in html

    def test_model_rows_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v1.html"
        render_html(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "gpt-4-turbo" in html
        assert "gpt-4o" in html

    def test_candidate_row_shows_already_on_badge(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v1.html"
        render_html(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "already on" in html
        assert "gpt-4o" in html

    def test_non_candidate_row_shows_swap_to(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v1.html"
        render_html(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "swap to" in html

    def test_total_row_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v1.html"
        render_html(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "Total" in html
        assert "56,100" in html

    def test_column_headers_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v1.html"
        render_html(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "% calls" in html
        assert "Current cost" in html
        assert "Action" in html


# ---------------------------------------------------------------------------
# HTML v2 — render_html_v2
# ---------------------------------------------------------------------------


class TestHtmlV2WholesaleSwapPlan:
    """The v2 HTML swap-plan section is present for wholesale runs."""

    def test_swap_plan_section_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.html"
        render_html_v2(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "Swap plan" in html

    def test_model_rows_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.html"
        render_html_v2(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "gpt-4-turbo" in html
        assert "gpt-4o" in html

    def test_candidate_row_shows_already_on_target(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.html"
        render_html_v2(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "already on target" in html

    def test_non_candidate_row_shows_swap_to(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.html"
        render_html_v2(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "swap to" in html

    def test_total_row_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.html"
        render_html_v2(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "Total" in html
        assert "56,100" in html

    def test_below_plan_full_section(self, tmp_path: Path) -> None:
        """The v2 swap plan uses the plan-full below-the-fold section class."""
        out = tmp_path / "ws.v2.html"
        render_html_v2(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        assert "plan-full" in html


# ---------------------------------------------------------------------------
# Markdown v1 — render_markdown
# ---------------------------------------------------------------------------


class TestMarkdownV1WholesaleSwapPlan:
    """The Markdown v1 ## Swap plan section is present for wholesale runs."""

    def test_swap_plan_heading_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Swap plan" in md

    def test_model_rows_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "`gpt-4-turbo`" in md
        assert "`gpt-4o`" in md

    def test_candidate_row_shows_already_on(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "already on" in md

    def test_non_candidate_row_shows_swap_to(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "→ swap to" in md

    def test_total_row_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "**Total**" in md
        assert "56,100" in md

    def test_table_headers_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "% calls" in md
        assert "Current cost" in md
        assert "Action" in md


# ---------------------------------------------------------------------------
# Markdown v2 — render_markdown_v2
# ---------------------------------------------------------------------------


class TestMarkdownV2WholesaleSwapPlan:
    """The Markdown v2 ## Swap plan section is present for wholesale runs."""

    def test_swap_plan_heading_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.md"
        render_markdown_v2(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Swap plan" in md

    def test_model_rows_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.md"
        render_markdown_v2(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "`gpt-4-turbo`" in md
        assert "`gpt-4o`" in md

    def test_candidate_row_shows_already_on(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.md"
        render_markdown_v2(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "already on" in md

    def test_non_candidate_row_shows_swap_to(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.md"
        render_markdown_v2(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "→ swap to" in md

    def test_total_row_present(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.v2.md"
        render_markdown_v2(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "**Total**" in md
        assert "56,100" in md


# ---------------------------------------------------------------------------
# Numeric parity
# ---------------------------------------------------------------------------


def _parse_swap_plan_rows_md(md: str) -> tuple[list[int], list[Decimal], int, Decimal]:
    """Parse the swap-plan table from rendered Markdown output.

    Returns (row_calls, row_costs, total_calls, total_cost) where row_* are
    per-model values rendered in data rows and total_* come from the Total row.
    """
    row_calls: list[int] = []
    row_costs: list[Decimal] = []
    total_calls = 0
    total_cost = Decimal("0")
    in_plan = False
    for line in md.splitlines():
        if line.strip() == "## Swap plan":
            in_plan = True
            continue
        if not in_plan:
            continue
        if line.startswith("##"):
            break
        if line.startswith("| `"):
            cols = [c.strip() for c in line.split("|")]
            row_calls.append(int(cols[2].replace(",", "")))
            row_costs.append(Decimal(cols[4].lstrip("$")))
        elif line.startswith("| **Total**"):
            cols = [c.strip() for c in line.split("|")]
            total_calls = int(cols[2].replace("**", "").replace(",", ""))
            total_cost = Decimal(cols[4].replace("**", "").lstrip("$"))
    return row_calls, row_costs, total_calls, total_cost


class TestNumericParity:
    """Calls and costs in the swap plan reconcile to the rendered Total row."""

    def test_calls_sum_to_priced_calls_md(self, tmp_path: Path) -> None:
        """Rendered swap-plan row calls sum to the rendered Total row calls."""
        out = tmp_path / "parity_calls.md"
        render_markdown(_result_wholesale(), out)
        row_calls, _, total_calls, _ = _parse_swap_plan_rows_md(out.read_text(encoding="utf-8"))
        assert sum(row_calls) == total_calls

    def test_cost_sum_to_total_cost_md(self, tmp_path: Path) -> None:
        """Rendered swap-plan row costs sum to the rendered Total row cost."""
        out = tmp_path / "parity_cost.md"
        render_markdown(_result_wholesale(), out)
        _, row_costs, _, total_cost = _parse_swap_plan_rows_md(out.read_text(encoding="utf-8"))
        assert sum(row_costs) == total_cost

    def test_swap_plan_md_contains_correct_calls(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        assert "46,100" in md
        assert "10,000" in md

    def test_swap_plan_md_contains_correct_costs(self, tmp_path: Path) -> None:
        out = tmp_path / "ws.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        # _fmt_usd formats with 4 decimal places
        assert "777.0000" in md
        assert "23.0000" in md
        assert "800.0000" in md


# ---------------------------------------------------------------------------
# Cross-surface reconciliation
# ---------------------------------------------------------------------------


class TestCrossSurfaceReconciliation:
    """Swap-plan total cost matches the headline cost on every rendered surface."""

    def test_html_v1_swap_plan_total_matches_current_cost(self, tmp_path: Path) -> None:
        out = tmp_path / "rec.v1.html"
        render_html(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        swap_m = re.search(
            r'class="row-total".*?100\.0%.*?<strong>(\$[\d.]+)</strong>',
            html,
        )
        head_m = re.search(
            r'stat-label">Current cost</div><div class="stat-value">(\$[\d.]+)',
            html,
        )
        assert swap_m is not None, "swap-plan Total row not found in HTML v1"
        assert head_m is not None, "Current cost stat not found in HTML v1"
        assert swap_m.group(1) == head_m.group(1)

    def test_html_v2_swap_plan_total_matches_cost_by_model(self, tmp_path: Path) -> None:
        out = tmp_path / "rec.v2.html"
        render_html_v2(_result_wholesale(), out)
        html = out.read_text(encoding="utf-8")
        swap_m = re.search(
            r'<td class="num r tnum c-cost">(\$[\d.]+)</td><td class="c-status"></td>',
            html,
        )
        cbm_m = re.search(
            r'<td class="lbl">Total</td>.*?<td class="num r tnum">(\$[\d.]+)</td>'
            r'<td class="num r tnum">100\.0%</td>',
            html,
        )
        assert swap_m is not None, "swap-plan Total cost not found in HTML v2"
        assert cbm_m is not None, "cost-by-model Total not found in HTML v2"
        assert swap_m.group(1) == cbm_m.group(1)

    def test_md_v1_swap_plan_total_matches_current_cost(self, tmp_path: Path) -> None:
        out = tmp_path / "rec.md"
        render_markdown(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        swap_m = re.search(
            r'\| \*\*Total\*\* \| \*\*[\d,]+\*\* \| \*\*100\.0%\*\* \| \*\*(\$[\d.]+)\*\* \| \|',
            md,
        )
        cur_m = re.search(r'\*\*Current cost:\*\* (\$[\d.]+)', md)
        assert swap_m is not None, "swap-plan Total row not found in MD v1"
        assert cur_m is not None, "Current cost not found in MD v1 Summary"
        assert swap_m.group(1) == cur_m.group(1)

    def test_md_v2_swap_plan_total_matches_cost_by_model(self, tmp_path: Path) -> None:
        out = tmp_path / "rec.v2.md"
        render_markdown_v2(_result_wholesale(), out)
        md = out.read_text(encoding="utf-8")
        swap_m = re.search(
            r'\| \*\*Total\*\* \| \*\*[\d,]+\*\* \| \*\*100\.0%\*\* \| \*\*(\$[\d.]+)\*\* \| \|',
            md,
        )
        cbm_m = re.search(
            r'\| \*\*Total\*\* \| \*\*[\d,]+\*\* \| \*\*(\$[\d.]+)\*\* \| \*\*100\.0%\*\* \|',
            md,
        )
        assert swap_m is not None, "swap-plan Total row not found in MD v2"
        assert cbm_m is not None, "cost-by-model Total row not found in MD v2"
        assert swap_m.group(1) == cbm_m.group(1)


# ---------------------------------------------------------------------------
# Split runs unchanged
# ---------------------------------------------------------------------------


class TestSplitRunsUnchanged:
    """Split runs do not render a swap plan; routing plan is present instead."""

    def test_html_v1_no_swap_plan_for_split(self, tmp_path: Path) -> None:
        out = tmp_path / "split.v1.html"
        render_html(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        assert "Swap plan" not in html

    def test_html_v1_routing_plan_present_for_split(self, tmp_path: Path) -> None:
        out = tmp_path / "split.v1.html"
        render_html(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        # HTML v1 split uses "Split routing" eyebrow (not "Routing plan")
        assert "Split routing" in html or "tbl-plan" in html

    def test_html_v2_no_swap_plan_for_split(self, tmp_path: Path) -> None:
        out = tmp_path / "split.v2.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        assert "Swap plan" not in html

    def test_html_v2_routing_plan_present_for_split(self, tmp_path: Path) -> None:
        out = tmp_path / "split.v2.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        assert "Routing plan" in html

    def test_markdown_v1_no_swap_plan_for_split(self, tmp_path: Path) -> None:
        out = tmp_path / "split.md"
        render_markdown(_result_with_split(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Swap plan" not in md

    def test_markdown_v1_routing_plan_present_for_split(self, tmp_path: Path) -> None:
        out = tmp_path / "split.md"
        render_markdown(_result_with_split(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Routing plan" in md

    def test_markdown_v2_no_swap_plan_for_split(self, tmp_path: Path) -> None:
        out = tmp_path / "split.v2.md"
        render_markdown_v2(_result_with_split(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Swap plan" not in md

    def test_markdown_v2_routing_plan_present_for_split(self, tmp_path: Path) -> None:
        out = tmp_path / "split.v2.md"
        render_markdown_v2(_result_with_split(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Routing plan" in md


# ---------------------------------------------------------------------------
# No-candidate guard
# ---------------------------------------------------------------------------


class TestNoCandidateGuard:
    """When candidate_model is None, no swap plan table is rendered."""

    def _result_no_candidate(self) -> AnalysisResult:
        return _result_wholesale(candidate_model=None, projected_cost=Decimal("0"))

    def test_html_v1_no_swap_plan_without_candidate(self, tmp_path: Path) -> None:
        out = tmp_path / "nc.v1.html"
        render_html(self._result_no_candidate(), out)
        html = out.read_text(encoding="utf-8")
        assert "Swap plan" not in html

    def test_html_v2_no_swap_plan_without_candidate(self, tmp_path: Path) -> None:
        out = tmp_path / "nc.v2.html"
        render_html_v2(self._result_no_candidate(), out)
        html = out.read_text(encoding="utf-8")
        assert "Swap plan" not in html

    def test_markdown_v1_no_swap_plan_without_candidate(self, tmp_path: Path) -> None:
        out = tmp_path / "nc.md"
        render_markdown(self._result_no_candidate(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Swap plan" not in md

    def test_markdown_v2_no_swap_plan_without_candidate(self, tmp_path: Path) -> None:
        out = tmp_path / "nc.v2.md"
        render_markdown_v2(self._result_no_candidate(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Swap plan" not in md
