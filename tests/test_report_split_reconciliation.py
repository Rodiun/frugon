"""Numeric reconciliation guard: every report variant's split figures must
reconcile to the FULL analyzed dataset, identical to the terminal panel.

This is the guard that should have existed for the flagged correctness bug
(the HTML/Markdown reports reported the baseline-model-only spend under
"Current", dropped the already-on-a-cheaper-model bucket, and divided the saving
by the baseline cost instead of the total — so a $800 dataset rendered as
"$777 → 35% saved" instead of the correct "$800 → 34% saved").

The tests below assert, for the real ``--demo`` result (56,100 priced calls) AND
for both projected and non-projected synthetic fixtures, that in EACH of the four
renderers (Markdown v1/v2, HTML v1/v2):

  * the rendered "Current" equals ``result.monthly_cost`` (projected) /
    ``result.total_cost`` (otherwise) — i.e. the sum of the Cost-by-model rows,
    NOT ``split.monthly_baseline`` / ``split.baseline_cost``;
  * the rendered saving percent equals the terminal's saving percent (both
    computed from the same shared :func:`_split_report_figures`);
  * the routing-plan bucket call counts sum to ``result.priced_calls`` (no
    analyzed call silently vanishes); and
  * the already-on-a-cheaper-model bucket is present whenever
    ``already_cheap > 0``.

The cross-check tests assert report-Current == terminal-Current and
report-saving% == terminal-saving% on the demo result, pinning the renderers to
the terminal so they can never drift apart again.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import frugon
from frugon.cost import AnalysisResult, analyze_records, iter_records
from frugon.report import (
    _split_report_figures,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
)
from frugon.routing import SplitRouting

# Resolve the bundled demo sample exactly as the CLI does (``--demo``).
assert frugon.__file__ is not None
_SAMPLE = Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"


def _demo_result() -> AnalysisResult:
    """Build the real ``--demo`` AnalysisResult via the same engine the CLI uses."""
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(
        records,
        skipped_malformed=skipped,
        split_routing=True,
    )


# --- Synthetic fixtures: one non-projected, one projected -------------------


def _split(**kw: Any) -> SplitRouting:
    defaults: dict[str, Any] = {
        "baseline_model": "gpt-4-turbo",
        "candidate_model": "gpt-4o-mini",
        "routed_count": 24,
        "kept_count": 3,
        # Production-magnitude figures so the TOTAL vs baseline-only distinction survives
        # whole-dollar rounding in html v2 (the value the buggy renderers got
        # wrong).  baseline_cost $750 != total_cost $800 (the $50 already-cheap
        # bucket); blended_cost $501.62 is the baseline-only blended.
        "routed_cost": Decimal("12.3400"),
        "kept_cost": Decimal("489.2802"),
        "baseline_cost": Decimal("750.0000"),
        "blended_cost": Decimal("501.6202"),
        "easy_threshold": Decimal("0.35"),
    }
    defaults.update(kw)
    return SplitRouting(**defaults)


def _result(**kw: Any) -> AnalysisResult:
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
        "split": _split(),
    }
    defaults.update(kw)
    return AnalysisResult(**defaults)


def _projected_result(**kw: Any) -> AnalysisResult:
    """Same shape but WITH a monthly projection (the projected figures path)."""
    return _result(
        monthly_cost=Decimal("1600.0000"),
        monthly_projected=Decimal("660.0000"),
        split=_split(
            monthly_baseline=Decimal("1500.0000"),
            monthly_blended=Decimal("1003.2404"),
        ),
        **kw,
    )


_RENDERERS = {
    "md_v1": (render_markdown, "r.md", False),
    "md_v2": (render_markdown_v2, "r.md", False),
    "html_v1": (render_html, "r.html", True),
    "html_v2": (render_html_v2, "r.html", True),
}


def _render(name: str, result: AnalysisResult, tmp_path: Path) -> str:
    renderer, fname, _ = _RENDERERS[name]
    out = tmp_path / fname
    renderer(result, out)
    return out.read_text(encoding="utf-8")


# Map a renderer name to the call-count tokens that, summed, must equal
# priced_calls.  For markdown the routing-plan rows carry the counts in a
# pipe-table; for HTML they carry them in c-calls cells (v2) or stat values (v1).


def _bucket_counts(name: str, html: str) -> list[int]:
    if name.startswith("md"):
        # Routing-plan rows: "| Routed · easy | 36,100 | ...", "| Keep · already
        # optimal | 10,000 | ...", "| Kept · hard | 10,000 | ...".  Counts carry
        # comma thousand-separators on every surface, so strip them before int().
        # Exclude the **Blended** total row.
        counts: list[int] = []
        for line in html.splitlines():
            m = re.match(
                r"\| (Routed · easy|Kept · hard|Keep · already optimal) \| ([\d,]+) \|", line
            )
            if m:
                counts.append(int(m.group(2).replace(",", "")))
        return counts
    if name == "html_v2":
        # Routing-plan c-calls cells, excluding the total row (class="total").
        rows = re.findall(r"<tr(?: class=\"total\")?>.*?</tr>", html, re.S)
        counts = []
        for row in rows:
            if 'class="total"' in row:
                continue
            m = re.search(r'c-calls">([\d,]+)</td>', row)
            if m and ("bucket" in row):
                counts.append(int(m.group(1).replace(",", "")))
        return counts
    # html_v1: the routing-plan table — each bucket row leads with a
    # <td class="bucket">…</td> label then the Calls <td class="num">N</td>.
    # Excludes the Blended total row (class="row-total").  This is the same
    # information-parity table the Markdown and HTML v2 surfaces carry.
    counts = []
    for row in re.findall(r"<tr(?: class=\"row-total\")?>.*?</tr>", html, re.S):
        if 'class="row-total"' in row:
            continue
        if 'class="bucket"' not in row:
            continue
        m = re.search(r'<td class="num">([\d,]+)</td>', row)
        if m:
            counts.append(int(m.group(1).replace(",", "")))
    return counts


def _headline_current(name: str, html: str) -> Decimal:
    """Extract the rendered HEADLINE 'Current' dollar figure for a variant.

    This targets the decision headline specifically (the bottom-line before/after
    for markdown, the Current stat for html v1, the before/after Current block for
    html v2) — NOT the cost-by-model table, which legitimately repeats per-model
    dollar figures that can coincide with the baseline cost in a small fixture.
    """
    if name.startswith("md"):
        # Bottom line: "... save ~N.N% ($<current>/?mo → $<blended>/?mo)."  The
        # saving percent renders at one decimal, so match a float here.
        m = re.search(r"save ~[\d.]+% \(\$([\d,.]+)(?:/mo)? &?(?:rarr|→)", html)
        if m is None:
            m = re.search(r"save ~[\d.]+% \(\$([\d,.]+)", html)
        assert m, f"no markdown headline current in:\n{html[:400]}"
        return Decimal(m.group(1).replace(",", ""))
    if name == "html_v1":
        m = re.search(
            r"Current cost</div><div class=\"stat-value\">\$([\d,.]+)", html
        )
        assert m, "no html v1 Current stat"
        return Decimal(m.group(1).replace(",", ""))
    # html_v2 before/after Current block.
    m = re.search(
        r"ba-label\">Current</span><span class=\"ba-value tnum\">\$([\d,.]+)", html
    )
    assert m, "no html v2 Current block"
    return Decimal(m.group(1).replace(",", ""))


def _rendered_saving_pcts(html: str) -> list[float]:
    """All saving-percent figures a renderer emits, parsed as floats.

    The renderers diverge by DESIGN on display precision: the HTML v2 routing-plan
    delta now renders the saving to one decimal ("-34.5%") while the terminal and
    the other surfaces render the integer ("34%").  This guard therefore compares
    the underlying NUMERIC value (parsed from whatever precision is rendered),
    never the rendered string — so the intended divergence cannot fail it.

    The saving percent appears as the negative delta ("-34.5%" / "−-style minus"),
    the hero "N% saved", the markdown "(−N%)" / "save ~N%", or the v1
    saving-hero.  The upper-bound figure is glossed "~N%" with an explicit
    "Upper bound" / "moving every call" preface; we exclude it so only the saving
    is compared.
    """
    # Strip the upper-bound sentence so its "~70%" never reads as a saving.
    cleaned = re.sub(r"[Uu]pper bound:.*?recommendation", " ", html, flags=re.S)
    cleaned = re.sub(r"moving every call.*?saves ~\d+(?:\.\d+)?%", " ", cleaned, flags=re.S)
    # Normalise the HTML minus entity and any stray markup so the numbers parse.
    cleaned = cleaned.replace("&minus;", "-").replace("−", "-")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    out: list[float] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", cleaned):
        out.append(float(m.group(1)))
    return out


@pytest.mark.parametrize("name", list(_RENDERERS))
@pytest.mark.parametrize(
    "builder", [_result, _projected_result], ids=["non_projected", "projected"]
)
class TestSplitReportReconciliation:
    def test_current_equals_total_not_baseline(
        self, name: str, builder: Any, tmp_path: Path
    ) -> None:
        """Rendered HEADLINE 'Current' == result total (monthly_cost / total_cost),
        NOT the baseline-model-only figure."""
        result = builder()
        fig = _split_report_figures(result, result.split)  # type: ignore[arg-type]
        baseline_only = (
            result.split.monthly_baseline if fig.projected else result.split.baseline_cost  # type: ignore[union-attr]
        )
        # Fixture sanity: the total and the baseline-only figure genuinely differ,
        # so this test cannot pass vacuously.
        assert fig.current != baseline_only
        html = _render(name, result, tmp_path)
        rendered = _headline_current(name, html)
        # The headline Current reconciles to the TOTAL (within the variant's
        # display rounding), and is NOT the baseline-only figure.
        assert abs(rendered - fig.current) <= Decimal("0.5")
        assert abs(rendered - baseline_only) > Decimal("0.5")

    def test_saving_pct_matches_terminal(
        self, name: str, builder: Any, tmp_path: Path
    ) -> None:
        """Rendered saving percent == the terminal's saving percent (both from the
        shared full-dataset figures), NOT split.saving_pct (baseline-only)."""
        result = builder()
        fig = _split_report_figures(result, result.split)  # type: ignore[arg-type]
        terminal_val = float(fig.total_pct)
        baseline_val = float(result.split.saving_pct)  # type: ignore[union-attr]
        html = _render(name, result, tmp_path)
        rendered = _rendered_saving_pcts(html)
        # NUMERIC comparison, not string: a rendered saving figure must match the
        # terminal's saving percent within display rounding (the HTML v2 routing
        # plan may render one decimal while the terminal renders the integer).
        assert any(abs(v - terminal_val) <= 0.05 or round(v) == round(terminal_val) for v in rendered), (
            name, rendered, terminal_val,
        )
        # Guard the fixture genuinely distinguishes the two percentages so this
        # test cannot pass vacuously.
        assert round(terminal_val) != round(baseline_val)
        # The baseline-basis saving must NOT be the rendered saving (numeric).
        assert not any(abs(v - baseline_val) <= 0.05 for v in rendered), (name, rendered, baseline_val)

    def test_buckets_sum_to_priced_calls(
        self, name: str, builder: Any, tmp_path: Path
    ) -> None:
        """routed + kept + already-cheap (the rendered routing buckets) ==
        result.priced_calls — no analyzed call silently vanishes."""
        result = builder()
        html = _render(name, result, tmp_path)
        counts = _bucket_counts(name, html)
        assert sum(counts) == result.priced_calls, (name, counts)

    def test_already_cheap_bucket_present(
        self, name: str, builder: Any, tmp_path: Path
    ) -> None:
        """The already-on-a-cheaper-model bucket renders whenever already_cheap > 0."""
        result = builder()
        fig = _split_report_figures(result, result.split)  # type: ignore[arg-type]
        assert fig.already_cheap > 0  # fixture precondition
        html = _render(name, result, tmp_path)
        # Each variant names the already-optimal bucket distinctly.
        assert ("already optimal" in html.lower()) or ("Already optimal" in html)


# --- Cross-check on the REAL demo result: report == terminal ----------------


class TestDemoCrossCheck:
    """The flagged numbers, pinned on the real --demo dataset (56,100 calls)."""

    def test_demo_figures_are_the_known_good_values(self) -> None:
        result = _demo_result()
        assert result.split is not None
        fig = _split_report_figures(result, result.split)
        # The cost engine: total == sum of cost-by-model rows == monthly_cost.
        assert result.monthly_cost is not None
        assert sum(result.cost_by_model.values(), Decimal("0")) == result.total_cost
        assert fig.current == result.monthly_cost
        # Known-good headline figures (whole-dollar).
        assert round(fig.current) == 390
        assert round(fig.blended) == 254
        assert round(fig.saved) == 136
        assert round(float(fig.total_pct)) == 35
        # Routing buckets reconcile to ALL analyzed calls.
        assert result.priced_calls == 56100
        assert (
            result.split.routed_count + result.split.kept_count + fig.already_cheap
            == result.priced_calls
        )
        assert fig.already_cheap == 10000

    @pytest.mark.parametrize("name", list(_RENDERERS))
    def test_demo_report_current_equals_terminal_current(
        self, name: str, tmp_path: Path
    ) -> None:
        """report-Current == terminal-Current (== monthly_cost == $390) for --demo."""
        result = _demo_result()
        assert result.split is not None
        assert result.monthly_cost is not None
        html = _render(name, result, tmp_path)
        # The HEADLINE Current reconciles to the terminal's Current (== $390),
        # NOT the baseline-only monthly figure.
        rendered = _headline_current(name, html)
        # Every surface now renders money at the full _fmt_usd 4-dp precision —
        # the terminal panel, both Markdown styles, and both HTML styles read
        # IDENTICALLY (e.g. "$389.8849"), so the headline Current equals the
        # terminal's monthly_cost at 4-dp on every variant.
        assert rendered == result.monthly_cost.quantize(Decimal("0.0001"))
        assert round(rendered) == 390
        assert abs(rendered - result.split.monthly_baseline) > Decimal("1")  # type: ignore[union-attr]

    @pytest.mark.parametrize("name", list(_RENDERERS))
    def test_demo_report_saving_pct_equals_terminal(
        self, name: str, tmp_path: Path
    ) -> None:
        """report-saving% == terminal-saving% (== ~34.8%) for --demo.

        The chatgpt-4o-latest demo fixture has total_pct=34.83% and
        split.saving_pct=34.95% — both round to 35%, so the integer test is no
        longer diagnostic.  Instead we verify that at least one rendered figure
        is within 0.5% of the terminal_val, and that the rendered figure equals
        the total-basis saving (fig.total_pct) not a materially different figure.
        """
        result = _demo_result()
        fig = _split_report_figures(result, result.split)  # type: ignore[arg-type]
        html = _render(name, result, tmp_path)
        terminal_val = float(fig.total_pct)  # 34.83%
        rendered = _rendered_saving_pcts(html)
        # At least one rendered saving figure is within 0.5% of the terminal value.
        assert any(abs(v - terminal_val) < 0.5 for v in rendered), (
            name, rendered, terminal_val
        )
        # Basis guard (figure-source level): the report's saving is the TOTAL-basis
        # figure (fig.total_pct ~34.83%), a DISTINCT computation from the baseline-
        # basis split saving (result.split.saving_pct ~34.95%). For THIS fixture the
        # two bases coincide once rounded to 35%, so the rendered string alone cannot
        # discriminate them — assert the distinction at the source so a regression
        # that collapses the report onto the baseline basis is still caught. A
        # basis-DIVERGENT demo fixture would let the rendered string discriminate
        # directly (tracked in ROADMAP).
        assert result.split is not None
        assert fig.total_pct != result.split.saving_pct

    @pytest.mark.parametrize("name", list(_RENDERERS))
    def test_demo_report_buckets_sum_to_priced_calls(
        self, name: str, tmp_path: Path
    ) -> None:
        result = _demo_result()
        html = _render(name, result, tmp_path)
        counts = _bucket_counts(name, html)
        assert sum(counts) == result.priced_calls == 56100, (name, counts)


class TestHtmlV1InformationParity:
    """HTML v1 carries the SAME figures as the Markdown + HTML v2 surfaces (F4).

    The v1 surface previously omitted the saving dollar amount, the per-bucket
    costs, the per-bucket call shares, the routing-plan blended total row, and the
    cost-by-model Total row — so the same --demo dataset told a thinner story in
    v1.  These guard the restored parity.
    """

    # The exact full-precision figures the --demo dataset produces (4-dp money,
    # commas, 1-dp percent), shared verbatim across every surface.
    # chatgpt-4o-latest baseline: saving=$135.7892, routed=$4.9105,
    # kept=$247.8250, already-optimal=$1.3602, call shares 64.4%/17.8%.
    _DEMO_FIGURES = ("135.7892", "4.9105", "247.8250", "1.3602", "64.4%", "17.8%")

    def test_v1_html_contains_saving_and_per_bucket_figures(
        self, tmp_path: Path
    ) -> None:
        html = _render("html_v1", _demo_result(), tmp_path)
        # Saving dollar amount beside the hero.
        assert "135.7892" in html
        # Per-bucket costs.
        assert "4.9105" in html  # routed
        assert "247.8250" in html  # kept
        assert "1.3602" in html  # already-optimal
        # Per-bucket call shares + the share bar markup.
        assert "64.4%" in html
        assert "17.8%" in html
        assert "share-bar" in html
        # Routing-plan blended total row + cost-by-model Total row.
        assert "Blended" in html
        assert ">Total<" in html or "Total</strong>" in html

    def test_demo_money_figures_string_equal_across_surfaces(
        self, tmp_path: Path
    ) -> None:
        result = _demo_result()
        md = _render("md_v1", result, tmp_path)
        v1 = _render("html_v1", result, tmp_path)
        v2 = _render("html_v2", result, tmp_path)
        for fig in self._DEMO_FIGURES:
            assert fig in md, (fig, "md")
            assert fig in v1, (fig, "html_v1")
            assert fig in v2, (fig, "html_v2")
