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
    """Build the real ``--demo`` AnalysisResult via the same engine the CLI uses.

    ``candidates=None`` matches the CLI: ``--demo`` now runs the SAME default
    ``_ROUTING_CANDIDATES`` pool as a real analyze (no demo-only pin).
    """
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(
        records,
        skipped_malformed=skipped,
        split_routing=True,
        candidates=None,
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
        # fig.current and fig.blended are now pre-rounded to 2 dp inside
        # _split_report_figures (the reconciling rounding).
        assert fig.current == Decimal("549.46")
        assert fig.blended == Decimal("343.91")
        # SAVING == Current − New exactly (no independent rounding divergence).
        assert fig.saved == fig.current - fig.blended
        assert fig.saved == Decimal("205.55")
        assert round(float(fig.total_pct)) == 37  # ~37.4%
        # Routing buckets reconcile to ALL analyzed calls.
        assert result.priced_calls == 56100
        assert (
            result.split.routed_count + result.split.kept_count + fig.already_cheap
            == result.priced_calls
        )
        assert fig.already_cheap == 10000

    def test_demo_saving_reconciles_with_displayed_current_and_new(self) -> None:
        """SAVING == Current − New, verifiable from the printed 2-dp figures.

        This is the core reconciling invariant: because _split_report_figures
        rounds current and blended (≡ New) to 2 dp BEFORE deriving saved, the
        SAVING printed on every report surface equals the difference the reader
        can compute mentally from the Current and New figures.  For the current
        demo dataset (gpt-5.5 → deepseek-v4-flash), Current=$549.46, New=$343.91,
        SAVING=$205.55 — these happen not to straddle a rounding boundary, so
        independent rounding of the raw saving would also produce $205.55 (no
        $0.01 divergence).  The invariant still guards against a regression that
        derives saved independently rather than from the shared rounded components.
        """
        result = _demo_result()
        assert result.split is not None
        fig = _split_report_figures(result, result.split)
        # The figures are already at 2 dp precision (quantized in the helper).
        assert fig.current == fig.current.quantize(Decimal("0.01"))
        assert fig.blended == fig.blended.quantize(Decimal("0.01"))
        # SAVING is derived — never independently rounded.
        assert fig.saved == fig.current - fig.blended

    def test_demo_terminal_panel_saving_reconciles_to_printed_current_minus_new(
        self,
    ) -> None:
        """Terminal --demo panel: printed SAVING == printed Current − printed New.

        Regression lock for the divergence where ``_render_split_panel`` computed
        ``saved`` from RAW components and rounded it independently.  The terminal
        must agree with the reports to the cent (it now consumes
        ``_split_report_figures``, the one shared figure source).

        Demo dataset (gpt-5.5 → deepseek-v4-flash): Current=$549.46, New=$343.91,
        SAVING=$205.55, 37.4% lower.  The new figures happen not to straddle a
        rounding boundary (raw saved≈$205.55 also rounds to $205.55), but the
        invariant guard — asserting that printed SAVING == printed Current − New —
        still catches any regression that re-derives saved independently rather than
        consuming the shared rounded components from _split_report_figures.
        """
        from rich.console import Console

        from frugon import report

        result = _demo_result()
        assert result.split is not None
        console = Console(width=200, force_terminal=False, no_color=True)
        captured: list[str] = []
        original = report.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            with console.capture() as cap:
                console.print(*args, **kwargs)
            captured.append(cap.get())

        report.rprint = _rp  # type: ignore[assignment]
        try:
            report._render_split_panel(result, result.split)
        finally:
            report.rprint = original

        out = " ".join("".join(captured).split())
        assert "$549.46" in out  # Current
        assert "$343.91" in out  # New (blended)
        assert "$205.55" in out  # SAVING == 549.46 − 343.91 (reconciles)
        # Invariant: the reconciled SAVING is present; no independently-computed
        # value that disagrees with Current − New should appear.  For this dataset
        # the raw saving rounds to the same $205.55, so the "wrong" value is also
        # $205.55 — the guard is the positive assertion above, which fails if a
        # regression produces a different value (e.g. $205.54 or $205.56).
        assert "37.4%" in out

    @pytest.mark.parametrize("name", list(_RENDERERS))
    def test_demo_report_current_equals_terminal_current(
        self, name: str, tmp_path: Path
    ) -> None:
        """report-Current == terminal-Current (== $549.46) for --demo.

        Since v0.1.3, money displays at 2 dp.  The headline Current on every
        surface is $549.46 (the 2-dp quantization of the raw monthly_cost for the
        gpt-5.5 → deepseek-v4-flash demo dataset).
        """
        result = _demo_result()
        assert result.split is not None
        assert result.monthly_cost is not None
        fig = _split_report_figures(result, result.split)
        html = _render(name, result, tmp_path)
        # The HEADLINE Current reconciles to the terminal's Current ($549.46),
        # NOT the baseline-only monthly figure ($548.55 — the already-cheap
        # slice is now priced on the cheaper deepseek-v4-flash, so the two
        # totals sit only ~$0.91 apart; the regression this guards against is
        # rendering the WRONG one, not a large gap, so the guard asserts they
        # are genuinely distinct values rather than pinning an arbitrary gap
        # size that a cheaper candidate can shrink below).
        rendered = _headline_current(name, html)
        # All surfaces render at 2-dp precision.
        assert rendered == Decimal("549.46")
        assert rendered == fig.current
        assert round(rendered) == 549
        assert rendered != result.split.monthly_baseline.quantize(Decimal("0.01"))  # type: ignore[union-attr]

    @pytest.mark.parametrize("name", list(_RENDERERS))
    def test_demo_report_saving_pct_equals_terminal(
        self, name: str, tmp_path: Path
    ) -> None:
        """report-saving% == terminal-saving% (== ~37.4%) for --demo.

        The gpt-5.5 → deepseek-v4-flash demo fixture has total_pct≈37.41% and
        split.saving_pct≈37.47% — both round to 37%, so the integer test
        is not diagnostic.  Instead we verify that at least one rendered figure
        is within 0.5% of the terminal_val, and that the rendered figure equals
        the total-basis saving (fig.total_pct) not a materially different figure.
        """
        result = _demo_result()
        fig = _split_report_figures(result, result.split)  # type: ignore[arg-type]
        html = _render(name, result, tmp_path)
        terminal_val = float(fig.total_pct)  # ~37.41%
        rendered = _rendered_saving_pcts(html)
        # At least one rendered saving figure is within 0.5% of the terminal value.
        assert any(abs(v - terminal_val) < 0.5 for v in rendered), (
            name, rendered, terminal_val
        )
        # Basis guard (figure-source level): the report's saving is the TOTAL-basis
        # figure (fig.total_pct ~37.41%), a DISTINCT computation from the baseline-
        # basis split saving (result.split.saving_pct ~37.47%). For THIS fixture the
        # two bases coincide once rounded to 37%, so the rendered string alone cannot
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


class TestVerboseSplitPctMatchesPanel:
    """Render-level regression: the verbose 'Upper bound' note's printed split-%
    equals the hero panel's printed split-%, even when the raw (unrounded)
    saving-% and the rounded-component saving-% diverge at 1 decimal place.

    This is the percent analogue of the $135.78-vs-$135.79 dollar-rounding case:
    both the panel hero and the verbose note read _split_report_figures().total_pct,
    so they always agree; a naive raw re-derivation inside the note would produce a
    different 1dp string and the user would read contradictory figures.

    Fixture design:
      total_cost = 2.005  → quantize(0.01, ROUND_HALF_UP) = 2.01
      blended_raw = total_cost - (baseline_cost - blended_cost)
                  = 2.005 - (2.000 - 1.000) = 1.005
                  → quantize(0.01, ROUND_HALF_UP) = 1.01
      raw  saving-% = (2.005 - 1.005) / 2.005 * 100 = 49.875... → 1dp "49.9%"
      rec. saving-% = (2.01  - 1.01)  / 2.01  * 100 = 49.7512...
                    → floored (never rounded up) 1dp "49.7%"
    These two differ at 1dp, proving the fixture exercises the divergence.
    """

    # The fixture costs (class constants so the divergence proof re-uses them).
    _TOTAL_COST = Decimal("2.005")
    _BASELINE_COST = Decimal("2.000")
    _BLENDED_COST = Decimal("1.000")

    def _divergence_fixture(self) -> tuple[AnalysisResult, SplitRouting]:
        """Build result + split engineered so raw-% and reconciled-% differ at 1dp."""
        split = _split(
            baseline_model="gpt-4o",
            candidate_model="gpt-4o-mini",
            routed_count=20,
            kept_count=5,
            routed_cost=Decimal("0.400"),
            kept_cost=Decimal("0.600"),
            baseline_cost=self._BASELINE_COST,
            blended_cost=self._BLENDED_COST,
            easy_threshold=Decimal("0.35"),
        )
        result = _result(
            total_calls=25,
            priced_calls=25,
            unpriced_calls=0,
            total_cost=self._TOTAL_COST,
            cost_by_model={"gpt-4o": self._TOTAL_COST},
            calls_by_model={"gpt-4o": 25},
            # projected_cost drives the wholesale upper-bound note; must produce a
            # wholesale_saving > split_total_pct (49.8%) so the note renders.
            # (2.005 - 0.900) / 2.005 * 100 = 55.1% > 49.8% ✓
            projected_cost=Decimal("0.900"),
            candidate_model="gpt-4o-mini",
            observed_span_days=7.0,
            split=split,
        )
        return result, split

    def _capture_verbose(self, result: AnalysisResult) -> str:
        """Render _render_split_verbose via the rprint-monkeypatch capture pattern."""
        from rich.console import Console

        from frugon import report

        console = Console(width=200, force_terminal=False, no_color=True)
        captured: list[str] = []
        original = report.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            with console.capture() as cap:
                console.print(*args, **kwargs)
            captured.append(cap.get())

        report.rprint = _rp  # type: ignore[assignment]
        try:
            assert result.split is not None
            report._render_split_verbose(result, result.split)
        finally:
            report.rprint = original
        return " ".join("".join(captured).split())

    def _capture_panel(self, result: AnalysisResult) -> str:
        """Render _render_split_panel via the rprint-monkeypatch capture pattern."""
        from rich.console import Console

        from frugon import report

        console = Console(width=200, force_terminal=False, no_color=True)
        captured: list[str] = []
        original = report.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            with console.capture() as cap:
                console.print(*args, **kwargs)
            captured.append(cap.get())

        report.rprint = _rp  # type: ignore[assignment]
        try:
            assert result.split is not None
            report._render_split_panel(result, result.split)
        finally:
            report.rprint = original
        return " ".join("".join(captured).split())

    def test_verbose_split_pct_equals_panel_pct_on_divergent_fixture(
        self,
    ) -> None:
        """Verbose 'Upper bound' note's split-% == panel hero's split-%, and both
        equal _split_report_figures().total_pct, on a fixture where the raw
        (unrounded) saving-% and the reconciled saving-% diverge at 1dp.

        This is a render-level test: both surfaces are rendered via the
        rprint-monkeypatch capture pattern and the printed XX.X% strings are
        compared — not an assertion on the helper's return value alone.
        """
        result, split = self._divergence_fixture()

        # --- Part 1: prove the fixture genuinely produces a 1dp divergence. ---
        # If this assertion fails, the fixture no longer exercises the bug class
        # and the rest of the test would be a no-op.
        current_raw = self._TOTAL_COST  # 2.005
        blended_raw = current_raw - (self._BASELINE_COST - self._BLENDED_COST)  # 1.005
        raw_pct_value = float(
            (current_raw - blended_raw) / current_raw * Decimal("100")
        )
        raw_pct_1dp = round(raw_pct_value, 1)  # 49.9

        from frugon.cost import _display_pct
        from frugon.report import _split_report_figures

        fig = _split_report_figures(result, split)
        reconciled_pct_value = float(fig.total_pct)
        # The ACTUAL displayed figure floors (truncates toward zero), not a
        # naive round-half-even/up — this must match what every renderer prints.
        displayed_pct = _display_pct(fig.total_pct)  # Decimal("49.7")

        assert raw_pct_1dp != float(displayed_pct), (
            f"Fixture no longer exercises a 1dp divergence: "
            f"raw={raw_pct_1dp}% displayed={displayed_pct}% — pick new costs."
        )

        # --- Part 2: capture both rendered surfaces. ---
        verbose_out = self._capture_verbose(result)
        panel_out = self._capture_panel(result)

        # --- Part 3: extract the XX.X% the verbose note prints for "split above". ---
        # The note template is:
        #   "the {_display_pct(split_total_pct):.1f}% split above is the conservative, ..."
        verbose_match = re.search(r"the (\d+\.\d+)% split above", verbose_out)
        assert verbose_match is not None, (
            f"'the X.X% split above' not found in verbose output:\n{verbose_out}"
        )
        verbose_pct_str = verbose_match.group(1) + "%"  # e.g. "49.7%"

        # --- Part 4: extract the XX.X% the panel hero prints for "% lower". ---
        # The panel template is: f"{pct} lower" where pct = f"{_display_pct(total_pct):.1f}%"
        panel_match = re.search(r"(\d+\.\d+)% lower", panel_out)
        assert panel_match is not None, (
            f"'X.X% lower' not found in panel output:\n{panel_out}"
        )
        panel_pct_str = panel_match.group(1) + "%"  # e.g. "49.7%"

        # --- Part 5: the expected string from the shared helper. ---
        expected_pct_str = f"{displayed_pct:.1f}%"  # "49.7%"

        # Both printed strings must equal the shared helper's figure.
        assert verbose_pct_str == expected_pct_str, (
            f"Verbose note printed '{verbose_pct_str}' but expected '{expected_pct_str}' "
            f"(from _split_report_figures.total_pct={reconciled_pct_value:.4f}%)"
        )
        assert panel_pct_str == expected_pct_str, (
            f"Panel hero printed '{panel_pct_str}' but expected '{expected_pct_str}' "
            f"(from _split_report_figures.total_pct={reconciled_pct_value:.4f}%)"
        )

        # Explicit agreement between the two rendered surfaces — the core guard.
        assert verbose_pct_str == panel_pct_str, (
            f"Verbose note printed '{verbose_pct_str}' but panel printed '{panel_pct_str}': "
            f"the two surfaces disagree on the split saving percent."
        )


class TestUpperBoundGateAgreement:
    """Regression: the DEFAULT-view Upper-bound hint and the VERBOSE Upper-bound
    note must appear/disappear in LOCKSTEP — they must NEVER disagree on whether
    the hint is shown.

    The gate predicate on both surfaces is ``wholesale_saving > split_total_pct``.
    The fix (v0.1.3) routes BOTH surfaces through
    ``_split_report_figures(...).total_pct`` (the reconciled figure).  Before the
    fix, ``_render_split_upper_bound_row`` re-derived its gate value from the raw
    unrounded components, which could differ from the reconciled ``total_pct`` at
    full Decimal precision.

    Boundary straddle fixture
    ─────────────────────────
    Re-uses the ``TestVerboseSplitPctMatchesPanel`` cost constants so the two
    sets of guards share the same divergence point:

      total_cost = 2.005  → current_rounded = 2.01  (ROUND_HALF_UP)
      blended_raw = 2.005 − (2.000 − 1.000) = 1.005
                  → blended_rounded = 1.01 (ROUND_HALF_UP)

      reconciled total_pct = (2.01 − 1.01) / 2.01 × 100 = 49.751 2…%
      raw total_pct        = (2.005 − 1.005) / 2.005 × 100 = 49.875 3…%

    We choose ``projected_cost = Decimal("1.007")`` so that:

      wholesale_saving = (2.005 − 1.007) / 2.005 × 100 = 49.776…%

    This value satisfies:
      reconciled_total_pct (49.751…) < wholesale (49.776…) ≤ raw_total_pct (49.875…)

    Under the bug the DEFAULT-view gate fires on raw (49.875 > 49.776 → FALSE →
    hide), while the verbose gate fires on reconciled (49.751 < 49.776 → TRUE →
    show) → MISMATCH.  With the fix both fire on reconciled → both show → AGREE.
    """

    # Shared with TestVerboseSplitPctMatchesPanel — same divergence origin.
    _TOTAL_COST = Decimal("2.005")
    _BASELINE_COST = Decimal("2.000")
    _BLENDED_COST = Decimal("1.000")
    # chosen so wholesale sits strictly between reconciled_pct and raw_pct
    _PROJECTED_COST = Decimal("1.007")

    def _boundary_fixture(self) -> tuple[AnalysisResult, SplitRouting]:
        """Build a result+split where wholesale_saving straddles the raw/reconciled gap."""
        split = _split(
            baseline_model="gpt-4o",
            candidate_model="gpt-4o-mini",
            routed_count=20,
            kept_count=5,
            routed_cost=Decimal("0.400"),
            kept_cost=Decimal("0.600"),
            baseline_cost=self._BASELINE_COST,
            blended_cost=self._BLENDED_COST,
            easy_threshold=Decimal("0.35"),
        )
        result = _result(
            total_calls=25,
            priced_calls=25,
            unpriced_calls=0,
            total_cost=self._TOTAL_COST,
            cost_by_model={"gpt-4o": self._TOTAL_COST},
            calls_by_model={"gpt-4o": 25},
            projected_cost=self._PROJECTED_COST,
            candidate_model="gpt-4o-mini",
            observed_span_days=7.0,
            split=split,
        )
        return result, split

    def _capture_default_row(self, result: AnalysisResult, split: SplitRouting) -> str:
        """Render _render_split_upper_bound_row (detail_shown=False) via rprint capture."""
        from rich.console import Console

        from frugon import report

        console = Console(width=200, force_terminal=False, no_color=True)
        captured: list[str] = []
        original = report.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            with console.capture() as cap:
                console.print(*args, **kwargs)
            captured.append(cap.get())

        report.rprint = _rp  # type: ignore[assignment]
        try:
            report._render_split_upper_bound_row(result, split, detail_shown=False)
        finally:
            report.rprint = original
        return " ".join("".join(captured).split())

    def _capture_verbose(self, result: AnalysisResult, split: SplitRouting) -> str:
        """Render _render_split_verbose via rprint capture."""
        from rich.console import Console

        from frugon import report

        console = Console(width=200, force_terminal=False, no_color=True)
        captured: list[str] = []
        original = report.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            with console.capture() as cap:
                console.print(*args, **kwargs)
            captured.append(cap.get())

        report.rprint = _rp  # type: ignore[assignment]
        try:
            assert result.split is not None
            report._render_split_verbose(result, result.split)
        finally:
            report.rprint = original
        return " ".join("".join(captured).split())

    def test_upper_bound_gate_agreement_at_boundary(self) -> None:
        """DEFAULT-view Upper-bound row and VERBOSE Upper-bound note agree on
        presence/absence when wholesale_saving straddles the raw/reconciled gap.

        On a fixture where ``wholesale_saving`` sits strictly between the raw
        ``total_pct`` (which the pre-fix default row gated on) and the reconciled
        ``total_pct`` (which the verbose note always used), a regression would show
        the hint in one view and hide it in the other.  With the fix (both gate on
        reconciled) the two must be CONSISTENT.
        """
        result, split = self._boundary_fixture()

        # --- Part 1: verify the fixture is a genuine boundary straddle. --------
        # If this fails, the fixture is degenerate and the test becomes a no-op.
        from frugon.cost import compute_saving_pct
        from frugon.report import _split_report_figures

        # Raw total_pct: (baseline_cost − blended_cost) / total_cost × 100.
        # This is the pre-fix derivation that _render_split_upper_bound_row used.
        raw_numerator = self._BASELINE_COST - self._BLENDED_COST
        raw_total_pct = raw_numerator / self._TOTAL_COST * Decimal("100")

        # Reconciled total_pct: what _split_report_figures returns.
        fig = _split_report_figures(result, split)
        reconciled_total_pct = fig.total_pct

        # Wholesale saving: exactly what both gate functions compute.
        wholesale = compute_saving_pct(result.total_cost, result.projected_cost)
        assert wholesale is not None, "projected_cost must be set for this fixture"

        # Self-check: divergence exists at 1dp.
        raw_1dp = round(float(raw_total_pct), 1)
        reconciled_1dp = round(float(reconciled_total_pct), 1)
        assert raw_1dp != reconciled_1dp, (
            f"Fixture no longer straddles: raw={raw_1dp}% reconciled={reconciled_1dp}% "
            f"— pick new costs. (raw_total_pct={float(raw_total_pct):.6f}%, "
            f"reconciled_total_pct={float(reconciled_total_pct):.6f}%)"
        )

        # Self-check: wholesale sits strictly between reconciled and raw.
        lo = min(raw_total_pct, reconciled_total_pct)
        hi = max(raw_total_pct, reconciled_total_pct)
        assert lo < wholesale <= hi, (
            f"wholesale_saving ({float(wholesale):.6f}%) must satisfy "
            f"min(raw,reconciled) < wholesale <= max(raw,reconciled): "
            f"lo={float(lo):.6f}%, hi={float(hi):.6f}%. "
            f"Adjust _PROJECTED_COST."
        )

        # --- Part 2: render both surfaces. ------------------------------------
        default_out = self._capture_default_row(result, split)
        verbose_out = self._capture_verbose(result, split)

        # --- Part 3: detect Upper-bound disclosure presence. ------------------
        # The default row emits text containing "a full swap to" and/or "saves ~".
        # The verbose note emits "moving every call to" and/or "saves ~" / "split above".
        # Either surface is silent (empty string) when the gate closes.
        _UPPER_BOUND_MARKERS = (
            "a full swap to",
            "moving every call to",
            "saves ~",
            "split above",
            "Upper bound",
        )

        def _has_upper_bound(text: str) -> bool:
            return any(marker in text for marker in _UPPER_BOUND_MARKERS)

        default_shows = _has_upper_bound(default_out)
        verbose_shows = _has_upper_bound(verbose_out)

        # --- Part 4: assert lockstep. -----------------------------------------
        assert default_shows == verbose_shows, (
            f"Upper-bound gate MISMATCH on boundary fixture:\n"
            f"  default-view shows={default_shows!r}, verbose shows={verbose_shows!r}\n"
            f"  wholesale_saving = {float(wholesale):.6f}%\n"
            f"  reconciled total_pct = {float(reconciled_total_pct):.6f}% "
            f"(gate threshold — both surfaces MUST use this)\n"
            f"  raw total_pct = {float(raw_total_pct):.6f}% "
            f"(pre-fix threshold — default row was gating here)\n"
            f"  default output: {default_out!r}\n"
            f"  verbose output: {verbose_out!r}"
        )

        # --- Part 5: confirm the gate decision is correct (show, not hide). ---
        # With wholesale > reconciled_total_pct, BOTH surfaces must show the hint.
        assert wholesale > reconciled_total_pct, (
            f"Fixture design error: wholesale ({float(wholesale):.6f}%) must be "
            f"> reconciled_total_pct ({float(reconciled_total_pct):.6f}%)"
        )
        assert default_shows, (
            f"Upper-bound hint should be SHOWN (wholesale {float(wholesale):.6f}% "
            f"> reconciled {float(reconciled_total_pct):.6f}%) but default view is silent.\n"
            f"default output: {default_out!r}"
        )
        assert verbose_shows, (
            f"Upper-bound hint should be SHOWN (wholesale {float(wholesale):.6f}% "
            f"> reconciled {float(reconciled_total_pct):.6f}%) but verbose is silent.\n"
            f"verbose output: {verbose_out!r}"
        )


class TestHtmlV1InformationParity:
    """HTML v1 carries the SAME figures as the Markdown + HTML v2 surfaces (F4).

    The v1 surface previously omitted the saving dollar amount, the per-bucket
    costs, the per-bucket call shares, the routing-plan blended total row, and the
    cost-by-model Total row — so the same --demo dataset told a thinner story in
    v1.  These guard the restored parity.
    """

    # The exact 2-dp figures the --demo dataset produces (FRG-OSS-034 Phase 3+4
    # un-pinning + seed refresh, 2026-07-02).
    # Demo: gpt-5.5 → deepseek-v4-flash, 56,100 calls.
    # Headline: current=$549.46, blended=$343.91, saving=$205.55.
    # Per-bucket costs: routed=$3.29, kept=$339.71, already-optimal=$0.91.
    # Call shares (unchanged): 64.4%/17.8%.
    _DEMO_FIGURES = ("205.55", "3.29", "339.71", "0.91", "64.4%", "17.8%")

    def test_v1_html_contains_saving_and_per_bucket_figures(
        self, tmp_path: Path
    ) -> None:
        html = _render("html_v1", _demo_result(), tmp_path)
        # Saving dollar amount beside the hero (2 dp).
        assert "205.55" in html
        # Per-bucket costs (2 dp).
        assert "3.29" in html   # routed
        assert "339.71" in html  # kept
        assert "0.91" in html   # already-optimal
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
