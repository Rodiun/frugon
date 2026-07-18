"""Tests for the per-call split-routing report rendering (terminal + HTML + MD).

The split is the headline FRUGON-R1 feature: routed N -> cheaper candidate
(within tolerance), kept M -> baseline, blended cost + saving.  These tests pin
the routed/kept/blended/saving shape the landing hero mirrors, the honest split
caveat, and the canonical emerald (#10B981) saving colour.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from frugon.cost import AnalysisResult
from frugon.report import (
    QUALITY_NOT_VERIFIED,
    SAVING_GREEN,
    SPLIT_CAVEAT,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
    render_terminal,
)
from frugon.routing import SplitRouting


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
    defaults: dict[str, Any] = {
        "total_calls": 37,
        "priced_calls": 37,
        "unpriced_calls": 0,
        "total_cost": Decimal("0.0676"),
        "cost_by_model": {"gpt-4-turbo": Decimal("0.0650"), "gpt-4o": Decimal("0.0026")},
        "calls_by_model": {"gpt-4-turbo": 27, "gpt-4o": 10},
        "projected_cost": Decimal("0.0222"),
        "candidate_model": "gpt-4o",  # wholesale upper-bound candidate
        "observed_span_days": 7.0,
        "split": _split(),
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def _render_to_text(result: AnalysisResult, **kwargs: Any) -> str:
    """Render the terminal split view to plain text via a fixed-width console.

    Uses an 88-column, colour-free console so layout assertions are stable
    regardless of the test runner's terminal, and so the structure (the one
    bordered panel + the borderless table) renders the same way a recording or a
    screenshot would.
    """
    import io
    import sys

    from rich.console import Console

    # Resolve the EXACT module object render_terminal lives in — robust even if a
    # prior test reloaded frugon.report via sys.modules (which would make a plain
    # ``import frugon.report`` bind a different module instance).
    report_mod = sys.modules[render_terminal.__module__]

    buf = io.StringIO()
    # legacy_windows=False forces Rich's Unicode box-drawing on every OS: without
    # it Rich auto-detects a legacy Windows console and falls back to SQUARE
    # corners, so the rounded-corner assertions below would pass on Linux/macOS
    # but spuriously fail on Windows.  It is a no-op on *nix (already False).
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
    # The responsive hanging-indent lines measure their wrap width from
    # _render_console(); point it at this fixed-width console so measurement and
    # emission agree on 88 columns (otherwise the wrap would be computed for the
    # ambient console and printed here, mismatching the width).
    report_mod._render_console = lambda: console  # type: ignore[attr-defined]
    try:
        render_terminal(result, **kwargs)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    return buf.getvalue()


class TestTerminalSplit:
    """The redesigned terminal split view (CLI redesign — bordered panel).

    Layout: ONE rounded, cyan-bordered panel carries the summary, the
    plain-English Route/Keep decision, and the SAVING hero; beneath it sit a
    borderless cost-by-model table, muted accounting lines, and a quiet
    three-line footer (quality caveat / privacy / one upsell).
    """

    def test_terminal_shows_plain_english_routing_plan(self, capsys: Any) -> None:
        """Plain-English 'Route N easy → mini / Keep M hard → premium'."""
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert "Route 24 easy calls" in out
        assert "Keep 3 hard calls" in out
        assert "gpt-4o-mini" in out
        assert "gpt-4-turbo" in out

    def test_terminal_hero_is_the_saving(self, capsys: Any) -> None:
        """The hero is the SAVING line — the money won + the percent.

        Every panel figure reconciles to the FULL dataset AND to the printed,
        2-dp-rounded dollars (the v0.1.3 reconciling design): "Current" is the
        TOTAL spend (cost_by_model sums to 0.0676 → printed $0.07), "New" is that
        total after routing the baseline's easy calls (0.0676 - (0.0650 - 0.0439)
        = 0.0465 → printed $0.05), so the printed SAVING is $0.07 - $0.05 = $0.02
        and the percent is derived from those rounded components: $0.02 / $0.07 =
        28.5714…%, floored (never rounded up) to 28.5% — verifiable straight from
        the printed figures, the SAME figure the HTML/MD reports now show (every
        surface shares the one rounded source _split_report_figures, so no
        surface can contradict another).
        """
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert "SAVING" in out
        # Saving over the TOTAL current spend, derived from the 2-dp-rounded
        # dollars: ($0.07 − $0.05) / $0.07 = 28.5714…%, floored to 28.5% (1-dp) —
        # ROUND_HALF_UP would overstate this as 28.6%.
        assert "28.5% lower" in out
        # Current and new-spend figures both appear (full _fmt_usd precision).
        assert "Current spend" in out
        assert "New spend" in out

    def test_terminal_within_tolerance_shown_on_route_line(self, capsys: Any) -> None:
        """'within tolerance' is shown on the route line (and stays neutral).

        The neutral-colour guarantee is enforced by
        test_terminal_colour_discipline_green_only_on_saving; here we only assert
        the phrase is present where the spec puts it.
        """
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert "within tolerance" in out

    def test_terminal_panel_is_the_only_box(self) -> None:
        """Exactly one framed box on screen: the rounded panel.

        The decision panel uses rounded box-drawing (╭ ╮ ╰ ╯) — that is the ONE
        intended frame.  No heavy/square table borders appear anywhere, in either
        the default view OR --verbose (where the cost-by-model table appears, and
        must still be borderless).
        """
        for verbose in (False, True):
            text = _render_to_text(_result_with_split(), verbose=verbose)
            # The rounded panel corners are present (the one intended frame).
            for corner in "╭╮╰╯":
                assert corner in text, f"expected rounded panel corner {corner!r}"
            # No square/heavy table borders leak from the borderless table.
            for glyph in "┌┐└┘┏┓┗┛┃━╔╗╚╝":
                assert glyph not in text, f"unexpected box glyph {glyph!r} (verbose={verbose})"

    def test_terminal_cost_by_model_table_is_verbose_only(self, capsys: Any) -> None:
        """The per-model cost table is demoted to --verbose, gone from the default.

        The default split view is the decision panel + accounting + footer; the
        per-model breakdown is supporting detail, not headline (CLI redesign
        clarity pass).
        """
        render_terminal(_result_with_split())
        default_out = " ".join(capsys.readouterr().out.split())
        assert "Cost by model" not in default_out

        render_terminal(_result_with_split(), verbose=True)
        verbose_out = " ".join(capsys.readouterr().out.split())
        assert "Cost by model" in verbose_out

    def test_terminal_table_is_borderless(self) -> None:
        """The cost-by-model table renders without its own box frame (--verbose).

        Its rows sit beneath the panel with no intervening box-drawing other than
        the panel's own border.  The table is verbose-only now, so this renders the
        verbose view to reach it.
        """
        text = _render_to_text(_result_with_split(), verbose=True)
        lines = text.splitlines()
        # Locate the table body (the per-model rows) and confirm none carry a
        # vertical table border glyph.
        body = [ln for ln in lines if "gpt-4-turbo" in ln and "$" in ln]
        assert body, "expected a cost-by-model row for gpt-4-turbo"
        for ln in body:
            assert "│" not in ln, f"table row carried a box border: {ln!r}"

    def test_terminal_footer_is_exactly_three_lines(self) -> None:
        """The footer is exactly three logical lines: caveat / privacy / upsell."""
        text = _render_to_text(_result_with_split())
        lines = [ln.rstrip() for ln in text.splitlines()]
        # Caveat line (amber ⚠), privacy line, upsell line — find each once.
        caveat = [i for i, ln in enumerate(lines) if "Quality is not verified" in ln]
        privacy = [i for i, ln in enumerate(lines) if "never leaves your machine" in ln]
        upsell = [i for i, ln in enumerate(lines) if "frugon.rodiun.io" in ln]
        assert len(caveat) == 1, f"expected one quality-caveat line, got {len(caveat)}"
        assert len(privacy) == 1, f"expected one privacy line, got {len(privacy)}"
        assert len(upsell) == 1, f"expected one upsell line, got {len(upsell)}"
        # They appear in order and are the last three content lines.
        assert caveat[0] < privacy[0] < upsell[0]

    def test_terminal_verbose_carries_the_upsell_exactly_once(self) -> None:
        """--verbose must not duplicate the site CTA: the Notes block used to
        carry its own 'Automate' row on top of the shared footer's upsell line,
        so the same https://frugon.rodiun.io pitch printed twice.  The footer is
        the ONE place it belongs, in both default and --verbose views."""
        text = _render_to_text(_result_with_split(), verbose=True)
        lines = [ln.rstrip() for ln in text.splitlines()]
        upsell = [i for i, ln in enumerate(lines) if "frugon.rodiun.io" in ln]
        assert len(upsell) == 1, f"expected one upsell line, got {len(upsell)}"
        assert "Automate" not in text

    def test_terminal_colour_discipline_green_only_on_saving(self) -> None:
        """The emerald saving colour (#10B981) wraps ONLY money-win text.

        Renders to Rich Segments (the structured representation behind the ANSI)
        and collects every segment whose style carries the emerald colour — as a
        foreground colour OR a background, on the off chance a future change tints
        a fill.  Each emerald segment must belong to the saving story (SAVING, the
        blended 'after', a dollar figure, or the percent-lower).  No model name,
        command, caveat, or 'within tolerance' may be emerald (CLI redesign
        point 2: green means the money win, and only the money win).
        """
        import io
        import sys

        from rich.console import Console

        report_mod = sys.modules[render_terminal.__module__]

        console = Console(
            file=io.StringIO(), width=88, force_terminal=True, color_system="truecolor"
        )
        captured: list[Any] = []
        original = report_mod.rprint

        def _capture(*args: Any, **kw: Any) -> None:
            for renderable in args:
                captured.extend(console.render(renderable))

        report_mod.rprint = _capture  # type: ignore[attr-defined]
        try:
            render_terminal(_result_with_split())
        finally:
            report_mod.rprint = original  # type: ignore[attr-defined]

        green_text: list[str] = []
        for seg in captured:
            style = seg.style
            if style is None:
                continue
            colours = [c for c in (style.color, style.bgcolor) if c is not None]
            if any(
                col.triplet is not None and col.triplet.hex.lower() == SAVING_GREEN.lower()
                for col in colours
            ):
                green_text.append(seg.text)

        assert green_text, "expected the saving to be rendered in emerald green"
        joined = "".join(green_text)
        # The money win is what is allowed to be green.
        assert "SAVING" in joined
        forbidden = [
            "gpt-4o-mini",
            "gpt-4-turbo",
            "within tolerance",
            "--measure",
            "Route",
            "Keep",
            "Analyzed",
        ]
        for token in forbidden:
            assert token not in joined, (
                f"{token!r} was coloured saving-green — green is for the money win only"
            )

    def test_terminal_two_load_bearing_caveats_present(self, capsys: Any) -> None:
        """The quality caveat and the privacy line are both present by default."""
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert " ".join(QUALITY_NOT_VERIFIED.split()) in out
        assert "never leaves your machine" in out

    def test_terminal_quality_caveat_reconciles_within_tolerance(
        self, capsys: Any
    ) -> None:
        """The footer caveat glosses 'within tolerance' so the tool never reads as
        self-contradicting.

        The Route line carries the muted 'within tolerance' band; the footer caveat
        names it explicitly as an offline estimate that --measure confirms — one
        clause reconciling the two phrases (CLI redesign clarity pass).
        """
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert "within tolerance" in out  # the band on the Route line
        # The single caveat clause ties 'within tolerance' to 'not verified' and
        # to --measure — no contradiction left dangling.
        assert "Quality is not verified" in out
        assert "'within tolerance' is an offline estimate" in out
        assert "--measure" in out

    def test_terminal_caveats_suppressed_with_measure(self, capsys: Any) -> None:
        render_terminal(_result_with_split(), suppress_caveat=True)
        out = " ".join(capsys.readouterr().out.split())
        assert " ".join(QUALITY_NOT_VERIFIED.split()) not in out
        # The whole footer (incl. the upsell) is suppressed under --measure.
        assert "frugon.rodiun.io" not in out

    def test_terminal_default_view_omits_supporting_detail(self, capsys: Any) -> None:
        """The heuristic explanation stays under --verbose; the default view does
        NOT carry the easy/hard method note or the aggressive-vs-conservative
        wholesale explanation.

        The DEFAULT view now carries ONE dim Upper-bound line (range information),
        but the full explanation — the easy/hard heuristic, the
        aggressive-vs-conservative gloss, and the ``--wholesale --measure
        --candidates`` command — remains verbose-only.  (The single footer upsell
        line stays.)
        """
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        # The verbose-only explanation is absent from the default view.
        assert "heuristic" not in out
        assert "the conservative, quality-respecting recommendation" not in out
        assert "--wholesale --measure --candidates" not in out

    def test_terminal_default_view_has_one_line_upper_bound(self, capsys: Any) -> None:
        """The default split view promotes the Upper-bound to ONE dim line.

        Range information the prior design hid in --verbose: a single dim line in
        the Accounting block names the full-swap saving and points at --verbose for
        the detail.  It is one line — no second green number — and reconciles with
        the verbose note + wholesale panel (same helpers; see
        test_terminal_upper_bound_reconciles_across_views).
        """
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert "Upper bound" in out
        assert "a full swap to gpt-4o saves ~" in out
        assert "run with --verbose for detail" in out
        # One line only — the verbose explanatory tail must NOT be present.
        assert "the conservative, quality-respecting recommendation" not in out

    def test_terminal_verbose_reveals_supporting_detail(self, capsys: Any) -> None:
        """--verbose carries upper bound + heuristic; the footer still carries
        the one upsell line (see test_terminal_verbose_carries_the_upsell_exactly_once
        for the no-duplicate regression)."""
        render_terminal(_result_with_split(), verbose=True)
        out = " ".join(capsys.readouterr().out.split())
        assert "Upper bound" in out
        assert "gpt-4o" in out  # wholesale upper-bound candidate
        assert "heuristic" in out
        assert "frugon.rodiun.io" in out

    def test_terminal_verbose_upper_bound_hint_points_to_notes(self, capsys: Any) -> None:
        """Under --verbose the Upper-bound hint redirects to the Notes block.

        The detail it points at (the aggressive-vs-conservative breakdown) is
        already rendered in the verbose Notes block, so re-suggesting --verbose
        would point at a flag already in force.  The hint must read "see notes
        below for detail" and must NOT re-suggest --verbose.
        """
        render_terminal(_result_with_split(), verbose=True)
        out = " ".join(capsys.readouterr().out.split())
        assert "a full swap to gpt-4o saves ~" in out
        assert "see notes below for detail" in out
        assert "run with --verbose for detail" not in out

    def test_terminal_accounting_reconciles(self, capsys: Any) -> None:
        """Every analyzed call is visibly accounted for.

        routed (24) + kept (3) + already-cheap (10) == analyzed (37).
        """
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert "24 routed" in out
        assert "3 kept" in out
        assert "10 already on cheaper" in out
        assert "gpt-4o" in out  # the already-cheap model is named
        assert "= 37 analyzed" in out

    def test_terminal_split_legacy_caveat_constant_still_exported(self) -> None:
        """SPLIT_CAVEAT remains the canonical HTML/MD caveat (terminal pared it)."""
        assert "Quality is not verified" in SPLIT_CAVEAT

    def test_terminal_already_optimal_line_present(self, capsys: Any) -> None:
        """The already-on-a-cheaper-model calls read as 'already optimal', in-panel.

        The reader never has to wonder where the non-routable, non-baseline calls
        went: they are named in the decision block as already optimal — no action.
        """
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert "10 already on gpt-4o" in out
        assert "already optimal — no action" in out

    def test_terminal_route_keep_shares_are_of_all_analyzed_calls(
        self, capsys: Any
    ) -> None:
        """Route/Keep/already-optimal %s share ONE denominator with the reports:
        share of ALL analyzed calls, at 1-dp, summing to exactly 100.0.

        Fixture: routed 24 + kept 3 + already-optimal 10 = 37 analyzed.  Share of
        all (largest-remainder, 1-dp) = 64.9% / 8.1% / 27.0% (sums to 100.0).  The
        retired behaviour divided route/keep by routed+kept ONLY (27), printing
        89%/11% — a different story for the same data.  This guards against that
        regression: the panel and the MD/HTML reports must tell one story.
        """
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        # Each bucket carries its share of ALL analyzed calls at 1-dp (the panel
        # right-justifies the figures to a common width, so the assertions target
        # the figure not its surrounding padding).
        assert "Route 24 easy calls" in out
        assert "64.9%" in out
        assert "Keep 3 hard calls" in out
        assert "8.1%" in out
        # The already-optimal line now carries its share too (it had none before).
        assert "10 already on gpt-4o" in out
        assert "27.0%" in out
        # The retired routed+kept-only denominator (89%/11%) must not reappear.
        assert "89%" not in out
        assert "11%" not in out
        # The three shares reconcile to exactly 100.0.
        assert 64.9 + 8.1 + 27.0 == 100.0

    def test_terminal_tier_note_is_on_its_own_line(self) -> None:
        """The 'no known quality tier' note sits on its OWN line, not trailing the
        quality caveat sentence (Problem 3 — a tidy, scannable footer).

        The reconciled caveat ("...'within tolerance' is an offline estimate; run
        --measure...") is longer and may soft-wrap across two visual lines in the
        fixed-width console.  The durable invariant is unchanged: the tier note is
        a DISTINCT line that is NOT part of the caveat sentence, and it follows the
        caveat block immediately — i.e. the very next line after the caveat's last
        wrapped line carries the tier note (no other content sneaks between them).
        """
        # The note only appears when the baseline model is unrated (as gpt-4-turbo
        # is in the bundled demo) — set that flag so the footer renders it.
        text = _render_to_text(_result_with_split(baseline_is_unrated=True))
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        caveat_start = next(
            i for i, ln in enumerate(lines) if "Quality is not verified" in ln
        )
        tier_idx = next(
            i for i, ln in enumerate(lines) if "has no known quality tier" in ln
        )
        # The tier note is a DISTINCT line — never sharing a line with the caveat.
        assert tier_idx != caveat_start
        assert "has no known quality tier" not in lines[caveat_start]
        # The caveat sentence ends at "...before you switch." — find its last
        # (possibly wrapped) line; the tier note must be the very next line.
        caveat_end = next(
            i for i, ln in enumerate(lines) if "before you switch." in ln
        )
        assert caveat_end >= caveat_start  # the caveat may span one or more lines
        assert tier_idx == caveat_end + 1
        # And nothing between the caveat start and its end is anything but caveat
        # text (the wrap stays within the caveat sentence — no foreign content).
        for ln in lines[caveat_start : caveat_end + 1]:
            assert "has no known quality tier" not in ln


class TestTerminalSplitReconciliation:
    """The panel must reconcile to the FULL analyzed dataset — every dollar and
    every call accounted for, no displayed figure contradicting another.

    This is the flagged invariant: "Current" is the TOTAL spend (the sum of the
    Cost-by-model rows), "Blended" is that total after routing, "SAVING" is their
    difference, and the routed + kept + already-optimal calls sum to the analyzed
    total.  The old framing put the baseline-model-only figure under "Current"
    while the breakdown summed to something larger — a silent contradiction.
    """

    def test_current_equals_sum_of_cost_by_model_rows(self) -> None:
        """Current (the panel total) == the exact sum of the per-model costs.

        Renders the panel + table and asserts the data-level invariant the display
        depends on: the result's total_cost is exactly the sum of cost_by_model —
        so "Current" can never silently exceed or undercut the breakdown.
        """
        result = _result_with_split()
        assert sum(result.cost_by_model.values(), Decimal("0")) == result.total_cost

    def test_panel_current_blended_saving_reconcile(self) -> None:
        """Current - Blended == SAVING, and the saving is the baseline routing win.

        The panel's TOTAL blended is the total current minus the routing reduction
        on the baseline model's easy calls (baseline_cost - blended_cost); the
        saving is exactly that reduction.  Asserting the arithmetic guards the
        framing against drift.
        """
        result = _result_with_split()
        split = result.split
        assert split is not None
        current = result.total_cost  # no monthly projection on this fixture
        baseline_reduction = split.baseline_cost - split.blended_cost
        blended_total = current - baseline_reduction
        saving = current - blended_total
        assert saving == baseline_reduction
        # The saving percent is over the TOTAL, not the baseline alone.
        total_pct = saving / current * Decimal("100")
        assert int(total_pct) == 31  # 0.0211 / 0.0676

    def test_full_call_reconciliation(self) -> None:
        """routed + kept + already-optimal == analyzed (no call vanishes)."""
        result = _result_with_split()
        split = result.split
        assert split is not None
        already_optimal = result.priced_calls - split.total_count
        assert already_optimal >= 0
        assert (
            split.routed_count + split.kept_count + already_optimal
            == result.priced_calls
        )


# ---------------------------------------------------------------------------
# HTML (v1 + v2)
# ---------------------------------------------------------------------------


class TestHtmlSplit:
    def test_html_v1_split_renders_emerald_and_shape(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        assert SAVING_GREEN in html  # emerald token defined
        assert "gpt-4o-mini" in html
        assert "within tolerance" in html
        # Total-basis saving derived from the 2-dp-rounded dollars
        # ($0.07 − $0.05) / $0.07 = 28.5714…%, floored (never rounded up) to
        # 28.5% (1-dp) — not the unrounded 31.2%, the retired baseline-only 32%,
        # nor the ROUND_HALF_UP-overstated 28.6%.
        assert "28.5%" in html
        assert SPLIT_CAVEAT.split(".")[0] in html

    def test_html_v2_split_renders_routing_plan(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        assert "Routing plan" in html
        assert "Routed" in html
        assert "Kept" in html
        assert "Blended" in html
        assert "gpt-4o-mini" in html
        assert "within tolerance" in html
        # Total-basis saving derived from the 2-dp-rounded dollars
        # ($0.07 − $0.05) / $0.07 = 28.5714…%, floored (never rounded up) to
        # 28.5% (1-dp) — not the unrounded 31.2%, the retired baseline-only 32%,
        # nor the ROUND_HALF_UP-overstated 28.6%.
        assert "28.5%" in html

    def test_html_v2_split_routing_table_column_law(self, tmp_path: Path) -> None:
        """Assert: the routing-plan table carries the SIX-column law with a share bar.

        Regression guard for the reported collision ("Keep · already o10000 gpt-4o")
        AND the MODEL/STATUS dead band: the fixed layout left MODEL as the only
        un-sized column, so it absorbed all spare width and opened a wide empty
        band before STATUS. The fix gives each datum its own column — Bucket |
        Calls | % calls | Model | Status | Cost — and switches the table to
        ``table-layout:auto`` so every column sizes to its content and packs left
        with even spacing (no dead band). The reclaimed width carries a per-bucket
        SHARE bar (% of all analyzed calls), the % shown as TEXT (accessible —
        never colour-only).
        """
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        # The routing-plan table is distinguished from the cost-by-model table.
        assert 'class="tbl tbl-plan"' in html
        # Six header columns, in order — the SHARE column sits next to CALLS.
        assert (
            '<th class="c-bucket">Bucket</th><th class="r c-calls">Calls</th>'
            '<th class="c-share">% calls</th>'
            '<th class="c-model">Model</th><th class="c-status">Status</th>'
            '<th class="r c-cost">Cost</th>' in html
        )
        # Bucket cells carry the bucket class and are held to one line by CSS.
        assert '<td class="bucket">Routed &middot; easy</td>' in html
        assert '<td class="bucket">Kept &middot; hard</td>' in html
        assert ".tbl-plan td.bucket,.tbl-plan th.c-bucket{white-space:nowrap}" in html
        # AUTO layout (not fixed) is what removes the MODEL/STATUS dead band.
        assert ".tbl-plan{table-layout:auto;width:100%;max-width:100%;border-collapse:collapse}" in html
        # Model and Status are SEPARATE cells — the model name carries no badge.
        assert '<td class="c-model"><span class="route-to">' in html
        assert '<td class="c-status"><span class="badge">within tolerance</span></td>' in html
        # The old crammed route-cell is gone entirely.
        assert "route-cell" not in html
        # SHARE column: a proportional fill bar driven by the --share property,
        # with the % as accessible text beside it.  Fixture: routed 24 / 37 calls
        # => 64.9% (largest-remainder rounded so the three buckets sum to 100.0%).
        assert '<span class="share-bar" style="--share:64.9%"><span class="share-fill"></span></span>' in html
        assert '<span class="share-pct tnum">64.9%</span>' in html
        # All three non-blended bucket shares are present and reconcile to 100%.
        assert '<span class="share-pct tnum">8.1%</span>' in html
        assert '<span class="share-pct tnum">27.0%</span>' in html
        # The Blended total row anchors the column at 100%.
        assert '<td class="num r tnum c-share">100.0%</td>' in html
        # Mobile: the share column shrinks (compact bar) so it never crowds MODEL.
        assert ".tbl-plan .share-bar{width:2.2rem}" in html

    def test_html_v2_split_long_model_name_contained(
        self, tmp_path: Path
    ) -> None:
        """Assert: a long candidate model name renders inside its own MODEL cell
        and is allowed to wrap, never overflowing or colliding with the status.

        The model name owns the single flexible column; ``overflow-wrap:anywhere``
        lets an extreme id wrap within that column rather than overflow, and the
        status badge sits in a separate cell so it can never be pushed off.
        """
        long_name = "openrouter/anthropic/claude-3-5-sonnet-20241022"
        split = _split()
        split.candidate_model = long_name
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(split=split, candidate_model=long_name), out)
        html = out.read_text(encoding="utf-8")
        # The long name renders inside its own MODEL cell.
        assert f'<td class="c-model"><span class="route-to">{long_name}</span></td>' in html
        # The status badge stays in its own cell, unaffected by the long name.
        assert '<td class="c-status"><span class="badge">within tolerance</span></td>' in html
        # The model name is allowed to wrap within its column for extreme ids.
        assert "overflow-wrap:anywhere" in html

    def test_html_v2_split_saving_uses_emerald(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        # --green is the emerald and the hero figure is coloured with it.
        assert f"--green:{SAVING_GREEN}" in html

    def test_html_v2_split_no_external_network(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        # Only the funnel link is allowed; no CDNs / font URLs / analytics.
        for bad in ("googleapis", "cdn.", "<script"):
            assert bad not in html


# ---------------------------------------------------------------------------
# Markdown (v1 + v2)
# ---------------------------------------------------------------------------


class TestMarkdownSplit:
    def test_markdown_v1_split_shape(self, tmp_path: Path) -> None:
        out = tmp_path / "r.md"
        render_markdown(_result_with_split(), out)
        md = out.read_text(encoding="utf-8")
        assert "Routing plan" in md
        assert "Routed" in md
        assert "Kept" in md
        assert "Blended" in md
        assert "gpt-4o-mini" in md
        # Total-basis saving derived from the 2-dp-rounded dollars
        # ($0.07 − $0.05) / $0.07 = 28.5714…%, floored (never rounded up) to
        # 28.5% (1-dp) — not the unrounded 31.2%, the retired baseline-only 32%,
        # nor the ROUND_HALF_UP-overstated 28.6%.
        assert "28.5%" in md

    def test_markdown_v2_split_shape(self, tmp_path: Path) -> None:
        out = tmp_path / "r.md"
        render_markdown_v2(_result_with_split(), out)
        md = out.read_text(encoding="utf-8")
        assert "## Bottom line" in md
        assert "Route 24 of 37 analyzed calls" in md
        assert "gpt-4o-mini" in md
        assert "within tolerance" in md
        assert "You save" in md

    def test_markdown_split_includes_caveat_and_funnel(self, tmp_path: Path) -> None:
        out = tmp_path / "r.md"
        render_markdown_v2(_result_with_split(), out)
        md = out.read_text(encoding="utf-8")
        assert "Before you switch" in md
        assert "frugon.rodiun.io" in md


# ---------------------------------------------------------------------------
# Fallback: a split with no routed calls must NOT become the headline.
# ---------------------------------------------------------------------------


class TestSplitFallback:
    def test_zero_routed_split_falls_back_to_wholesale(self, capsys: Any) -> None:
        zero = SplitRouting(
            baseline_model="gpt-4-turbo",
            candidate_model="gpt-4o-mini",
            routed_count=0,
            kept_count=5,
            routed_cost=Decimal("0"),
            kept_cost=Decimal("0.05"),
            baseline_cost=Decimal("0.05"),
            blended_cost=Decimal("0.05"),
            easy_threshold=Decimal("0.35"),
        )
        render_terminal(_result_with_split(split=zero))
        out = " ".join(capsys.readouterr().out.split())
        # Wholesale path renders the swap candidate, not the split scoreboard.
        assert "Routing plan" not in out
        assert "within tolerance" not in out


# ---------------------------------------------------------------------------
# Data-quality disclosures (skipped_malformed + approximated_calls)
# ---------------------------------------------------------------------------


class TestDataQualityNotes:
    def test_terminal_shows_skipped_and_approximated(self, capsys: Any) -> None:
        render_terminal(_result_with_split(skipped_malformed=4, approximated_calls=2))
        out = " ".join(capsys.readouterr().out.split())
        assert "4 malformed record(s) skipped" in out
        assert "2 call(s) used approximate token counts" in out

    def test_terminal_omits_notes_when_clean(self, capsys: Any) -> None:
        render_terminal(_result_with_split())
        out = " ".join(capsys.readouterr().out.split())
        assert "malformed record" not in out
        assert "approximate token counts" not in out

    def test_html_v2_shows_data_quality_notes(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(skipped_malformed=4, approximated_calls=2), out)
        html = out.read_text(encoding="utf-8")
        assert "4 malformed record(s) skipped" in html
        assert "approximate token counts" in html

    def test_markdown_v2_shows_data_quality_notes(self, tmp_path: Path) -> None:
        out = tmp_path / "r.md"
        render_markdown_v2(_result_with_split(skipped_malformed=4, approximated_calls=2), out)
        md = out.read_text(encoding="utf-8")
        assert "4 malformed record(s) skipped" in md
        assert "approximate token counts" in md


class TestSplitReportBrandPolish:
    """Regression guards for the UI Art Director hero/report polish pass.

    Pins three verified outcomes for the v2 split report:
      1. the model name and the decision badge live in SEPARATE columns of the
         five-column routing-plan table, so they can never collide;
      2. the FRUGON wordmark carries NO trailing period (the cyan period is
         reserved for the parent RODIUN. brand), while the three-dot mark stays;
      3. the methodology provenance line is emitted as a single element so it
         reads as one row at desktop.
    """

    def test_model_and_status_are_separate_columns(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        # Model name and the decision badge live in SEPARATE table cells, so they
        # can never collide (the superseded inline-pill cram is gone).
        assert (
            '<td class="c-model"><span class="route-to">gpt-4o-mini</span></td>'
            '<td class="c-status"><span class="badge">within tolerance</span></td>' in html
        )
        # The crammed route-cell construct is fully removed.
        assert "route-cell" not in html

    def test_wordmark_has_no_trailing_period(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        # The three-dot mark stays (cyan via currentColor + --cyan).
        assert '<svg class="brand-mark"' in html
        assert ".brand-mark{" in html
        assert "color:var(--cyan)" in html
        # The wordmark text is FRUGON with NO trailing period and no .dot span.
        assert "FRUGON" in html
        assert "FRUGON." not in html
        assert '<span class="dot">' not in html

    def test_methodology_provenance_is_single_element(self, tmp_path: Path) -> None:
        out = tmp_path / "r.html"
        render_html_v2(_result_with_split(), out)
        html = out.read_text(encoding="utf-8")
        # The full provenance string is one <p class="meta"> element (one row).
        provenance = (
            "methodology &middot; tokencost &middot; LiteLLM registry "
            "&middot; LMArena quality tiers, RouteLLM-style routing &middot; "
            "0 LLM calls made for this analysis"
        )
        assert f'<p class="meta">{provenance}</p>' in html
        # It is emitted exactly once in the split report footer.
        assert html.count("0 LLM calls made for this analysis") == 1


def _render_at_width(result: AnalysisResult, width: int, **kwargs: Any) -> list[str]:
    """Render the terminal split view at a fixed *width* and return its lines.

    Drives the renderer through a width-pinned, colour-free console (used for both
    measurement and emission) so the responsive wrapping assertions are stable and
    reproduce exactly what a terminal of that width would show.
    """
    import io
    import sys

    from rich.console import Console

    report_mod = sys.modules[render_terminal.__module__]

    buf = io.StringIO()
    # legacy_windows=False forces Rich's Unicode box-drawing on every OS so the
    # responsive-wrapping assertions (which key off rounded ╭/│/╰ panel glyphs)
    # behave identically on Linux, macOS and Windows.  No-op on *nix.
    console = Console(
        file=buf,
        width=width,
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
        render_terminal(result, **kwargs)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    # Strip any residual ANSI (dim is a style attr that survives no_color) and
    # Rich's right-padding so column assertions are about the leading indent only.
    import re

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    return [ansi.sub("", line).rstrip() for line in buf.getvalue().splitlines()]


class TestResponsiveFooterWrapping:
    """The footer caveat/accounting/upsell reflow to the console width with a
    hanging indent — wrapped continuation lines never bleed into the left margin.
    """

    def test_quality_caveat_action_is_on_its_own_line(self) -> None:
        """The 'run --measure …' sentence is a deliberate second line, not a
        tail wrapped onto the assertion — at a comfortable width where the
        assertion fits on one line.
        """
        lines = _render_at_width(_result_with_split(), width=100)
        assertion_idx = next(
            i for i, ln in enumerate(lines) if "Quality is not verified" in ln
        )
        # The assertion line ends at the semicolon (the deliberate break point)
        # and does NOT carry the call to action.
        assert lines[assertion_idx].rstrip().endswith("offline estimate;")
        assert "run --measure" not in lines[assertion_idx]
        # The very next line is the call to action, indented to the caveat text
        # (column 2 — under the body, after the column-0 ⚠ marker).
        action_line = lines[assertion_idx + 1]
        assert action_line.startswith("  run --measure to confirm")
        assert "before you switch." in action_line

    def test_narrow_width_hangs_under_text_never_into_margin(self) -> None:
        """At a narrow width every footer/accounting continuation line hangs at
        its indent column — never at column 0 (the margin bleed bug).
        """
        lines = _render_at_width(_result_with_split(), width=60)

        # Collect the footer/accounting region (everything after the panel box).
        body = [ln for ln in lines if ln and not ln.startswith(("╭", "│", "╰"))]

        # The caveat assertion + its continuation, the action line, privacy, and
        # the upsell all live at indent 2 or 4.  No continuation line may start
        # at column 0 with footer/accounting prose.  We assert the specific
        # offenders from the old soft-wrap bug are gone:
        joined = "\n".join(lines)
        # The URL must never appear at column 0 (it bled there before the fix).
        url_lines = [ln for ln in lines if "frugon.rodiun.io" in ln]
        assert url_lines, "the upsell URL must be present"
        for ln in url_lines:
            assert ln.startswith("  "), f"URL must hang at indent, got {ln!r}"

        # The accounting reconciliation continuation must hang under the body
        # column (15), not back under the label or at column 0.
        analyzed_lines = [ln for ln in lines if "analyzed" in ln and "Accounting" not in ln]
        for ln in analyzed_lines:
            assert ln.startswith("               "), (
                f"accounting continuation must hang at col 15, got {ln!r}"
            )

        # Every non-empty body line either is a deliberate margin marker (⚠ / →
        # sit in the column-0 gutter, like the ✓ checkpoints) or carries leading
        # indent — no PROSE/continuation line bleeds flush to column 0.
        for ln in body:
            if ln.startswith(("⚠", "→")):
                continue  # markers intentionally occupy the column-0 margin
            assert ln.startswith(" "), f"footer/accounting prose bled to margin: {ln!r}"
        assert joined  # sanity: we actually rendered something

    def test_caveat_marker_in_gutter_text_hangs_under_it(self) -> None:
        """The amber ⚠ sits in the column-0 margin (like the ✓ checkpoints) and
        the caveat text/continuation hangs at column 2 — under the body, not the
        marker.
        """
        lines = _render_at_width(_result_with_split(), width=60)
        caveat_idx = next(
            i for i, ln in enumerate(lines) if "Quality is not verified" in ln
        )
        assert lines[caveat_idx].startswith("⚠ Quality")
        # When the assertion wraps at width 60 its continuation hangs at col 2.
        cont = lines[caveat_idx + 1]
        if "offline estimate;" not in lines[caveat_idx]:
            assert cont.startswith("  "), f"caveat continuation must hang at col 2: {cont!r}"


# ---------------------------------------------------------------------------
# Freshness rows (Prices + Quality) — split path
# ---------------------------------------------------------------------------


class TestSplitFreshnessRows:
    """The Accounting block discloses both pricing.json and quality.json freshness.

    The Quality row mirrors the Prices row: a dim ``synced <date>`` label, omitted
    entirely when its date is absent, and annotated amber (with the cyan refresh
    command) when the table is stale — pricing at >30 days, quality at >60.
    """

    def test_split_shows_prices_and_quality_synced(self, capsys: Any) -> None:
        render_terminal(
            _result_with_split(
                pricing_json_last_synced="2026-06-04",
                quality_json_last_synced="2026-06-04",
            )
        )
        out = " ".join(capsys.readouterr().out.split())
        assert "Prices synced 2026-06-04" in out
        assert "Quality synced 2026-06-04" in out

    def test_split_quality_row_omitted_when_date_absent(self, capsys: Any) -> None:
        """No quality date → no Quality row (a table predating the freshness stamp)."""
        render_terminal(
            _result_with_split(
                pricing_json_last_synced="2026-06-04",
                quality_json_last_synced=None,
            )
        )
        out = " ".join(capsys.readouterr().out.split())
        assert "Prices synced 2026-06-04" in out
        assert "Quality synced" not in out

    def test_split_fresh_dates_carry_no_amber_annotation(self) -> None:
        """A recent sync renders no staleness annotation on either row."""
        from datetime import date, timedelta

        recent = (date.today() - timedelta(days=5)).isoformat()
        text = _render_to_text(
            _result_with_split(
                pricing_json_last_synced=recent,
                quality_json_last_synced=recent,
            )
        )
        assert f"synced {recent}" in text
        assert "days old" not in text
        assert "refresh with" not in text

    def test_split_stale_prices_and_quality_annotate_with_commands(self) -> None:
        """Old sync dates annotate BOTH rows amber with the right refresh commands.

        Pricing trips at >30 days (fabricated 70 days old), quality at >60 days
        (fabricated 70 days old — past the 60-day window).  Each annotation names
        the age and the row-specific command.
        """
        import re
        from datetime import date, timedelta

        old = (date.today() - timedelta(days=70)).isoformat()
        text = _render_to_text(
            _result_with_split(
                pricing_json_last_synced=old,
                quality_json_last_synced=old,
            )
        )
        # The styled spans (dim date / amber annotation / cyan command) interleave
        # SGR codes, so strip ANSI before the literal substring check.
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert f"Prices synced {old} — ⚠ 70 days old; refresh with frugon pricing update" in flat
        assert f"Quality synced {old} — ⚠ 70 days old; refresh with frugon quality update" in flat

    def test_split_quality_fresh_but_prices_stale(self) -> None:
        """The 60-day quality window is independent: a 45-day quality table is
        fresh while a 45-day pricing table is stale.

        Proves the two thresholds are distinct (30 vs 60), not a shared predicate.
        """
        import re
        from datetime import date, timedelta

        mid = (date.today() - timedelta(days=45)).isoformat()
        text = _render_to_text(
            _result_with_split(
                pricing_json_last_synced=mid,
                quality_json_last_synced=mid,
            )
        )
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        # Pricing is stale at 45 days (>30); quality is still fresh at 45 (<60).
        assert "refresh with frugon pricing update" in flat
        assert "refresh with frugon quality update" not in flat


class TestSplitWindowCaution:
    """The Accounting block warns when --window contradicts the observed span.

    ``--window N`` overrides the monthly-projection basis, so a window that
    materially disagrees with the log's real span silently scales the monthly
    figure.  The caution renders ONLY when --window was given, the span is known,
    and the two materially disagree — and is absent otherwise.  The split fixture's
    observed span is 7 days, so window=30 (ratio ~4.3) fires while window=7 (an
    exact match) and no-window do not.
    """

    def test_window_caution_renders_on_mismatch(self) -> None:
        """window=30 vs a 7-day span → the amber Window caution row appears."""
        import re

        text = _render_to_text(_result_with_split(window_days=30))
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert "Window" in flat
        assert "--window 30 overrides your log's actual ~7-day span" in flat
        assert "Drop --window to project from the real span." in flat

    def test_window_caution_absent_when_window_matches_span(self) -> None:
        """window=7 ≈ the 7-day span → no caution."""
        import re

        text = _render_to_text(_result_with_split(window_days=7))
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert "overrides your log's actual" not in flat

    def test_window_caution_absent_when_no_window_flag(self) -> None:
        """No --window (window_days None) → no caution even with a known span."""
        import re

        text = _render_to_text(_result_with_split(window_days=None))
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert "overrides your log's actual" not in flat

    def test_window_caution_absent_when_span_unknown(self) -> None:
        """No timestamps (span None) → no caution even with --window given."""
        import re

        text = _render_to_text(
            _result_with_split(window_days=30, observed_span_days=None)
        )
        flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())
        assert "overrides your log's actual" not in flat


class TestSplitUpperBoundLine:
    """The DEFAULT split view promotes the Upper-bound to one dim line.

    The line must reconcile EXACTLY with the verbose note and the wholesale panel —
    same helpers, same rounding — and must be a single dim line (model name cyan,
    no second green number).
    """

    def test_upper_bound_reconciles_with_verbose_note(self, capsys: Any) -> None:
        """The default Upper-bound %, the verbose note %, and the wholesale panel %
        all show the same figure (same helpers, same rounding).
        """
        # Default view percentage.
        render_terminal(_result_with_split())
        default_out = " ".join(capsys.readouterr().out.split())
        import re

        default_pct = re.search(r"a full swap to gpt-4o saves ~(\d+\.\d+)%", default_out)
        assert default_pct is not None, default_out

        # Verbose note percentage (the aggressive end).
        render_terminal(_result_with_split(), verbose=True)
        verbose_out = " ".join(capsys.readouterr().out.split())
        verbose_pct = re.search(r"moving every call to gpt-4o saves ~(\d+\.\d+)%", verbose_out)
        assert verbose_pct is not None, verbose_out

        assert default_pct.group(1) == verbose_pct.group(1)

        # The split headline (34%-class figure) is unchanged and distinct.
        assert "SAVING" in verbose_out

    def test_upper_bound_present_when_candidate_matches_split(self, capsys: Any) -> None:
        """Audit finding #2: the Upper-bound row STILL renders when the wholesale
        candidate == the split's easy-call target.

        The split routes only the easy baseline calls while the full swap moves
        every call, so the full-swap figure (~67% on this fixture) is materially
        larger than the conservative split (~31%): the user must be able to tell
        the larger figure is the aggressive full-swap basis, even though both name
        the same model.
        """
        render_terminal(
            _result_with_split(candidate_model="gpt-4o-mini")  # == split candidate
        )
        out = " ".join(capsys.readouterr().out.split())
        assert "Upper bound" in out
        assert "a full swap to gpt-4o-mini saves" in out

    def test_upper_bound_absent_when_full_swap_no_better_than_split(
        self, capsys: Any
    ) -> None:
        """The honest non-redundant case: when the full swap saves no more than the
        conservative split, no Upper-bound row is printed (no separate aggressive
        basis to surface — e.g. an effectively fully-routed split).
        """
        render_terminal(
            _result_with_split(
                candidate_model="gpt-4o-mini",
                projected_cost=Decimal("0.0500"),  # full swap ~26% < split ~31%
            )
        )
        out = " ".join(capsys.readouterr().out.split())
        assert "Upper bound" not in out

    def test_upper_bound_line_is_not_green(self) -> None:
        """The Upper-bound line carries NO emerald — green stays the saving hero's.

        Collects every emerald (#10B981) segment and asserts none of the
        Upper-bound prose is among them (the line is dim, model name cyan only).
        """
        import io
        import sys

        from rich.console import Console

        report_mod = sys.modules[render_terminal.__module__]
        console = Console(
            file=io.StringIO(), width=88, force_terminal=True, color_system="truecolor"
        )
        captured: list[Any] = []
        original = report_mod.rprint

        def _capture(*args: Any, **kw: Any) -> None:
            for renderable in args:
                captured.extend(console.render(renderable))

        report_mod.rprint = _capture  # type: ignore[attr-defined]
        try:
            render_terminal(_result_with_split())
        finally:
            report_mod.rprint = original  # type: ignore[attr-defined]

        green_text = "".join(
            seg.text
            for seg in captured
            if seg.style is not None
            and any(
                c.triplet is not None and c.triplet.hex.lower() == SAVING_GREEN.lower()
                for c in (seg.style.color, seg.style.bgcolor)
                if c is not None
            )
        )
        assert "full swap" not in green_text
        assert "--verbose" not in green_text


class TestVerboseLogSpan:
    """The --verbose Notes block discloses the observed log time span (Item 1).

    Default (non-verbose) output never shows it; verbose shows a dim, labelled
    row only when both span dates are present.
    """

    def test_verbose_shows_log_span_row_when_dates_present(self) -> None:
        # Arrange — a result carrying both span dates and a fractional span.
        result = _result_with_split(
            observed_span_days=30.0,
            observed_span_start="2026-05-11",
            observed_span_end="2026-06-10",
        )

        # Act
        text = _render_to_text(result, verbose=True)
        collapsed = " ".join(text.split())

        # Assert — labelled row with both dates and the day count.
        assert "Log span" in collapsed, text
        assert "2026-05-11" in collapsed, text
        assert "2026-06-10" in collapsed, text
        assert "(30.0 days)" in collapsed, text
        # The arrow glyph ties the two dates together as a span.
        assert "→" in text, text

    def test_default_view_omits_log_span_row(self) -> None:
        # Arrange — same dates, but the default (non-verbose) view.
        result = _result_with_split(
            observed_span_days=30.0,
            observed_span_start="2026-05-11",
            observed_span_end="2026-06-10",
        )

        # Act
        text = _render_to_text(result)  # verbose defaults to False

        # Assert — the span row only ever appears under --verbose.
        assert "Log span" not in text, text

    def test_verbose_omits_log_span_row_when_dates_absent(self) -> None:
        # Arrange — no span bounds (a timestamp-free log).
        result = _result_with_split(
            observed_span_days=None,
            observed_span_start=None,
            observed_span_end=None,
        )

        # Act
        text = _render_to_text(result, verbose=True)

        # Assert — no empty or partial span row.
        assert "Log span" not in text, text
