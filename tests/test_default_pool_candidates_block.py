"""Default-pool "Candidates considered" block (PD-directed 2026-07-02).

Before this change, the "Candidates considered" block only fired when the user
passed an explicit ``--candidates`` list with more than one model — a real user
(or the un-pinned demo) running the default pool never saw what else was
considered, only the winner.  This suite pins the NEW behaviour: when the
DEFAULT pool (no ``--candidates``) produces a split recommendation and the pool
has more than one priced/rated model, the block renders the recommended
candidate plus the next-4-cheapest candidates that also beat the baseline (5
rows max) — never the full 23-model roster — with a cap caption explaining the
truncation.  The explicit ``--candidates`` path is untouched: see
``test_candidate_headline_block_agreement.py`` for the regression pin proving
byte-identical behaviour there.

A P0-class follow-up (same day) closed a caption-truth gap: the block's caption
claimed "the cheapest split is the headline recommendation," yet on the demo
fixture two OTHER candidates (llama-4-scout-17b-16e-instruct, gpt-4.1-nano)
render a cheaper dollar figure than the recommended deepseek-v4-flash — because
all three TIE at the 1dp precision the report actually prints (37.4%).  The
fix (PD-ratified): a quality-aware tie-break in
:func:`frugon.cost._select_cheapest_eligible` — among candidates that tie at
DISPLAY precision, the higher quality tier wins; a `tier_label` column on every
surface makes the tie-break provable from the printed table alone.  A second
review pass (same day) reworded the caption once more to name the axis the
pick is ACTUALLY made on ("the biggest saving is the headline recommendation"
— never "cheapest", a dollar-column claim the recommended row can visibly
contradict) and fixed a rounding-fidelity gap: the renderer now shares the
SAME Decimal ROUND_HALF_UP quantizer (``frugon.cost._display_pct``) the
selector's tie-break reads, instead of Python's binary-float ``.1f``
round-half-to-EVEN, which disagreed at exact .x5 boundaries.
``TestCaptionTruthInvariant`` and ``TestSelectCheapestEligibleTieBreak`` pin
this.
"""

from __future__ import annotations

import sys as _sys
from decimal import Decimal
from pathlib import Path

import pytest

import frugon
from frugon.cost import (
    _ROUTING_CANDIDATES,
    AnalysisResult,
    _get_model_tier,
    _select_cheapest_eligible,
    analyze_records,
    iter_records,
)
from frugon.report import (
    _candidate_cap_caption,
    _candidates_considered_html,
    _candidates_considered_md_lines,
    _fmt_candidate_saving,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
    render_terminal,
)

_sys.path.insert(0, str(Path(__file__).parent))
from conftest import install_unrated_sentinel  # noqa: E402

assert frugon.__file__ is not None
_SAMPLE = Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"


@pytest.fixture(autouse=True)
def _sentinel_pricing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Mirrors test_candidate_headline_block_agreement.py: decouples the default-
    # pool candidate arithmetic from live registry drift.
    install_unrated_sentinel(monkeypatch, tmp_path)
    yield
    import frugon.pricing as _p

    _p.clear_pricing_cache()


def _demo_result() -> AnalysisResult:
    """Analyze the bundled demo log with the DEFAULT pool (candidates=None)."""
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(list(records), skipped_malformed=skipped, split_routing=True)


# ---------------------------------------------------------------------------
# Population — cap to <=5 rows, recommended first, ranked by split New-spend.
# ---------------------------------------------------------------------------


class TestDefaultPoolBlockPopulation:
    def test_default_pool_block_caps_at_five_rows(self) -> None:
        """The default-pool block never shows more than 5 rows."""
        result = _demo_result()
        assert result.used_default_pool is True
        assert result.split is not None
        assert 1 < len(result.candidate_projections) <= 5

    def test_default_pool_block_recommended_row_is_split_target(self) -> None:
        """Row 0 is the split's own routing target, tagged 'recommended'."""
        result = _demo_result()
        assert result.split is not None
        assert result.candidate_projections[0].model == result.split.candidate_model
        assert result.candidate_projections[0].status == "recommended"

    def test_default_pool_block_only_eligible_and_beats_baseline_rows(self) -> None:
        """Every non-recommended row is ELIGIBLE (rated) AND beats the baseline.

        The default-pool cap never pads with more-expensive, unpriced, or
        unrated rows — only the recommended candidate plus other RATED
        candidates that beat the baseline on the full-dataset New-spend basis
        are shown (:func:`frugon.cost._select_cheapest_eligible`'s own
        eligibility rule: priced, rated, strictly cheaper than baseline).
        """
        from frugon.quality import is_unrated

        result = _demo_result()
        for proj in result.candidate_projections[1:]:
            assert proj.status in ("recommended", "considered"), proj.status
            assert proj.status != "more_expensive"
            assert proj.status != "unpriced"
            assert not is_unrated(proj.model), (
                f"{proj.model} is unrated — the default-pool block must only "
                "show eligible (rated) candidates"
            )

    def test_default_pool_block_ranked_cheapest_first_among_the_rest(self) -> None:
        """Rows after the recommended one are sorted by ascending New-spend."""
        result = _demo_result()
        rest = result.candidate_projections[1:]
        costs = [
            (p.monthly_cost if p.monthly_cost is not None else p.observed_cost)
            for p in rest
        ]
        assert all(c is not None for c in costs)
        assert costs == sorted(costs)  # type: ignore[type-var]

    def test_default_pool_block_rows_come_from_the_built_in_pool(self) -> None:
        """Every candidate shown is drawn from _ROUTING_CANDIDATES."""
        result = _demo_result()
        for proj in result.candidate_projections:
            assert proj.model in _ROUTING_CANDIDATES

    def test_default_pool_block_absent_when_no_split_recommendation(self) -> None:
        """No split -> empty candidate_projections (--wholesale disables the split)."""
        records, skipped = iter_records(_SAMPLE)
        result = analyze_records(
            list(records), skipped_malformed=skipped, split_routing=False
        )
        assert result.split is None
        assert result.candidate_projections == []

    def test_default_pool_block_absent_when_pool_has_one_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single-model pool never renders the transparency block."""
        import frugon.cost as cost_mod

        monkeypatch.setattr(cost_mod, "_ROUTING_CANDIDATES", ["deepseek-v4-flash"])
        records, skipped = iter_records(_SAMPLE)
        result = analyze_records(
            list(records), skipped_malformed=skipped, split_routing=True
        )
        assert len(result.candidate_projections) <= 1


# ---------------------------------------------------------------------------
# Section 2a reconciliation — per-row figures reconcile from printed values.
# ---------------------------------------------------------------------------


class TestDefaultPoolBlockReconciliation:
    def test_recommended_row_equals_headline_new_spend_to_the_cent(self) -> None:
        """The block's recommended row IS the headline New-spend (same Decimal)."""
        from frugon.report import _split_current_and_blended

        result = _demo_result()
        assert result.split is not None
        _current, headline_newspend, _projected = _split_current_and_blended(
            result, result.split
        )
        rec = result.candidate_projections[0]
        assert rec.monthly_cost is not None
        assert rec.monthly_cost == headline_newspend

    def test_every_row_saving_pct_derives_from_rounded_dollar_components(self) -> None:
        """§2a: each row's saving% == (current - new) / current, from PRINTED $."""
        result = _demo_result()
        full_current = result.monthly_cost
        assert full_current is not None
        for proj in result.candidate_projections:
            assert proj.monthly_cost is not None
            assert proj.saving_pct is not None
            expected_saving = full_current - proj.monthly_cost
            assert proj.monthly_saving == expected_saving
            expected_pct = (expected_saving / full_current) * Decimal("100")
            assert proj.saving_pct == expected_pct


# ---------------------------------------------------------------------------
# Rendering parity — terminal + MD v1/v2 + HTML v1/v2 all show the block.
# ---------------------------------------------------------------------------


class TestDefaultPoolBlockRendersOnEverySurface:
    def test_terminal_renders_the_block_and_cap_caption(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _demo_result()
        render_terminal(result)
        out = " ".join(capsys.readouterr().out.split())
        assert "Candidates considered" in out
        assert result.split is not None
        assert result.split.candidate_model in out
        assert "candidates considered" in out.lower()
        assert "next-cheapest" in out
        assert "--candidates to compare specific models" in out

    def test_cap_caption_derives_count_from_candidate_pool_size(self) -> None:
        """The cap caption's pool count comes from candidate_pool_size, never hardcoded."""
        result = _demo_result()
        caption = _candidate_cap_caption(result)
        assert caption is not None
        assert str(result.candidate_pool_size) in caption
        assert str(len(result.candidate_projections) - 1) in caption

    def test_cap_caption_none_on_explicit_candidates_path(self) -> None:
        """The cap caption never fires for an explicit --candidates run."""
        records, skipped = iter_records(_SAMPLE)
        result = analyze_records(
            list(records),
            candidates=["gpt-4o", "gpt-4.1-mini"],
            skipped_malformed=skipped,
            split_routing=True,
        )
        assert _candidate_cap_caption(result) is None

    def test_markdown_v1_renders_the_block(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r.md"
        render_markdown(result, out_path)
        md = out_path.read_text(encoding="utf-8")
        assert "## Candidates considered" in md
        assert result.split is not None
        assert result.split.candidate_model in md
        assert "next-cheapest" in md

    def test_markdown_v2_renders_the_block(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r2.md"
        render_markdown_v2(result, out_path)
        md = out_path.read_text(encoding="utf-8")
        assert "## Candidates considered" in md
        assert "next-cheapest" in md

    def test_html_v1_renders_the_block(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r.html"
        render_html(result, out_path)
        html = out_path.read_text(encoding="utf-8")
        assert "candidates-considered" in html
        assert result.split is not None
        assert result.split.candidate_model in html
        assert "next-cheapest" in html

    def test_html_v2_renders_the_block(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r2.html"
        render_html_v2(result, out_path)
        html = out_path.read_text(encoding="utf-8")
        assert "candidates-considered" in html
        assert "next-cheapest" in html

    def test_md_lines_helper_includes_cap_caption(self) -> None:
        result = _demo_result()
        lines = _candidates_considered_md_lines(result)
        joined = "\n".join(lines)
        assert "next-cheapest" in joined

    def test_html_helper_includes_cap_caption(self) -> None:
        result = _demo_result()
        html = _candidates_considered_html(result, lambda s: s)
        assert "next-cheapest" in html


# ---------------------------------------------------------------------------
# CLI end-to-end — --demo (un-pinned) and a real log both show the block.
# ---------------------------------------------------------------------------


class TestDefaultPoolBlockViaCLI:
    def test_demo_cli_run_shows_the_block(self) -> None:
        from typer.testing import CliRunner

        from frugon.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["analyze", "--demo", "--no-progress"])
        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())
        assert "Candidates considered" in flat
        assert "next-cheapest" in flat
        # The existing pool-source notice stays, verbatim, alongside the new block.
        assert "Recommendations use a curated set" in flat

    def test_real_log_run_shows_the_block(self, tmp_path: Path) -> None:
        import json as _json

        from typer.testing import CliRunner

        from frugon.cli import app

        log_file = tmp_path / "real.jsonl"
        records = [
            {
                "model": "gpt-5.5",
                "request": {"messages": [{"role": "user", "content": "classify this"}]},
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                },
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }
            for _ in range(200)
        ]
        with log_file.open("w") as fh:
            for rec in records:
                fh.write(_json.dumps(rec) + "\n")

        runner = CliRunner()
        result = runner.invoke(app, ["analyze", str(log_file), "--no-progress"])
        assert result.exit_code == 0, result.output
        flat = " ".join(result.output.split())
        assert "Candidates considered" in flat
        assert "next-cheapest" in flat


# ---------------------------------------------------------------------------
# Explicit --candidates regression pin — unchanged (see also
# test_candidate_headline_block_agreement.py for the fuller regression suite).
# ---------------------------------------------------------------------------


class TestExplicitCandidatesPathUnchanged:
    def test_explicit_candidates_never_capped(self) -> None:
        """An explicit --candidates run shows every passed candidate, uncapped."""
        records, skipped = iter_records(_SAMPLE)
        explicit = [
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4o-mini",
            "gpt-4.1-nano",
            "claude-haiku-4-5",
            "grok-3-mini",
        ]
        result = analyze_records(
            list(records),
            candidates=explicit,
            skipped_malformed=skipped,
            split_routing=True,
        )
        # 6 requested minus the dominant model (gpt-5.5, not in this list) == 6 rows.
        assert len(result.candidate_projections) == len(explicit)

    def test_explicit_candidates_no_cap_caption_in_terminal(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        records, skipped = iter_records(_SAMPLE)
        result = analyze_records(
            list(records),
            candidates=["gpt-4o", "gpt-4.1-mini"],
            skipped_malformed=skipped,
            split_routing=True,
        )
        render_terminal(result)
        out = " ".join(capsys.readouterr().out.split())
        assert "next-cheapest" not in out


# ---------------------------------------------------------------------------
# C4 — caption-truth invariant: provable from the PRINTED values alone.
#
# The block's caption claims "the cheapest split is the headline
# recommendation" (plus the tie-break clause).  This must be checkable purely
# from what a reader sees on the page: the recommended row's percent is >= every
# other row's percent, and among rows that TIE its percent at display
# precision, no row carries a strictly better (lower-numbered) quality tier.
# ---------------------------------------------------------------------------


def _rendered_pct(saving_pct: Decimal) -> Decimal:
    """Parse the ACTUAL printed cell (`_fmt_candidate_saving`'s output) back to
    a Decimal, so every invariant test below is proven against what a reader
    literally sees — not a second, test-local re-implementation of the
    rounding rule that could silently drift from the renderer (P1-2: a prior
    version of this suite quantized locally and so could never have caught the
    P1-1 rounding-fidelity bug, where the renderer's binary-float `.1f`
    disagreed with the selector's Decimal ROUND_HALF_UP at exact .x5
    boundaries).  ``_fmt_candidate_saving`` renders "X.X% lower" / "X.X% higher"
    — parse the leading number and re-apply the sign so the returned value
    compares directly against another rendered cell's parsed value.
    """
    cell = _fmt_candidate_saving(saving_pct)
    number_str, _, suffix = cell.partition("%")
    magnitude = Decimal(number_str)
    return magnitude if "lower" in suffix else -magnitude


class TestCaptionTruthInvariant:
    def test_recommended_row_percent_is_at_least_every_other_row(self) -> None:
        """§2a-style proof #1: recommended's PRINTED % >= every row's PRINTED %."""
        result = _demo_result()
        rec = result.candidate_projections[0]
        assert rec.status == "recommended"
        assert rec.saving_pct is not None
        rec_pct = _rendered_pct(rec.saving_pct)
        for proj in result.candidate_projections:
            assert proj.saving_pct is not None
            assert rec_pct >= _rendered_pct(proj.saving_pct), (
                f"{proj.model} prints a HIGHER saving% "
                f"({_fmt_candidate_saving(proj.saving_pct)}) than the "
                f"recommended {rec.model} ({_fmt_candidate_saving(rec.saving_pct)}) "
                "— the caption's claim is not provable from the printed table"
            )

    def test_no_display_tie_beats_the_recommended_row_on_tier(self) -> None:
        """§2a-style proof #2: among rows tying the recommended row's PRINTED
        percent, none has a strictly BETTER (lower-numbered) quality tier."""
        result = _demo_result()
        rec = result.candidate_projections[0]
        assert rec.saving_pct is not None
        rec_pct = _rendered_pct(rec.saving_pct)
        rec_tier = _get_model_tier(rec.model)
        for proj in result.candidate_projections[1:]:
            assert proj.saving_pct is not None
            if _rendered_pct(proj.saving_pct) == rec_pct:
                proj_tier = _get_model_tier(proj.model)
                assert proj_tier >= rec_tier, (
                    f"{proj.model} ties the recommended row's PRINTED percent "
                    f"but has a BETTER tier ({proj_tier} < {rec_tier}) — the "
                    "recommendation is not honest"
                )

    def test_demo_fixture_actually_exercises_a_display_tie(self) -> None:
        """Precondition guard: prove the demo fixture has a REAL, PRINTED-cell
        tie, so the invariant tests above are not vacuously true."""
        result = _demo_result()
        pcts = [
            _rendered_pct(p.saving_pct)
            for p in result.candidate_projections
            if p.saving_pct is not None
        ]
        assert len(pcts) != len(set(pcts)), (
            "expected at least one display-precision tie among the demo's "
            "candidate rows — if this fails, the fixture or pricing/quality "
            "data has drifted and the caption-truth tests need a new fixture"
        )

    def test_renderer_and_selector_quantize_identically_at_an_x5_boundary(
        self,
    ) -> None:
        """P1-1 regression pin: a genuine .x5 boundary must round the SAME way
        through the selector's Decimal quantizer and the renderer's printed
        string — proving selector-tie <=> printed-string equality.

        37.25 is exactly the boundary where binary-float ``.1f`` (round-half-
        to-EVEN: "37.2") disagrees with Decimal ROUND_HALF_UP ("37.3"). Two
        candidates whose raw saving_pct both round to 37.25 must render the
        IDENTICAL cell string (proving they are the same printed tie the
        selector's tie-break key also sees) at the ROUND_HALF_UP answer, not
        the round-half-even one.
        """
        from frugon.cost import _display_pct

        a = Decimal("37.25")
        b = Decimal("37.2500001")  # rounds to the same 1dp cell as `a`
        assert _display_pct(a) == _display_pct(b) == Decimal("37.3")
        cell_a = _fmt_candidate_saving(a)
        cell_b = _fmt_candidate_saving(b)
        assert cell_a == cell_b == "37.3% lower", (
            f"selector-tie candidates rendered DIFFERENT cells: {cell_a!r} vs "
            f"{cell_b!r} — the renderer and selector have diverged at an .x5 "
            "boundary"
        )
        # The old float `.1f` behaviour (round-half-to-EVEN) would have printed
        # "37.2%" here — pin that the renderer is NOT doing that any more.
        assert f"{float(a):.1f}" == "37.2", (
            "fixture precondition: 37.25 must be a genuine round-half-even vs "
            "ROUND_HALF_UP divergence point, or this test proves nothing"
        )
        assert "37.2%" not in cell_a

    def test_tier_label_present_on_every_row(self) -> None:
        """Every row carries a tier_label so the tie-break is visible in the table."""
        result = _demo_result()
        for proj in result.candidate_projections:
            assert proj.tier_label
            assert proj.tier_label != ""

    def test_caption_states_the_tie_break_rule(self) -> None:
        from frugon.report import _candidate_caption

        caption = _candidate_caption(has_judge_section=False)
        assert "higher quality tier" in caption

    def test_terminal_renders_quality_tier_column(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _demo_result()
        render_terminal(result)
        out = " ".join(capsys.readouterr().out.split())
        assert "Quality tier" in out
        # The recommended row's own tier name must appear in the rendered table.
        rec = result.candidate_projections[0]
        assert rec.tier_label in out

    def test_markdown_renders_quality_tier_column(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r.md"
        render_markdown(result, out_path)
        md = out_path.read_text(encoding="utf-8")
        assert "Quality tier" in md

    def test_html_renders_quality_tier_column(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r.html"
        render_html(result, out_path)
        html = out_path.read_text(encoding="utf-8")
        assert "Quality tier" in html


# ---------------------------------------------------------------------------
# _select_cheapest_eligible — the shared selection rule, unit-tested directly.
# ---------------------------------------------------------------------------


class TestSelectCheapestEligibleTieBreak:
    def test_cheapest_wins_when_no_display_tie(self) -> None:
        """No tie -> plain cheapest-New-spend wins, regardless of tier."""
        newspend = {"a": Decimal("100"), "b": Decimal("50")}
        winner = _select_cheapest_eligible(
            ["a", "b"],
            cand_split_newspend=newspend,
            baseline_newspend=Decimal("200"),
            rated_only=False,
        )
        assert winner == "b"

    def test_higher_tier_wins_a_display_precision_tie(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two candidates whose New-spend rounds to the SAME 1dp percent: the
        better (lower-numbered) tier wins even though it is not the numerically
        cheapest."""
        import frugon.cost as cost_mod

        # baseline_newspend=1000; New-spend 620 -> saving 38.0%; New-spend
        # 619.9 -> saving 38.01% -> rounds to 38.0% too: a genuine display tie
        # where "b" is a hair cheaper than "a" but must lose to a better tier.
        newspend = {"a": Decimal("620.0"), "b": Decimal("619.9")}
        monkeypatch.setattr(
            cost_mod,
            "_get_model_tier",
            lambda m: {"a": 0, "b": 1}[m],
        )
        monkeypatch.setattr(cost_mod, "_is_unrated", lambda m: False)
        winner = _select_cheapest_eligible(
            ["a", "b"],
            cand_split_newspend=newspend,
            baseline_newspend=Decimal("1000"),
            rated_only=False,
        )
        assert winner == "a", "the better tier (0) must win the display tie"

    def test_name_tie_break_when_percent_and_tier_both_tie(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import frugon.cost as cost_mod

        newspend = {"z-cand": Decimal("500"), "a-cand": Decimal("500")}
        monkeypatch.setattr(cost_mod, "_get_model_tier", lambda m: 1)
        monkeypatch.setattr(cost_mod, "_is_unrated", lambda m: False)
        winner = _select_cheapest_eligible(
            ["z-cand", "a-cand"],
            cand_split_newspend=newspend,
            baseline_newspend=Decimal("1000"),
            rated_only=False,
        )
        assert winner == "a-cand"

    def test_rated_only_excludes_unrated_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import frugon.cost as cost_mod

        newspend = {"cheap-unrated": Decimal("10"), "rated": Decimal("50")}
        monkeypatch.setattr(cost_mod, "_get_model_tier", lambda m: 1)
        monkeypatch.setattr(
            cost_mod, "_is_unrated", lambda m: m == "cheap-unrated"
        )
        winner = _select_cheapest_eligible(
            ["cheap-unrated", "rated"],
            cand_split_newspend=newspend,
            baseline_newspend=Decimal("1000"),
            rated_only=True,
        )
        assert winner == "rated"

    def test_returns_none_when_nothing_beats_baseline(self) -> None:
        newspend = {"a": Decimal("2000")}
        winner = _select_cheapest_eligible(
            ["a"],
            cand_split_newspend=newspend,
            baseline_newspend=Decimal("1000"),
            rated_only=False,
        )
        assert winner is None

    def test_rated_candidate_beats_unrated_on_display_tie_even_with_rated_only_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P2-2(a): the unrated tier-sentinel (10_000) path.

        Two candidates tie at display precision; one is unrated. Even with
        rated_only=False (both are eligible to be RANKED), the rated candidate
        must still win the tie — the unrated tier sentinel (10_000) is always
        worse than any real tier, so it never wins a display tie against a
        rated candidate, regardless of the rated_only gate.
        """
        import frugon.cost as cost_mod

        newspend = {"rated-cand": Decimal("500"), "unrated-cand": Decimal("500")}
        monkeypatch.setattr(
            cost_mod, "_get_model_tier", lambda m: 2 if m == "rated-cand" else -1
        )
        monkeypatch.setattr(
            cost_mod, "_is_unrated", lambda m: m == "unrated-cand"
        )
        winner = _select_cheapest_eligible(
            ["unrated-cand", "rated-cand"],
            cand_split_newspend=newspend,
            baseline_newspend=Decimal("1000"),
            rated_only=False,
        )
        assert winner == "rated-cand", (
            "a rated candidate must beat an unrated one on a display tie even "
            "when rated_only=False lets both be ranked"
        )

    def test_candidate_exactly_at_baseline_is_excluded(self) -> None:
        """P2-2(b): New-spend == baseline (no strict improvement) is excluded.

        _select_cheapest_eligible's eligibility gate is ``ns < baseline_newspend``
        (strict) — a candidate whose New-spend exactly equals the baseline saves
        nothing and must never be selected, matching §6 "never inflate, never
        pretend a non-saving is a saving."
        """
        newspend = {"exactly-baseline": Decimal("1000"), "genuinely-cheaper": Decimal("900")}
        winner = _select_cheapest_eligible(
            ["exactly-baseline"],
            cand_split_newspend=newspend,
            baseline_newspend=Decimal("1000"),
            rated_only=False,
        )
        assert winner is None, (
            "a candidate at exactly the baseline New-spend must be excluded, "
            "not selected as a zero-saving 'winner'"
        )
        # Sanity: the SAME baseline with a genuinely cheaper candidate present
        # still selects normally — proving the exclusion is specific to the
        # exact-equality case, not a bug that excludes everything.
        winner_with_real_saving = _select_cheapest_eligible(
            ["exactly-baseline", "genuinely-cheaper"],
            cand_split_newspend=newspend,
            baseline_newspend=Decimal("1000"),
            rated_only=False,
        )
        assert winner_with_real_saving == "genuinely-cheaper"
