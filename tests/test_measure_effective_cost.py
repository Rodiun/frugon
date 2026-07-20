"""Tests for effective cost per judged success (price / P(judged success)).

The judge already produces win/tie/loss verdicts and the cost analysis
already produces a $ figure for the candidate, but the two render in separate
sections -- nobody sees price / P(judged success), the metric that shows
whether a cheaper model that fails more often is actually cheaper per
successful outcome.

Denominator honesty: success_count = wins + (ties NOT flagged both_failed) --
a tie the pointwise check flagged both-failed is a shared failure, not a
judged success (the FRG-OSS-068 dependency this item builds on).

The reconciling invariant: the effective $ figure is always derived from the
price this run ALREADY PRINTS elsewhere (the split/wholesale headline "New"
figure, or the "Candidates considered" block's own projection) -- never a
freshly-computed price that could disagree with the panel.
"""

from __future__ import annotations

import io
import re
from decimal import Decimal
from typing import Any

import pytest
from rich.console import Console

from frugon.cost import AnalysisResult, CandidateProjection
from frugon.measure import MeasureResult, Tier1Tally
from frugon.report import (
    _candidate_shown_price,
    _candidates_use_monthly_basis,
    _check_error_footnote_text,
    _effective_cost_per_success_text,
    _fmt_usd,
    _judged_success_summary,
    _quality_section_html,
    _quality_section_md,
    _quantize_usd_for_display,
    _reconciled_effective_cost_per_success,
    _split_priced_effective_cost_footnote_text,
    render_quality_terminal,
)
from frugon.routing import SplitRouting

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _split_result() -> tuple[AnalysisResult, SplitRouting]:
    """A split-headline AnalysisResult whose "New" figure is $0.02 exactly.

    baseline_cost=$0.10, blended_cost=$0.02 -> saving 80% (positive, so
    _has_split is True) on the observed (non-monthly) basis.
    """
    split = SplitRouting(
        baseline_model="gpt-4-turbo",
        candidate_model="gpt-4o-mini",
        routed_count=8,
        kept_count=2,
        routed_cost=Decimal("0.004"),
        kept_cost=Decimal("0.016"),
        baseline_cost=Decimal("0.10"),
        blended_cost=Decimal("0.02"),
        easy_threshold=Decimal("0.35"),
    )
    result = AnalysisResult(
        total_calls=10,
        priced_calls=10,
        unpriced_calls=0,
        total_cost=Decimal("0.10"),
        cost_by_model={"gpt-4-turbo": Decimal("0.10")},
        calls_by_model={"gpt-4-turbo": 10},
        split=split,
    )
    return result, split


def _wholesale_result() -> AnalysisResult:
    """A no-split AnalysisResult whose wholesale "New" figure is $0.50."""
    return AnalysisResult(
        total_calls=10,
        priced_calls=10,
        unpriced_calls=0,
        total_cost=Decimal("1.00"),
        cost_by_model={"claude-3-opus": Decimal("1.00")},
        calls_by_model={"claude-3-opus": 10},
        projected_cost=Decimal("0.50"),
        candidate_model="claude-haiku",
    )


def _multi_candidate_result() -> AnalysisResult:
    """A result whose "Candidates considered" block prices a non-headline model."""
    return AnalysisResult(
        total_calls=10,
        priced_calls=10,
        unpriced_calls=0,
        total_cost=Decimal("1.00"),
        cost_by_model={"claude-3-opus": Decimal("1.00")},
        calls_by_model={"claude-3-opus": 10},
        projected_cost=Decimal("0.50"),
        candidate_model="claude-haiku",
        candidate_projections=[
            CandidateProjection(
                model="claude-haiku", status="recommended", observed_cost=Decimal("0.50")
            ),
            CandidateProjection(
                model="gpt-4o-mini", status="considered", observed_cost=Decimal("0.30")
            ),
            CandidateProjection(
                model="frugon-eval-unrated-x1", status="unpriced"
            ),
        ],
    )


def _tally(
    candidate: str = "gpt-4o-mini",
    wins: int = 6,
    losses: int = 1,
    ties: int = 3,
    errors: int = 0,
    both_failed_ties: int = 1,
) -> Tier1Tally:
    return Tier1Tally(
        candidate=candidate,
        wins=wins,
        losses=losses,
        ties=ties,
        errors=errors,
        both_failed_ties=both_failed_ties,
    )


# ---------------------------------------------------------------------------
# Tier1Tally.judged_success_count / .verdict_count -- both-failed exclusion
# ---------------------------------------------------------------------------


class TestTallyCounting:
    def test_judged_success_count_excludes_both_failed_ties(self) -> None:
        tally = Tier1Tally(candidate="x", wins=3, losses=1, ties=2, both_failed_ties=1)
        assert tally.judged_success_count == 4  # 3 wins + (2 ties - 1 both-failed)
        assert tally.verdict_count == 6  # wins + losses + ties, errors excluded

    def test_judged_success_count_zero_when_every_tie_is_both_failed(self) -> None:
        tally = Tier1Tally(candidate="x", wins=0, losses=0, ties=3, both_failed_ties=3)
        assert tally.judged_success_count == 0

    def test_judged_success_count_counts_clean_tie_as_success(self) -> None:
        tally = Tier1Tally(candidate="x", wins=0, losses=0, ties=1, both_failed_ties=0)
        assert tally.judged_success_count == 1

    def test_verdict_count_excludes_errors(self) -> None:
        tally = Tier1Tally(candidate="x", wins=1, losses=1, ties=1, errors=5)
        assert tally.verdict_count == 3


# ---------------------------------------------------------------------------
# run_measure integration -- both_failed_ties aggregates onto Tier1Tally
# ---------------------------------------------------------------------------


class TestRunMeasureAggregatesBothFailedTies:
    @pytest.fixture(autouse=True)
    def _provider_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

    def test_run_measure_tally_counts_both_failed_tie(self) -> None:
        from unittest.mock import MagicMock, patch

        from frugon.cost import LogRecord
        from frugon.measure import run_measure

        mock_litellm = MagicMock()
        resp = MagicMock()
        resp.choices[0].message.content = "model output"
        mock_litellm.completion.return_value = resp
        record = LogRecord(
            model="gpt-4o",
            messages=[{"role": "user", "content": "prompt"}],
            completion_text="ok",
            prompt_tokens=10,
            completion_tokens=5,
            timestamp=None,
        )

        with (
            patch("frugon.measure._import_litellm", return_value=mock_litellm),
            patch("frugon.measure._judge_pair", return_value="tie"),
            patch("frugon.measure._judge_addressed", return_value=False),
        ):
            result = run_measure(
                [record],
                "gpt-4o",
                ["gpt-4o-mini"],
                n_samples=1,
                use_judge=True,
                judge_model="gpt-4o",
                concurrency=1,
                seed=0,
            )

        assert result.tier1_tallies is not None
        tally = result.tier1_tallies[0]
        assert tally.ties == 1
        assert tally.both_failed_ties == 1
        assert tally.judged_success_count == 0


# ---------------------------------------------------------------------------
# _reconciled_effective_cost_per_success -- golden vectors
# ---------------------------------------------------------------------------


class TestReconciledEffectiveCostGoldenVectors:
    def test_exact_division(self) -> None:
        # price=$0.02, 8/10 judged successes -> $0.02 / 0.8 = $0.025 exactly.
        effective = _reconciled_effective_cost_per_success(Decimal("0.02"), 8, 10)
        assert effective == Decimal("0.025")

    def test_sub_cent_price_quantizes_to_four_dp_before_dividing(self) -> None:
        # Raw price $0.005 (sub-cent) quantizes to $0.0050 (4dp tier) BEFORE
        # dividing -- not the raw, unquantized value.
        effective = _reconciled_effective_cost_per_success(Decimal("0.005"), 5, 10)
        assert effective == Decimal("0.0100")

    def test_hundred_percent_success_equals_price_exactly(self) -> None:
        # At 100% judged success, the effective figure MUST equal the
        # (quantized) price exactly -- the reconciling invariant's anchor case.
        effective = _reconciled_effective_cost_per_success(Decimal("3.33"), 10, 10)
        assert effective == Decimal("3.33")

    def test_one_success_inflates_price_by_verdict_count(self) -> None:
        # 1 success out of 10 verdicts -> price / 0.1 = price * 10.
        effective = _reconciled_effective_cost_per_success(Decimal("1.00"), 1, 10)
        assert effective == Decimal("10.00")

    def test_zero_success_is_none_never_a_divide_by_zero(self) -> None:
        assert _reconciled_effective_cost_per_success(Decimal("1.00"), 0, 10) is None

    def test_zero_verdict_count_is_none(self) -> None:
        assert _reconciled_effective_cost_per_success(Decimal("1.00"), 0, 0) is None


# ---------------------------------------------------------------------------
# _candidate_shown_price -- resolution order
# ---------------------------------------------------------------------------


class TestCandidateShownPrice:
    def test_none_result_yields_none(self) -> None:
        assert _candidate_shown_price(None, "gpt-4o-mini") is None

    def test_split_headline_price(self) -> None:
        result, split = _split_result()
        priced = _candidate_shown_price(result, split.candidate_model)
        assert priced is not None
        price, is_monthly = priced
        assert price == Decimal("0.02")
        assert is_monthly is False

    def test_wholesale_headline_price(self) -> None:
        result = _wholesale_result()
        priced = _candidate_shown_price(result, "claude-haiku")
        assert priced is not None
        price, is_monthly = priced
        assert price == Decimal("0.50")
        assert is_monthly is False

    def test_candidate_projections_observed_price(self) -> None:
        result = _multi_candidate_result()
        priced = _candidate_shown_price(result, "gpt-4o-mini")
        assert priced is not None
        price, is_monthly = priced
        assert price == Decimal("0.30")
        assert is_monthly is False

    def test_candidate_projections_monthly_preferred_over_observed(self) -> None:
        result = _multi_candidate_result()
        result.candidate_projections.append(
            CandidateProjection(
                model="mistral-small",
                status="considered",
                monthly_cost=Decimal("9.00"),
                observed_cost=Decimal("0.30"),
            )
        )
        priced = _candidate_shown_price(result, "mistral-small")
        assert priced == (Decimal("9.00"), True)

    def test_unpriced_candidate_projection_yields_none(self) -> None:
        result = _multi_candidate_result()
        assert _candidate_shown_price(result, "frugon-eval-unrated-x1") is None

    def test_unknown_candidate_yields_none(self) -> None:
        result = _wholesale_result()
        assert _candidate_shown_price(result, "some-other-model") is None


# ---------------------------------------------------------------------------
# _effective_cost_per_success_text -- combines price resolution + division
# ---------------------------------------------------------------------------


class TestEffectiveCostPerSuccessText:
    def test_none_result_omits_the_metric_entirely(self) -> None:
        assert _effective_cost_per_success_text(_tally(), None) is None

    def test_unpriced_candidate_renders_unpriced(self) -> None:
        result = _wholesale_result()
        text = _effective_cost_per_success_text(_tally(candidate="unknown"), result)
        assert text == "n/a (unpriced)"

    def test_zero_judged_success_renders_zero_successes(self) -> None:
        result, split = _split_result()
        tally = _tally(
            candidate=split.candidate_model, wins=0, losses=5, ties=2, both_failed_ties=2
        )
        text = _effective_cost_per_success_text(tally, result)
        assert text == "n/a (0 judged successes)"

    def test_priced_candidate_renders_effective_dollar_figure(self) -> None:
        result, split = _split_result()
        tally = _tally(candidate=split.candidate_model)  # 8/10 judged successes
        text = _effective_cost_per_success_text(tally, result)
        # $0.02 / 0.8 = $0.025 -> _fmt_usd rounds HALF_UP to 2dp -> $0.03
        assert text == "$0.03"

    def test_monthly_price_appends_per_month_suffix(self) -> None:
        result = _multi_candidate_result()
        result.candidate_projections.append(
            CandidateProjection(
                model="mistral-small", status="considered", monthly_cost=Decimal("9.00")
            )
        )
        tally = _tally(candidate="mistral-small", wins=10, losses=0, ties=0, both_failed_ties=0)
        text = _effective_cost_per_success_text(tally, result)
        assert text == "$9.00/mo"


# ---------------------------------------------------------------------------
# Surface parity -- terminal / Markdown / HTML
# ---------------------------------------------------------------------------


def _capture_terminal(measure_result: MeasureResult, result: AnalysisResult | None) -> str:
    import frugon.report as report_mod

    buf = io.StringIO()
    console = Console(
        file=buf, width=140, no_color=True, force_terminal=True, highlight=False
    )
    original = report_mod.rprint

    def _patched(*args: Any, **kw: Any) -> None:
        console.print(*args, **kw)

    report_mod.rprint = _patched  # type: ignore[attr-defined]
    try:
        render_quality_terminal(measure_result, result=result)
    finally:
        report_mod.rprint = original  # type: ignore[attr-defined]
    return _ANSI_RE.sub("", buf.getvalue())


def _measure_result(candidate: str, tally: Tier1Tally) -> MeasureResult:
    return MeasureResult(
        samples_requested=10,
        samples_taken=10,
        current_model="gpt-4-turbo",
        candidates=[candidate],
        comparisons=[],
        tier1_tallies=[tally],
    )


class TestSurfaceParity:
    def test_terminal_omits_column_without_result(self) -> None:
        mr = _measure_result("gpt-4o-mini", _tally())
        out = _capture_terminal(mr, None)
        assert "Eff. $/success" not in out

    def test_terminal_shows_column_with_result(self) -> None:
        result, split = _split_result()
        mr = _measure_result(split.candidate_model, _tally(candidate=split.candidate_model))
        out = _capture_terminal(mr, result)
        assert "Eff. $/success" in out
        assert "$0.03" in out

    def test_markdown_omits_column_without_result(self) -> None:
        mr = _measure_result("gpt-4o-mini", _tally())
        lines = _quality_section_md(mr)
        md = "\n".join(lines)
        assert "Eff. $/success" not in md
        assert "| Candidate | Win | Loss | Tie | Error | Summary |" in md

    def test_markdown_shows_column_with_result(self) -> None:
        result, split = _split_result()
        mr = _measure_result(split.candidate_model, _tally(candidate=split.candidate_model))
        md = "\n".join(_quality_section_md(mr, result=result))
        assert "| Candidate | Win | Loss | Tie | Error | Summary | Eff. $/success |" in md
        assert "$0.03" in md

    def test_html_omits_column_without_result(self) -> None:
        mr = _measure_result("gpt-4o-mini", _tally())
        html = _quality_section_html(mr, style="v1")
        assert "Eff. $/success" not in html

    def test_html_shows_column_with_result(self) -> None:
        result, split = _split_result()
        mr = _measure_result(split.candidate_model, _tally(candidate=split.candidate_model))
        html = _quality_section_html(mr, style="v1", result=result)
        assert "<th>Eff. $/success</th>" in html
        assert "$0.03" in html

    def test_html_v2_shows_column_with_result(self) -> None:
        result, split = _split_result()
        mr = _measure_result(split.candidate_model, _tally(candidate=split.candidate_model))
        html = _quality_section_html(mr, style="v2", result=result)
        assert "<th>Eff. $/success</th>" in html
        assert "$0.03" in html

    def test_zero_success_renders_on_every_surface(self) -> None:
        result, split = _split_result()
        tally = _tally(
            candidate=split.candidate_model, wins=0, losses=5, ties=2, both_failed_ties=2
        )
        mr = _measure_result(split.candidate_model, tally)

        term = _capture_terminal(mr, result)
        md = "\n".join(_quality_section_md(mr, result=result))
        html = _quality_section_html(mr, style="v1", result=result)

        assert "n/a (0 judged successes)" in term
        assert "n/a (0 judged successes)" in md
        assert "n/a (0 judged successes)" in html


# ---------------------------------------------------------------------------
# Reconciliation -- the printed effective-$ never disagrees with the printed
# price and the tally's own win/tie/loss counts
# ---------------------------------------------------------------------------


class TestReconciliation:
    def test_printed_effective_cost_matches_price_and_success_rate(self) -> None:
        result, split = _split_result()
        tally = _tally(candidate=split.candidate_model)  # wins=6 losses=1 ties=3 bf=1
        text = _effective_cost_per_success_text(tally, result)
        assert text is not None

        # Independently recompute from the SAME publicly-visible tally fields
        # and the SAME price this run's split panel prints as "New:".
        price, _is_monthly = _candidate_shown_price(result, split.candidate_model)  # type: ignore[misc]
        success_count = tally.wins + tally.ties - tally.both_failed_ties
        verdict_count = tally.wins + tally.losses + tally.ties
        expected = _fmt_usd(price / (Decimal(success_count) / Decimal(verdict_count)))

        assert text == expected
        # And the price itself is exactly the $0.02 the split panel's "New:"
        # line would print via _fmt_usd -- never a different, freshly-derived
        # number.
        assert _fmt_usd(price) == "$0.02"

    def test_effective_cost_equals_price_at_full_confirmation(self) -> None:
        """When every scored comparison is a clean win, effective == price."""
        result, split = _split_result()
        tally = _tally(
            candidate=split.candidate_model, wins=10, losses=0, ties=0, both_failed_ties=0
        )
        text = _effective_cost_per_success_text(tally, result)
        assert text == "$0.02"

    def test_printed_summary_and_effective_cost_use_the_same_success_count(self) -> None:
        """C1: the Summary cell and the Eff. $/success cell on the SAME row
        must reconcile -- a READER parsing the printed row and recomputing
        price / (N/M) must land on the exact rendered effective-cost text.
        """
        result, split = _split_result()
        tally = _tally(candidate=split.candidate_model)  # wins=6 losses=1 ties=3 bf=1
        mr = _measure_result(split.candidate_model, tally)
        term = _capture_terminal(mr, result)

        summary_lines = [ln for ln in term.splitlines() if "equivalent or better" in ln]
        assert len(summary_lines) == 1
        line = summary_lines[0]
        assert "1 tie both failed" in line

        match = re.search(r"(\d+)/(\d+) equivalent or better", line)
        assert match is not None
        n, m = int(match.group(1)), int(match.group(2))

        dollar_matches = re.findall(r"\$[\d,]+\.\d+", line)
        assert dollar_matches, "expected an effective-cost figure on the same row"
        rendered_effective = dollar_matches[-1]

        priced = _candidate_shown_price(result, split.candidate_model)
        assert priced is not None
        price, _is_monthly = priced
        expected = _fmt_usd(_quantize_usd_for_display(price) / (Decimal(n) / Decimal(m)))
        assert rendered_effective == expected


# ---------------------------------------------------------------------------
# _judged_success_summary (W2 dependency + C1) -- single-sourced Summary cell
# ---------------------------------------------------------------------------


class TestJudgedSuccessSummary:
    def test_zero_verdicts_renders_em_dash(self) -> None:
        tally = Tier1Tally(candidate="x", wins=0, losses=0, ties=0, errors=5)
        assert _judged_success_summary(tally) == "—"

    def test_no_both_failed_ties_omits_parenthetical(self) -> None:
        tally = Tier1Tally(candidate="x", wins=6, losses=1, ties=3, both_failed_ties=0)
        assert _judged_success_summary(tally) == "9/10 equivalent or better"

    def test_one_both_failed_tie_singular_wording(self) -> None:
        tally = Tier1Tally(candidate="x", wins=6, losses=1, ties=3, both_failed_ties=1)
        assert (
            _judged_success_summary(tally)
            == "8/10 equivalent or better (1 tie both failed)"
        )

    def test_multiple_both_failed_ties_plural_wording(self) -> None:
        tally = Tier1Tally(candidate="x", wins=4, losses=1, ties=5, both_failed_ties=2)
        assert (
            _judged_success_summary(tally)
            == "7/10 equivalent or better (2 ties both failed)"
        )


# ---------------------------------------------------------------------------
# _quantize_usd_for_display (W1) -- single-sourced display-precision ladder
# ---------------------------------------------------------------------------


class TestQuantizeUsdForDisplay:
    def test_quantize_matches_fmt_usd_across_all_three_tiers(self) -> None:
        for amount in (Decimal("0.00003"), Decimal("0.005"), Decimal("389.884")):
            assert _fmt_usd(amount) == "$" + str(_quantize_usd_for_display(amount))

    def test_zero_quantizes_to_two_dp(self) -> None:
        assert _quantize_usd_for_display(Decimal("0")) == Decimal("0.00")


# ---------------------------------------------------------------------------
# W2 -- a candidate with NO verdicts at all (every comparison errored) must
# read differently from one with verdicts but zero judged successes
# ---------------------------------------------------------------------------


class TestZeroVerdictCountVsZeroSuccesses:
    def test_zero_verdict_count_renders_no_verdicts_not_zero_successes(self) -> None:
        result, split = _split_result()
        tally = _tally(
            candidate=split.candidate_model,
            wins=0,
            losses=0,
            ties=0,
            errors=10,
            both_failed_ties=0,
        )
        text = _effective_cost_per_success_text(tally, result)
        assert text == "n/a (no verdicts)"

    def test_verdicts_present_but_zero_successes_still_renders_zero_successes(
        self,
    ) -> None:
        result, split = _split_result()
        tally = _tally(
            candidate=split.candidate_model,
            wins=0,
            losses=5,
            ties=2,
            both_failed_ties=2,
        )
        text = _effective_cost_per_success_text(tally, result)
        assert text == "n/a (0 judged successes)"


# ---------------------------------------------------------------------------
# W4 -- check_errors marks the cell and footnotes the run
# ---------------------------------------------------------------------------


class TestCheckErrorMarker:
    def test_check_errors_appends_trailing_marker(self) -> None:
        result, split = _split_result()
        tally = _tally(candidate=split.candidate_model)
        tally.check_errors = 1
        text = _effective_cost_per_success_text(tally, result)
        assert text is not None
        assert text.endswith("~")

    def test_no_check_errors_omits_marker(self) -> None:
        result, split = _split_result()
        tally = _tally(candidate=split.candidate_model)
        text = _effective_cost_per_success_text(tally, result)
        assert text is not None
        assert not text.endswith("~")

    def test_footnote_none_when_no_check_errors(self) -> None:
        mr = _measure_result("gpt-4o-mini", _tally())
        assert _check_error_footnote_text(mr) is None

    def test_footnote_present_and_pluralised_when_check_errors_nonzero(self) -> None:
        tally = _tally()
        tally.check_errors = 2
        mr = _measure_result("gpt-4o-mini", tally)
        note = _check_error_footnote_text(mr)
        assert note is not None
        assert "2 judged comparisons could not complete a shared-failure check" in note

    def test_terminal_renders_check_error_footnote(self) -> None:
        result, split = _split_result()
        tally = _tally(candidate=split.candidate_model)
        tally.check_errors = 2
        out = _capture_terminal(_measure_result(split.candidate_model, tally), result)
        assert "could not complete" in out
        assert "blended spend" in out  # W5 footnote, same surface

    def test_markdown_renders_both_footnotes(self) -> None:
        result, split = _split_result()
        tally = _tally(candidate=split.candidate_model)
        tally.check_errors = 2
        md = "\n".join(_quality_section_md(_measure_result(split.candidate_model, tally), result=result))
        assert "> ~ 2 judged comparisons" in md
        assert "_Eff. $/success for the routed candidate uses blended spend" in md

    def test_html_renders_both_footnotes(self) -> None:
        result, split = _split_result()
        tally = _tally(candidate=split.candidate_model)
        tally.check_errors = 2
        html = _quality_section_html(_measure_result(split.candidate_model, tally), result=result)
        assert 'class="quality-check-error"' in html
        assert 'class="quality-split-price-note"' in html


# ---------------------------------------------------------------------------
# W5 -- the split path's Eff. $/success basis is footnoted, not restructured
# ---------------------------------------------------------------------------


class TestSplitPricedFootnote:
    def test_footnote_fires_for_the_split_routed_candidate(self) -> None:
        result, split = _split_result()
        mr = _measure_result(split.candidate_model, _tally(candidate=split.candidate_model))
        note = _split_priced_effective_cost_footnote_text(mr, result)
        assert note is not None
        assert "blended spend" in note

    def test_footnote_absent_without_a_split(self) -> None:
        result = _wholesale_result()
        mr = _measure_result("claude-haiku", _tally(candidate="claude-haiku"))
        assert _split_priced_effective_cost_footnote_text(mr, result) is None

    def test_footnote_absent_when_none_result(self) -> None:
        mr = _measure_result("gpt-4o-mini", _tally())
        assert _split_priced_effective_cost_footnote_text(mr, None) is None


# ---------------------------------------------------------------------------
# W6 -- one uniform price basis (monthly XOR observed) across the whole
# Eff. $/success column, never mixed row-to-row
# ---------------------------------------------------------------------------


class TestUniformPriceBasis:
    def test_monthly_basis_true_when_every_candidate_is_monthly(self) -> None:
        result = _multi_candidate_result()
        for proj in result.candidate_projections:
            if proj.status != "unpriced":
                proj.monthly_cost = Decimal("9.00")
        result.candidate_projections.append(
            CandidateProjection(
                model="mistral-small",
                status="considered",
                monthly_cost=Decimal("9.00"),
                observed_cost=Decimal("0.90"),
            )
        )
        mr = _measure_result(
            "mistral-small", _tally(candidate="mistral-small", wins=10, losses=0, ties=0)
        )
        assert _candidates_use_monthly_basis(mr, result) is True

    def test_monthly_basis_false_when_no_candidate_is_monthly(self) -> None:
        result = _multi_candidate_result()
        mr = _measure_result(
            "gpt-4o-mini", _tally(candidate="gpt-4o-mini", wins=10, losses=0, ties=0)
        )
        assert _candidates_use_monthly_basis(mr, result) is False

    def test_one_observed_only_candidate_forces_observed_basis_for_all(self) -> None:
        result = _multi_candidate_result()
        result.candidate_projections.append(
            CandidateProjection(
                model="mistral-small",
                status="considered",
                monthly_cost=Decimal("9.00"),
                observed_cost=Decimal("0.90"),
            )
        )
        mr = _measure_result("x", _tally(candidate="mistral-small"))
        mr.tier1_tallies = [
            _tally(candidate="mistral-small", wins=10, losses=0, ties=0, both_failed_ties=0),
            _tally(candidate="gpt-4o-mini", wins=10, losses=0, ties=0, both_failed_ties=0),
        ]
        assert _candidates_use_monthly_basis(mr, result) is False
        for tally in mr.tier1_tallies:
            text = _effective_cost_per_success_text(tally, result, prefer_monthly=False)
            assert text is not None
            assert "/mo" not in text
