"""Measure-aware report copy (Items C, D, E) + no-measure byte-identity guard.

Once --measure / --judge has run, the report CONTAINS quality results, so the
pre-measure copy becomes contradictory or untrue.  These tests pin the honest
reframing across all four report variants and prove the measure-conditional
paths never touch the no-measure output (the no-measure path stays the canonical
"fully local, zero LLM calls" surface it has always been).

  * Item C — the "Quality is not verified - run --measure ..." caveat is dropped
    on the Tier-1 (judge) path and softened to a "run --judge" nudge on Tier-0.
  * Item D — the "No LLM calls. No network. No data leaves your machine."
    absolute and the "0 LLM calls" methodology tail are reframed honestly once a
    measurement ran (it DID call the user's own provider), while the
    "never to Frugon" guarantee is preserved in both cases.
  * Item E — in HTML v2 the synced dates ride their own meta line, separate from
    the calls-priced count.

All tests run fully offline - the MeasureResult is constructed directly.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import pytest

from frugon.cost import AnalysisResult, LogRecord
from frugon.measure import Comparison, MeasureResult, SampledOutput, Tier1Tally
from frugon.report import (
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
)
from frugon.routing import SplitRouting

_RENDERERS = {
    "md_v1": render_markdown,
    "md_v2": render_markdown_v2,
    "html_v1": render_html,
    "html_v2": render_html_v2,
}


def _split() -> SplitRouting:
    # Production-magnitude monthly figures: the routing-plan cost figures that overran
    # the v2 COST column ($496.7250 / $501.6202) before Item F.
    return SplitRouting(
        baseline_model="gpt-4-turbo",
        candidate_model="gpt-4o-mini",
        routed_count=24,
        kept_count=3,
        routed_cost=Decimal("12.3400"),
        kept_cost=Decimal("489.2802"),
        baseline_cost=Decimal("750.0000"),
        blended_cost=Decimal("501.6202"),
        easy_threshold=Decimal("0.35"),
        monthly_baseline=Decimal("750.0000"),
        monthly_blended=Decimal("501.6202"),
    )


def _result() -> AnalysisResult:
    return AnalysisResult(
        total_calls=37,
        priced_calls=37,
        unpriced_calls=0,
        total_cost=Decimal("496.7250"),
        cost_by_model={
            "gpt-4-turbo": Decimal("489.2802"),
            "gpt-4o-mini": Decimal("7.4448"),
        },
        calls_by_model={"gpt-4-turbo": 27, "gpt-4o-mini": 10},
        projected_cost=Decimal("330.0000"),
        candidate_model="gpt-4o-mini",
        observed_span_days=7.0,
        pricing_json_last_synced="2026-06-01",
        quality_json_last_synced="2026-05-15",
        split=_split(),
    )


def _record(prompt: str) -> LogRecord:
    return LogRecord(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _tier1() -> MeasureResult:
    """Tier-1 judge run: 0W / 2L / 3T over five sampled prompts."""
    verdicts = ["tie", "loss", "tie", "loss", "tie"]
    comparisons = [
        Comparison(
            record=_record(f"prompt {i}"),
            current_output=SampledOutput("gpt-4-turbo", "base"),
            candidate_outputs=[SampledOutput("gpt-4o-mini", f"cand {i}")],
            verdicts=[verdicts[i]],
        )
        for i in range(5)
    ]
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=2, ties=3)],
    )


def _tier0() -> MeasureResult:
    """Tier-0 (--measure, no --judge): two raw side-by-side samples, no verdict."""
    comparisons = [
        Comparison(
            record=_record("Name three primary colours."),
            current_output=SampledOutput("gpt-4-turbo", "Red, green, blue."),
            candidate_outputs=[SampledOutput("gpt-4o-mini", "Red, blue, yellow.")],
        ),
        Comparison(
            record=_record("Capital of France?"),
            current_output=SampledOutput("gpt-4-turbo", "Paris."),
            candidate_outputs=[SampledOutput("gpt-4o-mini", "Paris, France.")],
        ),
    ]
    return MeasureResult(
        samples_requested=2,
        samples_taken=2,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=None,
    )


def _render(fnname: str, mr: MeasureResult | None, tmp_path: Path) -> str:
    out = tmp_path / "r"
    _RENDERERS[fnname](_result(), out, measure_result=mr)
    return out.read_text(encoding="utf-8")


def _body(text: str) -> str:
    """Strip the inlined ``<style>`` block so assertions see only the rendered
    body, not stylesheet code comments (which legitimately quote the caveat copy
    when documenting the overflow-wrap CSS)."""
    return re.sub(r"<style>.*?</style>", "", text, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Item C - the "Quality is not verified - run --measure" caveat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(_RENDERERS))
def test_tier1_drops_quality_not_verified_caveat(variant: str, tmp_path: Path) -> None:
    # Act — assert against the rendered BODY (stylesheet code comments may quote
    # the caveat copy when documenting the overflow-wrap CSS; only the visible
    # body matters).
    text = _body(_render(variant, _tier1(), tmp_path))

    # Assert - the contradictory "Quality is not verified ... run --measure"
    # caveat sentence is gone (the report now carries a scored verdict).
    assert "Quality is not verified" not in text
    assert "run --measure to sample real outputs" not in text
    assert "run --measure to confirm it" not in text
    # The verdict the report DOES carry stands in for it.  Fix B emphasises just
    # the status word in place (bold in MD, a tally-class span in HTML).
    assert "Estimate " in text
    assert "borderline" in text
    assert "**borderline**" in text or '"verdict-tie">borderline' in text


@pytest.mark.parametrize("variant", list(_RENDERERS))
def test_tier0_softens_caveat_to_run_judge(variant: str, tmp_path: Path) -> None:
    # Act
    text = _render(variant, _tier0(), tmp_path)

    # Assert - no "not verified" claim; a measure-aware "--judge" nudge instead.
    assert "Quality is not verified" not in text
    judge_nudge = (
        "run <code>--judge</code> for a scored verdict"
        if variant.startswith("html")
        else "run --judge for a scored verdict"
    )
    assert judge_nudge in text


@pytest.mark.parametrize("variant", list(_RENDERERS))
def test_no_measure_keeps_quality_not_verified_caveat(
    variant: str, tmp_path: Path
) -> None:
    # The no-measure path is unchanged: the caveat still shows (today's wording).
    text = _render(variant, None, tmp_path)
    assert "Quality is not verified" in text


# ---------------------------------------------------------------------------
# Item D - measure-aware privacy / methodology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(_RENDERERS))
def test_measure_reframes_privacy_honestly(variant: str, tmp_path: Path) -> None:
    # Act
    text = _render(variant, _tier1(), tmp_path)

    # Assert - the unconditional "no data leaves your machine" absolute is gone
    # (a measurement DID send prompts to the user's provider); the honest split
    # owns the provider calls while keeping the "never to Frugon" guarantee.
    assert "No LLM calls. No network. No data leaves your machine." not in text
    assert "never to Frugon" in text
    assert "your own provider" in text
    # The sampled-prompt count is the real one (5).
    assert "5 prompts sent to your own provider" in text
    # The "0 LLM calls made for this analysis" methodology tail (v2 + Markdown
    # footers; html v1 only carries the methodology note) is reframed too.
    if variant != "html_v1":
        assert "0 LLM calls made for this analysis" not in text
        assert "0 LLM calls for the cost analysis" in text


@pytest.mark.parametrize("variant", list(_RENDERERS))
def test_no_measure_keeps_absolute_privacy(variant: str, tmp_path: Path) -> None:
    # The no-measure path keeps the strong unconditional privacy line.
    text = _render(variant, None, tmp_path)
    assert "No LLM calls. No network. No data leaves your machine." in text
    # The "0 LLM calls made for this analysis" tail rides the v2 + Markdown
    # footers; html v1 carries only the methodology note (which contains the
    # absolute privacy line asserted above).
    if variant != "html_v1":
        assert "0 LLM calls made for this analysis" in text


def test_tier0_privacy_uses_singular_prompt_count(tmp_path: Path) -> None:
    # One sampled prompt -> singular "1 prompt sent..." (count comes from the run).
    one = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[
            Comparison(
                record=_record("hi"),
                current_output=SampledOutput("gpt-4-turbo", "a"),
                candidate_outputs=[SampledOutput("gpt-4o-mini", "b")],
            )
        ],
        tier1_tallies=None,
    )
    text = _render("md_v1", one, tmp_path)
    assert "1 prompt sent to your own provider" in text


# ---------------------------------------------------------------------------
# Item E - HTML v2 synced dates on their own meta line
# ---------------------------------------------------------------------------


def test_html_v2_synced_dates_on_separate_meta_line(tmp_path: Path) -> None:
    # Act - no measure run (Item E is a layout fix, independent of measure).
    html = _render("html_v2", None, tmp_path)

    # Assert - the calls-priced count and the synced dates are SEPARATE
    # <p class="meta-line"> paragraphs (so a large call count cannot push the
    # synced dates past the container edge).
    metas = re.findall(r'<p class="meta-line">(.*?)</p>', html)
    calls_line = next(m for m in metas if "calls priced" in m)
    synced_line = next(m for m in metas if "pricing synced" in m)
    assert calls_line != synced_line
    assert "pricing synced" not in calls_line
    assert "quality synced" not in calls_line
    assert "pricing synced 2026-06-01" in synced_line
    assert "quality synced 2026-05-15" in synced_line


# ---------------------------------------------------------------------------
# Item F - HTML v2 routing-plan cost figures stay short (no overflow)
# ---------------------------------------------------------------------------


def test_html_v2_routing_plan_cost_renders_2dp(tmp_path: Path) -> None:
    # Act
    html = _render("html_v2", None, tmp_path)

    # Assert - the routing-plan COST cells render the standard 2dp _fmt_usd
    # figures (identical to the Cost-by-model table), NOT the whole-dollar
    # rounding that the overflow fix had shipped, contained inside a widened
    # COST column.
    #
    # The routing plan reconciles to the FULL analyzed dataset: Routed ($12.34)
    # + Kept ($489.28) + the already-on-a-cheaper bucket ($7.44, the 10
    # gpt-4o-mini calls) + the Blended TOTAL ($248.35 = total-after-routing).
    # Since v0.1.3, costs ≥ $0.01 display at 2 dp.
    costs = re.findall(r'c-cost">([^<]+)</td>', html)
    assert costs == ["$12.34", "$489.28", "$7.44", "$248.35"], costs
    # The CSS keeps the table inside the container at desktop width: AUTO layout
    # (no MODEL/STATUS dead band) under a max-width:100% clamp; the COST cell is
    # held to one line so the 2dp figure never wraps mid-number.
    assert ".tbl-plan{table-layout:auto;width:100%;max-width:100%;border-collapse:collapse}" in html
    assert ".tbl-plan .c-cost{white-space:nowrap}" in html


# ---------------------------------------------------------------------------
# No-measure byte-identity - the measure-conditional copy must not touch the
# no-measure output of the variants Items E/F do NOT restyle (md v1/v2, html v1).
# (html v2 intentionally changes under Items E/F - covered by the tests above.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["md_v1", "md_v2", "html_v1"])
def test_no_measure_unaffected_by_measure_conditional_copy(
    variant: str, tmp_path: Path
) -> None:
    # The measure-aware reframing is gated on measure_result; with None the
    # output carries the pre-measure copy verbatim - no honest-split sentence,
    # no "--judge" nudge leaks in.
    text = _render(variant, None, tmp_path)
    assert "never to Frugon" not in text
    assert "your own provider" not in text
    assert "run --judge for a scored verdict" not in text
