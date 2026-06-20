"""Tests for the routing-plan share column and the judge-provenance caption.

Two report refinements:

  * ``_call_share_pcts`` — each routing bucket's share of total analyzed calls,
    one-decimal, summing to EXACTLY 100.0% via largest-remainder rounding (so the
    share column reconciles with the 100%-anchored Blended total row). The MD and
    HTML v2 routing-plan tables carry these figures; the terminal is unchanged.

  * the judge-provenance CAPTION — "Judge: <model><qualifier> · A/B order
    randomised", rendered DIM directly under the "Quality sample — judge results
    …" title (above the tally) on the terminal, in Markdown and in HTML. The
    prompt count is omitted (the title carries it). Shared verbatim across the
    three surfaces via ``_judge_provenance_text``.

All tests run fully offline — MeasureResults are constructed directly.
"""

from __future__ import annotations

import re

from frugon.measure import MeasureResult, Tier1Tally
from frugon.report import (
    _call_share_pcts,
    _judge_provenance_text,
    _quality_section_html,
    _quality_section_md,
)

# ---------------------------------------------------------------------------
# _call_share_pcts — largest-remainder rounding sums to exactly 100.0
# ---------------------------------------------------------------------------


def test_call_share_pcts_demo_buckets_sum_to_100() -> None:
    """The demo's 36100/10000/10000 split sums to exactly 100.0% at one decimal.

    Naive per-bucket rounding gives 64.3 + 17.8 + 17.8 = 99.9; largest-remainder
    hands the leftover tenth to the routed bucket so the figures reconcile.
    """
    shares = _call_share_pcts([36100, 10000, 10000])
    assert shares == [64.4, 17.8, 17.8]
    assert round(sum(shares), 1) == 100.0


def test_call_share_pcts_fixture_buckets_sum_to_100() -> None:
    """24/3/10 (the report-split fixture) reconciles to exactly 100.0%."""
    shares = _call_share_pcts([24, 3, 10])
    assert shares == [64.9, 8.1, 27.0]
    assert round(sum(shares), 1) == 100.0


def test_call_share_pcts_two_buckets_sum_to_100() -> None:
    """A two-bucket split (no already-optimal calls) still sums to 100.0%."""
    shares = _call_share_pcts([1, 2])
    assert round(sum(shares), 1) == 100.0
    assert len(shares) == 2


def test_call_share_pcts_zero_total_is_all_zero() -> None:
    """A zero total (degenerate fixture) yields all-zero shares, never a divide."""
    assert _call_share_pcts([0, 0]) == [0.0, 0.0]
    assert _call_share_pcts([]) == []


def test_call_share_pcts_each_within_a_tenth_of_true_share() -> None:
    """Every displayed share stays within 0.1 of its exact value (no big skew)."""
    counts = [7, 11, 13, 19]
    total = sum(counts)
    shares = _call_share_pcts(counts)
    assert round(sum(shares), 1) == 100.0
    for c, s in zip(counts, shares, strict=True):
        assert abs(s - c / total * 100) <= 0.1 + 1e-9


# ---------------------------------------------------------------------------
# _judge_provenance_text — caption wording + Tier-0 absence
# ---------------------------------------------------------------------------


def _tier1(judge_model: str | None, *, self_judged: list[str], from_log: bool) -> MeasureResult:
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=4, losses=1, ties=0)],
        judge_model=judge_model,
        self_judged_models=self_judged,
        judge_from_log=from_log,
    )


def test_provenance_text_independent_judge() -> None:
    txt = _judge_provenance_text(_tier1("gpt-4o", self_judged=[], from_log=False))
    assert txt == "Judge: gpt-4o (independent) · A/B order randomised"
    assert "prompt" not in txt  # the count belongs in the title, not here


def test_provenance_text_log_best_autopick() -> None:
    txt = _judge_provenance_text(_tier1("gpt-4o", self_judged=[], from_log=True))
    assert txt == "Judge: gpt-4o (your highest-tier model) · A/B order randomised"


def test_provenance_text_explicit_compared_judge_unqualified() -> None:
    txt = _judge_provenance_text(
        _tier1("gpt-4o-mini", self_judged=["gpt-4o-mini"], from_log=False)
    )
    assert txt == "Judge: gpt-4o-mini · A/B order randomised"


def test_provenance_text_none_for_tier0() -> None:
    tier0 = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=None,
        judge_model=None,
        self_judged_models=[],
    )
    assert _judge_provenance_text(tier0) is None


# ---------------------------------------------------------------------------
# Provenance caption placement in MD + HTML (under the title, above the tally)
# ---------------------------------------------------------------------------


def test_md_provenance_caption_under_title_above_tally() -> None:
    md = "\n".join(_quality_section_md(_tier1("gpt-4o", self_judged=[], from_log=False)))
    assert "_Judge: gpt-4o (independent) · A/B order randomised_" in md
    # Caption sits under the "Judge results …" title and ABOVE the tally header.
    assert md.index("Judge results") < md.index("Judge: gpt-4o") < md.index("| Candidate |")


def test_html_provenance_caption_under_title_above_tally() -> None:
    html = _quality_section_html(
        _tier1("gpt-4o", self_judged=[], from_log=False), style="v2"
    )
    assert (
        '<p class="quality-provenance">Judge: gpt-4o (independent) '
        "· A/B order randomised</p>" in html
    )
    # Caption precedes the tally table.
    assert html.index("quality-provenance") < html.index("quality-tally")


def test_provenance_css_ships_in_full_html_report() -> None:
    """The dim ``.quality-provenance`` rule ships in the rendered stylesheet.

    The section markup carries the class; the rule itself is injected at the
    stylesheet level (``_QUALITY_HTML_CSS``), so a full report render must contain
    both — proving the caption is actually styled dim, not just labelled.
    """
    import tempfile
    from pathlib import Path

    from frugon.cost import analyze_records, iter_records
    from frugon.report import render_html_v2

    sample = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "frugon"
        / "data"
        / "sample_logs.jsonl.gz"
    )
    records, skipped = iter_records(sample)
    result = analyze_records(records, skipped_malformed=skipped, split_routing=True)

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.html"
        render_html_v2(
            result,
            out,
            measure_result=_tier1("gpt-4o", self_judged=[], from_log=False),
        )
        html = out.read_text(encoding="utf-8")

    assert ".quality-provenance{" in html
    assert '<p class="quality-provenance">Judge: gpt-4o (independent)' in html


def test_md_routing_share_column_present_and_reconciles() -> None:
    """The MD routing plan carries a '% calls' column reconciling to 100.0%."""
    from pathlib import Path

    from frugon.cost import analyze_records, iter_records

    sample = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "frugon"
        / "data"
        / "sample_logs.jsonl.gz"
    )
    records, skipped = iter_records(sample)
    result = analyze_records(records, skipped_malformed=skipped, split_routing=True)

    import tempfile

    from frugon.report import render_markdown

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "r.md"
        render_markdown(result, out)
        md = out.read_text(encoding="utf-8")

    assert "| Bucket | Calls | % calls | Route to | Cost |" in md
    # The Blended total row anchors the share column at 100.0% (bold).
    assert "| **Blended** | **56,100** | **100.0%** | — |" in md
    # The three non-blended bucket shares reconcile to 100.0%.
    routing = md[md.index("## Routing plan") : md.index("## Cost by model")]
    bucket_shares = [float(s) for s in re.findall(r"\| ([\d.]+)% \|", routing)]
    assert round(sum(bucket_shares), 1) == 100.0, bucket_shares
