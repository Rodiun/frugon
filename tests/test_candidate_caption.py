"""Tests for the conditional "Candidates considered" caption (Fix 1).

The caption sits under the multi-candidate block on every surface (terminal,
Markdown, HTML v1/v2).  Its trailing clause is CONDITIONAL on whether a
per-candidate judge tally (the Tier-1 quality measurement) actually renders
BELOW the block in the same output:

  * judge section follows  -> "...scored independently in the quality
    measurement below."  (the "below" reference is accurate)
  * no judge section       -> "Run --measure --judge to score each candidate's
    quality."  (actionable; no dangling "below")

The bug this guards: a cost-only report (``frugon analyze --candidates a,b``
with no ``--measure``) has NO judge section, so the old unconditional caption's
"...judged independently below." pointed at a section that did not exist.

All tests run fully offline — AnalysisResults come from ``analyze_logs`` over a
synthetic log; the judge presence is injected via a directly-constructed
Tier-1/Tier-0 MeasureResult, never a provider call.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from frugon.cost import AnalysisResult, LogRecord, analyze_logs
from frugon.measure import Comparison, MeasureResult, SampledOutput, Tier1Tally
from frugon.report import (
    _candidate_caption,
    _candidates_considered_html,
    _candidates_considered_md_lines,
    _render_candidates_considered_terminal,
)

# The two halves of the conditional caption — asserted as substrings so the
# tests pin the BEHAVIOUR (which clause renders) without being brittle about
# the invariant first sentence's exact whitespace.
_JUDGE_CLAUSE = "scored independently in the quality measurement below"
_NO_JUDGE_CLAUSE = "Run --measure --judge to score each candidate"
_BASE_SENTENCE = "the biggest saving is the headline recommendation"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _multi_candidate_result(tmp_path: Path) -> AnalysisResult:
    """An AnalysisResult with >1 candidate_projections (so the block renders)."""
    records = [
        {
            "model": "gpt-4-turbo",
            "request": {
                "messages": [{"role": "user", "content": "classify this ticket"}]
            },
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "billing"}}]
            },
            "usage": {"prompt_tokens": 200, "completion_tokens": 5},
        }
        for _ in range(100)
    ]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = analyze_logs(log, candidates=["gpt-4o-mini", "claude-haiku-4-5"])
    assert len(result.candidate_projections) >= 2
    return result


def _record(prompt: str) -> LogRecord:
    return LogRecord(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _tier1_measure_result() -> MeasureResult:
    """Tier-1 (--judge ran): a judge tally section WILL render below the block."""
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[
            Comparison(
                record=_record("classify this ticket"),
                current_output=SampledOutput(model="gpt-4-turbo", content="a"),
                candidate_outputs=[SampledOutput(model="gpt-4o-mini", content="b")],
                verdicts=["win"],
            )
        ],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=1, losses=0, ties=0)],
    )


def _tier0_measure_result() -> MeasureResult:
    """Tier-0 (--measure, no --judge): no per-candidate tally section follows."""
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[
            Comparison(
                record=_record("classify this ticket"),
                current_output=SampledOutput(model="gpt-4-turbo", content="a"),
                candidate_outputs=[SampledOutput(model="gpt-4o-mini", content="b")],
            )
        ],
        tier1_tallies=None,
    )


def _render_terminal_candidates(result: AnalysisResult, *, has_judge: bool) -> str:
    """Capture the terminal candidates block (incl. caption) as plain text."""
    console = Console(record=True, width=100, force_terminal=False, no_color=True)
    import frugon.report as report_mod

    original = report_mod.rprint
    report_mod.rprint = console.print  # type: ignore[assignment]
    try:
        _render_candidates_considered_terminal(result, has_judge_section=has_judge)
    finally:
        report_mod.rprint = original  # type: ignore[assignment]
    # Collapse Rich's soft-wrap whitespace so substring assertions are not
    # broken by the console folding the caption across lines at this width.
    return " ".join(console.export_text().split())


# ---------------------------------------------------------------------------
# _candidate_caption — pure function (both branches)
# ---------------------------------------------------------------------------


def test_candidate_caption_no_judge_offers_command_not_below() -> None:
    caption = _candidate_caption(False)
    assert _BASE_SENTENCE in caption
    assert _NO_JUDGE_CLAUSE in caption
    # The whole point: a cost-only caption must NOT dangle a "below" reference.
    assert "below" not in caption


def test_candidate_caption_with_judge_references_section_below() -> None:
    caption = _candidate_caption(True)
    assert _BASE_SENTENCE in caption
    assert _JUDGE_CLAUSE in caption
    assert caption.rstrip().endswith("quality measurement below.")


# ---------------------------------------------------------------------------
# Terminal surface
# ---------------------------------------------------------------------------


def test_render_terminal_caption_no_judge(tmp_path: Path) -> None:
    result = _multi_candidate_result(tmp_path)
    out = _render_terminal_candidates(result, has_judge=False)
    assert _NO_JUDGE_CLAUSE in out
    assert _JUDGE_CLAUSE not in out


def test_render_terminal_caption_with_judge(tmp_path: Path) -> None:
    result = _multi_candidate_result(tmp_path)
    out = _render_terminal_candidates(result, has_judge=True)
    assert _JUDGE_CLAUSE in out
    assert _NO_JUDGE_CLAUSE not in out


# ---------------------------------------------------------------------------
# Markdown surface
# ---------------------------------------------------------------------------


def test_md_caption_no_judge(tmp_path: Path) -> None:
    result = _multi_candidate_result(tmp_path)
    md = "\n".join(
        _candidates_considered_md_lines(result, has_judge_section=False)
    )
    assert _NO_JUDGE_CLAUSE in md
    assert _JUDGE_CLAUSE not in md


def test_md_caption_with_judge(tmp_path: Path) -> None:
    result = _multi_candidate_result(tmp_path)
    md = "\n".join(
        _candidates_considered_md_lines(result, has_judge_section=True)
    )
    assert _JUDGE_CLAUSE in md
    assert _NO_JUDGE_CLAUSE not in md


# ---------------------------------------------------------------------------
# HTML surface (shared inner table for v1 + v2)
# ---------------------------------------------------------------------------


def test_html_caption_no_judge(tmp_path: Path) -> None:
    result = _multi_candidate_result(tmp_path)
    html = _candidates_considered_html(result, str, has_judge_section=False)
    assert _NO_JUDGE_CLAUSE in html
    assert _JUDGE_CLAUSE not in html


def test_html_caption_with_judge(tmp_path: Path) -> None:
    result = _multi_candidate_result(tmp_path)
    html = _candidates_considered_html(result, str, has_judge_section=True)
    assert _JUDGE_CLAUSE in html
    assert _NO_JUDGE_CLAUSE not in html


# ---------------------------------------------------------------------------
# End-to-end: the flag is derived from measure_result (_is_tier1) in the report
# renderers, so a Tier-0 measure_result still yields the no-judge caption while
# a Tier-1 one yields the below-reference caption.  This guards the wiring, not
# just the helper.
# ---------------------------------------------------------------------------


def test_md_report_tier0_uses_no_judge_caption(tmp_path: Path) -> None:
    from frugon.report import render_markdown

    result = _multi_candidate_result(tmp_path)
    out = tmp_path / "r.md"
    render_markdown(result, out, measure_result=_tier0_measure_result())
    md = out.read_text(encoding="utf-8")
    assert _NO_JUDGE_CLAUSE in md
    assert _JUDGE_CLAUSE not in md.split("## Quality measurement")[0]


def test_md_report_tier1_uses_below_caption(tmp_path: Path) -> None:
    from frugon.report import render_markdown

    result = _multi_candidate_result(tmp_path)
    out = tmp_path / "r.md"
    render_markdown(result, out, measure_result=_tier1_measure_result())
    md = out.read_text(encoding="utf-8")
    # The candidates caption (above the quality section) references "below".
    candidates_region = md.split("## Quality measurement")[0]
    assert _JUDGE_CLAUSE in candidates_region


def test_html_report_tier0_uses_no_judge_caption(tmp_path: Path) -> None:
    from frugon.report import render_html

    result = _multi_candidate_result(tmp_path)
    out = tmp_path / "r.html"
    render_html(result, out, measure_result=_tier0_measure_result())
    html = out.read_text(encoding="utf-8")
    assert _NO_JUDGE_CLAUSE in html


def test_html_report_tier1_uses_below_caption(tmp_path: Path) -> None:
    from frugon.report import render_html

    result = _multi_candidate_result(tmp_path)
    out = tmp_path / "r.html"
    render_html(result, out, measure_result=_tier1_measure_result())
    html = out.read_text(encoding="utf-8")
    assert _JUDGE_CLAUSE in html
