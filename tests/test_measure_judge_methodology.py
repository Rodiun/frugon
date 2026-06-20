"""Tests for the judge methodology surfaced with a Tier-1 (--judge) sample.

Two trust-facing elements of render_quality_terminal's Tier-1 view:

  * a self-judge CAUTION — shown whenever the resolved judge IS one of the models
    it scored (judge == a candidate, or == the baseline), so the user is never
    misled into trusting a self-evaluated verdict.  Not gated on --verbose.
  * a judge-provenance CAPTION naming the judge and stating that the A/B order
    was randomised — "how it was measured".  It is rendered DIM as the caption
    directly under the "Quality sample — judge results …" title (above the
    tally), on EVERY run (verbose and non-verbose alike), so the cluster reads
    provenance → tally → verdict → caveat.  The prompt count is deliberately
    OMITTED here: the title already carries it ("… (N prompts, current: …)").

The caption format is ``Judge: <model><qualifier> · A/B order randomised`` where
the qualifier is "(your highest-tier model)" for a log-best auto-pick,
"(independent)" for an explicit external judge, or absent for an explicit judge
that is itself a compared model (the self-judge caution flags that bias).

All tests run fully offline — the MeasureResult is constructed directly, no
provider call is made.
"""

from __future__ import annotations

import io
import re
import sys

from rich.console import Console

from frugon.cost import LogRecord
from frugon.measure import (
    Comparison,
    MeasureResult,
    SampledOutput,
    Tier1Tally,
)
from frugon.report import render_quality_terminal

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


def _render(measure_result: MeasureResult, *, verbose: bool) -> str:
    """Render the quality section to a plain (ANSI-stripped) string."""
    report_mod = sys.modules[render_quality_terminal.__module__]
    buf = io.StringIO()
    console = Console(file=buf, width=200, no_color=True, highlight=False)
    original_rprint = report_mod.rprint
    original_render_console = report_mod._render_console
    report_mod.rprint = lambda *a, **k: console.print(*a, **k)  # type: ignore[attr-defined]
    report_mod._render_console = lambda: console  # type: ignore[attr-defined]
    try:
        render_quality_terminal(measure_result, verbose=verbose)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    return _ANSI_RE.sub("", buf.getvalue())


def _record(prompt: str) -> LogRecord:
    return LogRecord(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _result(
    *, judge_model: str, self_judged: list[str], judge_from_log: bool = False
) -> MeasureResult:
    comparisons = [
        Comparison(
            record=_record(f"prompt {i}"),
            current_output=SampledOutput("gpt-4o", f"baseline {i}"),
            candidate_outputs=[SampledOutput("gpt-4o-mini", f"candidate {i}")],
            verdicts=["win"],
        )
        for i in range(5)
    ]
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=4, losses=1, ties=0)],
        judge_model=judge_model,
        self_judged_models=self_judged,
        judge_from_log=judge_from_log,
    )


def test_provenance_caption_names_independent_judge_and_randomisation() -> None:
    """Arrange: independent judge (gpt-4o), verbose render.
    Assert: the dim provenance caption names the judge, marks it independent and
    states the A/B order was randomised — in the new ``Judge: …`` colon form.
    The prompt count is NOT repeated here (the title already carries it).
    """
    out = _render(_result(judge_model="gpt-4o", self_judged=[]), verbose=True)
    assert "Judge: gpt-4o (independent) · A/B order randomised" in out
    # The redundant per-line prompt count is gone (the title still carries it).
    assert "prompt(s)" not in out


def test_provenance_caption_shown_in_compact_view_too() -> None:
    """Assert: the provenance caption is a STANDING element of the cluster — it
    shows in the compact (non-verbose) view as well, since it is the caption
    under the title, not a verbose-only afterthought.  It sits ABOVE the tally.
    """
    out = _render(_result(judge_model="gpt-4o", self_judged=[]), verbose=False)
    assert "Judge: gpt-4o (independent) · A/B order randomised" in out
    # Caption is the caption: it precedes the tally table header row.
    assert out.index("Judge: gpt-4o") < out.index("Candidate")


def test_self_judge_caution_shown_even_without_verbose() -> None:
    """Arrange: judge == the candidate (self-evaluation).
    Assert: the caution fires in the COMPACT view (it is a trust signal, not a
    verbosity detail), names the offending model, and points at --judge-model.
    """
    out = _render(
        _result(judge_model="gpt-4o-mini", self_judged=["gpt-4o-mini"]),
        verbose=False,
    )
    assert "Caution" in out
    assert "grading a model it IS" in out
    assert "gpt-4o-mini" in out
    assert "--judge-model" in out


def test_self_judge_caution_absent_for_independent_judge() -> None:
    """Assert: an independent judge produces NO caution in either view."""
    for verbose in (False, True):
        out = _render(_result(judge_model="gpt-4o", self_judged=[]), verbose=verbose)
        assert "Caution" not in out
        assert "self-biased" not in out


def test_provenance_says_highest_tier_model_when_judge_from_log() -> None:
    """Arrange: the judge was auto-selected as the user's highest-tier LOG model
    and it happens to BE a compared model (judge_from_log=True, self-judged).
    Assert: the caption describes it as "your highest-tier model" and does NOT
    claim "(independent)" — the honest framing for a self-judge auto-pick.
    """
    out = _render(
        _result(
            judge_model="gpt-4o-mini",
            self_judged=["gpt-4o-mini"],
            judge_from_log=True,
        ),
        verbose=True,
    )
    assert "Judge: gpt-4o-mini (your highest-tier model)" in out
    assert "(independent)" not in out
    # The self-judge caution still fires (judge IS a compared model).
    assert "Caution" in out


def test_provenance_highest_tier_model_even_when_not_compared() -> None:
    """Arrange: judge auto-picked from the log (judge_from_log=True) but it is NOT
    one of the compared models (self_judged empty).
    Assert: it is still framed as "your highest-tier model" (where it came from),
    never "(independent)" — that label is reserved for an explicit external judge.
    """
    out = _render(
        _result(judge_model="gpt-4-turbo", self_judged=[], judge_from_log=True),
        verbose=True,
    )
    assert "Judge: gpt-4-turbo (your highest-tier model)" in out
    assert "(independent)" not in out
    assert "Caution" not in out


def test_provenance_independent_only_for_external_explicit_judge() -> None:
    """Assert: "(independent)" appears ONLY when the judge is neither a compared
    model NOR the log-best auto-pick — i.e. an explicit external --judge-model.
    """
    out = _render(
        _result(judge_model="gpt-4o", self_judged=[], judge_from_log=False),
        verbose=True,
    )
    assert "Judge: gpt-4o (independent)" in out
    assert "your highest-tier model" not in out


def test_tier0_has_no_judge_methodology() -> None:
    """Assert: a Tier-0 sample (no judge ran — judge_model=None) surfaces neither
    the caution nor the Note, even under --verbose.
    """
    comparisons = [
        Comparison(
            record=_record("prompt 0"),
            current_output=SampledOutput("gpt-4o", "baseline"),
            candidate_outputs=[SampledOutput("gpt-4o-mini", "candidate")],
            verdicts=[],
        )
    ]
    tier0 = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=None,
        judge_model=None,
        self_judged_models=[],
    )
    out = _render(tier0, verbose=True)
    assert "A/B order randomised" not in out
    assert "Caution" not in out
