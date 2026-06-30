"""Tests for the --measure → estimate synthesis lines (GAP 2).

render_quality_terminal must tie the measured result back to the offline
'within tolerance' estimate so the two are never left disconnected:

  * Tier-1 (--judge): a per-candidate verdict — "confirmed" when the candidate
    held quality (wins + ties dominate), "NOT confirmed" when it lost
    materially — naming the same candidate that was measured and reconciling
    with the scored-sample count.
  * Tier-0 (--measure, no --judge): a one-line framing that the outputs are raw
    and unscored, and that 'within tolerance' remains an offline estimate until
    --judge runs.

Colour law: green is the money saving ONLY.  "confirmed" reads in cyan,
"not confirmed" in amber; these assertions check the words, the synthesis
implementation owns the styling.

All tests run fully offline — the MeasureResult is constructed directly, no
provider call is made.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from rich.console import Console

import frugon.cost as cost
from frugon.cost import LogRecord
from frugon.measure import (
    Comparison,
    MeasureResult,
    SampledOutput,
    Tier1Tally,
)
from frugon.report import render_quality_terminal

sys.path.insert(0, str(Path(__file__).parent))
from conftest import install_synthetic_quality

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


def _capture(measure_result: MeasureResult) -> str:
    """Render *measure_result* to a wide, ANSI-stripped string for assertions."""
    console = Console(width=200, force_terminal=False, no_color=True)
    with console.capture() as cap:
        # Resolve the EXACT module object render_quality_terminal lives in, then
        # patch AND call through it — robust even if a prior test reloaded
        # frugon.report via sys.modules (a plain ``import frugon.report`` would
        # bind a different instance than the import-time render_quality_terminal,
        # so the patch and the call could land on different module objects).
        report = sys.modules[render_quality_terminal.__module__]

        original = report.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            console.print(*args, **kwargs)

        report.rprint = _rp  # type: ignore[assignment]
        try:
            report.render_quality_terminal(measure_result)
        finally:
            report.rprint = original  # type: ignore[assignment]
    return _ANSI_RE.sub("", cap.get())


# Tier → ANSI SGR foreground code, for asserting the verdict colour band.
# Rich renders the "cyan" / "yellow" styles as these codes; checking them lets a
# test assert the band (cyan = confirmed, yellow/amber = borderline + not-confirmed)
# without coupling to the exact wording.
_CYAN_SGR = "\x1b[36m"
_AMBER_SGR = "\x1b[33m"


def _capture_coloured(measure_result: MeasureResult) -> str:
    """Render *measure_result* to a colour-preserving string for band assertions."""
    console = Console(width=200, force_terminal=True, color_system="standard")
    with console.capture() as cap:
        # See _capture: resolve + patch + call through the one live module
        # instance so a prior frugon.report reload can't split patch from call.
        report = sys.modules[render_quality_terminal.__module__]

        original = report.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            console.print(*args, **kwargs)

        report.rprint = _rp  # type: ignore[assignment]
        try:
            report.render_quality_terminal(measure_result)
        finally:
            report.rprint = original  # type: ignore[assignment]
    return cap.get()


def _record(prompt: str = "Classify this ticket") -> LogRecord:
    return LogRecord(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


# ---------------------------------------------------------------------------
# Tier-1 — confirmed (candidate holds quality)
# ---------------------------------------------------------------------------


def test_tier1_synthesis_confirmed_when_candidate_holds() -> None:
    # Arrange — wins + ties dominate; zero losses.
    result = MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=2, losses=0, ties=3)],
    )

    # Act
    out = _capture(result)

    # Assert — confirmed verdict, names the measured candidate, reconciles 5/5.
    assert "confirmed" in out, out
    assert "NOT confirmed" not in out, out
    assert "borderline" not in out, out  # 0 losses → clean confirm, not borderline
    assert "gpt-4o-mini" in out, out
    assert "5/5" in out, out  # held 5 of 5 scored

    # A clean confirm reads in the cyan band (green is reserved for the saving).
    coloured = _capture_coloured(result)
    assert _CYAN_SGR in coloured, coloured


# ---------------------------------------------------------------------------
# Tier-1 — NOT confirmed (candidate loses materially)
# ---------------------------------------------------------------------------


def test_tier1_synthesis_not_confirmed_when_candidate_loses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange — losses dominate the scored prompts. This test exercises the
    # ESCALATION branch (a cheaper higher-tier "next rung up" exists), so PIN
    # BOTH the quality tiers AND the routing pool to a fixed synthetic universe:
    #
    #   * Tiers: gpt-4o=0 (Elite), gpt-4o-mini=2 (Capable), gpt-4-turbo=3 (Efficient)
    #   * Pool: ["gpt-4o"] — the only member that out-tiers the failed candidate
    #
    # Pinning the pool is the key drift-proofing step: next_rung_up reads
    # cost._ROUTING_CANDIDATES when no explicit pool is passed.  The live list
    # changes with each curation pass, which would break this test a third time.
    # With pool=["gpt-4o"] wired in via monkeypatch, the escalation resolution
    # is immune to future _ROUTING_CANDIDATES re-curation.
    #
    # gpt-4o is priced in the bundled seed at $2.5/1M input + $10/1M output
    # (blended ~$6.25/1M), which is strictly cheaper than gpt-4-turbo's
    # $10/1M + $30/1M (blended $20/1M) — so the escalation rung is valid.
    install_synthetic_quality(
        monkeypatch,
        tmp_path,
        {"gpt-4o": 0, "gpt-4o-mini": 2, "gpt-4-turbo": 3},
    )
    monkeypatch.setattr(cost, "_ROUTING_CANDIDATES", ["gpt-4o"])
    result = MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=1, losses=3, ties=1)],
    )

    # Act
    out = _capture(result)

    # Assert — caution verdict, names the failed candidate, the baseline, and the
    # actionable next rung up with a ready command.
    assert "NOT confirmed" in out, out
    assert "gpt-4o-mini" in out, out  # the failed candidate
    assert "3/5" in out, out  # worse in 3 of 5 scored
    assert "gpt-4-turbo" in out, out  # "cheaper than gpt-4-turbo"
    assert "try the next rung up" in out, out
    assert "frugon analyze --measure --candidates gpt-4o" in out, out

    # Not-confirmed reads in the amber caution band.
    coloured = _capture_coloured(result)
    assert _AMBER_SGR in coloured, coloured


# ---------------------------------------------------------------------------
# Tier-1 — borderline (held on balance, but a non-trivial share of losses)
# ---------------------------------------------------------------------------


def test_tier1_synthesis_borderline_when_held_but_losses_nontrivial() -> None:
    # Arrange — the reported case: 0 wins, 2 losses, 3 ties. held=3 > losses=2,
    # but losses/scored = 2/5 = 0.40 >= the 0.20 borderline fraction → a 3-2-ish
    # result must NOT read like a clean confirm.
    result = MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=2, ties=3)],
    )

    # Act
    out = _capture(result)

    # Assert — borderline band, NOT a flat confirm; held/losses reconcile to 5.
    assert "borderline" in out, out
    assert "confirmed" not in out, out  # neither "confirmed" nor "NOT confirmed"
    assert "gpt-4o-mini" in out, out
    assert "3/5" in out, out  # held 3 of 5
    assert "2/5" in out, out  # worse in 2 of 5

    # Borderline reads in the amber caution band.
    coloured = _capture_coloured(result)
    assert _AMBER_SGR in coloured, coloured


def test_tier1_synthesis_one_in_ten_loss_still_confirms() -> None:
    # Arrange — 9 ties, 1 loss: held=9 > losses=1, losses/scored = 0.10 < 0.20 →
    # a single stray loss in ten still reads as a clean confirm (the constant's
    # documented intent).
    result = MeasureResult(
        samples_requested=10,
        samples_taken=10,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=1, ties=9)],
    )

    # Act
    out = _capture(result)

    # Assert — clean confirm, not borderline.
    assert "confirmed" in out, out
    assert "NOT confirmed" not in out, out
    assert "borderline" not in out, out
    assert "9/10" in out, out  # held 9 of 10 scored


def test_tier1_synthesis_all_errors_states_unverified() -> None:
    # Arrange — every comparison errored: cannot confirm or refute.
    result = MeasureResult(
        samples_requested=3,
        samples_taken=3,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=0, ties=0, errors=3)],
    )

    # Act
    out = _capture(result)

    # Assert — no false verdict; explicitly "not verified".
    assert "not verified" in out, out
    assert "confirmed" not in out, out  # neither "confirmed" nor "NOT confirmed"


# ---------------------------------------------------------------------------
# Tier-1 — the BASELINE failed to sample on every prompt (not the candidate)
# ---------------------------------------------------------------------------


def _capture_verbose(measure_result: MeasureResult) -> str:
    """Render *measure_result* (verbose) to an ANSI-stripped string."""
    console = Console(width=200, force_terminal=False, no_color=True)
    with console.capture() as cap:
        report = sys.modules[render_quality_terminal.__module__]
        original = report.rprint

        def _rp(*args: object, **kwargs: object) -> None:
            console.print(*args, **kwargs)

        report.rprint = _rp  # type: ignore[assignment]
        try:
            report.render_quality_terminal(measure_result, verbose=True)
        finally:
            report.rprint = original  # type: ignore[assignment]
    return _ANSI_RE.sub("", cap.get())


def _baseline_failed_result() -> MeasureResult:
    """Every comparison errored because the CURRENT/baseline model failed to sample.

    The candidate's OWN output succeeded on every prompt; only the baseline call
    carries an error, so the comparison was impossible (nothing to compare
    against) — the judge tally counts neutral ``error`` verdicts.
    """
    comparisons = [
        Comparison(
            record=_record(f"prompt {i}"),
            current_output=SampledOutput(
                "gpt-4-turbo", "", error="[rate limited — 429]"
            ),
            candidate_outputs=[SampledOutput("gpt-4o-mini", f"candidate answer {i}")],
            verdicts=["error"],
        )
        for i in range(3)
    ]
    return MeasureResult(
        samples_requested=3,
        samples_taken=3,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=[
            Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=0, ties=0, errors=3)
        ],
    )


def test_tier1_synthesis_baseline_failed_blames_current_model() -> None:
    """Baseline errored on every prompt → "current model failed to sample", NOT
    "every <candidate> comparison errored" (which would blame the candidate)."""
    out = _capture(_baseline_failed_result())

    # The honest message names the CURRENT model as the thing that failed.
    assert "Could not verify" in out, out
    assert "your current model (gpt-4-turbo) failed to sample" in out, out
    # It must NOT wrongly imply the candidate failed.
    assert "every gpt-4o-mini comparison errored" not in out, out


def test_tier1_baseline_failed_per_prompt_reads_no_comparison_not_error() -> None:
    """In the verbose per-prompt detail, a candidate whose own output succeeded
    but couldn't be judged (baseline failed) reads [no comparison], not [error]."""
    out = _capture_verbose(_baseline_failed_result())

    assert "[no comparison]" in out, out
    # The candidate's own answer is present (it succeeded) — it isn't a failure.
    assert "candidate answer 0" in out, out
    # And the bare red [error] must NOT label this working candidate.
    assert "gpt-4o-mini [error]" not in out, out


def test_tier1_candidate_genuinely_errored_still_reads_error() -> None:
    """When the CANDIDATE itself errored (baseline fine), the per-prompt label
    stays [error] — only the baseline-failed case becomes [no comparison]."""
    comparisons = [
        Comparison(
            record=_record("prompt"),
            current_output=SampledOutput("gpt-4-turbo", "baseline answer"),
            candidate_outputs=[
                SampledOutput("gpt-4o-mini", "", error="[unavailable — no access]")
            ],
            verdicts=["error"],
        )
    ]
    result = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=[
            Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=0, ties=0, errors=1)
        ],
    )
    out = _capture_verbose(result)
    assert "[error]" in out, out
    assert "[no comparison]" not in out, out
    # The baseline succeeded here, so the synthesis is the candidate-errored
    # "not verified" wording — NOT the baseline-failed message.
    assert "Could not verify" not in out, out


def test_baseline_failed_message_renders_in_markdown_report() -> None:
    """The baseline-failed synthesis + [no comparison] label reach the report too.

    The Markdown quality section shares the SAME _classify_verdict /
    _refine_prompt_verdict source as the terminal, so the report carries the
    "current model failed" sentence and the neutral per-prompt label."""
    from frugon.report import _quality_section_md

    md = "\n".join(_quality_section_md(_baseline_failed_result()))
    assert "Could not verify" in md, md
    assert "your current model (gpt-4-turbo) failed to sample" in md, md
    assert "every gpt-4o-mini comparison errored" not in md, md
    assert "[no comparison]" in md, md


# ---------------------------------------------------------------------------
# Tier-0 — framing line (no --judge)
# ---------------------------------------------------------------------------


def test_tier0_framing_line_present_and_names_candidate() -> None:
    # Arrange — Tier-0 result (tier1_tallies is None).
    comp = Comparison(
        record=_record(),
        current_output=SampledOutput("gpt-4-turbo", "baseline answer"),
        candidate_outputs=[SampledOutput("gpt-4o-mini", "candidate answer")],
    )
    result = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[comp],
        tier1_tallies=None,
    )

    # Act
    out = _capture(result)

    # Assert — frames as raw/unscored, points to --judge, names the candidate,
    # keeps "within tolerance" as the muted offline estimate.
    assert "raw side-by-side" in out, out
    assert "--judge" in out, out
    assert "within tolerance" in out, out
    assert "offline estimate" in out, out
    assert "gpt-4o-mini" in out, out
    # Tier-0 must NOT claim a scored verdict.
    assert "confirmed" not in out, out
