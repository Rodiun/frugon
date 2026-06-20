"""Tests for the verbose Tier-1 per-prompt verdict view.

A fire-and-forget judge aggregates win/loss/tie counts, but the aggregate tally
alone cannot tell a user WHICH sampled prompts the candidate lost on.  Under
``--verbose`` the Tier-1 render gains a per-prompt detail block: each sampled
prompt's side-by-side outputs, with every candidate output labelled by its
per-prompt verdict (``gpt-4o-mini [LOSS]: …``) in the verdict colour band
(WIN = cyan, TIE = dim, LOSS = amber).  Non-verbose Tier-1 output is unchanged.

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

# Rich SGR foreground codes for the verdict colour bands.  These MATCH the
# judge tally table: green = WIN, red = LOSS, yellow = TIE.  Green-as-win is
# legitimate inside the quality detail (the cost panel keeps green=money).
# Asserting these pins the band without coupling to exact wording.
_GREEN_SGR = '\x1b[32m'
_RED_SGR = '\x1b[31m'
_YELLOW_SGR = '\x1b[33m'


def _render(measure_result: MeasureResult, *, verbose: bool, colour: bool) -> str:
    """Render to a fixed-width string; optionally preserve colour.

    Patches BOTH ``rprint`` and ``_render_console`` so the responsive
    hanging-indent rows measure and emit at the same width — otherwise the
    per-prompt detail would wrap for the ambient console and print here.
    """
    report_mod = sys.modules[render_quality_terminal.__module__]
    buf = io.StringIO()
    console = Console(
        file=buf,
        width=200,
        no_color=not colour,
        force_terminal=True,
        color_system="standard" if colour else None,
        highlight=False,
    )
    original_rprint = report_mod.rprint
    original_render_console = report_mod._render_console

    def _patched(*args: object, **kw: object) -> None:
        console.print(*args, **kw)

    report_mod.rprint = _patched  # type: ignore[attr-defined]
    report_mod._render_console = lambda: console  # type: ignore[attr-defined]
    try:
        render_quality_terminal(measure_result, verbose=verbose)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    out = buf.getvalue()
    return out if colour else _ANSI_RE.sub("", out)


def _record(prompt: str) -> LogRecord:
    return LogRecord(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _mixed_result() -> MeasureResult:
    """Build a Tier-1 result mirroring the reported case: 0 win / 2 loss / 3 tie.

    Five sampled prompts against one candidate, each carrying its own verdict so
    the per-prompt detail can label them.
    """
    verdicts = ["tie", "loss", "tie", "loss", "tie"]
    comparisons = [
        Comparison(
            record=_record(f"Classify ticket {i}"),
            current_output=SampledOutput("gpt-4-turbo", f"baseline answer {i}"),
            candidate_outputs=[SampledOutput("gpt-4o-mini", f"candidate answer {i}")],
            verdicts=[verdict],
        )
        for i, verdict in enumerate(verdicts)
    ]
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=2, ties=3)],
    )


# ---------------------------------------------------------------------------
# Verbose Tier-1 — per-prompt detail with verdict labels
# ---------------------------------------------------------------------------


def test_verbose_tier1_renders_per_prompt_detail_block() -> None:
    # Act
    out = _render(_mixed_result(), verbose=True, colour=False)

    # Assert — the aggregate table + synthesis are still present...
    assert "judge results" in out, out
    assert "review the losses" in out, out
    # ...and the verbose detail header introduces the per-prompt block.
    assert "Per-prompt detail (verbose):" in out, out
    # Every sampled prompt is surfaced side-by-side.
    for i in range(5):
        assert f"Classify ticket {i}" in out, out
        assert f"candidate answer {i}" in out, out
        assert f"baseline answer {i}" in out, out


def test_verbose_tier1_labels_each_candidate_with_its_verdict() -> None:
    # Act
    out = _render(_mixed_result(), verbose=True, colour=False)

    # Assert — exactly two LOSS labels and three TIE labels, matching the tally.
    assert out.count("[LOSS]") == 2, out
    assert out.count("[TIE]") == 3, out
    assert "[WIN]" not in out, out
    # The label rides on the candidate row, not the baseline.
    assert "gpt-4o-mini [LOSS]:" in out, out
    assert "gpt-4o-mini [TIE]:" in out, out


def test_verbose_tier1_verdict_colour_bands() -> None:
    # A result with one of each verdict so all three bands appear.
    verdicts = ["win", "tie", "loss"]
    comparisons = [
        Comparison(
            record=_record(f"prompt {v}"),
            current_output=SampledOutput("gpt-4-turbo", "base"),
            candidate_outputs=[SampledOutput("gpt-4o-mini", f"cand {v}")],
            verdicts=[v],
        )
        for v in verdicts
    ]
    result = MeasureResult(
        samples_requested=3,
        samples_taken=3,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=1, losses=1, ties=1)],
    )

    out = _render(result, verbose=True, colour=True)

    # All three verdict labels present (verified on the stripped text).
    stripped = _ANSI_RE.sub("", out)
    assert "[WIN]" in stripped, out
    assert "[TIE]" in stripped, out
    assert "[LOSS]" in stripped, out
    # WIN green, LOSS red, TIE yellow — matching the tally table; each label is
    # wrapped in the SGR of its band by Rich.
    assert f"{_GREEN_SGR}[WIN]" in out, out
    assert f"{_RED_SGR}[LOSS]" in out, out
    assert f"{_YELLOW_SGR}[TIE]" in out, out
    # Loss never reads in the money-green band.
    assert "\x1b[32m[LOSS]" not in out, out


def test_verbose_tier1_error_verdict_collapses_to_dim_error_label() -> None:
    comp = Comparison(
        record=_record("a prompt"),
        current_output=SampledOutput("gpt-4-turbo", "base"),
        candidate_outputs=[SampledOutput("gpt-4o-mini", "cand")],
        verdicts=["error"],
    )
    result = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[comp],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=0, ties=0, errors=1)],
    )

    out = _render(result, verbose=True, colour=False)

    assert "[error]" in out, out
    assert "[ERROR]" not in out, out


# ---------------------------------------------------------------------------
# Non-verbose Tier-1 — unchanged compact view
# ---------------------------------------------------------------------------


def test_non_verbose_tier1_has_no_per_prompt_detail() -> None:
    result = _mixed_result()

    out = _render(result, verbose=False, colour=False)

    # The compact table + synthesis remain...
    assert "judge results" in out, out
    assert "review the losses" in out, out
    # ...but NO per-prompt detail block, labels, or sampled outputs leak in.
    assert "Per-prompt detail (verbose):" not in out, out
    assert "[LOSS]" not in out, out
    assert "[TIE]" not in out, out
    assert "Classify ticket 0" not in out, out
    assert "candidate answer 0" not in out, out


def test_non_verbose_tier1_byte_identical_to_default() -> None:
    # The default (verbose omitted) must equal explicit verbose=False.
    result = _mixed_result()

    report_mod = sys.modules[render_quality_terminal.__module__]
    buf_default = io.StringIO()
    console_default = Console(
        file=buf_default, width=200, no_color=True, force_terminal=True, highlight=False
    )
    original_rprint = report_mod.rprint
    original_render_console = report_mod._render_console
    report_mod.rprint = lambda *a, **k: console_default.print(*a, **k)  # type: ignore[attr-defined]
    report_mod._render_console = lambda: console_default  # type: ignore[attr-defined]
    try:
        render_quality_terminal(result)  # verbose defaulted
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]

    explicit = _render(result, verbose=False, colour=False)
    assert _ANSI_RE.sub("", buf_default.getvalue()) == explicit


# ---------------------------------------------------------------------------
# Tier-0 — unaffected by verbose
# ---------------------------------------------------------------------------


def test_tier0_unaffected_by_verbose_flag() -> None:
    comp = Comparison(
        record=_record("Tier-0 prompt"),
        current_output=SampledOutput("gpt-4-turbo", "baseline answer"),
        candidate_outputs=[SampledOutput("gpt-4o-mini", "candidate answer")],
    )
    result = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[comp],
        tier1_tallies=None,  # Tier-0: no judge ran
    )

    quiet = _render(result, verbose=False, colour=False)
    loud = _render(result, verbose=True, colour=False)

    # Tier-0 shows its side-by-side regardless, and never invents verdict labels
    # (no judge ran) or the Tier-1-only detail header.
    assert "Tier-0 prompt" in quiet, quiet
    assert "Tier-0 prompt" in loud, loud
    assert "[LOSS]" not in loud, loud
    assert "[TIE]" not in loud, loud
    assert "[WIN]" not in loud, loud
    assert "Per-prompt detail (verbose):" not in loud, loud
    assert "raw side-by-side" in quiet, quiet
    assert "raw side-by-side" in loud, loud
