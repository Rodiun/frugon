"""Tests for the --preview-chars / --no-truncate display flags.

These flags are COSMETIC: they change only how long the per-prompt quality-sample
previews are when rendered, on the terminal AND in written reports.  They never
change what is sent to a provider (the measure engine always uses the full
prompt) and never cap call/token counts.

Invariants tested here:

  1. resolve_preview_limits() math — --preview-chars overrides the OUTPUT length
     on both surfaces and scales the PROMPT length by each surface's ratio;
     --no-truncate yields full text; neither flag yields the historical defaults.
  2. _truncate(no_truncate=True) returns text verbatim regardless of limit.
  3. PreviewLimits is immutable and stateless — two independent limits drive two
     concurrent renders with no shared global mutated between them.
  4. The terminal renderer honours both flags (shorter preview / full preview).
  5. The report renderers (md + html) honour both flags.
  6. Mutual exclusion is enforced at the CLI boundary.

Everything runs offline — MeasureResults are constructed directly.
"""

from __future__ import annotations

import concurrent.futures

from typer.testing import CliRunner

from frugon.cli import app
from frugon.cost import LogRecord
from frugon.measure import Comparison, MeasureResult, SampledOutput, Tier1Tally
from frugon.report import (
    PreviewLimits,
    _quality_section_html,
    _quality_section_md,
    _truncate,
    render_quality_terminal,
    resolve_preview_limits,
)

runner = CliRunner()

# A long, distinctive output so truncation is observable.  The tail marker only
# survives when the preview is long enough (or untruncated) to reach it.
_LONG_OUTPUT = "BEGIN-" + ("x" * 1000) + "-END-MARKER"
_LONG_PROMPT = "ASK-" + ("p" * 1000) + "-PROMPT-TAIL"


def _record() -> LogRecord:
    return LogRecord(
        model="gpt-4o",
        messages=[{"role": "user", "content": _LONG_PROMPT}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _tier0_long() -> MeasureResult:
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[
            Comparison(
                record=_record(),
                current_output=SampledOutput(model="gpt-4o", content=_LONG_OUTPUT),
                candidate_outputs=[
                    SampledOutput(model="gpt-4o-mini", content=_LONG_OUTPUT)
                ],
            )
        ],
        tier1_tallies=None,
    )


def _tier1_long() -> MeasureResult:
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[
            Comparison(
                record=_record(),
                current_output=SampledOutput(model="gpt-4o", content=_LONG_OUTPUT),
                candidate_outputs=[
                    SampledOutput(model="gpt-4o-mini", content=_LONG_OUTPUT)
                ],
                verdicts=["loss"],
            )
        ],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=1, ties=0)],
    )


# ---------------------------------------------------------------------------
# 1. resolve_preview_limits() math
# ---------------------------------------------------------------------------


def test_resolve_preview_limits_no_flags_returns_historical_defaults() -> None:
    terminal, report = resolve_preview_limits()
    assert terminal == PreviewLimits(prompt_chars=160, output_chars=240)
    assert report == PreviewLimits(prompt_chars=400, output_chars=800)
    assert terminal.no_truncate is False
    assert report.no_truncate is False


def test_resolve_preview_limits_preview_chars_overrides_output_and_scales_prompt() -> None:
    # --preview-chars 60: output → 60 on both surfaces; prompt scaled by ratio.
    # Terminal ratio 160/240 = 2/3 → round(60 * 160/240) = 40.
    # Report ratio 400/800 = 1/2 → round(60 * 400/800) = 30.
    terminal, report = resolve_preview_limits(preview_chars=60)
    assert terminal == PreviewLimits(prompt_chars=40, output_chars=60)
    assert report == PreviewLimits(prompt_chars=30, output_chars=60)


def test_resolve_preview_limits_tiny_n_floors_prompt_at_one() -> None:
    # A tiny N must never collapse the prompt preview to zero characters.
    terminal, report = resolve_preview_limits(preview_chars=20)
    # round(20 * 160/240) = round(13.33) = 13 ; round(20 * 1/2) = 10 — both >= 1.
    assert terminal.prompt_chars == 13
    assert terminal.output_chars == 20
    assert report.prompt_chars == 10
    assert report.output_chars == 20


def test_resolve_preview_limits_no_truncate_sets_flag_both_surfaces() -> None:
    terminal, report = resolve_preview_limits(no_truncate=True)
    assert terminal.no_truncate is True
    assert report.no_truncate is True


def test_resolve_preview_limits_no_truncate_wins_if_both_supplied() -> None:
    # Defensive: the CLI forbids both, but if both arrive, no_truncate wins.
    terminal, report = resolve_preview_limits(preview_chars=50, no_truncate=True)
    assert terminal.no_truncate is True
    assert report.no_truncate is True


# ---------------------------------------------------------------------------
# 2. _truncate(no_truncate=True)
# ---------------------------------------------------------------------------


def test_truncate_no_truncate_returns_verbatim_regardless_of_limit() -> None:
    text = "a" * 500
    assert _truncate(text, 10, no_truncate=True) == text
    # No ellipsis appended.
    assert "…" not in _truncate(text, 10, no_truncate=True)


def test_truncate_default_still_cuts_and_appends_ellipsis() -> None:
    assert _truncate("abcdef", 3) == "abc…"
    assert _truncate("abc", 3) == "abc"


# ---------------------------------------------------------------------------
# 3. Statelessness / concurrency safety
# ---------------------------------------------------------------------------


def test_preview_limits_render_concurrently_without_cross_contamination() -> None:
    """Two independent PreviewLimits drive two renders in parallel; each result
    reflects ONLY its own limits (no module global is mutated between them)."""
    short = PreviewLimits(prompt_chars=400, output_chars=30)
    full = PreviewLimits(prompt_chars=400, output_chars=800, no_truncate=True)
    result = _tier0_long()

    def render(limits: PreviewLimits) -> str:
        return "".join(_quality_section_md(result, limits=limits))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # Submit the SAME pair many times interleaved to surface any shared state.
        futures = [pool.submit(render, short if i % 2 == 0 else full) for i in range(20)]
        outputs = [f.result() for f in futures]

    short_outputs = [o for i, o in enumerate(outputs) if i % 2 == 0]
    full_outputs = [o for i, o in enumerate(outputs) if i % 2 == 1]
    # The short render never reaches the tail marker; the full render always does.
    assert all("END-MARKER" not in o for o in short_outputs)
    assert all("END-MARKER" in o for o in full_outputs)


# ---------------------------------------------------------------------------
# 4. Terminal renderer honours the flags
# ---------------------------------------------------------------------------


def _render_terminal_to_str(result: MeasureResult, limits: PreviewLimits) -> str:
    import io

    from rich.console import Console

    import frugon.report as report_mod

    buf = io.StringIO()
    console = Console(file=buf, width=400, no_color=True, highlight=False)
    original = report_mod.rprint
    report_mod.rprint = lambda *a, **k: console.print(*a, **k)  # type: ignore[assignment]
    try:
        render_quality_terminal(result, verbose=True, limits=limits)
    finally:
        report_mod.rprint = original  # type: ignore[assignment]
    return buf.getvalue()


def test_terminal_preview_chars_shortens_output_preview() -> None:
    out = _render_terminal_to_str(
        _tier0_long(), PreviewLimits(prompt_chars=160, output_chars=30)
    )
    assert "BEGIN-" in out  # the head is shown
    assert "END-MARKER" not in out  # the tail is cut
    assert "…" in out


def test_terminal_no_truncate_shows_full_output() -> None:
    out = _render_terminal_to_str(
        _tier0_long(),
        PreviewLimits(prompt_chars=160, output_chars=240, no_truncate=True),
    )
    assert "END-MARKER" in out  # the full output reaches the tail
    assert "PROMPT-TAIL" in out  # the full prompt reaches its tail too


def test_terminal_default_truncates_long_output() -> None:
    out = _render_terminal_to_str(_tier0_long(), PreviewLimits.terminal_default())
    assert "END-MARKER" not in out  # 240-char default cuts the 1000-char body


# ---------------------------------------------------------------------------
# 5. Report renderers honour the flags
# ---------------------------------------------------------------------------


def test_md_section_no_truncate_shows_full_output() -> None:
    full = PreviewLimits(prompt_chars=400, output_chars=800, no_truncate=True)
    md = "".join(_quality_section_md(_tier1_long(), limits=full))
    assert "END-MARKER" in md


def test_md_section_preview_chars_cuts_output() -> None:
    short = PreviewLimits(prompt_chars=400, output_chars=50)
    md = "".join(_quality_section_md(_tier1_long(), limits=short))
    assert "BEGIN-" in md
    assert "END-MARKER" not in md


def test_md_section_default_matches_report_default() -> None:
    explicit = "".join(
        _quality_section_md(_tier1_long(), limits=PreviewLimits.report_default())
    )
    implicit = "".join(_quality_section_md(_tier1_long()))
    assert explicit == implicit


def test_html_section_no_truncate_shows_full_output() -> None:
    full = PreviewLimits(prompt_chars=400, output_chars=800, no_truncate=True)
    html = _quality_section_html(_tier1_long(), style="v1", limits=full)
    assert "END-MARKER" in html


def test_html_section_preview_chars_cuts_output() -> None:
    short = PreviewLimits(prompt_chars=400, output_chars=50)
    html = _quality_section_html(_tier1_long(), style="v2", limits=short)
    assert "BEGIN-" in html
    assert "END-MARKER" not in html


# ---------------------------------------------------------------------------
# 6. CLI mutual exclusion + flag wiring (no measure run — fails at the gate)
# ---------------------------------------------------------------------------


def test_cli_preview_chars_and_no_truncate_are_mutually_exclusive() -> None:
    result = runner.invoke(
        app,
        ["analyze", "--demo", "--preview-chars", "60", "--no-truncate"],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.stdout


def test_cli_preview_chars_below_minimum_rejected() -> None:
    result = runner.invoke(app, ["analyze", "--demo", "--preview-chars", "10"])
    # typer min=20 → usage error (exit code 2), preview never starts.
    assert result.exit_code == 2


def test_cli_demo_with_no_truncate_runs_clean_without_measure() -> None:
    # Without --measure there is no quality section, so the flag is a harmless
    # no-op; it must NOT error and must NOT alter the cost analysis exit code.
    result = runner.invoke(app, ["analyze", "--demo", "--no-truncate"])
    assert result.exit_code == 0
