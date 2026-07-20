"""Tests for the pointwise "both failed" tie check (--measure --judge).

The pairwise judge (_judge_pair) defaults to TIE whenever it finds no MATERIAL
difference between two outputs -- a verdict that is silent about WHY: two
outputs can tie because they are equally GOOD, or because they equally FAILED
to address the prompt.  This module tests the separate, single-response
pointwise check (_judge_addressed) that flags the latter case as a
"both failed" tie, WITHOUT touching the calibrated pairwise preference signal
itself, plus its knock-on surfaces: the pre-run cost estimate and the
terminal/Markdown/HTML tie-row rendering.

All tests run fully offline -- LiteLLM is mocked or the check function itself
is stubbed; no real network call is made.
"""

from __future__ import annotations

import io
import re
import sys
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from frugon import cli
from frugon.cost import LogRecord
from frugon.measure import (
    Comparison,
    MeasureEstimate,
    MeasureResult,
    SampledOutput,
    Tier1Tally,
    _judge_addressed,
    _parse_addressed,
    max_check_call_count,
    run_measure,
)
from frugon.report import (
    _quality_section_md,
    _verdict_html_label,
    _verdict_label,
    _verdict_md_label,
    render_quality_terminal,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


@pytest.fixture(autouse=True)
def _provider_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")


def _make_litellm_mock(content: str = "mocked output") -> MagicMock:
    mock = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = content
    mock.completion.return_value = resp
    return mock


_RECORD_COUNTER = 0


def _make_record(prompt_text: str | None = None) -> LogRecord:
    global _RECORD_COUNTER
    if prompt_text is None:
        _RECORD_COUNTER += 1
        prompt_text = f"prompt {_RECORD_COUNTER}"
    return LogRecord(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_text}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


# ---------------------------------------------------------------------------
# _parse_addressed (unit) -- happy / edge
# ---------------------------------------------------------------------------


def test_parse_addressed_yes_returns_true() -> None:
    assert _parse_addressed("ADDRESSED: YES") is True


def test_parse_addressed_no_returns_false() -> None:
    assert _parse_addressed("ADDRESSED: NO") is False


def test_parse_addressed_lowercase_and_decorated_still_parses() -> None:
    assert _parse_addressed("**addressed: yes**") is True
    assert _parse_addressed("Reasoning...\nADDRESSED: NO") is False


def test_parse_addressed_ambiguous_reply_returns_none() -> None:
    """No 'ADDRESSED: YES|NO' token anywhere -> unparseable -> None.

    The caller (_judge_addressed) is the one that applies the honest
    "treat as addressed" default for a None result -- _parse_addressed itself
    stays a pure, caller-agnostic extractor.
    """
    assert _parse_addressed("I'm not sure how to answer that.") is None
    assert _parse_addressed("") is None


# ---------------------------------------------------------------------------
# _judge_addressed (unit)
# ---------------------------------------------------------------------------


def test_judge_addressed_yes_reply_is_true() -> None:
    mock_litellm = _make_litellm_mock("ADDRESSED: YES")
    result = _judge_addressed(
        mock_litellm, "gpt-4o", [{"role": "user", "content": "test"}], "a real answer"
    )
    assert result is True


def test_judge_addressed_no_reply_is_false() -> None:
    mock_litellm = _make_litellm_mock("ADDRESSED: NO")
    result = _judge_addressed(
        mock_litellm, "gpt-4o", [{"role": "user", "content": "test"}], ""
    )
    assert result is False


def test_judge_addressed_ambiguous_reply_defaults_true() -> None:
    """Honest default: an unparseable reply never flags a false 'both failed'."""
    mock_litellm = _make_litellm_mock("I cannot tell.")
    result = _judge_addressed(
        mock_litellm, "gpt-4o", [{"role": "user", "content": "test"}], "some output"
    )
    assert result is True


def test_judge_addressed_exhausted_retries_defaults_true() -> None:
    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = RuntimeError("network error")
    result = _judge_addressed(
        mock_litellm,
        "gpt-4o",
        [{"role": "user", "content": "test"}],
        "some output",
        backoff_s=0.0,
    )
    assert result is True


def test_judge_addressed_retries_once_then_succeeds() -> None:
    resp = MagicMock()
    resp.choices[0].message.content = "ADDRESSED: NO"
    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = [RuntimeError("rate limited"), resp]
    result = _judge_addressed(
        mock_litellm,
        "gpt-4o",
        [{"role": "user", "content": "test"}],
        "some output",
        backoff_s=0.0,
    )
    assert result is False
    assert mock_litellm.completion.call_count == 2


def test_judge_addressed_records_usage_on_returned_call() -> None:
    mock_litellm = _make_litellm_mock("ADDRESSED: YES")
    sink: list[Any] = []
    _judge_addressed(
        mock_litellm,
        "gpt-4o",
        [{"role": "user", "content": "test"}],
        "some output",
        usage_sink=sink,
    )
    assert len(sink) == 1
    assert sink[0].model == "gpt-4o"


def test_judge_addressed_never_reveals_pairwise_framing() -> None:
    """The pointwise prompt is a single ANSWER, never OUTPUT A / OUTPUT B --
    it must never be confused with (or leak into) the pairwise judge prompt.
    """
    mock_litellm = _make_litellm_mock("ADDRESSED: YES")
    _judge_addressed(
        mock_litellm, "gpt-4o", [{"role": "user", "content": "test"}], "some output"
    )
    sent = mock_litellm.completion.call_args.kwargs["messages"][0]["content"]
    assert "OUTPUT A" not in sent
    assert "OUTPUT B" not in sent
    assert "some output" in sent


# ---------------------------------------------------------------------------
# run_measure integration -- both_failed wiring
# ---------------------------------------------------------------------------


def test_run_measure_both_failed_true_when_neither_side_addresses() -> None:
    """Arrange: 1 prompt, 1 candidate; the pairwise judge ties; the pointwise
    check finds NEITHER side addressed the prompt.
    Act: run_measure.
    Assert: Comparison.both_failed is [True] for the tied candidate.
    """
    records = [_make_record()]
    mock_litellm = _make_litellm_mock("model output")

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", return_value="tie"),
        patch("frugon.measure._judge_addressed", return_value=False),
    ):
        result = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=1,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=1,
            seed=0,
        )

    assert len(result.comparisons) == 1
    comp = result.comparisons[0]
    assert comp.verdicts == ["tie"]
    assert comp.both_failed == [True]


def test_run_measure_both_failed_false_when_one_side_addresses() -> None:
    """A tie where only ONE side addressed the prompt must NOT be flagged --
    'both failed' requires BOTH sides to have failed the pointwise check.
    """
    records = [_make_record()]
    mock_litellm = _make_litellm_mock("model output")
    addressed_by_content = {"current output": True, "candidate output": False}

    def _stub_addressed(_litellm: object, _judge_model: str, _messages: Any, output_content: str, **_kwargs: object) -> bool:
        return addressed_by_content[output_content]

    def _stub_call_model(_litellm: object, model: str, _messages: Any, *, is_baseline: bool = False) -> SampledOutput:
        content = "current output" if is_baseline else "candidate output"
        return SampledOutput(model=model, content=content)

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._call_model", side_effect=_stub_call_model),
        patch("frugon.measure._judge_pair", return_value="tie"),
        patch("frugon.measure._judge_addressed", side_effect=_stub_addressed),
    ):
        result = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=1,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=1,
            seed=0,
        )

    assert result.comparisons[0].both_failed == [False]


@pytest.mark.parametrize("verdict", ["win", "loss", "error"])
def test_run_measure_pointwise_check_never_runs_for_non_tie_verdicts(
    verdict: str,
) -> None:
    """The pointwise check is a TIE-only check -- it must never fire (and
    both_failed must stay False) for win / loss / error verdicts, so a
    non-tie run never pays the extra call cost.
    """
    records = [_make_record()]
    mock_litellm = _make_litellm_mock("model output")

    def _judge_addressed_must_not_be_called(*_a: object, **_k: object) -> bool:
        raise AssertionError("_judge_addressed must not run for a non-tie verdict")

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", return_value=verdict),
        patch(
            "frugon.measure._judge_addressed",
            side_effect=_judge_addressed_must_not_be_called,
        ),
    ):
        result = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=1,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=1,
            seed=0,
        )

    assert result.comparisons[0].both_failed == [False]


def test_run_measure_baseline_addressed_check_cached_across_tied_candidates() -> None:
    """Arrange: 1 prompt, 2 candidates, BOTH tie against the baseline.
    Assert: the baseline's pointwise check runs ONCE (not once per tie) --
    the same baseline output is shared across both candidates for this prompt.
    """
    records = [_make_record()]
    mock_litellm = _make_litellm_mock("model output")

    def _stub_call_model(_litellm: object, model: str, _messages: Any, *, is_baseline: bool = False) -> SampledOutput:
        content = "baseline output" if is_baseline else f"{model} output"
        return SampledOutput(model=model, content=content)

    addressed_calls: list[str] = []

    def _stub_addressed(_litellm: object, _judge_model: str, _messages: Any, output_content: str, **_kwargs: object) -> bool:
        addressed_calls.append(output_content)
        return False

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._call_model", side_effect=_stub_call_model),
        patch("frugon.measure._judge_pair", return_value="tie"),
        patch("frugon.measure._judge_addressed", side_effect=_stub_addressed),
    ):
        result = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini", "claude-3-haiku-20240307"],
            n_samples=1,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=1,
            seed=0,
        )

    assert result.comparisons[0].both_failed == [True, True]
    # Baseline checked once, each of the 2 candidates checked once = 3 total.
    assert addressed_calls.count("baseline output") == 1
    assert len(addressed_calls) == 3


def test_run_measure_pointwise_usage_merged_into_measure_calls() -> None:
    """The pointwise check's token usage must reach MeasureResult.measure_calls
    exactly like the pairwise judge's -- the run's disclosed cost must never
    silently drop the extra calls.

    _judge_pair is stubbed (contributes no completion() call of its own) so
    every call that DOES hit the mocked litellm is either a sampling call or a
    REAL (unstubbed) _judge_addressed pointwise check -- isolating the
    pointwise leg's usage capture from the pairwise judge's.
    """
    resp = MagicMock()
    resp.choices[0].message.content = "ADDRESSED: NO"
    resp.usage = MagicMock(prompt_tokens=7, completion_tokens=3)
    mock_litellm = MagicMock()
    mock_litellm.completion.return_value = resp

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", return_value="tie"),
    ):
        result = run_measure(
            [_make_record()],
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=1,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=1,
            seed=0,
        )

    # 1 baseline sample + 1 candidate sample + 1 baseline-addressed pointwise
    # check + 1 candidate-addressed pointwise check = 4 calls.
    assert len(result.measure_calls) == 4
    assert all(
        c.prompt_tokens == 7 and c.completion_tokens == 3
        for c in result.measure_calls
    )


def test_run_measure_skips_candidate_check_when_baseline_addressed() -> None:
    """W3: once the baseline's pointwise check finds it DID address the
    prompt, 'both failed' is impossible regardless of the candidate -- the
    candidate-side _judge_addressed call must be skipped entirely (the check
    burns roughly half the calls it needs otherwise, in a COST tool).
    """
    records = [_make_record()]
    mock_litellm = _make_litellm_mock("model output")

    addressed_calls: list[str] = []

    def _stub_addressed(_litellm: object, _judge_model: str, _messages: Any, output_content: str, **_kwargs: object) -> bool:
        addressed_calls.append(output_content)
        return True  # baseline addressed the prompt

    def _stub_call_model(_litellm: object, model: str, _messages: Any, *, is_baseline: bool = False) -> SampledOutput:
        content = "baseline output" if is_baseline else "candidate output"
        return SampledOutput(model=model, content=content)

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._call_model", side_effect=_stub_call_model),
        patch("frugon.measure._judge_pair", return_value="tie"),
        patch("frugon.measure._judge_addressed", side_effect=_stub_addressed),
    ):
        result = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=1,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=1,
            seed=0,
        )

    assert result.comparisons[0].both_failed == [False]
    # Only the baseline check ran -- the candidate check was skipped entirely.
    assert addressed_calls == ["baseline output"]


def test_run_measure_check_errors_counts_retry_exhausted_pointwise_check() -> None:
    """W4: when the pointwise both-failed check exhausts its retries on a
    transient fault, the fault must not vanish silently -- Fail-Loud requires
    it to surface as Tier1Tally.check_errors so a judge-side outage cannot
    masquerade as a perfectly clean run.
    """
    records = [_make_record()]

    def _stub_call_model(_litellm: object, model: str, _messages: Any, *, is_baseline: bool = False) -> SampledOutput:
        content = "baseline output" if is_baseline else "candidate output"
        return SampledOutput(model=model, content=content)

    faulting_litellm = MagicMock()
    faulting_litellm.completion.side_effect = RuntimeError("network error")

    with (
        patch("frugon.measure._import_litellm", return_value=faulting_litellm),
        patch("frugon.measure._call_model", side_effect=_stub_call_model),
        patch("frugon.measure._judge_pair", return_value="tie"),
    ):
        result = run_measure(
            records,
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
    assert tally.check_errors == 1
    # The fault-defaulted "addressed" still counts as NOT both-failed --
    # the ambiguous-parse/fault default is honest and conservative, unchanged.
    assert tally.both_failed_ties == 0


# ---------------------------------------------------------------------------
# max_check_call_count (unit) + estimate_measure_cost wiring
# ---------------------------------------------------------------------------


def test_max_check_call_count_zero_without_judge() -> None:
    assert max_check_call_count(50, 2, use_judge=False) == 0


def test_max_check_call_count_worst_case_with_judge() -> None:
    # 1 baseline check + 2 candidate checks per prompt, worst case every tie.
    assert max_check_call_count(50, 2, use_judge=True) == 150


def test_estimate_measure_cost_reports_max_check_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from frugon.measure import estimate_measure_cost

    monkeypatch.setattr("frugon.pricing.get_model_price", lambda _m: None)
    records = [_make_record() for _ in range(4)]
    est = estimate_measure_cost(
        records,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        n_samples=4,
        use_judge=True,
        judge_model="gpt-4o",
    )
    # 4 prompts × (1 baseline + 1 candidate) = 8 worst-case check calls.
    assert est.max_check_calls == 8
    # planned_calls stays the EXACT sample+judge total, unchanged by the
    # worst-case (data-dependent) check leg.
    assert est.planned_calls == 12


def test_estimate_measure_cost_max_check_calls_zero_without_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from frugon.measure import estimate_measure_cost

    monkeypatch.setattr("frugon.pricing.get_model_price", lambda _m: None)
    records = [_make_record() for _ in range(4)]
    est = estimate_measure_cost(
        records, current_model="gpt-4o", candidates=["gpt-4o-mini"], n_samples=4
    )
    assert est.max_check_calls == 0


# ---------------------------------------------------------------------------
# CLI estimate rendering -- discloses the pointwise check's worst-case cost
# ---------------------------------------------------------------------------


def _render_estimate(estimate: MeasureEstimate, monkeypatch: pytest.MonkeyPatch) -> str:
    captured: list[str] = []
    monkeypatch.setattr(cli, "rprint", lambda s="": captured.append(str(s)))
    cli._render_measure_estimate(estimate)
    assert len(captured) == 1
    return captured[0].replace("[dim]", "").replace("[/dim]", "").strip()


def test_render_estimate_discloses_check_calls_when_judge_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = _render_estimate(
        MeasureEstimate(
            planned_calls=3,
            estimated_cost=Decimal("0.01"),
            unpriced_models=[],
            n_prompts=1,
            n_candidates=1,
            use_judge=True,
            max_check_calls=2,
        ),
        monkeypatch,
    )
    assert (
        "About to make 3 provider calls: "
        "2 to sample (1 prompt × 2 models: baseline + 1 candidate)"
        " + 1 to judge (1 prompt × 1 candidate), "
        "up to 2 more to check ties for shared failure. "
        "Estimated cost ~$0.01 on your keys."
    ) == line


def test_render_estimate_omits_check_clause_when_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_check_calls=0 (the default) leaves the estimate line byte-identical
    to before this feature existed -- no behaviour change for existing callers.
    """
    line = _render_estimate(
        MeasureEstimate(
            planned_calls=3,
            estimated_cost=Decimal("0.01"),
            unpriced_models=[],
            n_prompts=1,
            n_candidates=1,
            use_judge=True,
        ),
        monkeypatch,
    )
    assert "to check ties" not in line
    assert line == (
        "About to make 3 provider calls: "
        "2 to sample (1 prompt × 2 models: baseline + 1 candidate)"
        " + 1 to judge (1 prompt × 1 candidate). "
        "Estimated cost ~$0.01 on your keys."
    )


def test_render_estimate_check_clause_omitted_without_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = _render_estimate(
        MeasureEstimate(
            planned_calls=2,
            estimated_cost=Decimal("0.01"),
            unpriced_models=[],
            n_prompts=1,
            n_candidates=1,
            use_judge=False,
        ),
        monkeypatch,
    )
    assert "to check ties" not in line


# ---------------------------------------------------------------------------
# Report rendering -- [both failed] flag on TIE rows
# ---------------------------------------------------------------------------


def test_verdict_label_shows_both_failed_marker_only_on_tie() -> None:
    tie_flagged = _verdict_label("tie", both_failed=True)
    assert "[both failed]" in tie_flagged.plain

    tie_unflagged = _verdict_label("tie", both_failed=False)
    assert "[both failed]" not in tie_unflagged.plain

    # Never appended to a non-tie verdict, even if both_failed=True is passed.
    win_flagged = _verdict_label("win", both_failed=True)
    assert "[both failed]" not in win_flagged.plain


def test_verdict_md_label_shows_both_failed_marker_only_on_tie() -> None:
    assert _verdict_md_label("tie", both_failed=True) == "[TIE] [both failed]"
    assert _verdict_md_label("tie", both_failed=False) == "[TIE]"
    assert _verdict_md_label("loss", both_failed=True) == "[LOSS]"


def test_verdict_html_label_shows_both_failed_marker_only_on_tie() -> None:
    html = _verdict_html_label("tie", both_failed=True)
    assert "verdict-both-failed" in html
    assert "[both failed]" in html
    assert "verdict-both-failed" not in _verdict_html_label("tie", both_failed=False)
    assert "verdict-both-failed" not in _verdict_html_label("win", both_failed=True)


def _comparison(prompt: str, *, verdict: str, both_failed: bool) -> Comparison:
    return Comparison(
        record=_make_record(prompt),
        current_output=SampledOutput(model="gpt-4o", content="current output"),
        candidate_outputs=[SampledOutput(model="gpt-4o-mini", content="candidate output")],
        verdicts=[verdict],
        both_failed=[both_failed],
    )


def _both_failed_result() -> MeasureResult:
    return MeasureResult(
        samples_requested=2,
        samples_taken=2,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[
            _comparison("p1", verdict="tie", both_failed=True),
            _comparison("p2", verdict="tie", both_failed=False),
        ],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=0, ties=2)],
        judge_model="gpt-4o",
    )


def test_render_quality_terminal_verbose_shows_both_failed_flag() -> None:
    report_mod = sys.modules[render_quality_terminal.__module__]
    buf = io.StringIO()
    console = Console(file=buf, width=200, no_color=True, highlight=False)
    original_rprint = report_mod.rprint
    original_render_console = report_mod._render_console
    report_mod.rprint = lambda *a, **k: console.print(*a, **k)  # type: ignore[attr-defined]
    report_mod._render_console = lambda: console  # type: ignore[attr-defined]
    try:
        render_quality_terminal(_both_failed_result(), verbose=True)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    out = _ANSI_RE.sub("", buf.getvalue())
    assert "[both failed]" in out
    # p2's tie is NOT flagged -- the marker must not appear on every tie row,
    # only once (for p1's flagged tie).
    assert out.count("[both failed]") == 1


def test_render_quality_terminal_non_verbose_never_shows_flag() -> None:
    """Non-verbose Tier-1 output has no per-prompt detail at all -- the flag
    (a per-ROW marker) never appears there, matching the existing [WIN]/[TIE]/
    [LOSS] per-prompt labels which are verbose-only too.
    """
    report_mod = sys.modules[render_quality_terminal.__module__]
    buf = io.StringIO()
    console = Console(file=buf, width=200, no_color=True, highlight=False)
    original_rprint = report_mod.rprint
    original_render_console = report_mod._render_console
    report_mod.rprint = lambda *a, **k: console.print(*a, **k)  # type: ignore[attr-defined]
    report_mod._render_console = lambda: console  # type: ignore[attr-defined]
    try:
        render_quality_terminal(_both_failed_result(), verbose=False)
    finally:
        report_mod.rprint = original_rprint  # type: ignore[attr-defined]
        report_mod._render_console = original_render_console  # type: ignore[attr-defined]
    out = _ANSI_RE.sub("", buf.getvalue())
    assert "[both failed]" not in out


def test_quality_section_md_shows_both_failed_flag_on_flagged_tie_only() -> None:
    md = "\n".join(_quality_section_md(_both_failed_result()))
    assert "[TIE] [both failed]" in md
    # Exactly one flagged row (p1); p2's tie must render bare [TIE].
    assert md.count("[both failed]") == 1
