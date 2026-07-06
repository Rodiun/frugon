"""Tests for the frugon live-progress helper and its call-site wiring.

These tests pin the two load-bearing guarantees:

  1. **Gating** — progress chrome renders ONLY when stderr is a TTY, NO_COLOR is
     unset, and ``--no-progress`` was not passed.  Under any of: non-TTY,
     NO_COLOR, or ``--no-progress``, the reporter is a complete no-op (nothing
     on stderr, stdout untouched).
  2. **Per-record / per-prompt callbacks** — the pricing pass fires its
     ``(done, total)`` callback exactly once per record, and ``run_measure``
     fires its sampling/judge callbacks once per provider call.

No real network or TTY is needed: gating reads ``sys.stderr.isatty`` (patched)
and the callbacks are driven through the pure-Python code paths with LiteLLM
mocked, mirroring tests/test_measure.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from frugon._progress import (
    ProgressReporter,
    Stopwatch,
    progress_enabled,
    progress_reporter,
)
from frugon.cost import LogRecord, analyze_records

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    model: str = "gpt-4o",
    prompt: str = "hello",
    completion: str = "world",
    pt: int = 10,
    ct: int = 5,
) -> LogRecord:
    """Build a LogRecord with an explicit usage block (no tokenizer needed)."""
    return LogRecord(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        completion_text=completion,
        prompt_tokens=pt,
        completion_tokens=ct,
        timestamp=None,
        token_source="usage_block",
    )


# ---------------------------------------------------------------------------
# Gating — progress_enabled
# ---------------------------------------------------------------------------


def test_progress_enabled_tty_no_noscolor_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    assert progress_enabled(no_progress=False) is True


def test_progress_enabled_non_tty_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
    assert progress_enabled(no_progress=False) is False


def test_progress_enabled_no_progress_flag_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    assert progress_enabled(no_progress=True) is False


def test_progress_enabled_no_color_set_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr("sys.stderr.isatty", lambda: True, raising=False)
    assert progress_enabled(no_progress=False) is False


# ---------------------------------------------------------------------------
# No-op behaviour — disabled reporter writes nothing, anywhere
# ---------------------------------------------------------------------------


def test_disabled_reporter_writes_nothing_to_stdout_or_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reporter = ProgressReporter(enabled=False)
    with reporter.spinner("Reading…"):
        pass
    reporter.checkpoint("Read 10 records")
    with reporter.bar("Pricing", total=10) as task:
        for _ in range(10):
            task.advance(1)
    with reporter.counter("Sampling", total=3) as counter:
        for _ in range(3):
            counter.step("gpt-4o-mini")
    captured = capsys.readouterr()
    assert captured.out == "", f"disabled reporter wrote to stdout: {captured.out!r}"
    assert captured.err == "", f"disabled reporter wrote to stderr: {captured.err!r}"


def test_disabled_reporter_bar_advance_is_noop_but_callable() -> None:
    """A disabled bar still yields a task whose advance() is safe to call."""
    reporter = ProgressReporter(enabled=False)
    with reporter.bar("Pricing", total=0) as task:
        task.advance(1)  # must not raise even with total=0


def test_disabled_reporter_bar_relabel_is_noop_but_callable() -> None:
    """A disabled bar's relabel() is a safe no-op — no console to update."""
    reporter = ProgressReporter(enabled=False)
    with reporter.bar("Pricing", total=0) as task:
        task.relabel("Comparing 23 candidates…")  # must not raise


def test_rich_progress_task_relabel_changes_description() -> None:
    """FRG-OSS-039: _RichProgressTask.relabel() updates the Rich task's description.

    Used by the CLI to rename the still-open pricing bar to
    "Comparing N candidates…" once the per-record pricing phase completes but
    candidate comparison (an uncounted phase) is still running underneath.
    """
    from rich.console import Console
    from rich.progress import Progress, TextColumn

    from frugon._progress import _RichProgressTask

    progress = Progress(TextColumn("{task.description}"), console=Console(stderr=True))
    task_id = progress.add_task("Pricing", total=3)
    task = _RichProgressTask(progress, task_id)
    task.advance(3)
    assert progress.tasks[0].description == "Pricing"
    task.relabel("Comparing 23 candidates…")
    assert progress.tasks[0].description == "Comparing 23 candidates…"


def test_enabled_reporter_bar_relabel_writes_only_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """relabel(), like the rest of the bar chrome, never touches stdout."""
    reporter = ProgressReporter(enabled=True)
    with reporter.bar("Pricing", total=3) as task:
        for _ in range(3):
            task.advance(1)
        task.relabel("Comparing 23 candidates…")
    captured = capsys.readouterr()
    assert captured.out == "", f"relabel leaked to stdout: {captured.out!r}"


def test_progress_reporter_context_manager_gates_via_isatty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
    with progress_reporter(no_progress=False) as reporter:
        assert reporter.enabled is False


# ---------------------------------------------------------------------------
# Enabled reporter writes ONLY to stderr (never stdout)
# ---------------------------------------------------------------------------


def test_enabled_reporter_writes_only_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When enabled, every byte of chrome lands on stderr — stdout stays empty.

    This is the core invariant: the analysis result owns stdout; progress owns
    stderr.  We force-enable (bypassing the TTY gate) and drive a checkpoint +
    bar, then assert stdout captured nothing.
    """
    reporter = ProgressReporter(enabled=True)
    reporter.checkpoint("Read 56,100 records")
    with reporter.bar("Pricing", total=5) as task:
        for _ in range(5):
            task.advance(1)
    captured = capsys.readouterr()
    assert captured.out == "", f"enabled reporter leaked to stdout: {captured.out!r}"
    # The checkpoint text persists on stderr; the bar is transient (may clear),
    # so we only assert the persisted checkpoint reached stderr.
    assert "Read 56,100 records" in captured.err


def test_notice_writes_only_to_stderr_when_enabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The large-log heads-up lands on stderr only, never stdout."""
    reporter = ProgressReporter(enabled=True)
    reporter.notice("250,000 records — this may take a moment.")
    captured = capsys.readouterr()
    assert captured.out == "", f"notice leaked to stdout: {captured.out!r}"
    assert "this may take a moment" in captured.err


def test_notice_is_silent_when_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A disabled reporter prints no heads-up (piped / --no-progress / CI)."""
    reporter = ProgressReporter(enabled=False)
    reporter.notice("250,000 records — this may take a moment.")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# Pricing callback fires once per record
# ---------------------------------------------------------------------------


def test_pricing_callback_fires_once_per_record() -> None:
    """analyze_records invokes progress_cb exactly once per record, in order."""
    records = [_make_record() for _ in range(7)]
    calls: list[tuple[int, int]] = []
    analyze_records(records, progress_cb=lambda done, total: calls.append((done, total)))
    assert len(calls) == 7, f"expected 7 callback fires, got {len(calls)}"
    # done counts up 1..7; total is always the full count.
    assert [done for done, _ in calls] == [1, 2, 3, 4, 5, 6, 7]
    assert {total for _, total in calls} == {7}


def test_pricing_callback_none_is_allocation_free_path() -> None:
    """The default (progress_cb=None) path returns an identical result."""
    records = [_make_record() for _ in range(4)]
    result_no_cb = analyze_records(records)
    captured: list[tuple[int, int]] = []
    result_cb = analyze_records(
        records, progress_cb=lambda done, total: captured.append((done, total))
    )
    assert result_no_cb.total_cost == result_cb.total_cost
    assert result_no_cb.priced_calls == result_cb.priced_calls
    assert len(captured) == 4


def test_pricing_callback_empty_records_never_fires() -> None:
    calls: list[tuple[int, int]] = []
    analyze_records([], progress_cb=lambda done, total: calls.append((done, total)))
    assert calls == []


# ---------------------------------------------------------------------------
# run_measure callbacks fire once per provider call (LiteLLM mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider keys present for the run_measure callback tests (no real calls)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


def _mock_litellm() -> MagicMock:
    mock = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "VERDICT: TIE"
    mock.completion.return_value = resp
    return mock


@pytest.mark.usefixtures("_keys")
def test_run_measure_sample_cb_fires_once_per_prompt() -> None:
    """sample_cb fires once per sampled PROMPT — the same unit as the header.

    The baseline + each candidate are still called underneath; only the
    progress UNIT is the prompt, so the live "Sampling prompt n/N" counter and
    the "N prompt(s)" result header always agree.
    """
    from frugon.measure import run_measure

    records = [_make_record(prompt=f"p{i}") for i in range(3)]
    seen: list[tuple[int, int, str]] = []
    with patch("frugon.measure._import_litellm", return_value=_mock_litellm()):
        run_measure(
            records,
            current_model="gpt-4o",
            candidates=["gpt-4o-mini"],
            n_samples=3,
            sample_cb=lambda done, total, label: seen.append((done, total, label)),
        )
    # 3 prompts → 3 sampling steps (NOT 6 provider calls).
    assert len(seen) == 3
    # total is the prompt count, matching the "N prompt(s)" header.
    assert {total for _, total, _ in seen} == {3}
    # done advances 0, 1, 2 as each prompt's sampling begins.
    assert [done for done, _, _ in seen] == [0, 1, 2]
    # The label names the candidate(s) under comparison.
    assert {label for _, _, label in seen} == {"gpt-4o-mini"}


@pytest.mark.usefixtures("_keys")
def test_run_measure_sample_cb_label_joins_multiple_candidates() -> None:
    """With several candidates, the per-prompt label lists them comma-joined."""
    from frugon.measure import run_measure

    records = [_make_record(prompt=f"p{i}") for i in range(2)]
    seen: list[tuple[int, int, str]] = []
    with patch("frugon.measure._import_litellm", return_value=_mock_litellm()):
        run_measure(
            records,
            current_model="gpt-4o",
            candidates=["gpt-4o-mini", "claude-3-haiku-20240307"],
            n_samples=2,
            sample_cb=lambda done, total, label: seen.append((done, total, label)),
        )
    # Still one step per prompt regardless of candidate count.
    assert len(seen) == 2
    assert {total for _, total, _ in seen} == {2}
    assert {label for _, _, label in seen} == {
        "gpt-4o-mini, claude-3-haiku-20240307"
    }


@pytest.mark.usefixtures("_keys")
def test_run_measure_judge_cb_fires_once_per_prompt() -> None:
    """judge_cb fires once per sampled PROMPT when use_judge=True.

    Each candidate is still judged underneath; the progress unit is the prompt,
    so the "Judging n/N" counter counts the same prompts as the header.
    """
    from frugon.measure import run_measure

    records = [_make_record(prompt=f"p{i}") for i in range(2)]
    seen: list[tuple[int, int, str]] = []
    with patch("frugon.measure._import_litellm", return_value=_mock_litellm()):
        run_measure(
            records,
            current_model="gpt-4o",
            candidates=["gpt-4o-mini", "claude-3-haiku-20240307"],
            n_samples=2,
            use_judge=True,
            judge_cb=lambda done, total, label: seen.append((done, total, label)),
        )
    # 2 prompts → 2 judging steps (NOT 4 per-candidate comparisons).
    assert len(seen) == 2
    assert {total for _, total, _ in seen} == {2}
    assert [done for done, _, _ in seen] == [0, 1]


@pytest.mark.usefixtures("_keys")
def test_run_measure_no_callbacks_still_runs() -> None:
    """Callbacks are optional; omitting them does not change the result shape."""
    from frugon.measure import run_measure

    records = [_make_record()]
    with patch("frugon.measure._import_litellm", return_value=_mock_litellm()):
        result = run_measure(
            records, current_model="gpt-4o", candidates=["gpt-4o-mini"], n_samples=1
        )
    assert result.samples_taken == 1
    assert result.candidates == ["gpt-4o-mini"]


# ---------------------------------------------------------------------------
# Stopwatch
# ---------------------------------------------------------------------------


def test_stopwatch_measures_nonnegative_elapsed() -> None:
    with Stopwatch() as sw:
        sum(range(1000))
    assert sw.elapsed >= 0.0


# ---------------------------------------------------------------------------
# Counter step labels (enabled reporter, stderr only)
# ---------------------------------------------------------------------------


def test_counter_step_writes_only_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    reporter = ProgressReporter(enabled=True)
    with reporter.counter("Sampling", total=2) as counter:
        counter.step("gpt-4o-mini")
        counter.step("gpt-4o")
    captured = capsys.readouterr()
    assert captured.out == "", f"counter leaked to stdout: {captured.out!r}"


def test_counter_description_reads_prefix_count_label() -> None:
    """The rendered description is "<prefix> n/total · <label>".

    With the CLI's "Sampling prompt" prefix the user therefore sees
    "Sampling prompt 1/5 · gpt-4o-mini" — the same prompt count as the
    "5 prompt(s)" result header.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from frugon._progress import _RichStepCounter

    progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("{task.description}"),
        console=Console(stderr=True),
    )
    task_id = progress.add_task("Sampling prompt", total=5)
    counter = _RichStepCounter(progress, task_id, "Sampling prompt", 5)
    counter.step("gpt-4o-mini")
    assert progress.tasks[0].description == "Sampling prompt 1/5 · gpt-4o-mini"
    counter.step("gpt-4o-mini")
    assert progress.tasks[0].description == "Sampling prompt 2/5 · gpt-4o-mini"


def test_counter_step_flips_finished_on_last_step() -> None:
    """Regression for FRG-OSS-008: step() must report completed == total on
    the FINAL call so Rich's task.finished (completed >= total) flips True.

    Before the fix, ``completed=self._done - 1`` meant a 5-total counter never
    exceeded completed=4 even after all 5 step() calls, so `.finished` stayed
    False forever.  Masked in production because the call sites all use
    ``transient=True`` progress bars torn down unconditionally by the `with
    progress:` block — but the counter's own completion bookkeeping was wrong
    regardless of what tears the bar down.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from frugon._progress import _RichStepCounter

    progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("{task.description}"),
        console=Console(stderr=True),
    )
    task_id = progress.add_task("Sampling prompt", total=5)
    counter = _RichStepCounter(progress, task_id, "Sampling prompt", 5)

    for i in range(1, 5):
        counter.step(f"model-{i}")
        assert progress.tasks[0].completed == i
        assert not progress.tasks[0].finished

    counter.step("model-5")
    assert progress.tasks[0].completed == 5
    assert progress.tasks[0].finished
