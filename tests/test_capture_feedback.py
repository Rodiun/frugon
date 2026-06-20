"""Tests for the capture feedback stream (verbosity parameter).

Covers:
- quiet: no per-call output to stdout
- verbose: one newline-terminated line per call, with timestamp/model/tokens/status
- normal + TTY: carriage-return in-place counter line
- normal + non-TTY: no output (\\r rewrites must not reach non-interactive stdout)
- invalid verbosity: ValueError raised
- counter increments correctly across multiple calls

All tests exercise _FeedbackStream.on_call directly — no real sockets opened.
"""

from __future__ import annotations

import io
import sys
import threading
from typing import Any

import pytest

from frugon.capture import _FeedbackStream

# ---------------------------------------------------------------------------
# Synthetic call record matching _build_record output shape
# ---------------------------------------------------------------------------

_RECORD: dict[str, Any] = {
    "model": "gpt-4o",
    "request": {"messages": [{"role": "user", "content": "hi"}]},
    "response": {},
    "usage": {"prompt_tokens": 10, "completion_tokens": 22, "total_tokens": 32},
    "timestamp": "2025-01-01T00:00:00Z",
}

_STATUS = 200


# ---------------------------------------------------------------------------
# quiet mode
# ---------------------------------------------------------------------------


def test_feedback_quiet_produces_no_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: verbosity="quiet", capturing stdout.
    Act: call on_call once.
    Assert: nothing written to stdout.
    """
    # Arrange
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    stream = _FeedbackStream("quiet")

    # Act
    stream.on_call(_RECORD, _STATUS)

    # Assert
    assert buf.getvalue() == "", (
        "quiet mode must produce no per-call output"
    )


def test_feedback_quiet_produces_no_output_after_many_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="quiet".
    Act: call on_call five times.
    Assert: total stdout output is empty.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("quiet")

    for _ in range(5):
        stream.on_call(_RECORD, _STATUS)

    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# verbose mode
# ---------------------------------------------------------------------------


def test_feedback_verbose_writes_one_line_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose", capturing stdout.
    Act: call on_call three times.
    Assert: exactly three newline-terminated lines emitted.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    for _ in range(3):
        stream.on_call(_RECORD, _STATUS)

    lines = buf.getvalue().splitlines()
    assert len(lines) == 3, f"Expected 3 lines for 3 calls, got {len(lines)}"


def test_feedback_verbose_line_contains_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose".
    Act: on_call with a record that has a known timestamp.
    Assert: timestamp appears in the emitted line.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    stream.on_call(_RECORD, _STATUS)

    assert "2025-01-01T00:00:00Z" in buf.getvalue(), (
        "verbose output must include the record timestamp"
    )


def test_feedback_verbose_line_contains_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose".
    Act: on_call with model="gpt-4o".
    Assert: model name appears in the emitted line.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    stream.on_call(_RECORD, _STATUS)

    assert "gpt-4o" in buf.getvalue(), "verbose output must include the model name"


def test_feedback_verbose_line_contains_prompt_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose", record with prompt_tokens=10.
    Act: on_call.
    Assert: prompt token count appears in the emitted line.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    stream.on_call(_RECORD, _STATUS)

    assert "prompt=10" in buf.getvalue(), (
        "verbose output must include prompt token count as 'prompt=<N>'"
    )


def test_feedback_verbose_line_contains_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose", record with completion_tokens=22.
    Act: on_call.
    Assert: completion token count appears in the emitted line.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    stream.on_call(_RECORD, _STATUS)

    assert "completion=22" in buf.getvalue(), (
        "verbose output must include completion token count as 'completion=<N>'"
    )


def test_feedback_verbose_line_contains_upstream_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose", status=200.
    Act: on_call.
    Assert: HTTP status appears in the emitted line.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    stream.on_call(_RECORD, 200)

    assert "status=200" in buf.getvalue(), (
        "verbose output must include the upstream HTTP status"
    )


def test_feedback_verbose_lines_are_newline_terminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose".
    Act: on_call once.
    Assert: output ends with a newline (required for logfile scrolling).
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    stream.on_call(_RECORD, _STATUS)

    assert buf.getvalue().endswith("\n"), (
        "verbose output must be newline-terminated (not \\r)"
    )


def test_feedback_verbose_no_carriage_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose".
    Act: on_call.
    Assert: no carriage-return character in output (must not use \\r in verbose mode).
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    stream.on_call(_RECORD, _STATUS)

    assert "\r" not in buf.getvalue(), (
        "verbose output must not contain carriage-return characters"
    )


# ---------------------------------------------------------------------------
# normal mode + TTY
# ---------------------------------------------------------------------------


def test_feedback_normal_tty_writes_carriage_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="normal", stdout.isatty() returns True.
    Act: on_call once.
    Assert: output contains a carriage-return (in-place update pattern).
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(buf, "isatty", lambda: True)

    stream = _FeedbackStream("normal")
    stream.on_call(_RECORD, _STATUS)

    assert "\r" in buf.getvalue(), (
        "normal+TTY must emit a carriage-return for in-place counter update"
    )


def test_feedback_normal_tty_line_contains_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="normal", isatty=True.
    Act: on_call with model="gpt-4o".
    Assert: model name appears in the in-place counter line.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(buf, "isatty", lambda: True)

    stream = _FeedbackStream("normal")
    stream.on_call(_RECORD, _STATUS)

    assert "gpt-4o" in buf.getvalue()


def test_feedback_normal_tty_line_contains_total_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="normal", isatty=True, total_tokens=32.
    Act: on_call.
    Assert: total token count appears in the counter line.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(buf, "isatty", lambda: True)

    stream = _FeedbackStream("normal")
    stream.on_call(_RECORD, _STATUS)

    assert "32" in buf.getvalue(), (
        "normal+TTY counter line must include the total token count"
    )


def test_feedback_normal_tty_counter_increments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="normal", isatty=True.
    Act: call on_call three times, capturing all writes.
    Assert: counter values 1, 2, 3 each appear in one of the emitted writes.
    """
    writes: list[str] = []

    class _TrackingStream(io.StringIO):
        def isatty(self) -> bool:  # type: ignore[override]
            return True

        def write(self, s: str) -> int:  # type: ignore[override]
            writes.append(s)
            return len(s)

    buf = _TrackingStream()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("normal")

    for _ in range(3):
        stream.on_call(_RECORD, _STATUS)

    combined = "".join(writes)
    # Counter values 1..3 must each appear somewhere in the output
    for n in (1, 2, 3):
        assert str(n) in combined, (
            f"Counter value {n} not found in normal+TTY output after {n} calls"
        )


# ---------------------------------------------------------------------------
# normal mode + non-TTY (TTY degradation — critical)
# ---------------------------------------------------------------------------


def test_feedback_normal_non_tty_produces_no_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="normal", stdout.isatty() returns False (piped/redirected).
    Act: on_call once.
    Assert: nothing written to stdout — \\r rewrites must not reach non-TTY output.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(buf, "isatty", lambda: False)

    stream = _FeedbackStream("normal")
    stream.on_call(_RECORD, _STATUS)

    assert buf.getvalue() == "", (
        "normal+non-TTY must produce no output to avoid corrupting log files"
    )


def test_feedback_normal_non_tty_no_carriage_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="normal", isatty=False.
    Act: on_call five times.
    Assert: no carriage-return characters anywhere in stdout.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(buf, "isatty", lambda: False)

    stream = _FeedbackStream("normal")
    for _ in range(5):
        stream.on_call(_RECORD, _STATUS)

    assert "\r" not in buf.getvalue(), (
        "\\r must never appear in stdout when isatty() is False"
    )


# ---------------------------------------------------------------------------
# Invalid verbosity
# ---------------------------------------------------------------------------


def test_feedback_invalid_verbosity_raises_value_error() -> None:
    """Arrange: invalid verbosity string.
    Act: construct _FeedbackStream.
    Assert: ValueError raised with the bad value in the message.
    """
    with pytest.raises(ValueError, match="bogus"):
        _FeedbackStream("bogus")


# ---------------------------------------------------------------------------
# Thread safety — concurrent on_call does not corrupt the counter
# ---------------------------------------------------------------------------


def test_feedback_verbose_thread_safe_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose", 20 threads each calling on_call 5 times.
    Act: run all threads concurrently.
    Assert: exactly 100 lines emitted; no data corruption or missing writes.
    """
    lock = threading.Lock()
    lines_written: list[str] = []

    class _SafeStream(io.StringIO):
        def write(self, s: str) -> int:  # type: ignore[override]
            with lock:
                lines_written.append(s)
            return len(s)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdout", _SafeStream())

    stream = _FeedbackStream("verbose")
    threads = [
        threading.Thread(
            target=lambda: [stream.on_call(_RECORD, 200) for _ in range(5)]
        )
        for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    full_output = "".join(lines_written)
    line_count = full_output.count("\n")
    assert line_count == 100, (
        f"Expected 100 newline-terminated verbose lines from 20x5 concurrent calls, "
        f"got {line_count}"
    )


# ---------------------------------------------------------------------------
# Empty/missing fields degrade gracefully (no KeyError / TypeError)
# ---------------------------------------------------------------------------


def test_feedback_verbose_empty_record_no_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="verbose", empty record dict.
    Act: on_call.
    Assert: no exception raised; one line written containing 'unknown' model fallback.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    stream = _FeedbackStream("verbose")

    stream.on_call({}, 200)

    output = buf.getvalue()
    assert output.endswith("\n"), "verbose must still emit a newline-terminated line"
    assert "unknown" in output, "empty model must fall back to 'unknown'"


def test_feedback_normal_tty_empty_record_no_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: verbosity="normal", isatty=True, empty record.
    Act: on_call.
    Assert: no exception raised; \\r line emitted.
    """
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.setattr(buf, "isatty", lambda: True)
    stream = _FeedbackStream("normal")

    stream.on_call({}, 200)  # must not raise

    assert "\r" in buf.getvalue()
