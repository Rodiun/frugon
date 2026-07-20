"""``--judge`` with too few ``--samples`` produces an unreliable verdict.

The default ``--samples`` is 10 (kept as-is: silently bumping it to 25 would
surprise scripted callers and inflate cost without consent). Instead, frugon
warns on stderr when a judge run is about to use fewer than 20 samples --
the help text already recommends 25-30 for a confident verdict.

These tests exercise only the pure-argument guard, which fires BEFORE any
path resolution -- so an invocation against a missing log file is enough to
observe the warning (or its absence) without mocking the measure engine.
"""

from __future__ import annotations

import pathlib

from typer.testing import CliRunner

from frugon.cli import app

runner = CliRunner()

_ARGS_ENV = {"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"}


def test_judge_below_20_samples_warns_on_stderr(tmp_path: pathlib.Path) -> None:
    """Arrange: --judge --measure with --samples 5 (below the 20 threshold).
    Act: invoke analyze against a missing log (path resolution happens after
         the guard, so the run still exits non-zero, but the guard fires first).
    Assert: a warning naming the sample count appears on stderr.
    """
    missing = tmp_path / "missing.jsonl"

    result = runner.invoke(
        app,
        ["analyze", str(missing), "--measure", "--judge", "--samples", "5"],
        env=_ARGS_ENV,
    )

    assert "5" in result.stderr, f"sample count not named in stderr: {result.stderr!r}"
    assert "25-30" in result.stderr, f"guidance missing from stderr: {result.stderr!r}"


def test_judge_at_19_samples_warns(tmp_path: pathlib.Path) -> None:
    """Boundary: 19 is still below 20, so the warning fires."""
    missing = tmp_path / "missing.jsonl"

    result = runner.invoke(
        app,
        ["analyze", str(missing), "--measure", "--judge", "--samples", "19"],
        env=_ARGS_ENV,
    )

    assert "19" in result.stderr, f"expected warning at 19 samples: {result.stderr!r}"


def test_judge_at_20_samples_does_not_warn(tmp_path: pathlib.Path) -> None:
    """Boundary: 20 meets the threshold, so no warning fires."""
    missing = tmp_path / "missing.jsonl"

    result = runner.invoke(
        app,
        ["analyze", str(missing), "--measure", "--judge", "--samples", "20"],
        env=_ARGS_ENV,
    )

    assert result.stderr == "", f"unexpected warning at 20 samples: {result.stderr!r}"


def test_judge_above_20_samples_does_not_warn(tmp_path: pathlib.Path) -> None:
    """Arrange: --judge --measure with --samples 30 (comfortably above threshold).
    Assert: no warning on stderr.
    """
    missing = tmp_path / "missing.jsonl"

    result = runner.invoke(
        app,
        ["analyze", str(missing), "--measure", "--judge", "--samples", "30"],
        env=_ARGS_ENV,
    )

    assert result.stderr == "", f"unexpected warning at 30 samples: {result.stderr!r}"


def test_measure_without_judge_never_warns_regardless_of_samples(tmp_path: pathlib.Path) -> None:
    """Arrange: --measure alone (no --judge) with a low --samples.
    Assert: no warning -- there is no judge verdict to be noisy about.
    """
    missing = tmp_path / "missing.jsonl"

    result = runner.invoke(
        app,
        ["analyze", str(missing), "--measure", "--samples", "5"],
        env=_ARGS_ENV,
    )

    assert result.stderr == "", f"unexpected warning without --judge: {result.stderr!r}"


def test_judge_default_samples_warns(tmp_path: pathlib.Path) -> None:
    """The default --samples (10) is below 20, so a bare --judge --measure run
    (no explicit --samples) still warns -- the guard reads the resolved value,
    not just an explicit flag.
    """
    missing = tmp_path / "missing.jsonl"

    result = runner.invoke(
        app,
        ["analyze", str(missing), "--measure", "--judge"],
        env=_ARGS_ENV,
    )

    assert "10" in result.stderr, f"expected default-samples warning: {result.stderr!r}"
