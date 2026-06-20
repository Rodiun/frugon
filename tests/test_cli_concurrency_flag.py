"""CLI surface tests for the ``--concurrency`` flag on ``analyze``.

These assert the public contract of the flag WITHOUT making any provider call:

  * ``analyze --help`` advertises ``--concurrency`` with its default (5) and the
    documented help text;
  * an invalid value (``--concurrency 0``) is rejected with a clean, non-zero
    exit (typer's ``min=1`` boundary), not a traceback; and
  * a real ``--measure`` invocation THREADS the flag value through to
    ``run_measure(..., concurrency=...)`` — captured via a stubbed engine so no
    network egress occurs.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from typer.testing import CliRunner

from frugon import measure
from frugon.cli import app
from frugon.measure import MeasureResult

from .conftest import help_text

runner = CliRunner()

_WIDE_ENV = {"COLUMNS": "200", "TERM": "dumb", "OPENAI_API_KEY": "sk-test-not-real"}


def _priced_row(model: str) -> dict[str, object]:
    return {
        "model": model,
        "request": {"messages": [{"role": "user", "content": "classify: spam?"}]},
        "response": {"choices": [{"message": {"content": "no"}}]},
        "usage": {"prompt_tokens": 20, "completion_tokens": 3},
    }


def _write_log(path: pathlib.Path) -> pathlib.Path:
    rows = [_priced_row("gpt-4-turbo") for _ in range(6)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_analyze_help_advertises_concurrency_flag_and_default() -> None:
    """``analyze --help`` shows ``--concurrency``, its INTEGER type, the default
    (5), and the per-stage help text (sampling fans across providers; judging is
    capped).
    """
    # Render-independent help text (ANSI stripped, box borders flattened, whitespace
    # collapsed) so the contract holds under CI's forced-terminal 80-col rendering.
    out = help_text("analyze")
    assert "--concurrency" in out
    # The default is surfaced (show_default=True) ...
    assert "5" in out
    # ... and the documented help text is present.
    assert "Max concurrent provider calls per stage" in out
    assert "sampling fans across providers" in out
    assert "judging is capped to protect the single judge endpoint" in out


def test_analyze_rejects_zero_concurrency_cleanly(tmp_path: pathlib.Path) -> None:
    """``--concurrency 0`` is rejected by the ``min=1`` boundary with a non-zero
    exit and a clean usage error — never a Python traceback.
    """
    log = _write_log(tmp_path / "log.jsonl")
    result = runner.invoke(
        app,
        ["analyze", str(log), "--concurrency", "0", "--no-progress"],
        env=_WIDE_ENV,
    )
    assert result.exit_code != 0
    # Typer renders an Invalid value / range error; assert it is a clean usage
    # message, not a traceback.
    assert "Traceback" not in result.output
    combined = result.output.lower()
    assert "concurrency" in combined or "invalid" in combined


def test_analyze_rejects_negative_concurrency_cleanly(tmp_path: pathlib.Path) -> None:
    """``--concurrency -3`` is likewise rejected cleanly (the ``>= 1`` contract)."""
    log = _write_log(tmp_path / "log.jsonl")
    result = runner.invoke(
        app,
        ["analyze", str(log), "--concurrency", "-3", "--no-progress"],
        env=_WIDE_ENV,
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_analyze_threads_concurrency_into_run_measure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A ``--measure --concurrency 9`` run reaches ``run_measure`` with
    ``concurrency=9`` — proving the flag is wired end-to-end (stubbed engine; no
    provider call).
    """
    # Make the [measure] extra + key checks pass and skip the renderer/network.
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setattr(measure, "verify_measure_prerequisites", lambda *_a, **_k: None)
    monkeypatch.setattr("frugon.report.render_quality_terminal", lambda *_a, **_k: None)

    seen_concurrency: list[int] = []

    def _fake_run_measure(*_a: object, **kwargs: object) -> MeasureResult:
        seen_concurrency.append(int(kwargs["concurrency"]))  # type: ignore[call-overload]
        return MeasureResult(
            samples_requested=1,
            samples_taken=1,
            current_model="gpt-4-turbo",
            candidates=["gpt-4o-mini"],
            comparisons=[],
        )

    monkeypatch.setattr(measure, "run_measure", _fake_run_measure)

    log = _write_log(tmp_path / "log.jsonl")
    result = runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--measure",
            "--candidates",
            "gpt-4o-mini",
            "--concurrency",
            "9",
            "--no-progress",
        ],
        env=_WIDE_ENV,
        catch_exceptions=True,
    )

    assert result.exit_code == 0, result.output
    assert seen_concurrency == [9], (
        f"run_measure received concurrency={seen_concurrency}, expected [9] "
        f"-- the --concurrency flag was not threaded through"
    )


def test_analyze_default_concurrency_threads_five(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """With no ``--concurrency`` flag, ``run_measure`` receives the default (5)."""
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setattr(measure, "verify_measure_prerequisites", lambda *_a, **_k: None)
    monkeypatch.setattr("frugon.report.render_quality_terminal", lambda *_a, **_k: None)

    seen_concurrency: list[int] = []

    def _fake_run_measure(*_a: object, **kwargs: object) -> MeasureResult:
        seen_concurrency.append(int(kwargs["concurrency"]))  # type: ignore[call-overload]
        return MeasureResult(
            samples_requested=1,
            samples_taken=1,
            current_model="gpt-4-turbo",
            candidates=["gpt-4o-mini"],
            comparisons=[],
        )

    monkeypatch.setattr(measure, "run_measure", _fake_run_measure)

    log = _write_log(tmp_path / "log.jsonl")
    result = runner.invoke(
        app,
        ["analyze", str(log), "--measure", "--candidates", "gpt-4o-mini", "--no-progress"],
        env=_WIDE_ENV,
        catch_exceptions=True,
    )

    assert result.exit_code == 0, result.output
    assert seen_concurrency == [5]
