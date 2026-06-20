"""CLI ordering tests for --report + --measure (the reorder fix).

The bug this guards: when BOTH --report and --measure are set, the report used
to be written BEFORE run_measure ran, so it could never carry the judge verdict.
The fix runs --measure FIRST and passes the resulting MeasureResult into the
report renderer.  These tests assert, via mocks (no provider call, no real
spend):

  1. ordering — run_measure is called BEFORE the report renderer, and the
     renderer receives the exact MeasureResult run_measure produced;
  2. report-without-measure — the renderer is called with measure_result=None;
  3. measure-without-report — no renderer is called at all (terminal only).
"""

from __future__ import annotations

import json
import pathlib

import pytest
from typer.testing import CliRunner

from frugon import measure
from frugon.cli import app
from frugon.measure import Comparison, MeasureResult, SampledOutput, Tier1Tally

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


def _fake_measure_result() -> MeasureResult:
    comp = Comparison(
        record=__import__("frugon.cost", fromlist=["LogRecord"]).LogRecord(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": "classify: spam?"}],
            completion_text="no",
            prompt_tokens=20,
            completion_tokens=3,
            timestamp=None,
        ),
        current_output=SampledOutput(model="gpt-4-turbo", content="no"),
        candidate_outputs=[SampledOutput(model="gpt-4o-mini", content="not spam")],
        verdicts=["loss"],
    )
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[comp],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=1, ties=0)],
    )


def _patch_measure_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the [measure] extra + key checks pass and skip the network."""
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setattr(measure, "verify_measure_prerequisites", lambda *_a, **_k: None)
    monkeypatch.setattr("frugon.report.render_quality_terminal", lambda *_a, **_k: None)


def test_report_written_after_measure_and_receives_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """--report + --measure: run_measure runs BEFORE the renderer, which gets it.

    A shared call log records the order of (run_measure, renderer) and captures
    the measure_result the renderer was handed — proving both the ordering and
    the threading in one assertion.
    """
    _patch_measure_engine(monkeypatch)
    sentinel = _fake_measure_result()
    calls: list[str] = []
    seen_measure_result: list[object] = []

    def _fake_run_measure(*_a: object, **_k: object) -> MeasureResult:
        calls.append("measure")
        return sentinel

    def _fake_render_md(
        _result: object,
        _path: object,
        *,
        measure_result: object = None,
        **_kw: object,
    ) -> None:
        calls.append("render")
        seen_measure_result.append(measure_result)
        # Still write a file so the CLI's "Report written to" line is honest.
        pathlib.Path(_path).write_text("stub", encoding="utf-8")

    monkeypatch.setattr(measure, "run_measure", _fake_run_measure)
    # Pin --report-style v1 so the patched v1 markdown renderer is the dispatch
    # target.  The default is v2; this test only asserts the measure→render
    # ordering and result threading, which are renderer-agnostic.
    monkeypatch.setattr("frugon.report.render_markdown", _fake_render_md)

    log = _write_log(tmp_path / "log.jsonl")
    report = tmp_path / "out.md"

    # Act
    result = runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--measure",
            "--judge",
            "--report",
            str(report),
            "--report-style",
            "v1",
            "--no-progress",
        ],
        env=_WIDE_ENV,
        catch_exceptions=True,
    )

    # Assert
    assert result.exit_code == 0, result.output
    assert calls == ["measure", "render"], (
        f"report must be rendered AFTER measure; got order {calls}"
    )
    assert seen_measure_result == [sentinel], (
        "renderer did not receive the MeasureResult run_measure produced"
    )


def test_report_without_measure_passes_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """--report without --measure → renderer called once with measure_result=None."""
    seen: list[object] = []

    def _fake_render_md(
        _result: object,
        _path: object,
        *,
        measure_result: object = None,
        **_kw: object,
    ) -> None:
        seen.append(measure_result)
        pathlib.Path(_path).write_text("stub", encoding="utf-8")

    # Pin --report-style v1 so the patched v1 markdown renderer is the dispatch
    # target (the default is now v2).
    monkeypatch.setattr("frugon.report.render_markdown", _fake_render_md)

    log = _write_log(tmp_path / "log.jsonl")
    report = tmp_path / "out.md"

    result = runner.invoke(
        app,
        ["analyze", str(log), "--report", str(report), "--report-style", "v1", "--no-progress"],
        env={"COLUMNS": "200", "TERM": "dumb"},
        catch_exceptions=True,
    )

    assert result.exit_code == 0, result.output
    assert seen == [None], f"expected measure_result=None, got {seen}"


def test_measure_without_report_calls_no_renderer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """--measure without --report → no report renderer is invoked (terminal only)."""
    _patch_measure_engine(monkeypatch)
    monkeypatch.setattr(measure, "run_measure", lambda *_a, **_k: _fake_measure_result())

    rendered: list[str] = []
    for name in ("render_markdown", "render_markdown_v2", "render_html", "render_html_v2"):
        monkeypatch.setattr(
            f"frugon.report.{name}",
            lambda *_a, _n=name, **_k: rendered.append(_n),
        )

    log = _write_log(tmp_path / "log.jsonl")

    result = runner.invoke(
        app,
        ["analyze", str(log), "--measure", "--judge", "--no-progress"],
        env=_WIDE_ENV,
        catch_exceptions=True,
    )

    assert result.exit_code == 0, result.output
    assert rendered == [], f"no report renderer should run without --report; got {rendered}"
