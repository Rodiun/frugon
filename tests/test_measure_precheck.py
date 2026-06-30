"""Tests for the fail-fast --measure prerequisite pre-check.

These cover the three UX hardening behaviours:

1. A --measure / --judge run verifies its prerequisites (LiteLLM importable +
   provider keys present) BEFORE the expensive cost analysis runs, using a
   cheap distinct-model scan.
2. The two EXPECTED, actionable failures — missing [measure] extra and missing
   provider key — are rendered as clean, framed messages with NO Python
   traceback, exit code 1.
3. The pre-check ordering: the key check fires before analyze_logs processes
   the log.

All tests run in full isolation: no network, no real provider call.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import re

import pytest
from typer.testing import CliRunner

import frugon.cli as cli
import frugon.cost as cost
import frugon.measure as measure
from frugon.cli import app
from frugon.cost import scan_models
from frugon.measure import (
    MissingProviderKeyError,
    verify_measure_prerequisites,
)

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
_WIDE_ENV = {"COLUMNS": "200", "TERM": "dumb"}


def _clean(text: str) -> str:
    """Strip ANSI escape codes so substring assertions are renderer-independent."""
    return _ANSI_RE.sub("", text)


def _write_log(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """Write *rows* as JSONL to *path* and return it."""
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _priced_row(model: str) -> dict[str, object]:
    """A minimal priced log row with an explicit usage block."""
    return {
        "model": model,
        "request": {"messages": [{"role": "user", "content": "hi"}]},
        "response": {"choices": [{"message": {"content": "hello"}}]},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


# ---------------------------------------------------------------------------
# scan_models — the cheap distinct-model scan
# ---------------------------------------------------------------------------


def test_scan_models_returns_distinct_and_dominant(tmp_path: pathlib.Path) -> None:
    # Arrange — gpt-4-turbo appears 3×, gpt-4o once.
    log = _write_log(
        tmp_path / "log.jsonl",
        [
            _priced_row("gpt-4-turbo"),
            _priced_row("gpt-4o"),
            _priced_row("gpt-4-turbo"),
            _priced_row("gpt-4-turbo"),
        ],
    )

    # Act
    distinct, dominant = scan_models(log)

    # Assert
    assert distinct == ["gpt-4-turbo", "gpt-4o"], distinct
    assert dominant == "gpt-4-turbo", dominant


def test_scan_models_empty_log_returns_none_dominant(tmp_path: pathlib.Path) -> None:
    # Arrange — no usable model field on any line.
    log = _write_log(tmp_path / "log.jsonl", [{"no_model": True}])

    # Act
    distinct, dominant = scan_models(log)

    # Assert
    assert distinct == []
    assert dominant is None


def test_scan_models_reads_gzip(tmp_path: pathlib.Path) -> None:
    # Arrange — gzip-compressed log (mirrors the bundled --demo fixture).
    raw = json.dumps(_priced_row("gpt-4o-mini")) + "\n"
    gz = tmp_path / "log.jsonl.gz"
    gz.write_bytes(gzip.compress(raw.encode("utf-8")))

    # Act
    distinct, dominant = scan_models(gz)

    # Assert
    assert distinct == ["gpt-4o-mini"]
    assert dominant == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# verify_measure_prerequisites — import-then-key ordering
# ---------------------------------------------------------------------------


def test_verify_prerequisites_raises_import_error_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange — simulate the [measure] extra being absent.
    def _boom() -> object:
        raise ImportError("The --measure flag requires LiteLLM.")

    monkeypatch.setattr(measure, "_import_litellm", _boom)

    # Act / Assert — import check fires first, before any key check.
    with pytest.raises(ImportError):
        verify_measure_prerequisites(["gpt-4o"])


def test_verify_prerequisites_raises_missing_key_with_structured_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange — extra present, but the OpenAI key is absent.
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Act / Assert
    with pytest.raises(MissingProviderKeyError) as excinfo:
        verify_measure_prerequisites(["gpt-4o"])

    assert excinfo.value.missing_vars == ["OPENAI_API_KEY"], excinfo.value.missing_vars


# ---------------------------------------------------------------------------
# CLI — missing extra renders a clean framed message, no traceback, exit 1
# ---------------------------------------------------------------------------


def test_measure_missing_extra_clean_message_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange — pretend the [measure] extra is uninstalled (the import guard
    # path), even though it is installed in this venv.
    def _boom() -> object:
        raise ImportError("The --measure flag requires LiteLLM.")

    monkeypatch.setattr(measure, "_import_litellm", _boom)

    # Act — catch_exceptions=True so an uncaught traceback would surface in
    # result.exception; we assert it does NOT.
    result = runner.invoke(
        app, ["analyze", "--demo", "--measure"], env=_WIDE_ENV, catch_exceptions=True
    )

    # Assert
    assert result.exit_code == 1, f"expected exit 1, got {result.exit_code}"
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"unexpected traceback surfaced: {result.exception!r}"
    )
    out = _clean(result.output)
    assert "Traceback" not in out, f"traceback leaked:\n{out}"
    assert "measure" in out, out
    assert "LiteLLM" in out, out
    assert "frugon[measure]" in out, out


# ---------------------------------------------------------------------------
# CLI — missing key renders a clean framed message, BEFORE analysis, exit 1
# ---------------------------------------------------------------------------


def test_measure_missing_key_clean_message_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange — extra importable, but no OpenAI key.
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Act
    result = runner.invoke(
        app, ["analyze", "--demo", "--measure"], env=_WIDE_ENV, catch_exceptions=True
    )

    # Assert
    assert result.exit_code == 1, f"expected exit 1, got {result.exit_code}"
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"unexpected traceback surfaced: {result.exception!r}"
    )
    out = _clean(result.output)
    assert "Traceback" not in out, f"traceback leaked:\n{out}"
    assert "OPENAI_API_KEY" in out, out
    assert "your own provider" in out, out


def test_measure_key_check_fires_before_cost_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The expensive cost analysis must NOT run when the key pre-check fails.

    This is the core ordering guarantee: the user learns about a missing key
    within ~1s, before the full log is parsed and priced.  The CLI's analysis
    entry point is analyze_records (driven through a progress-aware read+price
    pass); spying on it proves the priced pass never starts.
    """
    # Arrange — extra importable, key absent, real small log.
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    log = _write_log(tmp_path / "log.jsonl", [_priced_row("gpt-4-turbo")])

    analyze_calls: list[object] = []
    real_analyze = cost.analyze_records

    def _spy_analyze(*args: object, **kwargs: object) -> object:
        analyze_calls.append(args)
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(cost, "analyze_records", _spy_analyze)

    # Act
    result = runner.invoke(
        app, ["analyze", str(log), "--measure"], env=_WIDE_ENV, catch_exceptions=True
    )

    # Assert — exited on the key check, with the cost analysis never invoked.
    assert result.exit_code == 1, f"expected exit 1, got {result.exit_code}"
    assert analyze_calls == [], (
        "cost analysis ran before the key pre-check — ordering regression"
    )
    assert "OPENAI_API_KEY" in _clean(result.output)


def test_measure_passes_precheck_when_keys_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """When the extra + keys are present, the pre-check passes and analysis runs."""
    # Arrange — extra importable, key present, run_measure stubbed so no real
    # provider call is made.
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    log = _write_log(tmp_path / "log.jsonl", [_priced_row("gpt-4-turbo")])

    analyze_calls: list[object] = []
    real_analyze = cost.analyze_records

    def _spy_analyze(*args: object, **kwargs: object) -> object:
        analyze_calls.append(args)
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(cost, "analyze_records", _spy_analyze)

    # Stub run_measure so the test makes no outbound call.
    sentinel = object()

    def _fake_run_measure(*_args: object, **_kwargs: object) -> object:
        return sentinel

    monkeypatch.setattr(measure, "run_measure", _fake_run_measure)
    monkeypatch.setattr(
        "frugon.report.render_quality_terminal", lambda *_a, **_k: None
    )

    # Act
    result = runner.invoke(
        app, ["analyze", str(log), "--measure"], env=_WIDE_ENV, catch_exceptions=True
    )

    # Assert — pre-check passed, analysis ran (the happy path).
    assert result.exit_code == 0, f"expected exit 0, got {result.exit_code}:\n{result.output}"
    assert analyze_calls, "cost analysis should have run once the pre-check passed"


# ---------------------------------------------------------------------------
# GAP 1 — the default measured candidate is the split-routing RECOMMENDATION
# (split.candidate_model), not a separately auto-selected wholesale model.
# ---------------------------------------------------------------------------


def _heavy_row(model: str) -> dict[str, object]:
    """A priced row whose token shape keeps it well under the easy threshold.

    Short prompt + short completion + single turn → difficulty score far below
    EASY_THRESHOLD, so the split routes it to the cheap candidate (the
    recommendation we want --measure to verify).
    """
    return {
        "model": model,
        "request": {"messages": [{"role": "user", "content": "classify: spam?"}]},
        "response": {"choices": [{"message": {"content": "no"}}]},
        "usage": {"prompt_tokens": 20, "completion_tokens": 3},
    }


def _capture_run_measure(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    """Patch run_measure to record its kwargs and skip the network; returns the log."""
    captured: list[dict[str, object]] = []

    def _fake_run_measure(*_args: object, **kwargs: object) -> object:
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(measure, "run_measure", _fake_run_measure)
    monkeypatch.setattr("frugon.report.render_quality_terminal", lambda *_a, **_k: None)
    return captured


def test_default_measured_candidate_is_split_recommendation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """No --candidates: --measure samples split.candidate_model, not the wholesale pick.

    The dominant baseline is gpt-4-turbo with easy calls; the split routes those
    easy calls to a cheaper rated candidate (gpt-4o-mini).  --measure must sample
    that recommended candidate so it verifies the actual switch frugon proposes.
    """
    # Arrange
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    log = _write_log(tmp_path / "log.jsonl", [_heavy_row("gpt-4-turbo") for _ in range(6)])
    captured = _capture_run_measure(monkeypatch)

    # Sanity: the split must actually recommend gpt-4.1-mini for this log.
    result = cost.analyze_logs(log)
    assert result.split is not None, "expected a split recommendation for this log"
    recommended = result.split.candidate_model
    assert recommended == "gpt-4.1-mini", recommended

    # Act
    invoke = runner.invoke(
        app, ["analyze", str(log), "--measure"], env=_WIDE_ENV, catch_exceptions=True
    )

    # Assert — run_measure was handed the split recommendation as the candidate.
    assert invoke.exit_code == 0, f"got {invoke.exit_code}:\n{invoke.output}"
    assert captured, "run_measure was never called"
    assert captured[0]["candidates"] == [recommended], captured[0]["candidates"]


def test_explicit_candidates_are_honoured_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """--candidates wins: the user's stated intent overrides the split recommendation."""
    # Arrange
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    log = _write_log(tmp_path / "log.jsonl", [_heavy_row("gpt-4-turbo") for _ in range(6)])
    captured = _capture_run_measure(monkeypatch)

    # Act — explicit candidates, deliberately NOT the split recommendation.
    invoke = runner.invoke(
        app,
        ["analyze", str(log), "--measure", "--candidates", "claude-3-haiku-20240307"],
        env=_WIDE_ENV,
        catch_exceptions=True,
    )

    # Assert — the explicit list is passed through verbatim.
    assert invoke.exit_code == 0, f"got {invoke.exit_code}:\n{invoke.output}"
    assert captured, "run_measure was never called"
    assert captured[0]["candidates"] == ["claude-3-haiku-20240307"], captured[0]["candidates"]


def test_precheck_panel_names_recommended_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The missing-key panel references the recommended candidate (proof of GAP 1).

    Both gpt-4-turbo and gpt-4.1-mini need OPENAI_API_KEY, so the var alone does
    not reveal which candidate is checked.  The panel must name gpt-4.1-mini so
    the user can see the key is for the switch frugon recommends.
    """
    # Arrange — extra importable, key absent → the friendly panel fires.
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    log = _write_log(tmp_path / "log.jsonl", [_heavy_row("gpt-4-turbo") for _ in range(6)])

    # Act
    invoke = runner.invoke(
        app, ["analyze", str(log), "--measure"], env=_WIDE_ENV, catch_exceptions=True
    )

    # Assert
    assert invoke.exit_code == 1, invoke.output
    out = _clean(invoke.output)
    assert "Traceback" not in out, out
    assert "gpt-4.1-mini" in out, f"recommended candidate not named in panel:\n{out}"


# ---------------------------------------------------------------------------
# Awareness — help + README mention the extra + key requirement
# ---------------------------------------------------------------------------


def test_analyze_help_mentions_measure_extra_and_key() -> None:
    # Act
    result = runner.invoke(app, ["analyze", "--help"], env=_WIDE_ENV)
    out = _clean(result.output)

    # Assert
    assert result.exit_code == 0, out
    assert "frugon[measure]" in out, f"extra not mentioned in --measure help:\n{out}"
    assert "API key" in out, f"API-key requirement not mentioned:\n{out}"


def test_readme_mentions_measure_extra_and_key() -> None:
    # Arrange
    readme = pathlib.Path(cli.__file__).resolve().parents[2] / "README.md"

    # Act
    text = readme.read_text(encoding="utf-8")

    # Assert
    assert "pip install 'frugon[measure]'" in text, "README missing measure extra install line"
    assert "OPENAI_API_KEY" in text, "README missing provider-key mention near measure"


# --- did-you-mean typo hint (OPEN_API_KEY -> OPENAI_API_KEY) ---------------
import frugon.measure as _m  # noqa: E402


def test_levenshtein_basic():
    assert _m._levenshtein("OPENAI_API_KEY", "OPEN_API_KEY") == 2
    assert _m._levenshtein("abc", "abc") == 0


def test_missing_key_suggests_typo_near_miss(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPEN_API_KEY", "dummy-not-real")  # the classic slip
    with pytest.raises(_m.MissingProviderKeyError) as ei:
        _m._check_provider_keys(["gpt-4-turbo"])
    assert ei.value.suggestions.get("OPENAI_API_KEY") == "OPEN_API_KEY"


def test_far_env_var_not_suggested(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_API_KEY", raising=False)
    monkeypatch.setenv("TOTALLY_UNRELATED_DB_URL", "x")
    with pytest.raises(_m.MissingProviderKeyError) as ei:
        _m._check_provider_keys(["gpt-4-turbo"])
    assert "OPENAI_API_KEY" not in ei.value.suggestions
