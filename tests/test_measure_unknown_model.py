"""Tests for the fail-fast unknown-model pre-check.

A model name a user passes via ``--candidates`` / ``--judge-model`` (or that
appears as the auto-detected baseline in their log) must be resolved against
frugon's pricing snapshot AND LiteLLM's own registry BEFORE any sampling call
is made.  Catches the witnessed bug: ``--candidates gpt-5.3`` was plumbed
through, 10 sampling calls were burned, all 10 errored with "bad request —
check model name", and 10 judge calls were attempted.  After this fix:

  * the unknown-name fails fast at the same pre-flight that verifies provider
    keys, BEFORE any provider call,
  * the user sees one clean amber panel naming the bad model(s) and the
    nearest pricing-table neighbours ("did you mean…?"),
  * the CLI exits with code 2 (config error) — distinct from code 1
    (real run failure).

All tests run with the network stubbed: the sampling/judging entry points
(_call_model, _judge_pair) are monkey-patched to record any invocation, and
the assertions prove the call lists stay empty.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
from typer.testing import CliRunner

import frugon.measure as measure
from frugon.cli import app
from frugon.measure import (
    UnknownModelError,
    _check_known_models,
    _suggest_models,
    verify_measure_prerequisites,
)

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
_WIDE_ENV = {"COLUMNS": "200", "TERM": "dumb"}


def _clean(text: str) -> str:
    """Strip ANSI escape codes so substring assertions are renderer-independent."""
    return _ANSI_RE.sub("", text)


def _priced_row(model: str) -> dict[str, object]:
    """A minimal priced log row with an explicit usage block."""
    return {
        "model": model,
        "request": {"messages": [{"role": "user", "content": "hi"}]},
        "response": {"choices": [{"message": {"content": "hello"}}]},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _write_log(path: pathlib.Path, rows: list[dict[str, object]]) -> pathlib.Path:
    """Write *rows* as JSONL to *path* and return it."""
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Unit — _check_known_models raises with structured payload
# ---------------------------------------------------------------------------


def test_check_known_models_raises_for_typoed_name() -> None:
    """A clearly-invalid model name (typo of a real one) raises with suggestions.

    `gpt-5.3` is not in the pricing snapshot or LiteLLM's registry, but
    `gpt-5`, `gpt-5.1`, `gpt-5.2` exist — the suggestion list surfaces those.
    """
    # Act / Assert
    with pytest.raises(UnknownModelError) as excinfo:
        _check_known_models(["gpt-5.3"])

    payload = excinfo.value.unknown_models
    assert len(payload) == 1, payload
    bad, suggestions = payload[0]
    assert bad == "gpt-5.3"
    assert len(suggestions) <= 3, (
        f"suggestion list should be capped at 3, got {len(suggestions)}: {suggestions}"
    )
    assert suggestions, "expected at least one suggestion for a near-miss typo"


def test_check_known_models_accepts_known_real_models() -> None:
    """Real models in the pricing snapshot pass without raising."""
    # Act — these MUST resolve via the pricing table; if not, the test would raise.
    _check_known_models(["gpt-4o-mini", "claude-3-5-sonnet-latest"])
    # Assert — reaching here is the assertion (no exception raised).


def test_check_known_models_reports_every_unknown_in_order() -> None:
    """Multiple unknowns in one call all surface in input order — one panel, one fix.

    A user who typo'd two ``--candidates`` should not have to iterate one at a
    time; both names appear in the structured payload.
    """
    # Arrange — first name is known, middle two are typos, last is known again.
    with pytest.raises(UnknownModelError) as excinfo:
        _check_known_models(["gpt-4o-mini", "gpt-5.3", "xqxqxqxq-9000", "gpt-4o"])

    names = [bad for bad, _suggestions in excinfo.value.unknown_models]
    assert names == ["gpt-5.3", "xqxqxqxq-9000"], names


def test_check_known_models_far_miss_yields_empty_suggestions() -> None:
    """A wholly invented name gets no suggestions — better than a misleading one."""
    with pytest.raises(UnknownModelError) as excinfo:
        _check_known_models(["xqxqxqxq-9000"])

    _bad, suggestions = excinfo.value.unknown_models[0]
    assert suggestions == [], suggestions


def test_check_known_models_dedups_repeated_input() -> None:
    """The same unknown name passed twice is reported once."""
    with pytest.raises(UnknownModelError) as excinfo:
        _check_known_models(["gpt-5.3", "gpt-5.3"])

    assert len(excinfo.value.unknown_models) == 1


def test_check_known_models_suggestion_ranking_is_deterministic() -> None:
    """Suggestion ordering is by distance then alphabetical — stable across runs."""
    # Act — call twice on the same input.
    first = _suggest_models("gpt-5.3", ["gpt-5.4", "gpt-5.1", "gpt-5.2", "gpt-5"])
    second = _suggest_models("gpt-5.3", ["gpt-5.4", "gpt-5.1", "gpt-5.2", "gpt-5"])
    # Assert — identical, capped at 3, ranked by edit distance ascending then name.
    assert first == second
    assert first == ["gpt-5.1", "gpt-5.2", "gpt-5.4"], first


def test_check_known_models_validates_judge_model() -> None:
    """A typo'd judge model is caught at the same gate as a typo'd candidate."""
    with pytest.raises(UnknownModelError) as excinfo:
        # Mixes the user's baseline + a candidate + a typo'd judge name.
        _check_known_models(["gpt-4o", "gpt-4o-mini", "gpt-judgee-typo"])

    names = [bad for bad, _ in excinfo.value.unknown_models]
    assert "gpt-judgee-typo" in names, names


# ---------------------------------------------------------------------------
# Integration with verify_measure_prerequisites — ordering: unknown before key
# ---------------------------------------------------------------------------


def test_verify_prerequisites_unknown_model_fires_before_key_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown-model surfaces BEFORE missing-key.

    Rationale: if the user typo'd ``--candidates gpt-5.3``, the right next step
    is "did you mean gpt-5?", not "set OPENAI_API_KEY".  Leading them through a
    pointless key-export ritual for a model that does not exist is worse UX
    than naming the typo.
    """
    # Arrange — both conditions would otherwise fire: name is bogus AND no key.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Act / Assert — UnknownModelError raises, not MissingProviderKeyError.
    with pytest.raises(UnknownModelError):
        verify_measure_prerequisites(["gpt-5.3"])


# ---------------------------------------------------------------------------
# CLI — bad --candidates renders amber panel, exit code 2, ZERO provider calls
# ---------------------------------------------------------------------------


def test_cli_unknown_candidate_renders_panel_exit_2_and_makes_no_provider_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The full reported bug, asserted end-to-end.

    `--candidates gpt-5.3`:
      * the CLI exits with code 2 (config error, NOT 1 = real run failure),
      * a clean amber panel names the bad model + nearest neighbours,
      * NO traceback leaks,
      * `_call_model` and `_judge_pair` are NEVER invoked — the exact regression
        being fixed.
    """
    # Arrange — key present so the only failure path left IS the unknown-model
    # gate (otherwise the missing-key panel would mask the bug).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    log = _write_log(tmp_path / "log.jsonl", [_priced_row("gpt-4o")])

    # Spy on the provider entry points.  Reaching either is the regression.
    sampling_calls: list[object] = []
    judging_calls: list[object] = []

    def _spy_sample(*args: object, **kwargs: object) -> object:
        sampling_calls.append((args, kwargs))
        raise AssertionError(
            "_call_model invoked after unknown-model pre-check — regression!"
        )

    def _spy_judge(*args: object, **kwargs: object) -> object:
        judging_calls.append((args, kwargs))
        raise AssertionError(
            "_judge_pair invoked after unknown-model pre-check — regression!"
        )

    monkeypatch.setattr(measure, "_call_model", _spy_sample)
    monkeypatch.setattr(measure, "_judge_pair", _spy_judge)

    # Act
    result = runner.invoke(
        app,
        ["analyze", str(log), "--measure", "--candidates", "gpt-5.3"],
        env=_WIDE_ENV,
        catch_exceptions=True,
    )

    # Assert — exit code 2 = config error, distinct from 1 = real failure.
    assert result.exit_code == 2, (
        f"expected exit 2 (config error), got {result.exit_code}:\n{result.output}"
    )
    # No traceback escaped to the user.
    out = _clean(result.output)
    assert "Traceback" not in out, f"traceback leaked:\n{out}"
    # Panel content — title, bad name, suggestion nudge, footer pointer.
    assert "Unknown model name" in out, out
    assert "gpt-5.3" in out, out
    assert "can't find" in out, out
    # At least one near-neighbour is named — the gpt-5* family is in the table.
    assert "gpt-5" in out, out
    # Footer hint points at the model-list discovery command.
    assert "frugon models" in out, out
    # ZERO provider calls — the whole point of the fix.
    assert sampling_calls == [], (
        f"sampling fired before pre-check: {len(sampling_calls)} call(s)"
    )
    assert judging_calls == [], (
        f"judging fired before pre-check: {len(judging_calls)} call(s)"
    )


def test_cli_far_miss_panel_says_no_close_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A wholly invented model name renders "(no close match in the pricing table)".

    Better to admit "I have nothing for this" than to mislead with an unrelated
    name beyond the edit-distance cap.
    """
    # Arrange
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    log = _write_log(tmp_path / "log.jsonl", [_priced_row("gpt-4o")])

    # Spy — still asserts zero provider calls on this code path.
    def _spy(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("provider call after unknown-model pre-check")

    monkeypatch.setattr(measure, "_call_model", _spy)
    monkeypatch.setattr(measure, "_judge_pair", _spy)

    # Act
    result = runner.invoke(
        app,
        ["analyze", str(log), "--measure", "--candidates", "xqxqxqxq-9000"],
        env=_WIDE_ENV,
        catch_exceptions=True,
    )

    # Assert
    assert result.exit_code == 2, result.output
    out = _clean(result.output)
    assert "xqxqxqxq-9000" in out, out
    assert "no close match" in out, out
