"""Tests for the key-aware judge fallback (kills the hardcoded-OpenAI bias).

When ``--judge`` runs and NO model in the user's log is quality-rated, frugon
used to hard-default to ``gpt-4o`` — demanding an OpenAI key from users who may
hold only Anthropic / Gemini / DeepSeek / OpenRouter keys.  The fallback now
scans the environment for present provider keys and picks the highest
quality-tier rated+priceable model whose provider key IS present.

The five scenarios from the spec are exercised against monkeypatched env (no
real keys) and the bundled quality + pricing tables (deterministic):

  * ANTHROPIC-only → best rated claude-*
  * GEMINI-only    → best rated gemini-*
  * multiple keys  → global best-tier among reachable
  * no keys        → None (CLI shows the fail-fast panel naming --judge-model)
  * OPENAI-only    → gpt-4o (or best rated gpt-*)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from frugon.cli import _resolve_judge_model
from frugon.measure import (
    _present_provider_key_vars,
    best_judge_for_available_keys,
)

sys.path.insert(0, str(Path(__file__).parent))
from conftest import install_synthetic_quality

# Every provider env var the fallback might read — cleared before each scenario
# so a leaked key from the developer's shell cannot taint the result.
_ALL_PROVIDER_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "TOGETHERAI_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "VERTEXAI_PROJECT",
    "DEEPSEEK_API_KEY",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every provider key before each test so a leaked shell key cannot
    taint the deterministic env scenarios."""
    for var in _ALL_PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def best_tier_pinned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin a synthetic quality table so the best-reachable-judge tests OWN the
    tier relationships they assert, immune to leaderboard re-anchors.

    The judge selector picks the lowest tier-integer reachable model (name tie-break
    ascending). These tests assert specific identities — ``claude-3-5-sonnet`` as the
    top reachable claude, and a ``claude-3-5-sonnet`` < ``gpt-4o`` cross-provider tie
    among two tier-0 models — which only holds for a controlled table. Pinning keeps
    that scenario stable while the real seed re-bands those models over time.
    ``claude-3-haiku`` / ``gpt-3.5-turbo`` are low-tier decoys so "best" is non-trivial.
    All four names are priced in the real seed, so the priceable-form filter keeps them.
    """
    install_synthetic_quality(
        monkeypatch,
        tmp_path,
        {"claude-3-5-sonnet": 0, "gpt-4o": 0, "claude-3-haiku": 3, "gpt-3.5-turbo": 3},
    )


# ---------------------------------------------------------------------------
# _present_provider_key_vars
# ---------------------------------------------------------------------------


def test_present_keys_reads_only_set_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "")  # empty → not present
    present = _present_provider_key_vars()
    assert "ANTHROPIC_API_KEY" in present
    assert "GEMINI_API_KEY" not in present
    assert "OPENAI_API_KEY" not in present


# ---------------------------------------------------------------------------
# best_judge_for_available_keys — the five scenarios
# ---------------------------------------------------------------------------


def test_anthropic_only_picks_best_rated_claude(
    monkeypatch: pytest.MonkeyPatch, best_tier_pinned: None
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    judge = best_judge_for_available_keys()
    assert judge is not None
    assert "claude-3-5-sonnet" in judge  # pinned tier-0 claude, concrete priceable form


def test_gemini_only_picks_best_rated_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    judge = best_judge_for_available_keys()
    assert judge is not None
    assert judge.startswith("gemini-")


def test_openai_only_picks_gpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    judge = best_judge_for_available_keys()
    assert judge is not None
    # The best rated OpenAI model — gpt-4o (or another rated gpt-*).
    assert judge.startswith(("gpt-", "o1", "o3", "o4"))


def test_multiple_keys_picks_global_best_tier_reachable(
    monkeypatch: pytest.MonkeyPatch, best_tier_pinned: None
) -> None:
    # Both Anthropic and OpenAI hold tier-0 models (pinned); the deterministic
    # tie-break is the rated NAME ascending, so claude-3-5-sonnet (< gpt-4o) wins.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    judge = best_judge_for_available_keys()
    assert judge is not None
    assert "claude-3-5-sonnet" in judge


def test_no_keys_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert best_judge_for_available_keys() is None


def test_result_is_deterministic_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    first = best_judge_for_available_keys()
    second = best_judge_for_available_keys()
    assert first == second


# ---------------------------------------------------------------------------
# _resolve_judge_model — full precedence chain wiring
# ---------------------------------------------------------------------------


def test_resolve_explicit_flag_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, from_log = _resolve_judge_model("my-judge", ["unrated-model"])
    assert model == "my-judge"
    assert from_log is False


def test_resolve_log_best_when_a_logged_model_is_rated(
    monkeypatch: pytest.MonkeyPatch, best_tier_pinned: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    # gpt-4o (pinned tier 0) outranks gpt-3.5-turbo (pinned tier 3) and is present
    # in the log → log-best path, from_log True.
    model, from_log = _resolve_judge_model(None, ["gpt-4o", "gpt-3.5-turbo"])
    assert model == "gpt-4o"
    assert from_log is True


def test_resolve_falls_back_to_key_aware_when_no_logged_model_rated(
    monkeypatch: pytest.MonkeyPatch, best_tier_pinned: None
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    # The logged model is unrated → no log-best → key-aware fallback to the best
    # reachable claude (pinned tier-0 claude-3-5-sonnet).
    model, from_log = _resolve_judge_model(None, ["some-unrated-model"])
    assert model is not None
    assert "claude-3-5-sonnet" in model
    assert from_log is False


def test_resolve_openai_default_only_when_openai_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenAI key present but no rated logged model and (hypothetically) no rated
    # key-reachable model would still land on gpt-4o via either the key-aware
    # path or the DEFAULT path — either way an OpenAI model.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    model, _ = _resolve_judge_model(None, ["unrated-only"])
    assert model is not None
    assert model.startswith(("gpt-", "o1", "o3", "o4"))


def test_resolve_returns_none_when_no_keys_and_no_rated_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, from_log = _resolve_judge_model(None, ["unrated-only"])
    assert model is None
    assert from_log is False


# ---------------------------------------------------------------------------
# CLI surface — fail-fast panel + the updated --judge help text
# ---------------------------------------------------------------------------


def test_cli_judge_with_no_reachable_judge_shows_failfast_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--judge`` with an unrated log and NO provider keys → the clean
    'No judge available' panel naming --judge-model, never a traceback."""
    import json

    from typer.testing import CliRunner

    from frugon import measure
    from frugon.cli import app

    runner = CliRunner()
    # Make the [measure] extra check pass so resolution (not the install panel)
    # is what fires.  The judge resolution happens BEFORE verify_measure_prereqs.
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())

    # A log of an UNRATED model (so best_judge_from_log returns None) — and the
    # env has no provider keys (the _clean_env fixture), so no judge is reachable.
    log = tmp_path / "log.jsonl"
    rows = [
        {
            "model": "some-obscure-unrated-model",
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {"choices": [{"message": {"content": "ok"}}]},
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["analyze", str(log), "--measure", "--judge", "--no-progress"],
        env={"COLUMNS": "200", "TERM": "dumb"},
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "No judge available" in result.output
    assert "--judge-model" in result.output


def test_cli_judge_help_no_longer_hardcodes_gpt4o_as_named_fallback() -> None:
    from typer.testing import CliRunner

    from frugon.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "best rated model your keys can reach" in out
