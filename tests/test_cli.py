from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from frugon.cli import app

runner = CliRunner()


def test_capture_allow_insecure_flag_reaches_run_capture(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_run = MagicMock()
    monkeypatch.setattr("frugon.capture.run_capture", mock_run, raising=False)
    out_file = tmp_path / "cap.jsonl"
    result = runner.invoke(
        app,
        ["capture", "--upstream", "http://192.168.1.50:11434",
         "--allow-insecure", "--out", str(out_file)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    mock_run.assert_called_once()
    kw = mock_run.call_args.kwargs
    assert kw.get("allow_insecure_upstream") is True, f"not True; got {kw!r}"


def test_capture_allow_insecure_upstream_alias_reaches_run_capture(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_run = MagicMock()
    monkeypatch.setattr("frugon.capture.run_capture", mock_run, raising=False)
    out_file = tmp_path / "cap.jsonl"
    result = runner.invoke(
        app,
        ["capture", "--upstream", "http://192.168.1.50:11434",
         "--allow-insecure-upstream", "--out", str(out_file)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    mock_run.assert_called_once()
    kw = mock_run.call_args.kwargs
    assert kw.get("allow_insecure_upstream") is True, f"alias not True; got {kw!r}"


def test_capture_default_no_allow_insecure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_run = MagicMock()
    monkeypatch.setattr("frugon.capture.run_capture", mock_run, raising=False)
    out_file = tmp_path / "cap.jsonl"
    result = runner.invoke(
        app,
        ["capture", "--upstream", "https://api.openai.com", "--out", str(out_file)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    mock_run.assert_called_once()
    kw = mock_run.call_args.kwargs
    assert kw.get("allow_insecure_upstream") is False, f"should be False; got {kw!r}"


# ---------------------------------------------------------------------------
# analyze --no-progress flag + progress gating under the (non-TTY) test runner
# ---------------------------------------------------------------------------


def test_analyze_demo_accepts_no_progress_flag() -> None:
    """The --no-progress flag is parsed and the analysis still succeeds."""
    result = runner.invoke(app, ["analyze", "--demo", "--no-progress"], catch_exceptions=False)
    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    # The result panel still renders to stdout.
    assert "cost analysis" in result.output


def test_analyze_demo_under_nontty_runner_has_no_progress_chrome() -> None:
    """Under the test runner stderr is NOT a TTY, so no progress chrome renders.

    CliRunner captures the program output; with a non-TTY stderr the helper is a
    no-op, so the transient spinner/bar phase labels never appear.  The result
    panel (stdout) is present.
    """
    result = runner.invoke(app, ["analyze", "--demo"], catch_exceptions=False)
    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    assert "cost analysis" in result.output
    for chrome in ("Reading logs", "Pricing  ", "Priced in"):
        assert chrome not in result.output, f"progress chrome leaked: {chrome!r}"


# ---------------------------------------------------------------------------
# _resolve_judge_model — flag > log-best > DEFAULT_JUDGE_MODEL
# ---------------------------------------------------------------------------


def test_resolve_judge_model_explicit_flag_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: an explicit --judge-model value is given.
    Assert: it is used verbatim and from_log is False — the user's stated intent
    overrides any log-derived pick, and best_judge_from_log is never consulted.
    """
    from frugon import cli

    called = False

    def _boom(_models: list[str]) -> str | None:
        nonlocal called
        called = True
        return "should-not-be-used"

    monkeypatch.setattr("frugon.cost.best_judge_from_log", _boom)
    resolved, from_log = cli._resolve_judge_model("claude-3-opus", ["gpt-4o"])
    assert resolved == "claude-3-opus"
    assert from_log is False
    assert called is False


def test_resolve_judge_model_falls_to_log_best_when_no_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: no --judge-model; the log has a rated model.
    Assert: the log-best model is chosen and from_log is True.
    """
    from frugon import cli

    monkeypatch.setattr(
        "frugon.cost.best_judge_from_log", lambda models: "gpt-4-turbo"
    )
    resolved, from_log = cli._resolve_judge_model(None, ["gpt-4o-mini", "gpt-4-turbo"])
    assert resolved == "gpt-4-turbo"
    assert from_log is True


def test_resolve_judge_model_key_aware_fallback_when_no_log_rated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: no --judge-model and no rated log model (best returns None), with
    ONLY an OpenAI key present.
    Assert: the key-aware fallback resolves to an OpenAI model (the best rated
    gpt-* the present key can reach — never demanding a key the user lacks) and
    from_log is False.
    """
    from frugon import cli

    # Clear every provider key, then set ONLY OpenAI, so the key-aware fallback
    # is constrained to OpenAI-reachable models.
    for var in (
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
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    monkeypatch.setattr("frugon.cost.best_judge_from_log", lambda models: None)
    resolved, from_log = cli._resolve_judge_model(None, ["some-unrated-model"])
    assert resolved is not None
    assert resolved.startswith(("gpt-", "o1", "o3", "o4"))
    assert from_log is False


# ---------------------------------------------------------------------------
# --report-style honesty for Markdown targets (F5)
# ---------------------------------------------------------------------------

_V2_MD_NOTICE = "Markdown has a single canonical layout"


def test_report_style_v2_markdown_emits_honest_notice(tmp_path: pathlib.Path) -> None:
    """md + --report-style v2 → a dim notice that v2 styles HTML only.

    Markdown has one canonical layout (v1 and v2 render identically), so the flag
    is a no-op for a .md target.  Rather than silently ignoring it, the CLI says
    so; the report is still written.
    """
    out = tmp_path / "r.md"
    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(out), "--report-style", "v2"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert _V2_MD_NOTICE in result.output
    assert out.exists()


def test_report_style_v2_html_emits_no_markdown_notice(tmp_path: pathlib.Path) -> None:
    """html + --report-style v2 → NO Markdown notice (v2 is meaningful for HTML)."""
    out = tmp_path / "r.html"
    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(out), "--report-style", "v2"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert _V2_MD_NOTICE not in result.output


def test_report_style_v1_markdown_emits_no_notice(tmp_path: pathlib.Path) -> None:
    """md + --report-style v1 → NO notice."""
    out = tmp_path / "r.md"
    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(out), "--report-style", "v1"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert _V2_MD_NOTICE not in result.output


def test_report_style_default_markdown_emits_no_notice(tmp_path: pathlib.Path) -> None:
    """md with NO --report-style → v2 is the default, but the notice must NOT
    fire: it is reserved for an EXPLICIT v2 choice on a Markdown target.

    Regression guard for the default-v2 transition: a bare default report must
    never spuriously claim the user asked for HTML styling.
    """
    out = tmp_path / "r.md"
    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(out)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert _V2_MD_NOTICE not in result.output
    assert out.exists()


# ---------------------------------------------------------------------------
# frugon update command
# ---------------------------------------------------------------------------


def test_update_command_invokes_both_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """frugon update calls both fetch_and_update_pricing and fetch_and_update_quality."""
    mock_pricing = MagicMock(return_value={"models_synced": 5})
    mock_quality = MagicMock(return_value={"models_synced": 10})
    monkeypatch.setattr("frugon.pricing.fetch_and_update_pricing", mock_pricing)
    monkeypatch.setattr("frugon.quality.fetch_and_update_quality", mock_quality)

    result = runner.invoke(app, ["update"], catch_exceptions=False)

    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    mock_pricing.assert_called_once()
    mock_quality.assert_called_once()


def test_update_command_pricing_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """frugon update exits 1 and prints error when pricing update fails."""
    from frugon.pricing import PricingUpdateError

    monkeypatch.setattr(
        "frugon.pricing.fetch_and_update_pricing",
        MagicMock(side_effect=PricingUpdateError("network timeout")),
    )

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "pricing update failed" in result.output


def test_update_command_quality_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """frugon update exits 1 when quality update fails after pricing succeeds."""
    from frugon.quality import QualityUpdateError

    monkeypatch.setattr(
        "frugon.pricing.fetch_and_update_pricing",
        MagicMock(return_value={"models_synced": 5}),
    )
    monkeypatch.setattr(
        "frugon.quality.fetch_and_update_quality",
        MagicMock(side_effect=QualityUpdateError("HF unavailable")),
    )

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 1
    assert "quality update failed" in result.output


def test_analyze_demo_pool_notice_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    """frugon analyze --demo renders the pool notice when a recommendation is made."""
    result = runner.invoke(
        app, ["analyze", "--demo", "--no-progress"], catch_exceptions=False
    )
    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    # The pool notice must appear whenever a split or candidate recommendation exists.
    # --demo consistently produces a recommendation from the bundled sample log.
    assert "Recommendations use a curated set" in result.output


def test_analyze_demo_sample_disclosure(monkeypatch: pytest.MonkeyPatch) -> None:
    """--demo discloses it is bundled sample data.

    FRG-OSS-034 Phase 3 un-pinned the demo's candidate pool — it now uses the
    SAME default roster as a real run, so only the DATA (not the candidate set)
    is illustrative.  The disclosure says so honestly and points the user at
    analysing their own logs.
    """
    result = runner.invoke(
        app, ["analyze", "--demo", "--no-progress"], catch_exceptions=False
    )
    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    flat = " ".join(result.output.split())  # collapse Rich line-wrapping
    assert "This is bundled sample data" in flat
    assert "fixed demo candidate set" not in flat
    assert "your own logs" in flat


def test_analyze_explicit_candidates_no_demo_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit --candidates run suppresses the demo-sample disclosure."""
    result = runner.invoke(
        app,
        ["analyze", "--demo", "--no-progress", "--candidates", "gpt-4.1"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    assert "bundled sample data" not in " ".join(result.output.split())
