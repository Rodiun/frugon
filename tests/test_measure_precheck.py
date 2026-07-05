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
    easy calls to a cheaper rated candidate from the default pool.  --measure must
    sample that recommended candidate so it verifies the actual switch frugon proposes.
    """
    # Arrange
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")  # for the gpt-4-turbo baseline
    log = _write_log(tmp_path / "log.jsonl", [_heavy_row("gpt-4-turbo") for _ in range(6)])
    captured = _capture_run_measure(monkeypatch)

    # Sanity: the split recommends a real, strictly-cheaper candidate drawn from
    # the default routing pool.  The exact model tracks the curated roster, so
    # derive it rather than pin a brittle literal.
    result = cost.analyze_logs(log)
    assert result.split is not None, "expected a split recommendation for this log"
    recommended = result.split.candidate_model
    assert recommended in cost._ROUTING_CANDIDATES, recommended
    assert recommended != "gpt-4-turbo", recommended

    # The recommendation may be cross-provider (e.g. a Gemini or Anthropic model),
    # so provide the key it needs — otherwise the precheck would block --measure.
    _needed_key = measure._required_key_for_model(recommended)
    if _needed_key:
        monkeypatch.setenv(_needed_key, "sk-test-not-real")

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
    """The missing-key panel references the PREDICTED candidate (proof of GAP 1).

    The panel must name a candidate — not just the missing env var — so the user
    can see which switch frugon is likely to propose. Since the 2026-07-02
    quality-aware tie-break fix, the cost analysis's REAL split recommendation
    (full-dataset New-spend + tie-break) can differ from this cheap pre-check's
    PREDICTION (select_easy_target's blended-per-token-price heuristic, the only
    basis available before the expensive per-call cost pass) — that divergence
    is exactly why the prediction is advisory, never blocking, in cli.py (see
    ``predicted_default_candidate``). This test asserts against the SAME cheap
    prediction the panel actually displays, not the expensive real answer.
    """
    # Arrange — extra importable, key absent → the friendly panel fires.
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    log = _write_log(tmp_path / "log.jsonl", [_heavy_row("gpt-4-turbo") for _ in range(6)])
    # Derive the PREDICTED candidate via the same cheap heuristic cli.py's
    # precheck uses, so the assertion tracks the curated roster instead of a
    # brittle literal — and stays honest about what the panel can actually know
    # before the expensive analysis pass.
    from frugon.cost import _ROUTING_CANDIDATES, scan_models
    from frugon.routing import select_easy_target

    _distinct, _dominant = scan_models(log)
    assert _dominant is not None
    predicted = select_easy_target(_dominant, _ROUTING_CANDIDATES)
    assert predicted is not None, "expected a predicted candidate for this log"

    # Act
    invoke = runner.invoke(
        app, ["analyze", str(log), "--measure"], env=_WIDE_ENV, catch_exceptions=True
    )

    # Assert
    assert invoke.exit_code == 1, invoke.output
    out = _clean(invoke.output)
    assert "Traceback" not in out, out
    assert predicted in out, f"predicted candidate not named in panel:\n{out}"


class TestAdvisoryPredictionNeverBlocksAWrongGuess:
    """The cheap default-pool prediction (select_easy_target) is ADVISORY.

    When it diverges from the real analysis's split recommendation
    (frugon.cost._select_cheapest_eligible, full-dataset New-spend +
    quality tie-break — a basis the cheap scan_models() pass cannot compute),
    a missing key for the WRONG (predicted-but-not-actually-needed) model must
    never block a run that does not actually need that key. run_measure's own
    authoritative _check_provider_keys(all_models) re-verifies against the REAL
    model list before any provider call and is the true gate.
    """

    def test_wrong_prediction_does_not_block_when_real_recommendation_needs_no_new_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        # Arrange — the classic divergent fixture: predicted candidate needs
        # DEEPSEEK_API_KEY, but the REAL split recommendation for this exact
        # log is gpt-4.1-nano (needs only OPENAI_API_KEY, already set for the
        # gpt-4-turbo baseline).
        from frugon.cost import _ROUTING_CANDIDATES, scan_models
        from frugon.routing import select_easy_target

        monkeypatch.setattr(measure, "_import_litellm", lambda: object())
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        log = _write_log(
            tmp_path / "log.jsonl", [_heavy_row("gpt-4-turbo") for _ in range(6)]
        )

        _distinct, _dominant = scan_models(log)
        assert _dominant is not None
        predicted = select_easy_target(_dominant, _ROUTING_CANDIDATES)
        assert predicted == "deepseek-v4-flash", (
            f"fixture precondition drifted: predicted={predicted!r}"
        )
        real_split = cost.analyze_logs(log).split
        assert real_split is not None
        assert real_split.candidate_model != predicted, (
            "fixture precondition: the cheap prediction must genuinely diverge "
            "from the real recommendation for this test to prove anything"
        )
        assert measure._required_key_for_model(real_split.candidate_model) == (
            "OPENAI_API_KEY"
        )

        captured = _capture_run_measure(monkeypatch)

        # Act
        invoke = runner.invoke(
            app, ["analyze", str(log), "--measure"], env=_WIDE_ENV, catch_exceptions=True
        )

        # Assert — the wrong prediction (needing DEEPSEEK_API_KEY) never
        # blocked the run; the REAL recommendation only needs the
        # already-present OPENAI_API_KEY, so the run completes successfully.
        assert invoke.exit_code == 0, (
            f"a wrong cheap prediction incorrectly blocked the run:\n{invoke.output}"
        )
        assert captured, "run_measure was never called"
        assert captured[0]["candidates"] == [real_split.candidate_model]

    def test_advisory_heads_up_line_appears_for_a_genuinely_missing_predicted_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """The non-fatal heads-up still surfaces so the fast-fail nicety survives."""
        monkeypatch.setattr(measure, "_import_litellm", lambda: object())
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        log = _write_log(
            tmp_path / "log.jsonl", [_heavy_row("gpt-4-turbo") for _ in range(6)]
        )
        _capture_run_measure(monkeypatch)

        invoke = runner.invoke(
            app, ["analyze", str(log), "--measure"], env=_WIDE_ENV, catch_exceptions=True
        )

        out = _clean(invoke.output)
        assert "Heads up" in out
        assert "DEEPSEEK_API_KEY" in out


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


# ---------------------------------------------------------------------------
# New-vendor roster gap (FRG-OSS-034 follow-up, 2026-07-02) — the 23-model
# default pool grew to 11 vendors but _PROVIDER_KEY_MAP / the LiteLLM routing
# table were never extended to match.  Every mapping added to close that gap
# gets its own precheck test here: (1) the correct env var is required, (2) the
# precheck actually FIRES MissingProviderKeyError when that var is absent, and
# (3) the bare roster name routes to a LiteLLM-resolvable provider-prefixed
# form (see _LITELLM_ROUTE_PREFIX in measure.py — verified against this repo's
# own .venv LiteLLM install, not guessed).
# ---------------------------------------------------------------------------

_NEW_VENDOR_KEY_CASES = [
    ("deepseek-v3.2", "DEEPSEEK_API_KEY"),
    ("deepseek-v4-flash", "DEEPSEEK_API_KEY"),
    ("deepseek-v4-pro", "DEEPSEEK_API_KEY"),
    ("grok-4", "XAI_API_KEY"),
    ("grok-3-mini", "XAI_API_KEY"),
    ("kimi-k2.6", "MOONSHOT_API_KEY"),
    ("glm-4.6", "ZAI_API_KEY"),
    ("glm-4.5-air", "ZAI_API_KEY"),
    ("minimax-m3", "MINIMAX_API_KEY"),
    ("qwen-max", "DASHSCOPE_API_KEY"),
    ("llama-4-maverick-17b-128e-instruct", "GROQ_API_KEY"),
    ("llama-4-scout-17b-16e-instruct", "GROQ_API_KEY"),
    ("mistral-large-3", "MISTRAL_API_KEY"),
]


class TestNewVendorKeyMap:
    @pytest.mark.parametrize(("model", "expected_var"), _NEW_VENDOR_KEY_CASES)
    def test_required_key_for_model_resolves(
        self, model: str, expected_var: str
    ) -> None:
        assert _m._required_key_for_model(model) == expected_var

    @pytest.mark.parametrize(("model", "expected_var"), _NEW_VENDOR_KEY_CASES)
    def test_precheck_fires_missing_key_when_env_var_absent(
        self, model: str, expected_var: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — extra present, the required var absent from the environment.
        monkeypatch.setattr(_m, "_import_litellm", lambda: object())
        monkeypatch.delenv(expected_var, raising=False)

        # Act / Assert — the precheck must name the SAME var the map declares,
        # BEFORE any provider call is attempted.
        with pytest.raises(MissingProviderKeyError) as excinfo:
            verify_measure_prerequisites([model])
        assert expected_var in excinfo.value.missing_vars, (
            f"{model} did not require {expected_var}: "
            f"got {excinfo.value.missing_vars!r}"
        )

    @pytest.mark.parametrize(("model", "expected_var"), _NEW_VENDOR_KEY_CASES)
    def test_precheck_passes_when_env_var_present(
        self, model: str, expected_var: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(_m, "_import_litellm", lambda: object())
        monkeypatch.setenv(expected_var, "dummy-not-real")

        # Act / Assert — no MissingProviderKeyError once the var is present.
        # (verify_measure_prerequisites also runs _check_known_models, which
        # this bundled roster's own pricing.json entries satisfy.)
        verify_measure_prerequisites([model])

    def test_no_roster_model_is_missing_a_key_mapping(self) -> None:
        """Regression pin for the exact gap this suite closes: EVERY model in
        the live 23-model default pool must resolve a required env var."""
        from frugon.cost import _ROUTING_CANDIDATES

        missing = [
            m for m in _ROUTING_CANDIDATES if _m._required_key_for_model(m) is None
        ]
        assert missing == [], (
            f"_ROUTING_CANDIDATES entries with no _PROVIDER_KEY_MAP mapping: "
            f"{missing}"
        )


class TestNewVendorLiteLLMRouting:
    """_route_for_measure prepends the provider prefix the new roster needs.

    Confirmed against this repo's OWN .venv LiteLLM install
    (``litellm.get_llm_provider``) that: (a) the bare roster name is NOT
    routable on its own, and (b) the routed (prefixed) form IS routable — so
    these tests pin frugon's OWN mapping logic, not LiteLLM's internals.
    """

    @pytest.mark.parametrize(
        ("bare", "expected_prefix"),
        [
            ("deepseek-v3.2", "deepseek/"),
            ("deepseek-v4-flash", "deepseek/"),
            ("grok-4", "xai/"),
            ("grok-3-mini", "xai/"),
            ("kimi-k2.6", "moonshot/"),
            ("glm-4.6", "zai/"),
            ("glm-4.5-air", "zai/"),
            ("minimax-m3", "minimax/"),
            ("qwen-max", "dashscope/"),
            ("mistral-large-3", "mistral/"),
        ],
    )
    def test_route_for_measure_prepends_expected_prefix(
        self, bare: str, expected_prefix: str
    ) -> None:
        assert measure._route_for_measure(bare) == expected_prefix + bare

    def test_route_for_measure_mistral_already_prefixed_not_double_routed(
        self,
    ) -> None:
        """A caller-supplied "mistral/..." name (already routable) must pass

        through unchanged — the "mistral-" bare-name prefix key must not
        double-prefix a name that already carries the provider segment.
        """
        already_routed = "mistral/mistral-large-latest"
        assert measure._route_for_measure(already_routed) == already_routed

    @pytest.mark.parametrize(
        "bare",
        [
            "llama-4-maverick-17b-128e-instruct",
            "llama-4-scout-17b-16e-instruct",
        ],
    )
    def test_route_for_measure_llama4_uses_groq_meta_llama_form(
        self, bare: str
    ) -> None:
        assert measure._route_for_measure(bare) == f"groq/meta-llama/{bare}"

    def test_route_for_measure_leaves_already_routable_names_unchanged(
        self,
    ) -> None:
        # Default-namespace models (OpenAI/Anthropic) and names the user
        # already prefixed themselves must pass through unchanged.
        for model in ("gpt-5.5", "claude-opus-4-8", "openrouter/openai/gpt-4o"):
            assert measure._route_for_measure(model) == model

    def test_stored_output_model_name_stays_bare_after_routing(self) -> None:
        """SampledOutput.model must be the ORIGINAL bare name, never the routed
        (provider-prefixed) form — reports and --candidates matching depend on
        the bare name."""
        from frugon.measure import SampledOutput, _call_model

        class _FakeMessage:
            content = "ok"

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]
            usage = None

        class _FakeLiteLLM:
            last_model_called: str | None = None

            def completion(self, model: str, messages: list[dict[str, str]]) -> _FakeResponse:
                self.last_model_called = model
                return _FakeResponse()

        fake = _FakeLiteLLM()
        out: SampledOutput = _call_model(fake, "deepseek-v3.2", [{"role": "user", "content": "hi"}])
        assert out.model == "deepseek-v3.2"  # bare, not "deepseek/deepseek-v3.2"
        assert fake.last_model_called == "deepseek/deepseek-v3.2"  # but routed on the wire

    @pytest.mark.parametrize(
        ("bare", "expected_prefix"),
        [
            ("deepseek-v3.2", "deepseek/"),
            ("grok-4", "xai/"),
            ("mistral-large-3", "mistral/"),
        ],
    )
    def test_get_llm_provider_resolves_the_routed_form(
        self, bare: str, expected_prefix: str
    ) -> None:
        """Live confirmation (not a guess) against the bundled LiteLLM install:
        the ROUTED form resolves a provider; the bare form does not."""
        litellm = pytest.importorskip("litellm")
        routed = measure._route_for_measure(bare)
        assert routed == expected_prefix + bare
        # Routed form resolves without raising.
        litellm.get_llm_provider(routed)
        # Bare form raises BadRequestError — proving the prefix is actually
        # necessary, not merely harmless decoration.
        with pytest.raises(litellm.exceptions.BadRequestError):
            litellm.get_llm_provider(bare)


class TestRoutePrefixNoOverlap:
    """P3-1: _LITELLM_ROUTE_PREFIX mixes vendor PREFIXES ("deepseek-") with
    FULL model names ("llama-4-scout-17b-16e-instruct"), and
    ``_route_for_measure`` matches via ``startswith`` — so if a shorter key
    ever became a proper prefix of a longer one, the match for the longer
    name would be nondeterministic (whichever key ``dict`` iteration reaches
    first), silently routing it through the WRONG vendor's prefix. This test
    is the standing guard: it must stay green as new entries are added.
    """

    def test_no_key_is_a_proper_prefix_of_another_key(self) -> None:
        from frugon.measure import _LITELLM_ROUTE_PREFIX

        keys = list(_LITELLM_ROUTE_PREFIX)
        offenders = [
            (short, long_)
            for short in keys
            for long_ in keys
            if short != long_ and long_.startswith(short)
        ]
        assert offenders == [], (
            f"_LITELLM_ROUTE_PREFIX keys overlap (a proper prefix relationship "
            f"exists): {offenders} — _route_for_measure's startswith() match "
            "would resolve these nondeterministically"
        )
