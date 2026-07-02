"""Asserts that frugon analyze --demo uses the SAME default recommendation pool
as a real run — demo == production, no demo-only illustrative pool.

FRG-OSS-034 Phase 3 un-pinned the demo: --demo now passes candidates=None to
analyze_records exactly like a real (non-demo) run, so the recommendation
reflects the live 23-model _ROUTING_CANDIDATES roster.  The ONE remaining pin
is narrower and different in kind: --demo --measure samples a single model
(_DEMO_MEASURE_CANDIDATE) so the try-out path needs only OPENAI_API_KEY — that
pin affects ONLY which model --measure samples, never the recommendation math
analyze_records computes (which always sees candidates=None for --demo).
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from typer.testing import CliRunner

from frugon.cli import app
from frugon.cost import _DEMO_MEASURE_CANDIDATE, AnalysisResult

runner = CliRunner()


def _minimal_analysis_result() -> AnalysisResult:
    """Return a minimal AnalysisResult that renders without crashing."""
    from decimal import Decimal

    return AnalysisResult(
        total_calls=0,
        priced_calls=0,
        unpriced_calls=0,
        total_cost=Decimal("0"),
        projected_cost=Decimal("0"),
        candidate_model=None,
        split=None,
        used_default_pool=False,
    )


class TestDemoUsesDefaultPool:
    def test_demo_uses_default_pool_not_a_demo_only_pin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When demo=True and no --candidates, analyze_records receives candidates=None
        — the same as a real (non-demo) run — so it falls back to the live
        _ROUTING_CANDIDATES pool internally.  Demo == production."""
        captured_candidates: list[list[str] | None] = []

        def _capturing_analyze_records(*args: Any, **kwargs: Any) -> AnalysisResult:
            captured_candidates.append(kwargs.get("candidates"))
            return _minimal_analysis_result()

        # Patch at the module level where cli.py imports it
        monkeypatch.setattr("frugon.cost.analyze_records", _capturing_analyze_records)

        result = runner.invoke(
            app,
            ["analyze", "--demo", "--no-progress"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
        assert len(captured_candidates) >= 1, "analyze_records was never called"
        candidates_used = captured_candidates[0]
        assert candidates_used is None, (
            f"Expected candidates=None (default pool, same as a real run), "
            f"got {candidates_used!r}. frugon analyze --demo must NOT pin a "
            "demo-only candidate pool — demo == production."
        )

    def test_real_analyze_uses_routing_candidates_not_demo(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When demo=False and no --candidates, analyze_records receives None (default pool)."""
        import json as _json

        captured_candidates: list[list[str] | None] = []

        def _capturing_analyze_records(*args: Any, **kwargs: Any) -> AnalysisResult:
            captured_candidates.append(kwargs.get("candidates"))
            return _minimal_analysis_result()

        monkeypatch.setattr("frugon.cost.analyze_records", _capturing_analyze_records)

        # Write a minimal single-record log so iter_records has something to parse
        log = tmp_path / "tiny.jsonl"
        record = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        log.write_text(_json.dumps(record) + "\n", encoding="utf-8")

        result = runner.invoke(
            app,
            ["analyze", str(log), "--no-progress"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
        assert len(captured_candidates) >= 1, "analyze_records was never called"
        # Both demo and non-demo paths pass None → analyze_records uses
        # _ROUTING_CANDIDATES internally. Demo and real runs are identical here.
        assert captured_candidates[0] is None, (
            f"Non-demo path should pass candidates=None, got {captured_candidates[0]!r}."
        )


class TestDemoMeasurePin:
    """--demo --measure samples a single pinned model so the try-out path needs
    only OPENAI_API_KEY.  This pin is scoped to WHICH MODEL --measure samples —
    it must never leak into the candidates= argument analyze_records receives
    (that is covered by TestDemoUsesDefaultPool above)."""

    def test_demo_measure_precheck_uses_single_pinned_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The --measure pre-flight key check for --demo --measure verifies only
        _DEMO_MEASURE_CANDIDATE's key — not whatever the live 23-model roster's
        recommendation happens to be (which could require any provider's key)."""
        captured_precheck_models: list[list[str]] = []

        def _capturing_verify(models: list[str]) -> None:
            captured_precheck_models.append(list(models))

        monkeypatch.setattr(
            "frugon.measure.verify_measure_prerequisites", _capturing_verify
        )
        # Ensure only OPENAI_API_KEY is "present" so a leak to a non-OpenAI model
        # would be caught by run_measure's own key check if it were ever reached
        # (the pre-check patch above prevents that call in this test).
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-not-real")

        runner.invoke(
            app,
            ["analyze", "--demo", "--measure", "--no-progress", "-y"],
        )

        assert len(captured_precheck_models) >= 1, (
            "verify_measure_prerequisites was never called for --demo --measure"
        )
        precheck_models = captured_precheck_models[0]
        assert _DEMO_MEASURE_CANDIDATE in precheck_models, (
            f"Expected the pinned {_DEMO_MEASURE_CANDIDATE!r} in the --demo "
            f"--measure precheck models, got {precheck_models!r}."
        )
        # dominant_model (gpt-5.5, the baseline) is always precked too — that's
        # expected. The invariant under test is narrower: the live roster's
        # actual RECOMMENDED candidate (whatever _ROUTING_CANDIDATES currently
        # resolves to for this baseline) must NOT appear, since --demo --measure
        # must never require a key for it.
        from frugon.cost import _ROUTING_CANDIDATES
        from frugon.routing import select_easy_target

        live_recommendation = select_easy_target("gpt-5.5", _ROUTING_CANDIDATES)
        assert live_recommendation is not None
        assert live_recommendation != _DEMO_MEASURE_CANDIDATE, (
            "test fixture assumption broken: the live roster's recommendation "
            "now coincides with the pinned measure candidate — pick a "
            "different assertion basis."
        )
        assert live_recommendation not in precheck_models, (
            f"--demo --measure precheck must not require a key for the live "
            f"roster's recommendation ({live_recommendation!r}); "
            f"got precheck models {precheck_models!r}."
        )
