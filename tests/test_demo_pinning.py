"""Asserts that frugon analyze --demo uses _DEMO_CANDIDATES, not _ROUTING_CANDIDATES.

When the default routing pool evolves, the demo's recommendation must remain
stable.  This test monkeypatches analyze_records to capture which candidates
are passed, then asserts they equal _DEMO_CANDIDATES exactly.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from typer.testing import CliRunner

from frugon.cli import app
from frugon.cost import _DEMO_CANDIDATES, AnalysisResult

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


class TestDemoUsesDemoCandidates:
    def test_demo_uses_demo_candidates_not_routing_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When demo=True and no --candidates, analyze_records receives _DEMO_CANDIDATES."""
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
        assert candidates_used == _DEMO_CANDIDATES, (
            f"Expected _DEMO_CANDIDATES={_DEMO_CANDIDATES!r}, "
            f"got {candidates_used!r}. "
            "frugon analyze --demo must pin the demo pool, not the live routing pool."
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
        # Non-demo path passes None → analyze_records uses _ROUTING_CANDIDATES internally
        assert captured_candidates[0] is None, (
            f"Non-demo path should pass candidates=None, got {captured_candidates[0]!r}. "
            "Only the --demo path must override with _DEMO_CANDIDATES."
        )
