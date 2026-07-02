"""``frugon analyze --demo --measure`` must not attribute the headline
recommendation's offline quality phrase to the pinned try-out candidate.

The demo un-pinned its RECOMMENDATION pool (FRG-OSS-034 Phase 3): --demo now
recommends against the same live _ROUTING_CANDIDATES roster a real run uses,
which currently resolves to ``deepseek-v4-flash`` for the bundled fixture. But
``--demo --measure`` still samples a single pinned model,
``_DEMO_MEASURE_CANDIDATE`` ("gpt-4.1-mini"), so the try-out path needs only
OPENAI_API_KEY. When the judge confirms that pinned candidate, the synthesis
must not say "(offline 'within tolerance' -> verified on your data)" -- that
offline phrase describes deepseek-v4-flash's relationship to the baseline, not
gpt-4.1-mini's. The synthesis must instead disclose that the measured model is
the demo's try-out sample, not the headline recommendation.

These tests mock run_measure (no provider call, no real spend) and assert:
  (a) the divergence disclosure fires when the measured candidate differs
      from the headline recommendation, and
  (b) no "verified your recommendation"-shaped claim (the offline-phrase
      back-reference) renders in that case.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from frugon import measure
from frugon.cli import app
from frugon.cost import _DEMO_MEASURE_CANDIDATE
from frugon.measure import Comparison, MeasureResult, SampledOutput, Tier1Tally

runner = CliRunner()

_WIDE_ENV = {
    "COLUMNS": "220",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "OPENAI_API_KEY": "sk-test-not-real",
}


def _fake_confirmed_measure_result() -> MeasureResult:
    """A MeasureResult where the pinned demo candidate CONFIRMS quality (holds
    every sampled prompt) — the exact shape that used to trigger the
    misattributed "(offline 'within tolerance' -> verified on your data)" line.
    """
    comp = Comparison(
        record=__import__("frugon.cost", fromlist=["LogRecord"]).LogRecord(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "classify this ticket"}],
            completion_text="technical",
            prompt_tokens=40,
            completion_tokens=5,
            timestamp=None,
        ),
        current_output=SampledOutput(model="gpt-5.5", content="technical"),
        candidate_outputs=[
            SampledOutput(model=_DEMO_MEASURE_CANDIDATE, content="technical")
        ],
        verdicts=["win"],
    )
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-5.5",
        candidates=[_DEMO_MEASURE_CANDIDATE],
        comparisons=[comp],
        tier1_tallies=[
            Tier1Tally(candidate=_DEMO_MEASURE_CANDIDATE, wins=1, losses=0, ties=0)
        ],
    )


def _patch_measure_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the [measure] extra + key checks pass and skip the network."""
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setattr(measure, "verify_measure_prerequisites", lambda *_a, **_k: None)


def test_demo_measure_confirmed_verdict_discloses_divergence_not_offline_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--demo --measure --judge, pinned candidate CONFIRMS: the synthesis names
    the divergence between the measured model and the headline recommendation,
    and never claims the offline estimate was "verified on your data" for a
    model that offline estimate does not describe.
    """
    _patch_measure_engine(monkeypatch)
    sentinel = _fake_confirmed_measure_result()
    monkeypatch.setattr(measure, "run_measure", lambda *_a, **_k: sentinel)

    result = runner.invoke(
        app,
        ["analyze", "--demo", "--measure", "--judge", "--no-progress", "-y"],
        env=_WIDE_ENV,
    )

    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    out = " ".join(result.output.split())

    # (a) the divergence disclosure fires — names the pinned model, the word
    # "recommendation", and points at the real headline candidate.
    assert _DEMO_MEASURE_CANDIDATE in out
    assert "try-out sample model" in out
    assert "not the headline recommendation" in out
    assert "verifies the --measure flow, not the recommended switch" in out

    # (b) no "verified your recommendation"-shaped claim: the offline-phrase
    # back-reference wording must NOT appear anywhere in this run's output.
    assert "verified on your data" not in out
    assert "within tolerance' →" not in out
    assert "same or better quality' →" not in out


def test_demo_measure_confirmed_verdict_still_shows_confirmed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The divergence disclosure REPLACES the offline back-reference clause,
    not the whole verdict — "Estimate confirmed" and the held/scored tally
    still render; only the misattributed offline-phrase clause is gone.
    """
    _patch_measure_engine(monkeypatch)
    sentinel = _fake_confirmed_measure_result()
    monkeypatch.setattr(measure, "run_measure", lambda *_a, **_k: sentinel)

    result = runner.invoke(
        app,
        ["analyze", "--demo", "--measure", "--judge", "--no-progress", "-y"],
        env=_WIDE_ENV,
    )

    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    out = " ".join(result.output.split())
    assert "Estimate confirmed" in out
    assert "held quality in" in out
    assert "1/1 sampled prompt" in out


def test_explicit_candidates_matching_recommendation_keeps_offline_backreference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: when the MEASURED candidate genuinely IS the headline
    recommendation (the common, non-demo case), the offline-phrase
    back-reference still renders — the divergence branch must not fire for
    every --measure run, only when the models actually differ.
    """
    import json
    import pathlib
    import tempfile

    _patch_measure_engine(monkeypatch)

    # A tiny real log whose single priced model is gpt-4-turbo baseline; with
    # no --candidates, the offline split recommends the cheapest quality-
    # preserving candidate from the live default pool — deepseek-v4-flash is
    # NOT reachable from gpt-4-turbo's tier here, so pin an explicit candidate
    # that IS the recommendation by construction: pass --candidates naming the
    # exact model we then measure, so measured == recommended by construction.
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    log = tmp_dir / "log.jsonl"
    rows = [
        {
            "model": "gpt-4-turbo",
            "request": {"messages": [{"role": "user", "content": "classify: spam?"}]},
            "response": {"choices": [{"message": {"content": "no"}}]},
            "usage": {"prompt_tokens": 20, "completion_tokens": 3},
        }
        for _ in range(6)
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

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
        candidate_outputs=[SampledOutput(model="gpt-4o-mini", content="no")],
        verdicts=["win"],
    )
    sentinel = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=[comp],
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=1, losses=0, ties=0)],
    )
    monkeypatch.setattr(measure, "run_measure", lambda *_a, **_k: sentinel)

    result = runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--candidates",
            "gpt-4o-mini",
            "--measure",
            "--judge",
            "--no-progress",
            "-y",
        ],
        env=_WIDE_ENV,
    )

    assert result.exit_code == 0, f"exited {result.exit_code}: {result.output}"
    out = " ".join(result.output.split())
    # measured == recommended here (both gpt-4o-mini), so the ORIGINAL
    # offline-phrase back-reference still renders — no divergence note.
    assert "verified on your data" in out
    assert "try-out sample model" not in out
