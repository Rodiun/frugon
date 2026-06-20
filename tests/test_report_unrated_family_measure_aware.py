"""Measurement-awareness for the unrated-recommendation caveat (finding #1).

Under the unified split basis every priced candidate (rated or not) competes on
its full-dataset split New-spend; the cheapest becomes the headline routing
target.  An unrated candidate is therefore never "held out of the split" or
forced into a wholesale fallback — the old finding-#4 "split-skipped /
wholesale-fallback" notes are obsolete and removed.  The ONLY remaining unrated
disclosure is the #1 caveat: when the chosen ROUTING TARGET is unrated, name it
and the verify command.

These tests pin the measurement-awareness of that #1 caveat:

  * For an unrated recommendation JUDGED in this run (a Tier-1 tally is present —
    the same condition under which a verdict renders in the quality section), the
    #1 caveat is SUPPRESSED: the measured verdict IS the verification, so a "run
    --measure to confirm" caveat would be contradictory.
  * For an unrated recommendation NOT judged this run (the default), the caveat
    is shown verbatim — the quality is genuinely unverified, a real caution.

It also pins :func:`judged_models_from_measure`, the bridge from a run to the
per-model judged set the suppression keys off.  All tests run fully offline.

The unrated role is a priced-but-unrated sentinel installed via the shared
``install_unrated_sentinel`` fixture (absent from every quality table, so always
"unrated"), keeping the behaviour independent of real-registry drift.
``gpt-4o-mini`` is Capable (rated) in the bundled quality table.
"""

from __future__ import annotations

import pathlib as _pathlib
import sys as _sys
from pathlib import Path
from typing import Any

import pytest

import frugon
from frugon.cost import AnalysisResult, analyze_records, iter_records
from frugon.measure import Comparison, MeasureResult, SampledOutput, Tier1Tally
from frugon.report import (
    _recommended_unrated_model,
    _unrated_family_messages,
    _unrated_recommendation_caveat,
    judged_models_from_measure,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
    render_terminal,
)

_sys.path.insert(0, str(_pathlib.Path(__file__).parent))
from conftest import FRUGON_TEST_UNRATED, install_unrated_sentinel

assert frugon.__file__ is not None
_SAMPLE = Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"

_UNRATED = FRUGON_TEST_UNRATED  # priced-but-unrated sentinel (drift-proof)
_RATED = "gpt-4o-mini"  # Capable tier in the bundled quality table

# The "run --measure --judge" imperative the #1 caveat carries when NOT judged.
_IMPERATIVE = f"--measure --judge --candidates {_UNRATED}"


@pytest.fixture(autouse=True)
def _sentinel_pricing(monkeypatch, tmp_path):
    install_unrated_sentinel(monkeypatch, tmp_path)
    yield
    import frugon.pricing as _p

    _p.clear_pricing_cache()


def _result(candidates: list[str]) -> AnalysisResult:
    """Build the real ``--demo`` AnalysisResult for *candidates* via the engine."""
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(
        list(records),
        candidates=candidates,
        skipped_malformed=skipped,
        split_routing=True,
    )


def _record(prompt: str) -> Any:
    from frugon.cost import LogRecord

    return LogRecord(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _tier1_judging(*judged: str) -> MeasureResult:
    """A Tier-1 judge run that scored each model in *judged* (all confirmed)."""
    comparisons = [
        Comparison(
            record=_record(f"prompt {i}"),
            current_output=SampledOutput("gpt-4-turbo", "base"),
            candidate_outputs=[SampledOutput(m, f"cand {i}") for m in judged],
            verdicts=["tie" for _ in judged],
        )
        for i in range(5)
    ]
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4-turbo",
        candidates=list(judged),
        comparisons=comparisons,
        tier1_tallies=[
            Tier1Tally(candidate=m, wins=5, losses=0, ties=0) for m in judged
        ],
    )


def _tier0() -> MeasureResult:
    """A Tier-0 sample (--measure without --judge): no tallies, no verdict."""
    return MeasureResult(
        samples_requested=2,
        samples_taken=2,
        current_model="gpt-4-turbo",
        candidates=[_UNRATED],
        comparisons=[
            Comparison(
                record=_record("hi"),
                current_output=SampledOutput("gpt-4-turbo", "a"),
                candidate_outputs=[SampledOutput(_UNRATED, "b")],
            )
        ],
        tier1_tallies=None,
    )


def _flat(text: str) -> str:
    """Collapse whitespace so wrapped terminal lines match the source string."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# judged_models_from_measure — the bridge from a run to the per-model set
# ---------------------------------------------------------------------------


def test_judged_models_from_measure_tier1_returns_scored_candidates() -> None:
    mr = _tier1_judging(_RATED, _UNRATED)
    assert judged_models_from_measure(mr) == frozenset({_RATED, _UNRATED})


def test_judged_models_from_measure_tier0_is_empty() -> None:
    assert judged_models_from_measure(_tier0()) == frozenset()


def test_judged_models_from_measure_none_is_empty() -> None:
    assert judged_models_from_measure(None) == frozenset()


# ---------------------------------------------------------------------------
# #1 caveat measurement-awareness on _unrated_family_messages
# ---------------------------------------------------------------------------


def _unrated_recommendation_result() -> AnalysisResult:
    """A run whose chosen routing target is the unrated model (single candidate)."""
    result = _result([_UNRATED])
    # Precondition: the unified split basis routes to the unrated candidate.
    assert _recommended_unrated_model(result) == _UNRATED
    return result


def test_unrated_recommendation_not_judged_keeps_caveat() -> None:
    """Not judged this run → the #1 caveat is shown verbatim (a real caution)."""
    result = _unrated_recommendation_result()
    items = _unrated_family_messages(result)  # default: not judged
    messages = [m for m, _sev in items]
    assert _unrated_recommendation_caveat(_UNRATED) in messages


def test_unrated_recommendation_judged_suppresses_caveat() -> None:
    """Judged this run → the #1 caveat is SUPPRESSED (the verdict IS the proof)."""
    result = _unrated_recommendation_result()
    items = _unrated_family_messages(result, frozenset({_UNRATED}))
    assert items == []


def test_tier0_measurement_treated_as_not_judged() -> None:
    """A Tier-0 sample (no verdict) keeps the caveat — nothing was scored."""
    result = _unrated_recommendation_result()
    items = _unrated_family_messages(
        result, judged_models_from_measure(_tier0())
    )
    messages = [m for m, _sev in items]
    assert _unrated_recommendation_caveat(_UNRATED) in messages


# ---------------------------------------------------------------------------
# Surface parity — every renderer honours the suppression
# ---------------------------------------------------------------------------


def test_terminal_not_judged_keeps_caveat(capsys: Any) -> None:
    render_terminal(_unrated_recommendation_result())
    out = _flat(capsys.readouterr().out)
    assert _flat(_unrated_recommendation_caveat(_UNRATED)) in out


def test_terminal_judged_suppresses_caveat(capsys: Any) -> None:
    # render_terminal takes the resolved judged-model set directly (the CLI
    # derives it from the measure run via judged_models_from_measure).
    render_terminal(
        _unrated_recommendation_result(),
        judged_models=frozenset({_UNRATED}),
    )
    out = _flat(capsys.readouterr().out)
    # The "run --measure --candidates X to confirm" imperative must not appear:
    # the model was just measured in the quality section.
    assert _flat(_unrated_recommendation_caveat(_UNRATED)) not in out


@pytest.mark.parametrize(
    ("renderer", "judged"),
    [
        (render_markdown, False),
        (render_markdown, True),
        (render_markdown_v2, False),
        (render_markdown_v2, True),
    ],
)
def test_markdown_caveat_suppression(
    renderer: Any, judged: bool, tmp_path: Path
) -> None:
    out = tmp_path / "r.md"
    renderer(
        _unrated_recommendation_result(),
        out,
        measure_result=_tier1_judging(_UNRATED) if judged else None,
    )
    text = out.read_text(encoding="utf-8")
    if judged:
        assert _unrated_recommendation_caveat(_UNRATED) not in text
    else:
        assert _unrated_recommendation_caveat(_UNRATED) in text


@pytest.mark.parametrize(
    ("renderer", "judged"),
    [
        (render_html, False),
        (render_html, True),
        (render_html_v2, False),
        (render_html_v2, True),
    ],
)
def test_html_caveat_suppression(
    renderer: Any, judged: bool, tmp_path: Path
) -> None:
    out = tmp_path / "r.html"
    renderer(
        _unrated_recommendation_result(),
        out,
        measure_result=_tier1_judging(_UNRATED) if judged else None,
    )
    text = out.read_text(encoding="utf-8")
    if judged:
        assert _IMPERATIVE not in text
    else:
        assert _IMPERATIVE in text
