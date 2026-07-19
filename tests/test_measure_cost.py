"""Tests for measurement-cost accounting and the pre-run estimate.

Two cost-transparency features:

  * Feature 2 — POST-run measurement cost: each provider call's token usage is
    captured (sampling + judge), aggregated on MeasureResult.measure_calls, and
    priced via frugon's own pricing table.  The rendered line discloses what the
    run cost the user (4-dp money, comma counts, unpriced flag).
  * Feature 3 — PRE-run estimate: before any call, the planned call count and a
    dollar estimate are computed from the sampled records' own token counts.

All tests run offline — LiteLLM is mocked and pricing is monkeypatched to fixed
per-token rates so the arithmetic is hand-verifiable.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from frugon.cost import LogRecord
from frugon.measure import (
    Comparison,
    MeasureCallUsage,
    MeasureResult,
    SampledOutput,
    _extract_usage,
    estimate_measure_cost,
    measurement_cost,
    planned_call_count,
    run_measure,
)
from frugon.pricing import ModelPrice
from frugon.report import _measurement_cost_text


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


# A fixed two-model price table: input $1/1k tokens, output $2/1k tokens for the
# baseline; cheaper for the candidate.  Hand-pickable so the arithmetic is exact.
_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(
        model="gpt-4o",
        input_cost_per_token=Decimal("0.001"),
        output_cost_per_token=Decimal("0.002"),
        source="test",
        pricing_json_last_synced=None,
    ),
    "gpt-4o-mini": ModelPrice(
        model="gpt-4o-mini",
        input_cost_per_token=Decimal("0.0001"),
        output_cost_per_token=Decimal("0.0002"),
        source="test",
        pricing_json_last_synced=None,
    ),
}


def _fixed_price(model: str) -> ModelPrice | None:
    return _PRICES.get(model)


_RECORD_COUNTER = 0


def _record(pt: int = 100, ct: int = 50, prompt: str | None = None) -> LogRecord:
    """Build a LogRecord with a UNIQUE prompt by default.

    sample_records dedups by prompt content, so identical fixture prompts
    would collapse the test universe to a single record.  An auto-incrementing
    counter keeps each call distinct unless the caller passes ``prompt=`` to
    force a duplicate (the dedup-aware tests do exactly that).
    """
    global _RECORD_COUNTER
    if prompt is None:
        _RECORD_COUNTER += 1
        prompt = f"Q{_RECORD_COUNTER}"
    return LogRecord(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        completion_text="A",
        prompt_tokens=pt,
        completion_tokens=ct,
        timestamp=None,
    )


def _usage_response(content: str, pt: int, ct: int) -> MagicMock:
    """A LiteLLM-shaped response carrying real integer usage."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = pt
    resp.usage.completion_tokens = ct
    return resp


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


def test_extract_usage_reads_prompt_and_completion_tokens() -> None:
    assert _extract_usage(_usage_response("x", 123, 45)) == (123, 45)


def test_extract_usage_missing_usage_block_degrades_to_zero() -> None:
    resp = MagicMock()
    resp.usage = None
    assert _extract_usage(resp) == (0, 0)


def test_extract_usage_non_integer_degrades_to_zero() -> None:
    resp = MagicMock()
    resp.usage.prompt_tokens = "not-a-number"
    resp.usage.completion_tokens = None
    assert _extract_usage(resp) == (0, 0)


# ---------------------------------------------------------------------------
# planned_call_count
# ---------------------------------------------------------------------------


def test_planned_call_count_sampling_only() -> None:
    # 50 prompts × (1 baseline + 2 candidates) = 150, no judge.
    assert planned_call_count(50, 2, use_judge=False) == 150


def test_planned_call_count_with_judge() -> None:
    # 50×(1+2) sampling + 50×2 judging = 150 + 100 = 250.
    assert planned_call_count(50, 2, use_judge=True) == 250


def test_planned_call_count_default_judge_run_is_thirty() -> None:
    # The default 10-sample single-candidate judge run is exactly 30 calls — the
    # documented gate floor (kept frictionless).
    assert planned_call_count(10, 1, use_judge=True) == 30


# ---------------------------------------------------------------------------
# measurement_cost — post-run accounting (Feature 2)
# ---------------------------------------------------------------------------


def test_measurement_cost_none_when_no_calls_captured() -> None:
    mr = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=None,
    )
    assert measurement_cost(mr) is None


def test_measurement_cost_exact_math(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frugon.pricing.get_model_price", _fixed_price)
    # 2 baseline gpt-4o calls (100 in @0.001, 50 out @0.002) = 2 × (0.1 + 0.1) = 0.4
    # 2 candidate gpt-4o-mini calls (100 in @0.0001, 50 out @0.0002) =
    #   2 × (0.01 + 0.01) = 0.04
    # Total = 0.44, across 4 calls, 0 unpriced.
    calls = [
        MeasureCallUsage(model="gpt-4o", prompt_tokens=100, completion_tokens=50),
        MeasureCallUsage(model="gpt-4o", prompt_tokens=100, completion_tokens=50),
        MeasureCallUsage(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50),
        MeasureCallUsage(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50),
    ]
    mr = MeasureResult(
        samples_requested=2,
        samples_taken=2,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=None,
        measure_calls=calls,
    )
    cost = measurement_cost(mr)
    assert cost is not None
    assert cost.total_cost == Decimal("0.44")
    assert cost.call_count == 4
    assert cost.unpriced_calls == 0


def test_measurement_cost_flags_unpriced_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frugon.pricing.get_model_price", _fixed_price)
    calls = [
        MeasureCallUsage(model="gpt-4o", prompt_tokens=100, completion_tokens=50),
        MeasureCallUsage(model="mystery-model", prompt_tokens=999, completion_tokens=999),
    ]
    mr = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["mystery-model"],
        comparisons=[],
        tier1_tallies=None,
        measure_calls=calls,
    )
    cost = measurement_cost(mr)
    assert cost is not None
    # Only the gpt-4o call is priced: 0.1 + 0.1 = 0.2.  The mystery model
    # contributes nothing and is flagged.
    assert cost.total_cost == Decimal("0.2")
    assert cost.call_count == 2
    assert cost.unpriced_calls == 1


# ---------------------------------------------------------------------------
# _measurement_cost_text — the rendered line (terminal/MD/HTML share it)
# ---------------------------------------------------------------------------


def _mr_with(calls: list[MeasureCallUsage]) -> MeasureResult:
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=None,
        measure_calls=calls,
    )


def test_cost_line_rendered_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frugon.pricing.get_model_price", _fixed_price)
    # 2 priced calls totalling exactly $0.0040 — hand-built: 2 gpt-4o-mini
    # calls of 10 in / 5 out each = 10×0.0001 + 5×0.0002 = 0.002 per call.
    one = MeasureCallUsage(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5)
    calls = [one] * 2  # 0.004 total, 2 calls
    text = _measurement_cost_text(_mr_with(calls))
    assert text == (
        "Measurement cost  ~$0.0040 · 2 calls "
        "(your spend at list prices — check your provider dashboard "
        "for the exact bill)"
    )


def test_cost_line_unpriced_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frugon.pricing.get_model_price", _fixed_price)
    calls = [
        MeasureCallUsage(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5),
        MeasureCallUsage(model="unknown-a", prompt_tokens=1, completion_tokens=1),
        MeasureCallUsage(model="unknown-b", prompt_tokens=1, completion_tokens=1),
    ]
    text = _measurement_cost_text(_mr_with(calls))
    assert text is not None
    # The unpriced clause now sits BEFORE the parenthetical so the user reads
    # the coverage caveat as part of the call-count clause, not trailing after
    # the explanation of what the figure represents.
    assert "3 calls · 2 calls unpriced (your spend at list prices" in text
    assert text.endswith(
        "(your spend at list prices — check your provider dashboard "
        "for the exact bill)"
    )


def test_cost_line_none_when_no_usage() -> None:
    assert _measurement_cost_text(_mr_with([])) is None


def test_cost_line_renders_in_md_and_html(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frugon.pricing.get_model_price", _fixed_price)
    from frugon.report import _quality_section_html, _quality_section_md

    rec = _record()
    comp = Comparison(
        record=rec,
        current_output=SampledOutput(model="gpt-4o", content="A"),
        candidate_outputs=[SampledOutput(model="gpt-4o-mini", content="B")],
        verdicts=["tie"],
    )
    calls = [MeasureCallUsage(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5)]
    mr = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[comp],
        tier1_tallies=None,
        measure_calls=calls,
    )
    md = "\n".join(_quality_section_md(mr))
    html = _quality_section_html(mr, style="v1")
    assert "Measurement cost" in md
    assert "your spend at list prices" in md
    assert "check your provider dashboard for the exact bill" in md
    assert "your bill, never Frugon's" not in md  # old wording must not regress
    assert "Measurement cost" in html
    assert 'class="quality-measure-cost"' in html
    assert "your spend at list prices" in html


# ---------------------------------------------------------------------------
# estimate_measure_cost — pre-run projection (Feature 3)
# ---------------------------------------------------------------------------


def test_estimate_planned_calls_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frugon.pricing.get_model_price", _fixed_price)
    # 4 records, 200 prompt / 80 completion tokens each.  1 candidate, judge on.
    records = [_record(pt=200, ct=80) for _ in range(4)]
    est = estimate_measure_cost(
        records,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        n_samples=4,
        use_judge=True,
        judge_model="gpt-4o",
    )
    # planned = 4×(1+1) + 4×1 = 12.
    assert est.planned_calls == 12
    # The plan inputs are surfaced so the CLI can render arithmetic that
    # reconciles to planned_calls exactly.
    assert est.n_prompts == 4
    assert est.n_candidates == 1
    assert est.use_judge is True
    # Per record:
    #   baseline gpt-4o: 200×0.001 + 80×0.002 = 0.2 + 0.16 = 0.36
    #   candidate gpt-4o-mini: 200×0.0001 + 80×0.0002 = 0.02 + 0.016 = 0.036
    #   judge gpt-4o: in = 200 + 80 + 80 = 360 ; out = 8 (fixed reply est)
    #     360×0.001 + 8×0.002 = 0.36 + 0.016 = 0.376
    #   per-record total = 0.36 + 0.036 + 0.376 = 0.772
    # × 4 records = 3.088
    assert est.estimated_cost == Decimal("3.088")
    assert est.unpriced_models == []


def test_estimate_unpriced_target_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("frugon.pricing.get_model_price", _fixed_price)
    records = [_record() for _ in range(3)]
    est = estimate_measure_cost(
        records,
        current_model="totally-unknown",
        candidates=["also-unknown"],
        n_samples=3,
    )
    # planned = 3×(1+1) = 6.  No target model is priceable → estimated_cost None,
    # both names flagged, but the call count is still meaningful.
    assert est.planned_calls == 6
    assert est.estimated_cost is None
    assert set(est.unpriced_models) == {"totally-unknown", "also-unknown"}


def test_estimate_samples_capped_to_available_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("frugon.pricing.get_model_price", _fixed_price)
    records = [_record() for _ in range(2)]  # only 2 records
    est = estimate_measure_cost(
        records,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        n_samples=10,  # asked for more than exist
    )
    # Only 2 prompts can be sampled → 2×(1+1) = 4 planned calls.  n_prompts is
    # capped to the 2 available records, NOT the 10 requested — so the rendered
    # arithmetic reconciles to planned_calls instead of overstating it.
    assert est.planned_calls == 4
    assert est.n_prompts == 2
    assert est.n_candidates == 1
    assert est.use_judge is False


# ---------------------------------------------------------------------------
# run_measure captures usage end-to-end (sampling + judge)
# ---------------------------------------------------------------------------


def _usage_litellm_mock() -> MagicMock:
    """A litellm mock whose every completion() carries integer usage (7 in / 3 out)."""
    mock = MagicMock()

    def _completion(*_a: Any, **_k: Any) -> MagicMock:
        return _usage_response("VERDICT: TIE", 7, 3)

    mock.completion.side_effect = _completion
    return mock


def test_run_measure_captures_sampling_and_judge_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock = _usage_litellm_mock()
    monkeypatch.setattr("frugon.measure._import_litellm", lambda: mock)
    records = [_record() for _ in range(3)]
    result = run_measure(
        records,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        n_samples=3,
        use_judge=True,
        judge_model="gpt-4o",
        concurrency=1,
        seed=7,
    )
    # 3 prompts × (1 baseline + 1 candidate) sampling = 6 ; 3 prompts × 1 judge = 3.
    # Every judge call ties ("VERDICT: TIE" from the mock), so the pointwise
    # "both failed" check also fires for every prompt: 1 baseline check
    # (cached per prompt) + 1 candidate check = 2, × 3 prompts = 6.
    # 6 + 3 + 6 = 15.
    assert len(result.measure_calls) == 15
    # Every captured call carries the stubbed usage (7 in / 3 out).
    assert all(c.prompt_tokens == 7 and c.completion_tokens == 3 for c in result.measure_calls)


def test_run_measure_failed_sampling_call_counts_as_zero_token_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock = MagicMock()

    def _completion(*, model: str, messages: Any) -> MagicMock:
        if model == "gpt-4o-mini":
            raise RuntimeError("provider down")  # candidate fails
        return _usage_response("ok", 11, 4)

    mock.completion.side_effect = _completion
    monkeypatch.setattr("frugon.measure._import_litellm", lambda: mock)
    records = [_record() for _ in range(2)]
    result = run_measure(
        records,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        n_samples=2,
        use_judge=False,
        concurrency=1,
    )
    # 2 baseline (priced, 11/4) + 2 candidate (failed → zero-token, still counted).
    assert len(result.measure_calls) == 4
    candidate_calls = [c for c in result.measure_calls if c.model == "gpt-4o-mini"]
    assert len(candidate_calls) == 2
    assert all(c.prompt_tokens == 0 and c.completion_tokens == 0 for c in candidate_calls)
