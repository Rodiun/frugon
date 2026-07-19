"""Tests for frugon.cost — token counting, cost computation, routing projection.

Covers:
  - Golden vectors with usage block present (P3-5 existing path)
  - Golden vector for no-usage block (count-from-messages path) (P3-5 new requirement)
  - Monthly extrapolation: --window path, real-span path, no-timestamps-no-flag path (P1-2)
  - ZeroDivisionError guard when all calls are unpriced (P3-4)
  - compute_saving_pct returns None when current == 0 (P3-4)
"""

from __future__ import annotations

import json
import pathlib
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from frugon.cost import (
    CallCost,
    LogRecord,
    _best_candidate,
    _safe_int,
    analyze_logs,
    analyze_records,
    compute_call_cost,
    compute_saving_pct,
    iter_records,
    parse_record,
    scan_models,
    window_contradicts_span,
)
from frugon.pricing import ModelPrice
from frugon.quality import UNRATED_TIER, get_model_tier

sys.path.insert(0, str(Path(__file__).parent))
from conftest import install_synthetic_quality

# ---------------------------------------------------------------------------
# Fixtures — frozen prices so tests are independent of tokencost network state
# ---------------------------------------------------------------------------

# These prices match the bundled pricing.json (models present there win by spec).
# Keep them in sync with src/frugon/data/pricing.json.
GPT4O_IN = Decimal("0.0000025")
GPT4O_OUT = Decimal("0.00001")
GPT4T_IN = Decimal("0.00001")
GPT4T_OUT = Decimal("0.00003")
MINI_IN = Decimal("0.00000015")
MINI_OUT = Decimal("0.0000006")


def _write_jsonl(records: list[dict[str, Any]], tmp_path: Path) -> Path:
    """Write records to a temp JSONL file and return the path."""
    p = tmp_path / "test_logs.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return p


# ---------------------------------------------------------------------------
# P3-5: Golden vectors — usage block present
# ---------------------------------------------------------------------------


class TestGoldenVectorsWithUsageBlock:
    """Golden vectors using records that include explicit usage blocks."""

    def test_single_gpt4o_call_correct_cost(self, tmp_path: Path) -> None:
        """Arrange: one gpt-4o call with known prompt/completion tokens.
        Act: analyze_logs.
        Assert: total_cost matches hand-calculated expected value.
        """
        # Arrange
        # 100 prompt tokens × $0.0000025 + 50 completion × $0.00001 = $0.00075
        record = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "test"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        path = _write_jsonl([record], tmp_path)

        # Act
        result = analyze_logs(path)

        # Assert
        expected = GPT4O_IN * 100 + GPT4O_OUT * 50
        assert result.total_cost == expected, (
            f"Expected ${expected}, got ${result.total_cost}"
        )
        assert result.priced_calls == 1
        assert result.unpriced_calls == 0

    def test_mixed_models_cost_aggregation(self, tmp_path: Path) -> None:
        """Arrange: one gpt-4o call + one gpt-4-turbo call.
        Act: analyze_logs.
        Assert: cost_by_model maps each model correctly.
        """
        # Arrange
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "summarize this"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "summary"}}]},
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "review code"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "lgtm"}}]},
                "usage": {"prompt_tokens": 30, "completion_tokens": 15},
            },
        ]
        path = _write_jsonl(records, tmp_path)

        # Act
        result = analyze_logs(path)

        # Assert
        expected_gpt4o = GPT4O_IN * 20 + GPT4O_OUT * 10
        expected_gpt4t = GPT4T_IN * 30 + GPT4T_OUT * 15
        assert result.cost_by_model["gpt-4o"] == expected_gpt4o
        assert result.cost_by_model["gpt-4-turbo"] == expected_gpt4t
        assert result.total_cost == expected_gpt4o + expected_gpt4t


# ---------------------------------------------------------------------------
# P3-5: Golden vector — NO usage block (count-from-messages path)
# ---------------------------------------------------------------------------


class TestGoldenVectorNoUsageBlock:
    """Golden vector for the count-from-messages code path (no usage key)."""

    def test_no_usage_block_uses_tokenizer(self, tmp_path: Path) -> None:
        """Arrange: record with NO usage block; messages and completion present.
        Act: analyze_logs.
        Assert: the call is priced (non-zero cost) and token_source is 'counted'.

        This golden vector verifies the count-from-messages path executes and
        produces a positive cost, preventing silent zero-cost fallback.
        """
        # Arrange — no 'usage' key; tokencost must count from messages
        record: dict[str, Any] = {
            "model": "gpt-4o",
            "request": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ]
            },
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "4"}}]
            },
            # No "usage" key
        }
        path = _write_jsonl([record], tmp_path)

        # Act
        result = analyze_logs(path)

        # Assert — cost must be positive (not silently zero)
        assert result.priced_calls == 1
        assert result.total_cost > Decimal("0"), (
            "Expected positive cost from count-from-messages path"
        )

    def test_no_usage_block_cost_within_expected_range(self, tmp_path: Path) -> None:
        """Arrange: known short messages, no usage block.
        Act: parse_record then compute_call_cost.
        Assert: token counts and cost are in the expected range (±5 tokens).

        Hand-verification: 'You are a helpful assistant.' + 'What is 2 + 2?' ~10-15
        prompt tokens; '4' ~1 completion token.  Total cost < $0.000001 for gpt-4o.
        """
        # Arrange
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        record: dict[str, Any] = {
            "model": "gpt-4o",
            "request": {"messages": messages},
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "4"}}]
            },
        }

        # Act
        parsed = parse_record(record)
        assert parsed is not None
        cc = compute_call_cost(parsed)

        # Assert — token source and range
        assert cc.token_source == "counted"
        # Prompt tokens for these messages should be 10–25
        assert 5 <= parsed.prompt_tokens <= 30, (
            f"Unexpected prompt token count: {parsed.prompt_tokens}"
        )
        # Completion token for "4" should be 1
        assert parsed.completion_tokens >= 1
        assert cc.total_cost > Decimal("0")


# ---------------------------------------------------------------------------
# P1-2: Monthly extrapolation — three paths
# ---------------------------------------------------------------------------


class TestMonthlyExtrapolation:
    """P1-2: extrapolation only happens with explicit --window or real timestamps."""

    def test_window_flag_sets_window_days(self, tmp_path: Path) -> None:
        """Arrange: records with NO timestamps, --window 7.
        Act: analyze_logs with window_days=7.
        Assert: result.window_days == 7, observed_span_days is None.
        """
        # Arrange
        record = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            # No 'timestamp' key
        }
        path = _write_jsonl([record], tmp_path)

        # Act
        result = analyze_logs(path, window_days=7)

        # Assert
        assert result.window_days == 7
        assert result.observed_span_days is None

    def test_real_timestamps_compute_span(self, tmp_path: Path) -> None:
        """Arrange: records with timestamps 2 days apart, no --window.
        Act: analyze_logs without window_days.
        Assert: observed_span_days is approximately 2, window_days is None.
        """
        # Arrange — timestamps exactly 2 days apart
        records = [
            {
                "model": "gpt-4o",
                "timestamp": "2026-01-01T00:00:00Z",
                "request": {"messages": [{"role": "user", "content": "first"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "a"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
            {
                "model": "gpt-4o",
                "timestamp": "2026-01-03T00:00:00Z",
                "request": {"messages": [{"role": "user", "content": "second"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "b"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        ]
        path = _write_jsonl(records, tmp_path)

        # Act
        result = analyze_logs(path)

        # Assert
        assert result.window_days is None
        assert result.observed_span_days is not None
        assert abs(result.observed_span_days - 2.0) < 0.01, (
            f"Expected span ~2.0 days, got {result.observed_span_days}"
        )
        # Span bounds expose the same min/max the span is derived from, as ISO dates.
        assert result.observed_span_start == "2026-01-01"
        assert result.observed_span_end == "2026-01-03"

    def test_no_timestamps_no_window_no_projection(self, tmp_path: Path) -> None:
        """Arrange: records with NO timestamps, no --window.
        Act: analyze_logs without window_days.
        Assert: window_days is None, observed_span_days is None (no projection invented).
        """
        # Arrange
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "classify"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "yes"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        # Act
        result = analyze_logs(path)

        # Assert — no projection invented
        assert result.window_days is None
        assert result.observed_span_days is None
        # No parseable timestamps → no span bounds to disclose.
        assert result.observed_span_start is None
        assert result.observed_span_end is None


# ---------------------------------------------------------------------------
# P3-4: Division-by-zero guard — all-unknown-model logs
# ---------------------------------------------------------------------------


class TestZeroDivisionGuard:
    """P3-4: compute_saving_pct and analyze_logs must not raise on zero cost."""

    def test_compute_saving_pct_returns_none_for_zero_current(self) -> None:
        """Arrange: current cost == 0.
        Act: compute_saving_pct(0, 0).
        Assert: returns None, no ZeroDivisionError.
        """
        # Act
        result = compute_saving_pct(Decimal("0"), Decimal("0"))

        # Assert
        assert result is None

    def test_compute_saving_pct_returns_none_for_zero_current_positive_projected(self) -> None:
        """Arrange: current == 0, projected > 0 (edge case).
        Act: compute_saving_pct.
        Assert: returns None, no error.
        """
        result = compute_saving_pct(Decimal("0"), Decimal("0.05"))
        assert result is None

    def test_analyze_all_unknown_models_returns_zero_priced(self, tmp_path: Path) -> None:
        """Arrange: all log records use an unknown model name.
        Act: analyze_logs.
        Assert: priced_calls == 0, total_cost == 0, no exception.
        """
        # Arrange
        records = [
            {
                "model": "unknown-model-xyz-9999",
                "request": {"messages": [{"role": "user", "content": "test"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        # Act
        result = analyze_logs(path)

        # Assert
        assert result.priced_calls == 0
        assert result.total_cost == Decimal("0")
        assert result.projected_cost == Decimal("0")
        assert result.candidate_model is None

    def test_analyze_no_priced_calls_no_zero_division(self, tmp_path: Path) -> None:
        """Arrange: mix of unknown models.
        Act: analyze_logs then compute_saving_pct.
        Assert: saving is None, no exception raised.
        """
        records = [
            {
                "model": "mystery-model-alpha",
                "request": {"messages": [{"role": "user", "content": "hello"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "world"}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)
        saving = compute_saving_pct(result.total_cost, result.projected_cost)
        assert saving is None


# ---------------------------------------------------------------------------
# P2-3: Pricing precedence
# ---------------------------------------------------------------------------


class TestPricingPrecedence:
    """P2-3: pricing.json wins over tokencost when model is present in both."""

    def test_pricing_json_model_uses_override_price(self, tmp_path: Path) -> None:
        """Arrange: a model present in pricing.json at a known price.
        Act: compute cost for that model.
        Assert: cost matches pricing.json rate, not tokencost rate.

        gpt-4o is in our pricing.json at $0.0000025/$0.00001.
        tokencost also carries gpt-4o. We verify our override wins.
        """
        # Arrange — 1000 prompt tokens, 0 completion tokens for clean math
        record = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "x"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": ""}}]},
            "usage": {"prompt_tokens": 1000, "completion_tokens": 0},
        }
        path = _write_jsonl([record], tmp_path)

        # Act
        result = analyze_logs(path)

        # Assert — cost = 1000 × $0.0000025 = $0.0025 exactly
        expected = GPT4O_IN * 1000
        assert result.total_cost == expected, (
            f"Expected ${expected} (pricing.json rate), got ${result.total_cost}"
        )

    def test_pricing_source_is_pricing_json_not_tokencost(self) -> None:
        """Arrange: resolve gpt-4o price.
        Act: get_model_price.
        Assert: source field is 'pricing.json'.
        """
        from frugon.pricing import get_model_price

        price = get_model_price("gpt-4o")
        assert price is not None
        assert price.source == "pricing.json", (
            f"Expected source='pricing.json', got '{price.source}'"
        )

    def test_unknown_model_falls_back_to_tokencost(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """Arrange: model not in pricing.json seed, not reachable via base-family
        folding, but known to tokencost — and user-data-dir is isolated to tmp_path
        so a developer's synced pricing.json cannot shadow the result.
        Act: get_model_price.
        Assert: source is 'tokencost'.

        o1-mini is present in tokencost and absent from the bundled seed.
        Its base_family is 'o1-mini' (no date suffix), which is also absent
        from the seed, so the base-family fallback step cannot resolve it via
        pricing.json.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import get_model_price

        # Isolate: point both paths at a location that does not exist so that
        # the real user-data-dir pricing.json (which may contain this model
        # after a 'frugon pricing update') never shadows the test.
        absent = pathlib.Path(tmp_path / "no_pricing.json")
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", absent)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", absent)

        price = get_model_price("o1-mini")
        assert price is not None, "o1-mini must be known to tokencost"
        assert price.source == "tokencost"


# ---------------------------------------------------------------------------
# compute_saving_pct — positive saving calculation
# ---------------------------------------------------------------------------


class TestComputeSavingPct:
    """Saving percentage arithmetic correctness."""

    def test_fifty_percent_saving(self) -> None:
        """Arrange: current=1.0, projected=0.5.
        Act: compute_saving_pct.
        Assert: 50%.
        """
        result = compute_saving_pct(Decimal("1.0"), Decimal("0.5"))
        assert result == Decimal("50"), f"Expected 50, got {result}"

    def test_zero_projected_means_100_percent_saving(self) -> None:
        """Arrange: current>0, projected=0.
        Act: compute_saving_pct.
        Assert: 100%.
        """
        result = compute_saving_pct(Decimal("1.0"), Decimal("0"))
        assert result == Decimal("100")

    def test_equal_costs_means_zero_saving(self) -> None:
        """Arrange: current == projected.
        Act: compute_saving_pct.
        Assert: 0%.
        """
        result = compute_saving_pct(Decimal("0.5"), Decimal("0.5"))
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# window_contradicts_span — the --window-vs-observed-span sanity predicate
# ---------------------------------------------------------------------------


class TestWindowContradictsSpan:
    """A ``--window`` override that materially disagrees with the real span fires.

    ``--window N`` overrides the monthly-projection basis (total_cost × 30/N), so a
    window much shorter/longer than the log's actual observed span silently scales
    the monthly figure.  The predicate must fire only when BOTH values are present
    and they differ by at least the ratio threshold (default 1.5) in either
    direction, and must never raise on missing/zero/negative inputs.
    """

    def test_fires_when_window_much_shorter_than_span(self) -> None:
        """Arrange: --window 7 on a ~30-day span (ratio ~4.3 >= 1.5).
        Act: window_contradicts_span.
        Assert: True.
        """
        assert window_contradicts_span(7, 30.0) is True

    def test_fires_when_window_much_longer_than_span(self) -> None:
        """Arrange: --window 30 on a ~7-day span (ratio ~4.3, other direction).
        Act: window_contradicts_span.
        Assert: True — the contradiction is symmetric.
        """
        assert window_contradicts_span(30, 7.0) is True

    def test_fires_exactly_at_threshold(self) -> None:
        """Arrange: window 30, span 20 → ratio exactly 1.5 (inclusive bound).
        Act: window_contradicts_span.
        Assert: True (>= threshold).
        """
        assert window_contradicts_span(30, 20.0) is True

    def test_no_fire_when_window_matches_span(self) -> None:
        """Arrange: --window 30 on a ~30-day span (ratio 1.0).
        Act: window_contradicts_span.
        Assert: False.
        """
        assert window_contradicts_span(30, 30.0) is False

    def test_no_fire_when_window_close_to_span(self) -> None:
        """Arrange: --window 28 on a 30-day span (ratio ~1.07 < 1.5).
        Act: window_contradicts_span.
        Assert: False — a near-match is not a contradiction.
        """
        assert window_contradicts_span(28, 30.0) is False

    def test_no_fire_just_below_threshold(self) -> None:
        """Arrange: window 29, span 20 → ratio 1.45 < 1.5.
        Act: window_contradicts_span.
        Assert: False (strictly below the inclusive bound).
        """
        assert window_contradicts_span(29, 20.0) is False

    def test_no_fire_when_span_is_none(self) -> None:
        """Arrange: --window given but no timestamps (span None).
        Act: window_contradicts_span.
        Assert: False — nothing to compare against.
        """
        assert window_contradicts_span(7, None) is False

    def test_no_fire_when_window_is_none(self) -> None:
        """Arrange: no --window flag (window None) even with a known span.
        Act: window_contradicts_span.
        Assert: False — no override to warn about.
        """
        assert window_contradicts_span(None, 30.0) is False

    def test_no_fire_when_both_none(self) -> None:
        """Arrange: neither value present.
        Act: window_contradicts_span.
        Assert: False.
        """
        assert window_contradicts_span(None, None) is False

    def test_no_fire_on_zero_window(self) -> None:
        """Arrange: degenerate window 0 (guards div-by-zero).
        Act: window_contradicts_span.
        Assert: False — never raises.
        """
        assert window_contradicts_span(0, 30.0) is False

    def test_no_fire_on_zero_span(self) -> None:
        """Arrange: degenerate span 0.0 (guards div-by-zero).
        Act: window_contradicts_span.
        Assert: False — never raises.
        """
        assert window_contradicts_span(7, 0.0) is False

    def test_no_fire_on_negative_span(self) -> None:
        """Arrange: nonsensical negative span.
        Act: window_contradicts_span.
        Assert: False.
        """
        assert window_contradicts_span(7, -5.0) is False

    def test_custom_threshold_respected(self) -> None:
        """Arrange: window 30, span 20 (ratio 1.5) with a stricter 2.0 threshold.
        Act: window_contradicts_span with ratio_threshold=2.0.
        Assert: False — below the raised bound.
        """
        assert window_contradicts_span(30, 20.0, ratio_threshold=2.0) is False


# ---------------------------------------------------------------------------
# Routing — custom candidates path
# ---------------------------------------------------------------------------


class TestCustomCandidates:
    """analyze_logs with an explicit candidates list."""

    def test_custom_candidate_overrides_auto_detection(self, tmp_path: Path) -> None:
        """Arrange: gpt-4-turbo baseline, explicit gpt-4o-mini candidate.
        Act: analyze_logs with candidates=['gpt-4o-mini'].
        Assert: candidate_model is gpt-4o-mini, projected cost is positive.
        """
        # Arrange
        records = [
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "classify this"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "billing"}}]},
                "usage": {"prompt_tokens": 40, "completion_tokens": 1},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        # Act
        result = analyze_logs(path, candidates=["gpt-4o-mini"])

        # Assert
        assert result.candidate_model == "gpt-4o-mini"
        assert result.projected_cost > Decimal("0")

    def test_unknown_custom_candidate_falls_back_gracefully(self, tmp_path: Path) -> None:
        """Arrange: gpt-4o baseline, candidate list with only unknown model.
        Act: analyze_logs with candidates=['imaginary-model-9999'].
        Assert: candidate_model is None (no priced candidate found).
        """
        # Arrange
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "test"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        # Act
        result = analyze_logs(path, candidates=["imaginary-model-9999"])

        # Assert
        assert result.candidate_model is None

    def test_more_expensive_custom_candidate_is_not_recommended(
        self, tmp_path: Path
    ) -> None:
        """Arrange: gpt-4o-mini baseline; explicit gpt-4o candidate costs more.
        Act: analyze_logs with candidates=['gpt-4o'].
        Assert: candidate_model is None so savings cannot go negative.
        """
        # Arrange
        records = [
            {
                "model": "gpt-4o-mini",
                "request": {"messages": [{"role": "user", "content": "classify"}]},
                "response": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "billing"}}
                    ]
                },
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        # Act
        result = analyze_logs(path, candidates=["gpt-4o"])

        # Assert
        assert result.candidate_model is None
        assert result.projected_cost == Decimal("0")


# ---------------------------------------------------------------------------
# Malformed record handling
# ---------------------------------------------------------------------------


class TestMalformedRecords:
    """analyze_logs must not crash on malformed or incomplete log lines."""

    def test_empty_file_produces_zero_results(self, tmp_path: Path) -> None:
        """Arrange: empty JSONL file.
        Act: analyze_logs.
        Assert: no exception; total_calls == 0.
        """
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")

        result = analyze_logs(path)
        assert result.total_calls == 0

    def test_invalid_json_lines_are_skipped(self, tmp_path: Path) -> None:
        """Arrange: file with one valid and one invalid JSON line.
        Act: analyze_logs.
        Assert: only the valid record is processed.
        """
        path = tmp_path / "mixed.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write('{"model": "gpt-4o", "request": {"messages": [{"role": "user", "content": "hi"}]}, "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}, "usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n')
            fh.write("this is not json\n")
            fh.write("\n")  # blank line

        result = analyze_logs(path)
        assert result.total_calls == 1

    def test_record_with_usage_but_no_messages_is_priced(self, tmp_path: Path) -> None:
        """Arrange: record with model and valid usage block but no messages.
        Act: analyze_logs.
        Assert: record IS priced — usage block supplies token counts directly so
                messages are not required (P1 fix: messages gate only for tokenizer).
        """
        import json as _json

        path = tmp_path / "no_messages.jsonl"
        path.write_text(
            _json.dumps({"model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
            + "\n",
            encoding="utf-8",
        )

        result = analyze_logs(path)
        assert result.total_calls == 1
        assert result.priced_calls == 1

    def test_record_without_messages_and_without_usage_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """Arrange: record with model but neither messages nor a usage block.
        Act: analyze_logs.
        Assert: record is skipped (no token counts available via any path).
        """
        import json as _json

        path = tmp_path / "no_messages_no_usage.jsonl"
        path.write_text(
            _json.dumps({"model": "gpt-4o"}) + "\n",
            encoding="utf-8",
        )

        result = analyze_logs(path)
        assert result.total_calls == 0

    def test_record_without_model_is_skipped(self, tmp_path: Path) -> None:
        """Arrange: record with no model field.
        Act: analyze_logs.
        Assert: record is skipped.
        """
        import json as _json

        path = tmp_path / "no_model.jsonl"
        path.write_text(
            _json.dumps({"request": {"messages": [{"role": "user", "content": "x"}]},
                         "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
            + "\n",
            encoding="utf-8",
        )

        result = analyze_logs(path)
        assert result.total_calls == 0

    def test_single_timestamp_produces_no_span(self, tmp_path: Path) -> None:
        """Arrange: only one record with a timestamp.
        Act: analyze_logs.
        Assert: observed_span_days is None (need 2+ timestamps for a span).
        """
        records = [
            {
                "model": "gpt-4o",
                "timestamp": "2026-01-01T00:00:00Z",
                "request": {"messages": [{"role": "user", "content": "hi"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)
        assert result.observed_span_days is None


# ---------------------------------------------------------------------------
# _best_candidate edge cases
# ---------------------------------------------------------------------------


class TestBestCandidateEdgeCases:
    """Edge cases in the routing candidate selection logic."""

    def test_baseline_model_same_as_candidate_is_skipped(self, tmp_path: Path) -> None:
        """Arrange: only one model and it IS gpt-4o (in routing candidates list).
        Act: analyze_logs with gpt-4o baseline.
        Assert: the dominant model (gpt-4o) is not chosen as its own candidate.
        """
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "test"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)
        # gpt-4o is in routing candidates, but it's also the baseline.
        # Candidate must not be the same as the baseline.
        assert result.candidate_model != "gpt-4o" or result.candidate_model is None

    def test_equal_blended_candidate_costs_tie_break_by_model_name(
        self, monkeypatch: Any
    ) -> None:
        """Arrange: two cheaper candidates with identical blended prices.
        Act: _best_candidate.
        Assert: model name provides the deterministic tie-break.
        """
        # Arrange
        prices = {
            "baseline": ModelPrice(
                "baseline",
                Decimal("0.000010"),
                Decimal("0.000030"),
                "test",
                None,
            ),
            "z-equal": ModelPrice(
                "z-equal",
                Decimal("0.000001"),
                Decimal("0.000003"),
                "test",
                None,
            ),
            "a-equal": ModelPrice(
                "a-equal",
                Decimal("0.000001"),
                Decimal("0.000003"),
                "test",
                None,
            ),
        }
        record = LogRecord(
            model="baseline",
            messages=[{"role": "user", "content": "hello"}],
            completion_text="world",
            prompt_tokens=10,
            completion_tokens=5,
            timestamp=None,
        )
        call_cost = CallCost(
            record=record,
            price=prices["baseline"],
            prompt_cost=Decimal("0.000100"),
            completion_cost=Decimal("0.000150"),
            total_cost=Decimal("0.000250"),
            token_source="usage_block",
        )

        monkeypatch.setattr("frugon.cost._ROUTING_CANDIDATES", ["z-equal", "a-equal"])
        monkeypatch.setattr("frugon.cost.get_model_price", prices.get)
        # Give the synthetic candidates known tiers (tier 0) so they pass the unrated filter.
        monkeypatch.setattr(
            "frugon.cost._get_model_tier",
            lambda m: {"baseline": 0, "z-equal": 0, "a-equal": 0}.get(m, -1),
        )

        # Act
        candidate, projected_cost = _best_candidate("baseline", [call_cost])

        # Assert — forward order
        assert candidate == "a-equal"
        assert projected_cost == Decimal("0.000025")

        # Assert — reversed insertion order yields the same winner: determinism is NOT
        # a side-effect of iteration order.
        monkeypatch.setattr("frugon.cost._ROUTING_CANDIDATES", ["a-equal", "z-equal"])
        candidate_reversed, _ = _best_candidate("baseline", [call_cost])
        assert candidate_reversed == "a-equal"


# ---------------------------------------------------------------------------
# Timestamp edge cases
# ---------------------------------------------------------------------------


class TestTimestampEdgeCases:
    """Timestamp parsing edge cases in span computation."""

    def test_invalid_timestamp_string_is_skipped(self, tmp_path: Path) -> None:
        """Arrange: record with an unparseable timestamp string.
        Act: analyze_logs.
        Assert: no exception; unparseable timestamp is skipped silently.
        """
        records = [
            {
                "model": "gpt-4o",
                "timestamp": "not-a-valid-date",
                "request": {"messages": [{"role": "user", "content": "hi"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            {
                "model": "gpt-4o",
                "timestamp": "also-not-valid",
                "request": {"messages": [{"role": "user", "content": "hi2"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "hello2"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)
        # All timestamps invalid → no span computed
        assert result.observed_span_days is None
        # Both calls priced correctly
        assert result.priced_calls == 2

    def test_text_completion_response_format_is_parsed(self, tmp_path: Path) -> None:
        """Arrange: record using text-completion response format (choices[0].text).
        Act: analyze_logs.
        Assert: call is priced; no exception.
        """
        import json as _json

        path = tmp_path / "text_completion.jsonl"
        path.write_text(
            _json.dumps({
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "complete this"}]},
                "response": {"choices": [{"text": "...completed text..."}]},
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            }) + "\n",
            encoding="utf-8",
        )

        result = analyze_logs(path)
        assert result.priced_calls == 1


# ---------------------------------------------------------------------------
# Quality-tier routing — P1-B acceptance tests
# ---------------------------------------------------------------------------


class TestQualityTierRouting:
    """_best_candidate respects quality tiers; demo savings are unaffected."""

    def test_best_candidate_gpt4o_baseline_does_not_auto_pick_gpt4o_mini(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: gpt-4o as the dominant baseline model.
        Act: _best_candidate directly (the WHOLESALE selector's own tier cap).
        Assert: it does NOT pick gpt-4o-mini — the quality gap is too large.

        This test exercises the max_tier_drop=1 guard (§6 honest savings) that
        lives specifically in :func:`frugon.cost._best_candidate` — the
        FULL-SWAP/wholesale selector, which caps the tier drop because a
        wholesale swap moves EVERY call. It PINS the tiers it needs rather than
        reading the live seed: gpt-4o (tier 0) → gpt-4o-mini (tier 2) is a
        two-tier drop > 1, so gpt-4o-mini must be excluded from wholesale
        auto-selection.

        Since the 2026-07-02 quality-aware tie-break fix unified
        ``result.candidate_model`` with ``result.split.candidate_model`` on the
        DEFAULT (split-active) analysis path, this test asserts directly
        against ``_best_candidate`` — the one function that actually owns this
        tier-cap guarantee — rather than ``analyze_logs(...).candidate_model``,
        which now reflects the split recommendation (intentionally UNCAPPED by
        tier, same as it always was via ``select_easy_target`` — the per-call
        easy/hard gate is that path's quality protection, not a tier cap; see
        ``test_wholesale_candidate_model_still_respects_tier_cap`` below for the
        end-to-end ``--wholesale`` proof that the cap still gates the real
        analysis output on that surface).
        """
        install_synthetic_quality(monkeypatch, tmp_path, {"gpt-4o": 0, "gpt-4o-mini": 2})

        # Arrange — 10 gpt-4o calls
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "classify this email"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "billing"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }
            for _ in range(10)
        ]
        records_parsed = [parse_record(r) for r in records]
        call_costs = [compute_call_cost(r) for r in records_parsed if r is not None]

        # Act
        candidate_model, _projected_cost = _best_candidate("gpt-4o", call_costs)

        # Assert
        assert candidate_model != "gpt-4o-mini", (
            "gpt-4o baseline must NOT auto-recommend gpt-4o-mini (two quality tiers down) "
            f"via the wholesale selector. Got candidate_model={candidate_model!r}"
        )

    def test_wholesale_candidate_model_still_respects_tier_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end proof: on the real --wholesale surface (split_routing=False),
        analyze_logs(...).candidate_model IS the tier-capped _best_candidate pick
        — the tier cap still gates the actual analysis output, just not on the
        default split-active surface (see the test above)."""
        install_synthetic_quality(monkeypatch, tmp_path, {"gpt-4o": 0, "gpt-4o-mini": 2})
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "classify this email"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "billing"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }
            for _ in range(10)
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path, split_routing=False)

        assert result.split is None
        assert result.candidate_model != "gpt-4o-mini", (
            "the --wholesale (split_routing=False) surface must still respect "
            f"the tier cap. Got candidate_model={result.candidate_model!r}"
        )

    def test_best_candidate_gpt4turbo_baseline_recommends_cheaper_in_tolerance_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: gpt-4-turbo as the dominant baseline model.
        Act: analyze_logs (auto-detect candidate).
        Assert: a rated candidate, within max_tier_drop=1 of the baseline AND
        cheaper, is recommended.

        Tiers are PINNED so the assertion binds to the REAL guard (the candidate
        stays within max_tier_drop of the baseline) rather than a frozen band
        literal — a leaderboard re-anchor that re-bands the routing pool can never
        re-break it. With gpt-4-turbo (tier 3) baseline the pool's cheapest
        in-tolerance rated model is gpt-4.1-mini (tier 2, within max_tier_drop=1).
        Pool members not listed in the synthetic table resolve as UNRATED and are
        excluded, so the table must include at least one new-pool member at tier ≤4.
        """
        install_synthetic_quality(
            monkeypatch,
            tmp_path,
            {
                "gpt-4-turbo": 3,
                "gpt-4o": 0,
                # New pool members — must be present so the tier guard finds a candidate.
                "claude-sonnet-4-5": 1,
                "gpt-4.1": 1,
                "claude-haiku-4-5": 1,
                "gemini-2.5-flash": 0,
                "gpt-4.1-mini": 2,
            },
        )

        # Arrange — 25 gpt-4-turbo calls
        records = [
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "summarize this report"}]},
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "Summary here."}}]
                },
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            }
            for _ in range(25)
        ]
        path = _write_jsonl(records, tmp_path)

        # Act
        result = analyze_logs(path)

        # Assert: a rated candidate, within the max_tier_drop=1 guard of the
        # baseline and cheaper than it, is recommended.
        assert result.candidate_model is not None, "Expected a candidate recommendation"
        cand_tier = get_model_tier(result.candidate_model)
        assert cand_tier != UNRATED_TIER, (
            f"Recommended candidate {result.candidate_model!r} must be quality-rated"
        )
        baseline_tier = get_model_tier("gpt-4-turbo")
        assert cand_tier - baseline_tier <= 1, (
            f"Auto candidate {result.candidate_model!r} (tier {cand_tier}) must stay "
            f"within max_tier_drop=1 of the gpt-4-turbo baseline (tier {baseline_tier})"
        )
        assert result.projected_cost > Decimal("0")

    def test_explicit_candidates_bypass_tier_constraint(self, tmp_path: Path) -> None:
        """Arrange: gpt-4o baseline; explicitly pass gpt-4o-mini as --candidates.
        Act: analyze_logs with candidates=['gpt-4o-mini'].
        Assert: candidate_model is gpt-4o-mini — explicit list overrides tier guard.

        Users who knowingly want a multi-tier recommendation can pass --candidates.
        """
        # Arrange
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "short answer"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "yes"}}]},
                "usage": {"prompt_tokens": 20, "completion_tokens": 2},
            }
            for _ in range(5)
        ]
        path = _write_jsonl(records, tmp_path)

        # Act — explicit candidates bypass tier guard
        result = analyze_logs(path, candidates=["gpt-4o-mini"])

        # Assert
        assert result.candidate_model == "gpt-4o-mini"
        assert result.projected_cost > Decimal("0")


# ---------------------------------------------------------------------------
# Routing honesty — unrated model gating, pool size, tier_drop (#3 / #4 / #10)
# ---------------------------------------------------------------------------


class TestRoutingHonestyMetadata:
    """AnalysisResult exposes honest metadata used by the CLI for disclosures."""

    def test_unrated_baseline_flagged_in_result(
        self, tmp_path: Path, unrated_model: str
    ) -> None:
        """Arrange: baseline model not in _QUALITY_TIERS (a priced-but-unrated model).
        Act: analyze_logs.
        Assert: result.baseline_is_unrated is True.
        """
        records = [
            {
                "model": unrated_model,
                "request": {"messages": [{"role": "user", "content": "hello"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "world"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            }
            for _ in range(5)
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)

        assert result.baseline_is_unrated is True

    def test_rated_baseline_not_flagged(self, tmp_path: Path) -> None:
        """Arrange: baseline model is gpt-4o (present in the quality table, any tier).
        Act: analyze_logs.
        Assert: result.baseline_is_unrated is False.

        The assertion is tier-value-agnostic — baseline_is_unrated only requires
        gpt-4o to carry SOME rating, so it survives any re-anchor (it would only
        break if gpt-4o were removed from the seed entirely).
        """
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "hello"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "world"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            }
            for _ in range(5)
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)

        assert result.baseline_is_unrated is False

    def test_unrated_model_excluded_from_auto_selection(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Arrange: routing candidate pool contains only an unrated model.
        Act: analyze_logs (no explicit candidates).
        Assert: candidate_model is None — unrated models must not be auto-recommended.
        """
        from frugon.pricing import ModelPrice as MP

        fake_prices: dict[str, MP] = {
            "gpt-4-turbo": MP("gpt-4-turbo", Decimal("0.00001"), Decimal("0.00003"), "test", None),
            "unlisted-cheap": MP("unlisted-cheap", Decimal("0.000001"), Decimal("0.000003"), "test", None),
        }
        monkeypatch.setattr("frugon.cost._ROUTING_CANDIDATES", ["unlisted-cheap"])
        # unlisted-cheap is not in the quality table → naturally unrated; no patch needed.
        monkeypatch.setattr("frugon.cost.get_model_price", fake_prices.get)

        records = [
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "test"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)

        assert result.candidate_model is None, (
            "Unrated model must not be auto-recommended as candidate. "
            f"Got candidate_model={result.candidate_model!r}"
        )

    def test_unrated_candidate_allowed_via_explicit_candidates(self, tmp_path: Path) -> None:
        """Arrange: gpt-4o baseline; explicit --candidates includes an unrated-but-priced model.
        Act: analyze_logs with candidates=['gpt-4o-mini'] (in tier map) — verifying the
             explicit-candidates path allows any priced model regardless of tier.
        Assert: candidate_model is set (explicit list bypasses auto-selection gating).
        """
        # gpt-4o-mini IS in the tier map, but the key invariant here is that the
        # explicit-candidates path NEVER calls _best_candidate (which enforces the
        # unrated exclusion), so any priced model can be selected.
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "test"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }
            for _ in range(3)
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path, candidates=["gpt-4o-mini"])

        assert result.candidate_model == "gpt-4o-mini"

    def test_pool_size_exposed_in_result(self, tmp_path: Path) -> None:
        """Arrange: any valid log.
        Act: analyze_logs without --candidates.
        Assert: result.candidate_pool_size == len(_ROUTING_CANDIDATES) and > 0.
        """
        from frugon.cost import _ROUTING_CANDIDATES

        records = [
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)

        assert result.candidate_pool_size == len(_ROUTING_CANDIDATES)
        assert result.candidate_pool_size > 0

    def test_used_default_pool_true_without_candidates(self, tmp_path: Path) -> None:
        """Arrange: analyze without --candidates.
        Act: analyze_logs.
        Assert: result.used_default_pool is True.
        """
        records = [
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)

        assert result.used_default_pool is True

    def test_used_default_pool_false_with_explicit_candidates(self, tmp_path: Path) -> None:
        """Arrange: analyze with explicit --candidates.
        Act: analyze_logs with candidates=['gpt-4o-mini'].
        Assert: result.used_default_pool is False.
        """
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path, candidates=["gpt-4o-mini"])

        assert result.used_default_pool is False

    def test_tier_drop_none_when_baseline_unrated(
        self, tmp_path: Path, unrated_model: str
    ) -> None:
        """Arrange: unrated baseline (a priced-but-unrated model) with any candidate.
        Act: analyze_logs.
        Assert: result.tier_drop is None (drop math is unreliable with unrated baseline).
        """
        # The baseline is unrated; even if a candidate is found, tier_drop stays None
        records = [
            {
                "model": unrated_model,
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            }
            for _ in range(5)
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path)

        # If a candidate was found, tier_drop must still be None (baseline unrated)
        if result.candidate_model is not None:
            assert result.tier_drop is None, (
                f"Expected tier_drop=None for unrated baseline, got {result.tier_drop}"
            )

    def test_tier_drop_computed_for_known_tier_drop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: explicit --candidates from gpt-4o (tier 0) to claude-3-haiku (tier 3).
        Act: analyze_logs.
        Assert: result.tier_drop == 3 (candidate_tier 3 - baseline_tier 0).

        Tiers are PINNED so the "known tier drop" stays a known 3 regardless of
        leaderboard re-anchors (which now place gpt-4o at tier 2). This verifies
        the production formula tier_drop = candidate_tier - baseline_tier.
        """
        install_synthetic_quality(
            monkeypatch, tmp_path, {"gpt-4o": 0, "claude-3-haiku": 3}
        )

        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "classify"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "a"}}]},
                "usage": {"prompt_tokens": 20, "completion_tokens": 2},
            }
            for _ in range(10)
        ]
        path = _write_jsonl(records, tmp_path)

        result = analyze_logs(path, candidates=["claude-3-haiku-20240307"])

        if result.candidate_model == "claude-3-haiku-20240307":
            assert result.tier_drop == 3, (
                f"Expected tier_drop=3 (tier 3 - tier 0), got {result.tier_drop}"
            )

    def test_tier_drop_none_when_no_candidate(self, tmp_path: Path) -> None:
        """Arrange: no candidate can be found cheaper than the baseline.
        Act: analyze_logs.
        Assert: result.tier_drop is None.
        """
        records = [
            {
                "model": "gpt-4o-mini",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ]
        path = _write_jsonl(records, tmp_path)

        # gpt-4o costs more — no cheaper candidate
        result = analyze_logs(path, candidates=["gpt-4o"])

        assert result.candidate_model is None
        assert result.tier_drop is None


# ---------------------------------------------------------------------------
# _safe_int helper
# ---------------------------------------------------------------------------


class TestSafeInt:
    """_safe_int handles null, non-numeric strings, and valid numerics."""

    def test_none_returns_none(self) -> None:
        """Arrange: None (JSON null).
        Assert: _safe_int returns None without raising.
        """
        assert _safe_int(None) is None

    def test_string_abc_returns_none(self) -> None:
        """Arrange: non-numeric string "abc".
        Assert: _safe_int returns None.
        """
        assert _safe_int("abc") is None

    def test_valid_int_string_returns_int(self) -> None:
        """Arrange: numeric string "42".
        Assert: _safe_int returns 42.
        """
        assert _safe_int("42") == 42

    def test_plain_int_returns_int(self) -> None:
        """Arrange: already an int (100).
        Assert: _safe_int returns 100.
        """
        assert _safe_int(100) == 100

    def test_float_truncates_to_int(self) -> None:
        """Arrange: float 3.9.
        Assert: _safe_int returns 3 (standard int() truncation).
        """
        assert _safe_int(3.9) == 3


# ---------------------------------------------------------------------------
# Malformed usage blocks — P1 crash fix
# ---------------------------------------------------------------------------


class TestMalformedUsageBlock:
    """parse_record must not crash when usage block contains null/non-numeric values."""

    def test_null_prompt_tokens_falls_back_to_tokenizer(self, tmp_path: Path) -> None:
        """Arrange: usage block with null prompt_tokens; messages present.
        Act: parse_record.
        Assert: record is parsed, token_source is 'counted' (tokenizer fallback used).
        """
        raw = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "hello"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
            "usage": {"prompt_tokens": None, "completion_tokens": 5},
        }
        rec = parse_record(raw)
        assert rec is not None
        assert rec.token_source == "counted"
        assert rec.prompt_tokens > 0

    def test_string_usage_tokens_falls_back_to_tokenizer(self, tmp_path: Path) -> None:
        """Arrange: usage block with string "abc" for prompt_tokens; messages present.
        Act: parse_record.
        Assert: record is parsed via tokenizer fallback, does not crash.
        """
        raw = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "classify this"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "billing"}}]},
            "usage": {"prompt_tokens": "abc", "completion_tokens": "def"},
        }
        rec = parse_record(raw)
        assert rec is not None
        assert rec.token_source == "counted"

    def test_null_usage_no_messages_returns_none(self) -> None:
        """Arrange: malformed usage block AND no messages.
        Act: parse_record.
        Assert: returns None (cannot price without either valid usage or messages).
        """
        raw = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": None, "completion_tokens": None},
        }
        rec = parse_record(raw)
        assert rec is None

    def test_malformed_usage_does_not_crash_analyze_logs(self, tmp_path: Path) -> None:
        """Arrange: file with a record whose usage block has null prompt_tokens.
        Act: analyze_logs.
        Assert: no exception; record is still priced (tokenizer fallback).
        """
        record = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "summarize"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            "usage": {"prompt_tokens": None, "completion_tokens": 5},
        }
        path = _write_jsonl([record], tmp_path)
        result = analyze_logs(path)
        assert result.priced_calls == 1
        assert result.total_cost > Decimal("0")


# ---------------------------------------------------------------------------
# Usage-only records (no messages) — P1 silent data-loss fix
# ---------------------------------------------------------------------------


class TestUsageOnlyRecords:
    """Records with a valid usage block are priced even without messages."""

    def test_usage_block_without_messages_is_priced(self, tmp_path: Path) -> None:
        """Arrange: record with model + usage but no messages.
        Act: analyze_logs.
        Assert: priced_calls == 1; cost is positive.
        """
        record = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }
        path = _write_jsonl([record], tmp_path)
        result = analyze_logs(path)
        assert result.priced_calls == 1
        assert result.total_cost > Decimal("0")

    def test_usage_block_without_messages_correct_cost(self, tmp_path: Path) -> None:
        """Arrange: usage-only record with known token counts.
        Act: analyze_logs.
        Assert: cost matches hand-calculated expected value using gpt-4o rates.
        """
        # 100 prompt × $0.0000025 + 20 completion × $0.00001 = $0.00025 + $0.0002 = $0.00045
        record = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
        path = _write_jsonl([record], tmp_path)
        result = analyze_logs(path)
        expected = GPT4O_IN * 100 + GPT4O_OUT * 20
        assert result.total_cost == expected

    def test_token_source_usage_block_for_usage_only_record(self, tmp_path: Path) -> None:
        """Arrange: usage-only record (no messages).
        Act: parse_record then compute_call_cost.
        Assert: token_source is 'usage_block' (not 'counted').
        """
        raw = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }
        rec = parse_record(raw)
        assert rec is not None
        cc = compute_call_cost(rec)
        assert cc.token_source == "usage_block"


# ---------------------------------------------------------------------------
# skipped_malformed count — P1 silent data-loss
# ---------------------------------------------------------------------------


class TestSkippedMalformed:
    """AnalysisResult.skipped_malformed counts lines that yielded no record."""

    def test_invalid_json_increments_skipped_malformed(self, tmp_path: Path) -> None:
        """Arrange: one valid record + one invalid JSON line.
        Act: analyze_logs.
        Assert: skipped_malformed == 1; valid record is still priced.
        """
        path = tmp_path / "mixed.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({
                    "model": "gpt-4o",
                    "request": {"messages": [{"role": "user", "content": "hi"}]},
                    "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }) + "\n"
            )
            fh.write("this is not json\n")
        result = analyze_logs(path)
        assert result.skipped_malformed == 1
        assert result.priced_calls == 1

    def test_no_model_record_increments_skipped_malformed(self, tmp_path: Path) -> None:
        """Arrange: valid JSON line that parse_record returns None for (missing model).
        Act: analyze_logs.
        Assert: skipped_malformed == 1.
        """
        record = {"request": {"messages": [{"role": "user", "content": "x"}]}}
        path = _write_jsonl([record], tmp_path)
        result = analyze_logs(path)
        assert result.skipped_malformed == 1
        assert result.total_calls == 0

    def test_zero_skipped_malformed_for_valid_records(self, tmp_path: Path) -> None:
        """Arrange: two well-formed records.
        Act: analyze_logs.
        Assert: skipped_malformed == 0.
        """
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "a"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "b"}}]},
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        ]
        path = _write_jsonl(records, tmp_path)
        result = analyze_logs(path)
        assert result.skipped_malformed == 0

    def test_non_object_json_lines_increment_skipped_malformed(
        self, tmp_path: Path
    ) -> None:
        """Arrange: a JSONL file where syntactically-valid non-object lines
        (a bare array, a bare string, a bare number) are interleaved with two
        valid records and one line that is invalid JSON entirely.
        Act: iter_records.
        Assert: no AttributeError — a non-dict parse result is counted in
        skipped_malformed exactly like a JSON syntax error, and both valid
        records still come through untouched (FRG-OSS-054).
        """
        path = tmp_path / "non_object_lines.jsonl"
        valid_record = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "hello"}}]
            },
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(valid_record) + "\n")
            fh.write(json.dumps([1, 2, 3]) + "\n")  # bare array
            fh.write(json.dumps("just a string") + "\n")  # bare string
            fh.write("this is not json\n")  # actual JSON syntax error
            fh.write(json.dumps(42) + "\n")  # bare number
            fh.write(json.dumps(valid_record) + "\n")

        records, skipped_malformed = iter_records(path)

        assert len(records) == 2
        # 4 = 3 non-object lines (array, string, number) + 1 syntax-error line
        assert skipped_malformed == 4


# ---------------------------------------------------------------------------
# scan_models — non-object JSON lines must not crash (FRG-OSS-054)
# ---------------------------------------------------------------------------


class TestScanModelsNonObjectLines:
    """scan_models must treat a non-dict parse result like invalid JSON."""

    def test_non_object_lines_do_not_crash_and_are_skipped(
        self, tmp_path: Path
    ) -> None:
        """Arrange: a JSONL log with a bare array, a bare string, and a bare
        number interleaved with valid model rows.
        Act: scan_models.
        Assert: no AttributeError; the distinct models and dominant model
        reflect only the valid rows, exactly as if the non-object lines were
        JSON syntax errors.
        """
        path = tmp_path / "non_object_models.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"model": "gpt-4o"}) + "\n")
            fh.write(json.dumps([1, 2, 3]) + "\n")  # bare array
            fh.write(json.dumps({"model": "gpt-4o"}) + "\n")
            fh.write(json.dumps("just a string") + "\n")  # bare string
            fh.write("this is not json\n")  # actual JSON syntax error
            fh.write(json.dumps(42) + "\n")  # bare number
            fh.write(json.dumps({"model": "gpt-4o-mini"}) + "\n")

        distinct_models, dominant = scan_models(path)

        assert distinct_models == ["gpt-4o", "gpt-4o-mini"]
        assert dominant == "gpt-4o"


# ---------------------------------------------------------------------------
# Monthly projection computation — P0 fix
# ---------------------------------------------------------------------------


class TestMonthlyProjection:
    """monthly_cost and monthly_projected are correctly computed from total_cost."""

    def test_monthly_cost_window_equals_observed_times_30_over_window(
        self, tmp_path: Path
    ) -> None:
        """Arrange: records with --window 10.
        Act: analyze_logs with window_days=10.
        Assert: monthly_cost == total_cost * 30 / 10 (exact Decimal math).
        """
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ]
        path = _write_jsonl(records, tmp_path)
        result = analyze_logs(path, window_days=10)

        assert result.monthly_cost is not None
        expected = result.total_cost * Decimal("30") / Decimal("10")
        assert result.monthly_cost == expected, (
            f"monthly_cost={result.monthly_cost}, expected={expected}"
        )

    def test_monthly_cost_window_30_equals_total_cost(self, tmp_path: Path) -> None:
        """Arrange: --window 30 (a full month).
        Act: analyze_logs with window_days=30.
        Assert: monthly_cost == total_cost (factor = 30/30 = 1).
        """
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "hi"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ho"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            }
        ]
        path = _write_jsonl(records, tmp_path)
        result = analyze_logs(path, window_days=30)

        assert result.monthly_cost is not None
        assert result.monthly_cost == result.total_cost

    def test_monthly_cost_span_correct_from_timestamps(self, tmp_path: Path) -> None:
        """Arrange: records with timestamps 5 days apart, no --window.
        Act: analyze_logs.
        Assert: monthly_cost == total_cost * 30 / 5 (±Decimal rounding).
        """
        records = [
            {
                "model": "gpt-4o",
                "timestamp": "2026-01-01T00:00:00Z",
                "request": {"messages": [{"role": "user", "content": "a"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "b"}}]},
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
            {
                "model": "gpt-4o",
                "timestamp": "2026-01-06T00:00:00Z",
                "request": {"messages": [{"role": "user", "content": "c"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "d"}}]},
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        ]
        path = _write_jsonl(records, tmp_path)
        result = analyze_logs(path)

        assert result.observed_span_days is not None
        assert result.monthly_cost is not None
        # span = 5 days → factor = 30/5 = 6
        expected = result.total_cost * Decimal("30") / Decimal(str(result.observed_span_days))
        assert abs(result.monthly_cost - expected) < Decimal("0.000001")

    def test_monthly_cost_none_when_no_window_no_span(self, tmp_path: Path) -> None:
        """Arrange: records without timestamps and no --window.
        Act: analyze_logs.
        Assert: monthly_cost is None (no extrapolation invented).
        """
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ]
        path = _write_jsonl(records, tmp_path)
        result = analyze_logs(path)
        assert result.monthly_cost is None
        assert result.monthly_projected is None

    def test_total_cost_unchanged_by_monthly_projection(self, tmp_path: Path) -> None:
        """Arrange: records with --window 7.
        Act: analyze_logs.
        Assert: total_cost is the raw observed total (not multiplied by the projection
                factor). monthly_cost != total_cost when window != 30.
        """
        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ]
        path = _write_jsonl(records, tmp_path)
        result = analyze_logs(path, window_days=7)

        raw_total = GPT4O_IN * 100 + GPT4O_OUT * 50
        assert result.total_cost == raw_total, (
            f"total_cost must remain the raw observed total; got {result.total_cost}"
        )
        assert result.monthly_cost is not None
        assert result.monthly_cost != result.total_cost

    def test_monthly_projected_computed_when_candidate_found(self, tmp_path: Path) -> None:
        """Arrange: gpt-4-turbo baseline (expensive) with --window 10.
        Act: analyze_logs with window_days=10.
        Assert: monthly_projected is not None when a candidate was found.
        """
        records = [
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "summarize"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            }
            for _ in range(5)
        ]
        path = _write_jsonl(records, tmp_path)
        result = analyze_logs(path, window_days=10)

        if result.candidate_model is not None:
            assert result.monthly_projected is not None
            expected = result.projected_cost * Decimal("30") / Decimal("10")
            assert result.monthly_projected == expected


# ---------------------------------------------------------------------------
# analyze_records — P2 double-read fix
# ---------------------------------------------------------------------------


class TestAnalyzeRecords:
    """analyze_records accepts pre-parsed LogRecords and returns the same result
    as analyze_logs, enabling callers to avoid re-reading the source file."""

    def test_analyze_records_matches_analyze_logs(self, tmp_path: Path) -> None:
        """Arrange: write a JSONL file and parse records manually.
        Act: call analyze_records with the parsed records.
        Assert: result matches analyze_logs on the same file.
        """
        records_raw = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "classify"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "billing"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }
            for _ in range(3)
        ]
        path = _write_jsonl(records_raw, tmp_path)

        result_from_file = analyze_logs(path)

        parsed = [parse_record(r) for r in records_raw]
        log_records = [r for r in parsed if r is not None]
        result_from_records = analyze_records(log_records)

        assert result_from_records.total_cost == result_from_file.total_cost
        assert result_from_records.priced_calls == result_from_file.priced_calls
        assert result_from_records.candidate_model == result_from_file.candidate_model

    def test_analyze_records_with_window_days(self, tmp_path: Path) -> None:
        """Arrange: pre-parsed records.
        Act: analyze_records with window_days=7.
        Assert: monthly_cost is computed correctly.
        """
        records_raw = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "hi"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ]
        parsed = [parse_record(r) for r in records_raw if parse_record(r) is not None]
        result = analyze_records(parsed, window_days=7)

        assert result.monthly_cost is not None
        expected = result.total_cost * Decimal("30") / Decimal("7")
        assert result.monthly_cost == expected

    def test_analyze_records_propagates_skipped_malformed(self) -> None:
        """Arrange: pre-parsed records with a skipped_malformed count.
        Act: analyze_records with skipped_malformed=3.
        Assert: result.skipped_malformed == 3.
        """
        raw = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "x"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "y"}}]},
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        rec = parse_record(raw)
        assert rec is not None
        result = analyze_records([rec], skipped_malformed=3)
        assert result.skipped_malformed == 3


# ---------------------------------------------------------------------------
# LogRecord.token_source field — P2 fix (remove private-attr stash)
# ---------------------------------------------------------------------------


class TestTokenSourceField:
    """token_source is a proper dataclass field on LogRecord (not a dynamic attr)."""

    def test_log_record_has_token_source_field(self) -> None:
        """Assert: LogRecord instances have a token_source attribute without setattr."""
        rec = LogRecord(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            completion_text="hi",
            prompt_tokens=10,
            completion_tokens=2,
            timestamp=None,
            token_source="usage_block",
        )
        assert rec.token_source == "usage_block"

    def test_parse_record_usage_block_sets_correct_source(self) -> None:
        """Arrange: record with a valid usage block.
        Act: parse_record.
        Assert: token_source == 'usage_block' set on the dataclass field.
        """
        raw = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }
        rec = parse_record(raw)
        assert rec is not None
        assert rec.token_source == "usage_block"
        # Confirm it is the real dataclass field, not a dynamic attribute stash
        assert "token_source" in rec.__dataclass_fields__

    def test_parse_record_no_usage_sets_counted_source(self) -> None:
        """Arrange: record with no usage block (tokenizer path).
        Act: parse_record.
        Assert: token_source == 'counted'.
        """
        raw = {
            "model": "gpt-4o",
            "request": {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                ]
            },
            "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]},
        }
        rec = parse_record(raw)
        assert rec is not None
        assert rec.token_source == "counted"

    def test_compute_call_cost_uses_token_source_field(self) -> None:
        """Assert: compute_call_cost propagates token_source from the dataclass field."""
        rec = LogRecord(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            completion_text="y",
            prompt_tokens=10,
            completion_tokens=2,
            timestamp=None,
            token_source="counted",
        )
        cc = compute_call_cost(rec)
        assert cc.token_source == "counted"


# ---------------------------------------------------------------------------
# Cost-math carve-out (§2a) — known-answer vectors at extreme scale + a
# monotonicity property test, plus parse/fallback edge cases.
# ---------------------------------------------------------------------------


class TestCostMathCarveout:
    """High-rigor golden + property tests for compute_call_cost."""

    def test_compute_call_cost_huge_token_counts_exact(self) -> None:
        """Arrange: 100,000,000 prompt + 100,000,000 completion tokens on gpt-4o.
        Act: compute_call_cost.
        Assert: exact Decimal cost (no float drift at scale).

        100e6 × $0.0000025 = $250; 100e6 × $0.00001 = $1000; total $1250.
        """
        rec = LogRecord(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            completion_text="y",
            prompt_tokens=100_000_000,
            completion_tokens=100_000_000,
            timestamp=None,
        )
        cc = compute_call_cost(rec)
        assert cc.prompt_cost == GPT4O_IN * 100_000_000
        assert cc.completion_cost == GPT4O_OUT * 100_000_000
        assert cc.total_cost == Decimal("1250")

    def test_cost_increases_monotonically_with_tokens(self) -> None:
        """Property: cost never decreases as prompt-token count increases."""
        prev = Decimal("-1")
        for pt in range(0, 100_001, 5_000):
            rec = LogRecord(
                model="gpt-4o",
                messages=[{"role": "user", "content": "x"}],
                completion_text="y",
                prompt_tokens=pt,
                completion_tokens=10,
                timestamp=None,
            )
            cost = compute_call_cost(rec).total_cost
            assert cost >= prev, f"cost dropped at prompt_tokens={pt}"
            prev = cost


class _FailingTokenizer:
    """Stand-in for tokencost whose every counter raises (forces char fallback)."""

    def count_message_tokens(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("tokenizer unavailable")

    def count_string_tokens(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("tokenizer unavailable")


class TestTokenApproximationDisclosure:
    """When the tokenizer fails, the char heuristic is flagged 'approximated'."""

    def test_parse_record_marks_approximated_on_tokenizer_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from frugon import cost as cost_module

        monkeypatch.setattr(cost_module, "_tc", _FailingTokenizer())
        raw = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "hello there friend"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        }
        rec = parse_record(raw)
        assert rec is not None
        assert rec.token_source == "approximated"
        assert rec.prompt_tokens > 0

    def test_analyze_counts_approximated_calls(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from frugon import cost as cost_module

        monkeypatch.setattr(cost_module, "_tc", _FailingTokenizer())
        record = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "summarize this please"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        }
        path = _write_jsonl([record], tmp_path)
        result = analyze_logs(path)
        assert result.approximated_calls == 1

    def test_usage_block_calls_are_not_approximated(self, tmp_path: Path) -> None:
        record = {
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        path = _write_jsonl([record], tmp_path)
        result = analyze_logs(path)
        assert result.approximated_calls == 0


class TestParseFallbackEdgeCases:
    """Named edge cases on the parse/fallback core (cost-math silent-corruption surface)."""

    def test_mixed_records_skipped_count_exact(self, tmp_path: Path) -> None:
        """2 valid + 2 invalid-JSON + 1 no-model → priced 2, skipped 3."""
        path = tmp_path / "mixed.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }) + "\n")
            fh.write("not json at all\n")
            fh.write(json.dumps({
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            }) + "\n")
            fh.write("{bad json\n")
            fh.write(json.dumps({"usage": {"prompt_tokens": 3, "completion_tokens": 1}}) + "\n")
        result = analyze_logs(path)
        assert result.priced_calls == 2
        assert result.skipped_malformed == 3

    def test_completion_tokens_null_only_falls_back_to_tokenizer(self) -> None:
        raw = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "classify this"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "billing"}}]},
            "usage": {"prompt_tokens": 10, "completion_tokens": None},
        }
        rec = parse_record(raw)
        assert rec is not None
        assert rec.token_source == "counted"
        assert rec.completion_tokens >= 1

    def test_both_token_fields_null_with_messages_falls_back(self) -> None:
        raw = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "what is 2+2?"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]},
            "usage": {"prompt_tokens": None, "completion_tokens": None},
        }
        rec = parse_record(raw)
        assert rec is not None
        assert rec.token_source == "counted"
        assert rec.prompt_tokens > 0

    def test_empty_completion_text_through_tokenizer_fallback(self) -> None:
        raw = {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "say nothing"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": ""}}]},
        }
        rec = parse_record(raw)
        assert rec is not None
        assert rec.token_source == "counted"
        assert rec.completion_tokens >= 0
        cc = compute_call_cost(rec)
        assert cc.total_cost >= Decimal("0")


class TestTokenCountMemoization:
    """The token-count memo that speeds up repetitive (templated) logs.

    The win is correctness-preserving: a cached count must equal the count the
    tokenizer would have produced, so identical prompts share one tokenization
    without changing any figure.
    """

    def test_repeated_prompt_tokenized_once(self) -> None:
        """Arrange: clear the memo, count the same message list twice.
        Act: read the lru_cache hit counter.
        Assert: the second call is a cache hit — the tokenizer ran once.
        """
        from frugon.cost import _count_message_tokens_cached, _count_prompt_tokens

        _count_message_tokens_cached.cache_clear()
        messages = [{"role": "user", "content": "translate this sentence to French"}]
        first_tokens, first_approx = _count_prompt_tokens(messages, "gpt-4o")
        hits_before = _count_message_tokens_cached.cache_info().hits
        second_tokens, second_approx = _count_prompt_tokens(messages, "gpt-4o")
        hits_after = _count_message_tokens_cached.cache_info().hits

        assert second_tokens == first_tokens
        assert second_approx == first_approx
        assert hits_after == hits_before + 1, "identical prompt must hit the memo"

    def test_completion_count_memoized_and_stable(self) -> None:
        """Arrange: clear the memo, count the same completion text twice.
        Act: compare counts and the hit counter.
        Assert: identical counts; the second call is a cache hit.
        """
        from frugon.cost import _count_completion_tokens, _count_string_tokens_cached

        _count_string_tokens_cached.cache_clear()
        text = "Voici la traduction de la phrase."
        first, first_approx = _count_completion_tokens(text, "gpt-4o")
        hits_before = _count_string_tokens_cached.cache_info().hits
        second, second_approx = _count_completion_tokens(text, "gpt-4o")
        hits_after = _count_string_tokens_cached.cache_info().hits

        assert second == first
        assert second_approx == first_approx
        assert hits_after == hits_before + 1

    def test_memoized_prompt_count_matches_direct_tokencost(self) -> None:
        """Arrange: a message list the tokenizer can count.
        Act: compare _count_prompt_tokens against tokencost directly.
        Assert: byte-identical — memoization is transparent to the figure.
        """
        import tokencost as tc

        from frugon.cost import _count_message_tokens_cached, _count_prompt_tokens

        _count_message_tokens_cached.cache_clear()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarise the quarterly report."},
        ]
        memoized, approximated = _count_prompt_tokens(messages, "gpt-4o")
        direct = int(tc.count_message_tokens(messages, "gpt-4o"))
        assert memoized == direct
        assert approximated is False


# ---------------------------------------------------------------------------
# quality_json_last_synced plumbing — mirrors pricing_json_last_synced
# ---------------------------------------------------------------------------


class TestQualityLastSyncedPlumbing:
    """analyze_records plumbs the quality table _last_synced into AnalysisResult.

    The "within tolerance" recommendation rests on the quality tiers, so the date
    those tiers were synced is decision-relevant and must reach the renderer the
    same way pricing freshness does.
    """

    def test_quality_last_synced_is_populated_from_table(
        self, monkeypatch: Any
    ) -> None:
        """The (tier_map, last_synced, attribution) tuple feeds quality_json_last_synced."""
        import frugon.cost as cost_mod

        monkeypatch.setattr(
            cost_mod,
            "_load_quality_table",
            lambda: ({"gpt-4o": 0, "gpt-4o-mini": 2}, "2026-06-04", "attr"),
        )
        rec = LogRecord(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello world"}],
            completion_text="hi there",
            prompt_tokens=10,
            completion_tokens=4,
            timestamp=None,
        )
        result = analyze_records([rec])
        assert result.quality_json_last_synced == "2026-06-04"

    def test_quality_last_synced_none_when_table_unstamped(
        self, monkeypatch: Any
    ) -> None:
        """A quality table with no _last_synced leaves the field None (no Quality row)."""
        import frugon.cost as cost_mod

        monkeypatch.setattr(
            cost_mod,
            "_load_quality_table",
            lambda: ({"gpt-4o": 0}, None, None),
        )
        rec = LogRecord(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello world"}],
            completion_text="hi there",
            prompt_tokens=10,
            completion_tokens=4,
            timestamp=None,
        )
        result = analyze_records([rec])
        assert result.quality_json_last_synced is None


# ---------------------------------------------------------------------------
# best_judge_from_log — default the --judge judge to the user's best LOG model
# ---------------------------------------------------------------------------
#
# Tier semantics: LOWER int = BETTER (0 = Elite); UNRATED_TIER (-1) means "no
# rating" and must be ignored.  Ties on tier break by name (ascending) for
# determinism.  These tests patch _get_model_tier so they do not depend on the
# live, network-synced quality table.


class TestBestJudgeFromLog:
    """Unit tests for cost.best_judge_from_log."""

    def test_picks_best_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Arrange: three rated models at different tiers.
        Assert: the lowest-tier-number (highest quality) model wins.
        """
        from frugon.cost import best_judge_from_log

        tiers = {"weak": 3, "best": 0, "mid": 2}
        monkeypatch.setattr(
            "frugon.cost._get_model_tier", lambda m: tiers.get(m, -1)
        )
        assert best_judge_from_log(["weak", "best", "mid"]) == "best"

    def test_ignores_unrated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Arrange: one rated model and several unrated (tier -1) models.
        Assert: the single rated model is chosen; unrated models are never picked
        even though -1 is numerically the lowest value.
        """
        from frugon.cost import best_judge_from_log

        tiers = {"rated": 2}
        monkeypatch.setattr(
            "frugon.cost._get_model_tier", lambda m: tiers.get(m, -1)
        )
        assert best_judge_from_log(["unrated-a", "rated", "unrated-b"]) == "rated"

    def test_tie_breaks_by_name_ascending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: two models at the SAME best tier.
        Assert: the name-ascending one wins, deterministically, regardless of
        iteration order.
        """
        from frugon.cost import best_judge_from_log

        monkeypatch.setattr("frugon.cost._get_model_tier", lambda m: 0)
        assert best_judge_from_log(["zeta", "alpha", "mu"]) == "alpha"
        # Order-independent: same answer when the input order is reversed.
        assert best_judge_from_log(["mu", "alpha", "zeta"]) == "alpha"

    def test_none_when_no_rated_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: every candidate is unrated.
        Assert: None — the caller then falls back to its own default judge.
        """
        from frugon.cost import best_judge_from_log

        monkeypatch.setattr("frugon.cost._get_model_tier", lambda m: -1)
        assert best_judge_from_log(["a", "b", "c"]) is None

    def test_none_on_empty_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Assert: an empty model list yields None (no judge to pick)."""
        from frugon.cost import best_judge_from_log

        monkeypatch.setattr("frugon.cost._get_model_tier", lambda m: 0)
        assert best_judge_from_log([]) is None


# ---------------------------------------------------------------------------
# Multi-candidate projection block (cost.py side)
# ---------------------------------------------------------------------------


class TestMultiCandidateProjections:
    """AnalysisResult.candidate_projections is populated only when --candidates lists >1.

    Single-candidate / no-candidate paths leave the field empty — the report layer
    keys off len > 1, so an empty list is the byte-identical no-op path that keeps
    --demo (no --candidates) unchanged.  This is the cost-side guarantee; the
    report-side rendering is covered in test_report_*.
    """

    def _gpt4t_log(self, tmp_path: Path) -> Path:
        """100 gpt-4-turbo calls so candidates beat / don't beat the baseline deterministically."""
        records = [
            {
                "model": "gpt-4-turbo",
                "request": {
                    "messages": [
                        {"role": "user", "content": "classify this support ticket"}
                    ]
                },
                "response": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "billing"}}
                    ]
                },
                "usage": {"prompt_tokens": 200, "completion_tokens": 5},
            }
            for _ in range(100)
        ]
        return _write_jsonl(records, tmp_path)

    def test_empty_when_single_candidate(self, tmp_path: Path) -> None:
        """Single --candidates entry: the headline picks it; no per-candidate block."""
        result = analyze_logs(
            self._gpt4t_log(tmp_path), candidates=["gpt-4o-mini"]
        )
        assert result.candidate_projections == []

    def test_default_pool_populates_capped_transparency_block(
        self, tmp_path: Path
    ) -> None:
        """No --candidates: the default-pool path now DOES populate a capped
        "Candidates considered" transparency block (PD-directed 2026-07-02) —
        real users and the un-pinned demo see what was considered, not just the
        winner. Capped to the recommended candidate plus the next-4-cheapest
        that also beat the baseline (5 rows max; see analyze_records)."""
        result = analyze_logs(self._gpt4t_log(tmp_path))
        assert result.used_default_pool is True
        assert 1 < len(result.candidate_projections) <= 5
        assert result.candidate_projections[0].status == "recommended"

    def test_multi_candidate_tags_recommended_considered_unpriced(
        self, tmp_path: Path
    ) -> None:
        """Multi-candidate run surfaces every candidate with the right status tag.

        gpt-4o-mini beats baseline AND is the cheapest -> recommended.
        claude-3-7-sonnet-latest beats baseline but is more expensive than 4o-mini -> considered.
        imaginary-model-9999 has no entry in the pricing table -> unpriced.
        """
        result = analyze_logs(
            self._gpt4t_log(tmp_path),
            candidates=[
                "gpt-4o-mini",
                "claude-3-7-sonnet-latest",
                "imaginary-model-9999",
            ],
        )
        assert len(result.candidate_projections) == 3
        by_model = {p.model: p for p in result.candidate_projections}
        assert by_model["gpt-4o-mini"].status == "recommended"
        assert by_model["claude-3-7-sonnet-latest"].status == "considered"
        assert by_model["imaginary-model-9999"].status == "unpriced"
        # The recommended row's observed_cost is now the split blended cost.
        # When all calls are easy (200-token prompts) the split blended == full-swap,
        # so this assertion still holds against projected_cost.
        assert (
            by_model["gpt-4o-mini"].observed_cost == result.projected_cost
        )

    def test_multi_candidate_tags_more_expensive_when_loses_to_baseline(
        self, tmp_path: Path
    ) -> None:
        """Candidate priced higher than baseline -> more_expensive (not silently dropped)."""
        # Baseline gpt-4o-mini is cheap; gpt-4o + gpt-4-turbo will be more expensive.
        records = [
            {
                "model": "gpt-4o-mini",
                "request": {
                    "messages": [{"role": "user", "content": "x"}]
                },
                "response": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "y"}}
                    ]
                },
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            }
            for _ in range(50)
        ]
        path = _write_jsonl(records, tmp_path)
        result = analyze_logs(
            path, candidates=["gpt-4o", "gpt-4-turbo"]
        )
        # No candidate beats the baseline so the headline has no recommendation.
        assert result.candidate_model is None
        # But both candidates were considered — they should show up tagged.
        statuses = {p.model: p.status for p in result.candidate_projections}
        assert statuses == {"gpt-4o": "more_expensive", "gpt-4-turbo": "more_expensive"}

    def test_multi_candidate_headline_math_unchanged(
        self, tmp_path: Path
    ) -> None:
        """The new block is rendering metadata only — projected_cost must equal the
        single-candidate path result for the same winner."""
        path = self._gpt4t_log(tmp_path)
        single = analyze_logs(path, candidates=["gpt-4o-mini"])
        multi = analyze_logs(
            path,
            candidates=[
                "gpt-4o-mini",
                "claude-3-7-sonnet-latest",
                "imaginary-model-9999",
            ],
        )
        assert single.candidate_model == multi.candidate_model
        assert single.projected_cost == multi.projected_cost
        assert single.total_cost == multi.total_cost


class TestNoPriceableCandidates:
    """AnalysisResult.no_priceable_candidates -- state (b) of the three-state
    no-candidate distinction (report._render_wholesale_panel and its md/html
    counterparts): (a) evaluated, none cheaper; (b) no priceable candidate in
    the pool (the cost race never ran); (c) a mixed pool (existing "unpriced"
    tag behaviour, untouched by this flag).
    """

    def _gpt4t_log(self, tmp_path: Path) -> Path:
        """100 gpt-4-turbo calls -- mirrors TestMultiCandidateProjections._gpt4t_log."""
        records = [
            {
                "model": "gpt-4-turbo",
                "request": {
                    "messages": [
                        {"role": "user", "content": "classify this support ticket"}
                    ]
                },
                "response": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "billing"}}
                    ]
                },
                "usage": {"prompt_tokens": 200, "completion_tokens": 5},
            }
            for _ in range(100)
        ]
        return _write_jsonl(records, tmp_path)

    def test_single_unpriced_candidate_flags_no_priceable_candidates(
        self, tmp_path: Path
    ) -> None:
        """--candidates with ONE model that has no list price (e.g. a local
        model) -- the bug scenario from the live report: candidate_model is
        None, but the cost race never ran, so no_priceable_candidates is True."""
        result = analyze_logs(
            self._gpt4t_log(tmp_path), candidates=["ollama/llama3.2:1b"]
        )
        assert result.candidate_model is None
        assert result.no_priceable_candidates is True
        assert result.unpriced_candidate_names == ["ollama/llama3.2:1b"]

    def test_multi_all_unpriced_candidates_flags_no_priceable_candidates(
        self, tmp_path: Path
    ) -> None:
        """Every explicit candidate unpriced -- still no priceable candidate,
        names preserved in user-supplied order."""
        result = analyze_logs(
            self._gpt4t_log(tmp_path),
            candidates=["imaginary-model-9999", "another-fake-model"],
        )
        assert result.candidate_model is None
        assert result.no_priceable_candidates is True
        assert result.unpriced_candidate_names == [
            "imaginary-model-9999",
            "another-fake-model",
        ]

    def test_mixed_pool_does_not_flag_no_priceable_candidates(
        self, tmp_path: Path
    ) -> None:
        """State (c): at least one candidate priced -- the race ran, so this is
        NOT state (b), even when the priced candidate loses to the baseline and
        another candidate is unpriced."""
        result = analyze_logs(
            self._gpt4t_log(tmp_path),
            candidates=["gpt-4o", "imaginary-model-9999"],
        )
        assert result.no_priceable_candidates is False
        assert result.unpriced_candidate_names == ["imaginary-model-9999"]

    def test_default_pool_never_flags_no_priceable_candidates(
        self, tmp_path: Path
    ) -> None:
        """No explicit --candidates: the built-in pool is always fully priced."""
        result = analyze_logs(self._gpt4t_log(tmp_path))
        assert result.used_default_pool is True
        assert result.no_priceable_candidates is False
        assert result.unpriced_candidate_names == []


class TestCandidateProjectionSplitBasis:
    """Candidate projections use SPLIT basis (not full-swap) and sign reads correctly."""

    def _gpt4t_log_with_timestamps(self, tmp_path: Path) -> Path:
        """100 gpt-4-turbo calls with timestamps so monthly projection fires."""
        import datetime
        import json

        records = []
        base = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
        for i in range(100):
            ts = (base + datetime.timedelta(hours=i * 0.5)).isoformat()
            records.append(
                {
                    "model": "gpt-4-turbo",
                    "request": {
                        "messages": [{"role": "user", "content": "classify this"}]
                    },
                    "response": {
                        "choices": [
                            {"message": {"role": "assistant", "content": "billing"}}
                        ]
                    },
                    "usage": {"prompt_tokens": 200, "completion_tokens": 5},
                    "timestamp": ts,
                }
            )
        path = tmp_path / "log.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return path

    def test_recommended_monthly_equals_headline_split(self, tmp_path: Path) -> None:
        """Recommended row monthly_cost must equal the headline New-spend exactly.

        The headline New-spend is the total-basis blended:
            result.monthly_cost - (split.monthly_baseline - split.monthly_blended)

        For this test fixture all 100 calls are easy (200 tokens each, well below
        EASY_THRESHOLD), so every call is routed to gpt-4o-mini and
        split.monthly_baseline == result.monthly_cost.  The formula therefore
        collapses to split.monthly_blended, which is the expected value we assert
        against.
        """
        result = analyze_logs(
            self._gpt4t_log_with_timestamps(tmp_path),
            candidates=["gpt-4o-mini", "claude-3-7-sonnet-latest"],
        )
        assert result.split is not None, "split should exist for cheaper-rated candidate"
        assert len(result.candidate_projections) == 2

        by_model = {p.model: p for p in result.candidate_projections}
        rec = by_model[result.split.candidate_model]
        assert rec.status == "recommended"

        # Compute expected headline New-spend via the total-basis formula.
        from decimal import Decimal

        split = result.split
        assert split.monthly_baseline is not None
        assert split.monthly_blended is not None
        assert result.monthly_cost is not None
        expected_headline = result.monthly_cost - (
            split.monthly_baseline - split.monthly_blended
        )

        # Core invariant: recommended row monthly_cost == headline New-spend.
        assert rec.monthly_cost == expected_headline, (
            f"recommended monthly {rec.monthly_cost} != headline New-spend "
            f"{expected_headline}"
        )
        # saving_pct: (baseline_reduction / total_monthly) * 100
        baseline_reduction = split.monthly_baseline - split.monthly_blended
        expected_saving_pct = (
            (baseline_reduction / result.monthly_cost) * Decimal("100")
            if result.monthly_cost > Decimal("0")
            else None
        )
        assert rec.saving_pct == expected_saving_pct

    def test_saving_sign_lower_on_cheaper_candidate(self) -> None:
        """Positive saving_pct renders as 'X.X% lower', not '+X.X%'."""
        from decimal import Decimal

        from frugon.report import _fmt_candidate_saving

        assert _fmt_candidate_saving(Decimal("34.5")) == "34.5% lower"
        assert _fmt_candidate_saving(Decimal("0.0")) == "0.0% lower"

    def test_saving_sign_higher_on_more_expensive(self) -> None:
        """Negative saving_pct renders as 'X.X% higher'."""
        from decimal import Decimal

        from frugon.report import _fmt_candidate_saving

        assert _fmt_candidate_saving(Decimal("-98.2")) == "98.2% higher"
        assert _fmt_candidate_saving(Decimal("-12.0")) == "12.0% higher"

    def test_multi_candidate_split_projections_status_tags(self, tmp_path: Path) -> None:
        """recommended/considered/more_expensive/unpriced tags correct under split basis.

        Baseline: gpt-4-turbo ($0.00001/$0.00003 per token).
        - gpt-4o-mini: very cheap ($1.5e-7/$6e-7) -> recommended.
        - gpt-4o: cheaper than turbo ($0.0000025/$0.00001) -> considered.
        - o1: more expensive than turbo ($0.000015/$0.00006) -> more_expensive.
        - imaginary-xyz: no price entry -> unpriced.
        """
        result = analyze_logs(
            self._gpt4t_log_with_timestamps(tmp_path),
            candidates=["gpt-4o-mini", "gpt-4o", "o1", "imaginary-xyz"],
        )
        by_model = {p.model: p for p in result.candidate_projections}
        assert by_model["gpt-4o-mini"].status == "recommended"
        assert by_model["gpt-4o"].status == "considered"
        assert by_model["o1"].status == "more_expensive"
        assert by_model["imaginary-xyz"].status == "unpriced"

        # Recommended row monthly == headline New-spend (core invariant).
        # Total-basis formula: result.monthly_cost - (split.monthly_baseline - split.monthly_blended)
        assert result.split is not None
        rec = by_model["gpt-4o-mini"]
        split = result.split
        if (
            rec.monthly_cost is not None
            and split.monthly_baseline is not None
            and split.monthly_blended is not None
            and result.monthly_cost is not None
        ):
            expected = result.monthly_cost - (
                split.monthly_baseline - split.monthly_blended
            )
            assert rec.monthly_cost == expected

    def test_demo_no_candidates_populates_capped_default_pool_block(
        self, tmp_path: Path
    ) -> None:
        """No --candidates: the default-pool block populates (PD-directed
        2026-07-02) — capped to the recommended candidate plus the
        next-4-cheapest that also beat the baseline (5 rows max)."""
        import json

        records = [
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "x"}]},
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "y"}}]
                },
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            }
        ]
        path = tmp_path / "simple.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        result = analyze_logs(path)
        assert 1 < len(result.candidate_projections) <= 5, (
            "default-pool run should populate a capped candidate_projections list"
        )
        assert result.candidate_projections[0].status == "recommended"
