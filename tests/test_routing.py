"""Tests for frugon.routing — per-call difficulty classifier + split-routing math.

This is cost-math carve-out territory: the difficulty score and
the blended split cost get golden / known-answer vectors AND property tests, with
>90% coverage gated by .coveragerc-strict.  A wrong blended number or an inflated
routed/kept count kills credibility on contact, so every figure here is hand-derived.

AAA pattern; naming test_<unit>_<scenario>_<expected>.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from frugon.cost import CallCost, LogRecord
from frugon.pricing import ModelPrice
from frugon.routing import (
    EASY_THRESHOLD,
    SplitRouting,
    compute_split,
    difficulty_score,
    is_easy,
    select_easy_target,
)

# ---------------------------------------------------------------------------
# Frozen synthetic prices — independent of registry/network state
# ---------------------------------------------------------------------------

# Baseline "premium" price (gpt-4-turbo-shaped): $10 / $30 per 1M tokens.
PREMIUM = ModelPrice("premium", Decimal("0.00001"), Decimal("0.00003"), "test", None)
# Cheap "mini" routing target: $0.10 / $0.40 per 1M tokens.
MINI = ModelPrice("mini", Decimal("0.0000001"), Decimal("0.0000004"), "test", None)


def _record(prompt_tokens: int, completion_tokens: int, n_messages: int = 2) -> LogRecord:
    """Build a LogRecord with explicit token counts and message depth."""
    messages = [{"role": "user", "content": "x"} for _ in range(n_messages)]
    return LogRecord(
        model="premium",
        messages=messages,
        completion_text="y",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        timestamp=None,
    )


def _call_cost(record: LogRecord, price: ModelPrice) -> CallCost:
    """Price *record* at *price* (the baseline cost the split keeps for hard calls)."""
    prompt_cost = price.input_cost_per_token * Decimal(record.prompt_tokens)
    completion_cost = price.output_cost_per_token * Decimal(record.completion_tokens)
    return CallCost(
        record=record,
        price=price,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        total_cost=prompt_cost + completion_cost,
        token_source="usage_block",
    )


# ---------------------------------------------------------------------------
# difficulty_score — golden vectors (hand-derived from the documented formula)
# ---------------------------------------------------------------------------


class TestDifficultyScoreGoldenVectors:
    """Exact difficulty scores for known inputs (formula 0.5·sp + 0.35·sc + 0.15·st)."""

    def test_zero_tokens_single_message_scores_zero(self) -> None:
        # sp=0, sc=0, turns=max(1-1,0)=0 → st=0 → 0.0
        assert difficulty_score(_record(0, 0, n_messages=1)) == Decimal("0")

    def test_saturated_signals_score_one(self) -> None:
        # pt=1000→sp=1, ct=600→sc=1, 7 messages→turns=6→st=1 → 0.5+0.35+0.15 = 1.0
        assert difficulty_score(_record(1000, 600, n_messages=7)) == Decimal("1.00")

    def test_oversaturated_signals_clamp_to_one(self) -> None:
        # pt and ct far above scale clamp to 1; turns clamp to 1 → still exactly 1.0
        assert difficulty_score(_record(100_000, 100_000, n_messages=50)) == Decimal("1.00")

    def test_half_scale_prompt_only_exact(self) -> None:
        # pt=500→sp=0.5, ct=0, 1 message→turns=0 → 0.5·0.5 = 0.25
        assert difficulty_score(_record(500, 0, n_messages=1)) == Decimal("0.250")

    def test_typical_classification_call_is_low(self) -> None:
        # pt=50, ct=1, 2 messages (turns=1): 0.5·0.05 + 0.35·(1/600) + 0.15·(1/6)
        score = difficulty_score(_record(50, 1, n_messages=2))
        # 0.025 + 0.000583… + 0.025 = ~0.0506 — comfortably easy
        assert score < EASY_THRESHOLD
        assert score < Decimal("0.06")


class TestDifficultyScoreBounds:
    """The score is always within [0, 1] regardless of input."""

    @pytest.mark.parametrize(
        ("pt", "ct", "n"),
        [(0, 0, 1), (1, 1, 1), (999, 599, 6), (1000, 600, 7), (10**9, 10**9, 999)],
    )
    def test_score_within_unit_interval(self, pt: int, ct: int, n: int) -> None:
        score = difficulty_score(_record(pt, ct, n_messages=n))
        assert Decimal("0") <= score <= Decimal("1")

    def test_negative_token_counts_clamp_to_zero_floor(self) -> None:
        # A corrupt negative count must not drive the score below zero.
        assert difficulty_score(_record(-50, -10, n_messages=1)) == Decimal("0")


class TestDifficultyScoreMonotonicity:
    """Property: the score never decreases when any signal increases."""

    def test_monotonic_non_decreasing_in_prompt_tokens(self) -> None:
        prev = Decimal("-1")
        for pt in range(0, 2001, 50):
            score = difficulty_score(_record(pt, 0, n_messages=1))
            assert score >= prev, f"score dropped at pt={pt}"
            prev = score

    def test_monotonic_non_decreasing_in_completion_tokens(self) -> None:
        prev = Decimal("-1")
        for ct in range(0, 1201, 30):
            score = difficulty_score(_record(0, ct, n_messages=1))
            assert score >= prev, f"score dropped at ct={ct}"
            prev = score

    def test_monotonic_non_decreasing_in_conversation_depth(self) -> None:
        prev = Decimal("-1")
        for n in range(1, 20):
            score = difficulty_score(_record(0, 0, n_messages=n))
            assert score >= prev, f"score dropped at n_messages={n}"
            prev = score


class TestIsEasyThreshold:
    """is_easy uses a strict < threshold so the boundary is hard (conservative)."""

    def test_short_call_is_easy(self) -> None:
        assert is_easy(_record(40, 2, n_messages=2)) is True

    def test_long_call_is_hard(self) -> None:
        assert is_easy(_record(1500, 800, n_messages=4)) is False

    def test_score_exactly_at_threshold_is_hard(self) -> None:
        # Construct a record whose score is exactly EASY_THRESHOLD (0.35):
        # pt=700 → sp=0.7 → 0.5·0.7 = 0.35, ct=0, 1 message → 0.0. Total 0.35.
        rec = _record(700, 0, n_messages=1)
        assert difficulty_score(rec) == EASY_THRESHOLD
        assert is_easy(rec) is False

    def test_custom_threshold_respected(self) -> None:
        rec = _record(700, 0, n_messages=1)  # score 0.35
        assert is_easy(rec, threshold=Decimal("0.40")) is True
        assert is_easy(rec, threshold=Decimal("0.30")) is False


# ---------------------------------------------------------------------------
# select_easy_target — cheapest rated, priced, strictly-cheaper candidate
# ---------------------------------------------------------------------------


class TestSelectEasyTarget:
    """Easy-call target selection against the real bundled pricing/quality tables."""

    def test_premium_baseline_picks_cheapest_rated_mini(self) -> None:
        # gpt-4-turbo baseline → gpt-4o-mini is the cheapest rated cheaper model.
        target = select_easy_target(
            "gpt-4-turbo",
            ["gpt-4o", "gpt-4o-mini", "claude-3-haiku-20240307"],
        )
        assert target == "gpt-4o-mini"

    def test_returns_none_when_no_cheaper_candidate(self) -> None:
        # gpt-4o-mini is already very cheap; gpt-4o is more expensive → no target.
        assert select_easy_target("gpt-4o-mini", ["gpt-4o"]) is None

    def test_unrated_candidate_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An unrated-but-cheap candidate must never be auto-selected as the target.
        from frugon import routing as routing_mod

        prices = {
            "premium": PREMIUM,
            "unrated-cheap": ModelPrice(
                "unrated-cheap", Decimal("0.0000001"), Decimal("0.0000001"), "test", None
            ),
        }
        monkeypatch.setattr(routing_mod, "get_model_price", prices.get)
        monkeypatch.setattr(routing_mod, "_is_unrated", lambda m: m == "unrated-cheap")
        assert select_easy_target("premium", ["unrated-cheap"]) is None

    def test_unknown_baseline_returns_none(self) -> None:
        assert select_easy_target("totally-unknown-model-xyz", ["gpt-4o-mini"]) is None

    def test_pool_containing_baseline_and_unpriced_candidate(self) -> None:
        # The baseline appearing in its own pool is skipped; an unpriced junk
        # candidate is skipped; gpt-4o-mini still wins.
        target = select_easy_target(
            "gpt-4-turbo",
            ["gpt-4-turbo", "frugon-no-such-priced-model-xyz", "gpt-4o-mini"],
        )
        assert target == "gpt-4o-mini"

    def test_rated_but_unpriced_candidate_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A candidate that is rated but has no price must be skipped, not crash.
        from frugon import routing as routing_mod

        prices = {"premium": PREMIUM, "rated-no-price": None}
        monkeypatch.setattr(routing_mod, "get_model_price", prices.get)
        monkeypatch.setattr(routing_mod, "_is_unrated", lambda m: False)
        assert select_easy_target("premium", ["rated-no-price"]) is None

    def test_equal_priced_candidates_tie_break_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two rated candidates with identical blended prices → name decides the
        # winner deterministically, regardless of pool order.
        from frugon import routing as routing_mod

        cheap = ModelPrice("c", Decimal("0.000001"), Decimal("0.000003"), "test", None)
        prices = {
            "premium": PREMIUM,
            "z-tie": cheap,
            "a-tie": cheap,
        }
        monkeypatch.setattr(routing_mod, "get_model_price", prices.get)
        monkeypatch.setattr(routing_mod, "_is_unrated", lambda m: False)
        assert select_easy_target("premium", ["z-tie", "a-tie"]) == "a-tie"
        assert select_easy_target("premium", ["a-tie", "z-tie"]) == "a-tie"


# ---------------------------------------------------------------------------
# compute_split — golden vectors (hand-derived blended cost)
# ---------------------------------------------------------------------------


class TestComputeSplitGolden:
    """Exact routed/kept counts and blended cost for a 2-easy + 1-hard mix."""

    def _fixture(self) -> list[CallCost]:
        easy1 = _call_cost(_record(10, 2, n_messages=1), PREMIUM)
        easy2 = _call_cost(_record(20, 5, n_messages=2), PREMIUM)
        hard1 = _call_cost(_record(2000, 1000, n_messages=2), PREMIUM)
        return [easy1, easy2, hard1]

    def test_routed_and_kept_counts_exact(self) -> None:
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=self._fixture(),
            candidate_price=MINI,
        )
        assert split.routed_count == 2
        assert split.kept_count == 1
        assert split.total_count == 3

    def test_routed_cost_priced_at_candidate_exact(self) -> None:
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=self._fixture(),
            candidate_price=MINI,
        )
        # easy1 @ mini: 0.0000001·10 + 0.0000004·2  = 0.0000018
        # easy2 @ mini: 0.0000001·20 + 0.0000004·5  = 0.0000040
        assert split.routed_cost == Decimal("0.0000058")

    def test_kept_cost_priced_at_baseline_exact(self) -> None:
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=self._fixture(),
            candidate_price=MINI,
        )
        # hard1 @ premium: 0.00001·2000 + 0.00003·1000 = 0.02 + 0.03 = 0.05
        assert split.kept_cost == Decimal("0.05")

    def test_baseline_and_blended_cost_exact(self) -> None:
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=self._fixture(),
            candidate_price=MINI,
        )
        # baseline: easy1 0.00016 + easy2 0.00035 + hard1 0.05 = 0.05051
        assert split.baseline_cost == Decimal("0.05051")
        # blended: routed 0.0000058 + kept 0.05 = 0.0500058
        assert split.blended_cost == Decimal("0.0500058")

    def test_saving_pct_matches_formula(self) -> None:
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=self._fixture(),
            candidate_price=MINI,
        )
        expected = (
            (split.baseline_cost - split.blended_cost) / split.baseline_cost * Decimal("100")
        )
        assert split.saving_pct == expected
        assert split.saving_pct is not None
        assert split.saving_pct > Decimal("0")


class TestComputeSplitEdgeCases:
    """All-easy, all-hard, and zero-cost edge cases stay honest."""

    def test_all_easy_keeps_nothing_and_blended_equals_routed(self) -> None:
        calls = [
            _call_cost(_record(10, 2, n_messages=1), PREMIUM),
            _call_cost(_record(15, 3, n_messages=1), PREMIUM),
        ]
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=calls,
            candidate_price=MINI,
        )
        assert split.kept_count == 0
        assert split.kept_cost == Decimal("0")
        assert split.blended_cost == split.routed_cost

    def test_all_hard_routes_nothing_and_saving_is_zero(self) -> None:
        calls = [
            _call_cost(_record(2000, 900, n_messages=3), PREMIUM),
            _call_cost(_record(1800, 800, n_messages=4), PREMIUM),
        ]
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=calls,
            candidate_price=MINI,
        )
        assert split.routed_count == 0
        assert split.routed_cost == Decimal("0")
        assert split.blended_cost == split.baseline_cost
        assert split.saving_pct == Decimal("0")

    def test_empty_baseline_cost_saving_is_none(self) -> None:
        split = SplitRouting(
            baseline_model="premium",
            candidate_model="mini",
            routed_count=0,
            kept_count=0,
            routed_cost=Decimal("0"),
            kept_cost=Decimal("0"),
            baseline_cost=Decimal("0"),
            blended_cost=Decimal("0"),
            easy_threshold=EASY_THRESHOLD,
        )
        assert split.saving_pct is None


class TestComputeSplitNeverInflates:
    """Property: blended cost is never greater than baseline cost (never inflate)."""

    @pytest.mark.parametrize(
        "specs",
        [
            [(10, 2, 1), (2000, 1000, 2)],
            [(50, 5, 2), (40, 3, 2), (3000, 1500, 5)],
            [(800, 400, 2)],
            [(5, 1, 1), (9, 1, 1), (12, 2, 1)],
        ],
    )
    def test_blended_le_baseline(self, specs: list[tuple[int, int, int]]) -> None:
        calls = [_call_cost(_record(pt, ct, n), PREMIUM) for pt, ct, n in specs]
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=calls,
            candidate_price=MINI,
        )
        assert split.blended_cost <= split.baseline_cost

    def test_inverted_axis_candidate_does_not_inflate_a_call(self) -> None:
        """A candidate cheaper on blended-average but pricier on the OUTPUT axis
        must NOT route a completion-heavy easy call (that would inflate it).

        baseline in=out=$1e-5; candidate in=$1e-6 (cheaper) but out=$1.8e-5
        (pricier).  An easy, completion-heavy call (pt=50, ct=300) costs $0.0035
        at baseline but $0.00545 at the candidate — routing it would raise cost.
        compute_split must keep it, so blended_cost <= baseline_cost holds.
        """
        # Symmetric baseline (in == out == $1e-5) so the inverted candidate is
        # genuinely pricier on the output axis for a completion-heavy call.
        sym_baseline = ModelPrice("sym", Decimal("0.00001"), Decimal("0.00001"), "test", None)
        inverted = ModelPrice("inv", Decimal("0.000001"), Decimal("0.000018"), "test", None)
        easy_completion_heavy = _call_cost(_record(50, 300, n_messages=2), sym_baseline)
        assert is_easy(easy_completion_heavy.record)  # difficulty ~0.225 → easy
        # baseline: 1e-5·50 + 1e-5·300 = $0.0035; candidate: 1e-6·50 + 1.8e-5·300 = $0.00545

        split = compute_split(
            baseline_model="premium",
            candidate_model="inv",
            baseline_call_costs=[easy_completion_heavy],
            candidate_price=inverted,
        )
        assert split.routed_count == 0, "must not route a call that costs more at the candidate"
        assert split.kept_count == 1
        assert split.blended_cost <= split.baseline_cost
        assert split.saving_pct == Decimal("0")


class TestComputeSplitMonthlyProjection:
    """Monthly projection follows the same window/span disclosure rules as cost.py."""

    def _calls(self) -> list[CallCost]:
        return [
            _call_cost(_record(10, 2, n_messages=1), PREMIUM),
            _call_cost(_record(2000, 1000, n_messages=2), PREMIUM),
        ]

    def test_window_projects_both_axes(self) -> None:
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=self._calls(),
            candidate_price=MINI,
            window_days=10,
        )
        assert split.monthly_baseline == split.baseline_cost * Decimal("30") / Decimal("10")
        assert split.monthly_blended == split.blended_cost * Decimal("30") / Decimal("10")

    def test_span_projects_when_no_window(self) -> None:
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=self._calls(),
            candidate_price=MINI,
            observed_span_days=5.0,
        )
        assert split.monthly_baseline == split.baseline_cost * Decimal("30") / Decimal("5.0")

    def test_no_window_no_span_no_projection(self) -> None:
        split = compute_split(
            baseline_model="premium",
            candidate_model="mini",
            baseline_call_costs=self._calls(),
            candidate_price=MINI,
        )
        assert split.monthly_baseline is None
        assert split.monthly_blended is None


