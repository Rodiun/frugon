"""Equivalence proof: the partition-reusing split path == the per-candidate path.

FRG-OSS-038 (perf): the default-pool "Candidates considered" block used to
call :func:`frugon.routing.compute_split` once per candidate (up to 23 times),
each call re-running the candidate-INDEPENDENT easy/hard difficulty
classification over the baseline model's own calls from scratch — on the
bundled 56,100-record demo this alone cost ~4s of redundant Decimal
arithmetic (1,014,200 repeated ``is_easy`` evaluations for a call set of only
46,100 records).

The fix: :func:`frugon.routing.partition_by_difficulty` classifies the
baseline call set ONCE into a :class:`~frugon.routing.DifficultyPartition`,
and :func:`frugon.routing.compute_split_from_partition` derives each
candidate's :class:`~frugon.routing.SplitRouting` from that shared partition
— reusing the hard-bucket aggregates in O(1) and pricing only the easy
subset per candidate, while the per-call "route only if actually cheaper for
THIS call" gate (§6 never-inflate) is preserved exactly.

This suite proves the two paths are byte-for-byte identical — every field of
every returned SplitRouting, for every candidate — on (a) the real bundled
demo fixture across the full 23-model roster, and (b) a randomized property
fixture with synthetic prices, BEFORE the fast path is trusted as a drop-in
replacement anywhere in the cost engine.
"""

from __future__ import annotations

import random as _random
from decimal import Decimal
from pathlib import Path

import frugon
from frugon.cost import _ROUTING_CANDIDATES, CallCost, LogRecord, compute_call_cost, iter_records
from frugon.pricing import ModelPrice, get_model_price, pinned_pricing_identity
from frugon.routing import (
    compute_split,
    compute_split_from_partition,
    partition_by_difficulty,
)

assert frugon.__file__ is not None
_SAMPLE = Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"

_SPLIT_FIELDS = [
    "baseline_model",
    "candidate_model",
    "routed_count",
    "kept_count",
    "routed_cost",
    "kept_cost",
    "baseline_cost",
    "blended_cost",
    "easy_threshold",
    "monthly_baseline",
    "monthly_blended",
]


def _assert_splits_identical(old: object, new: object, *, context: str) -> None:
    for field in _SPLIT_FIELDS:
        old_val = getattr(old, field)
        new_val = getattr(new, field)
        assert old_val == new_val, (
            f"{context}: field {field!r} differs — old={old_val!r} new={new_val!r}"
        )


class TestPartitionEquivalenceOnBundledDemo:
    """The real 56,100-record demo fixture, full 23-model roster."""

    def _baseline_call_costs(self) -> tuple[str, list[CallCost]]:
        records, _skipped = iter_records(_SAMPLE)
        with pinned_pricing_identity():
            call_costs = [compute_call_cost(r) for r in records]
        cost_by_model: dict[str, Decimal] = {}
        for cc in call_costs:
            if cc.price is not None:
                cost_by_model[cc.record.model] = (
                    cost_by_model.get(cc.record.model, Decimal("0")) + cc.total_cost
                )
        dominant = max(cost_by_model, key=lambda m: cost_by_model[m])
        baseline_call_costs = [
            cc for cc in call_costs if cc.record.model == dominant and cc.price is not None
        ]
        return dominant, baseline_call_costs

    def test_every_candidate_produces_an_identical_split(self) -> None:
        dominant, baseline_call_costs = self._baseline_call_costs()
        partition = partition_by_difficulty(baseline_call_costs)

        compared = 0
        for cand in _ROUTING_CANDIDATES:
            if cand == dominant:
                continue
            price = get_model_price(cand)
            if price is None:
                continue
            old = compute_split(
                baseline_model=dominant,
                candidate_model=cand,
                baseline_call_costs=baseline_call_costs,
                candidate_price=price,
            )
            new = compute_split_from_partition(
                baseline_model=dominant,
                candidate_model=cand,
                partition=partition,
                candidate_price=price,
            )
            _assert_splits_identical(old, new, context=f"candidate={cand}")
            compared += 1

        # Precondition guard: the roster must actually exercise >1 priced
        # candidate, or this test would pass vacuously.
        assert compared >= 15, (
            f"expected most of the {len(_ROUTING_CANDIDATES)}-model roster to be "
            f"priced against the demo's dominant model; only {compared} were "
            "comparable — the fixture may have drifted"
        )

    def test_with_monthly_projection_basis_also_identical(self) -> None:
        """Same proof, but with a --window projection basis engaged (monthly_*
        fields populated) — the projection multiplies the same aggregates, so
        this guards that path too."""
        dominant, baseline_call_costs = self._baseline_call_costs()
        partition = partition_by_difficulty(baseline_call_costs)

        for cand in _ROUTING_CANDIDATES[:6]:
            if cand == dominant:
                continue
            price = get_model_price(cand)
            if price is None:
                continue
            old = compute_split(
                baseline_model=dominant,
                candidate_model=cand,
                baseline_call_costs=baseline_call_costs,
                candidate_price=price,
                window_days=30,
            )
            new = compute_split_from_partition(
                baseline_model=dominant,
                candidate_model=cand,
                partition=partition,
                candidate_price=price,
                window_days=30,
            )
            _assert_splits_identical(old, new, context=f"windowed candidate={cand}")


class TestPartitionEquivalenceRandomProperty:
    """Randomized synthetic records + synthetic prices — a property-style proof
    that does not depend on the live pricing/quality registry."""

    @staticmethod
    def _rand_record(rng: _random.Random) -> LogRecord:
        prompt_tokens = rng.randint(0, 3000)
        completion_tokens = rng.randint(0, 2000)
        turns = rng.randint(1, 12)
        messages = [
            {"role": "user" if j % 2 == 0 else "assistant", "content": "x" * 10}
            for j in range(turns)
        ]
        return LogRecord(
            model="baseline-model",
            messages=messages,
            completion_text="y" * completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timestamp=None,
        )

    @staticmethod
    def _call_cost(record: LogRecord, price: ModelPrice) -> CallCost:
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

    def test_two_hundred_random_candidate_prices_all_match(self) -> None:
        rng = _random.Random(42)
        baseline_price = ModelPrice(
            "baseline-model", Decimal("0.000005"), Decimal("0.000015"), "test", None
        )
        records = [self._rand_record(rng) for _ in range(2000)]
        call_costs = [self._call_cost(r, baseline_price) for r in records]
        partition = partition_by_difficulty(call_costs)

        price_rng = _random.Random(7)
        for _trial in range(200):
            input_price = Decimal(str(round(price_rng.uniform(0.0000001, 0.00002), 10)))
            output_price = Decimal(str(round(price_rng.uniform(0.0000001, 0.00002), 10)))
            cand_price = ModelPrice(
                "candidate", input_price, output_price, "test", None
            )
            old = compute_split(
                baseline_model="baseline-model",
                candidate_model="candidate",
                baseline_call_costs=call_costs,
                candidate_price=cand_price,
            )
            new = compute_split_from_partition(
                baseline_model="baseline-model",
                candidate_model="candidate",
                partition=partition,
                candidate_price=cand_price,
            )
            _assert_splits_identical(old, new, context=f"trial input={input_price} output={output_price}")

    def test_mixed_prompt_completion_ratios_route_differently_per_candidate(self) -> None:
        """Precondition guard: prove the per-call 'route only if cheaper for
        THIS call' gate is actually exercised by this fixture (calls with very
        different prompt/completion ratios can flip differently for the same
        candidate) — otherwise the equivalence proof above would be too easy.
        """
        cheap_input_price = ModelPrice(
            "cand", Decimal("0.0000001"), Decimal("0.00002"), "test", None
        )
        # Prompt-heavy easy call (score 0.256, still < 0.35 threshold): favours
        # the cheap-input candidate — routes.
        prompt_heavy = LogRecord(
            model="baseline-model",
            messages=[{"role": "user", "content": "x"}],
            completion_text="y",
            prompt_tokens=500,
            completion_tokens=10,
            timestamp=None,
        )
        # Completion-heavy easy call (score 0.180, still < 0.35 threshold):
        # disfavours the expensive-output candidate — does NOT route.
        completion_heavy = LogRecord(
            model="baseline-model",
            messages=[{"role": "user", "content": "x"}],
            completion_text="y",
            prompt_tokens=10,
            completion_tokens=300,
            timestamp=None,
        )
        baseline_price = ModelPrice(
            "baseline-model", Decimal("0.000005"), Decimal("0.000005"), "test", None
        )
        call_costs = [
            self._call_cost(prompt_heavy, baseline_price),
            self._call_cost(completion_heavy, baseline_price),
        ]
        partition = partition_by_difficulty(call_costs)
        split = compute_split_from_partition(
            baseline_model="baseline-model",
            candidate_model="cand",
            partition=partition,
            candidate_price=cheap_input_price,
        )
        # One of the two easy calls should route, the other should not —
        # proving the per-call gate genuinely discriminates within one
        # candidate's split (not just across candidates).
        assert split.routed_count == 1, (
            "expected exactly one of the two easy calls to route to the "
            "cheap-input/expensive-output candidate — if both or neither "
            "route, this fixture does not exercise the per-call gate"
        )

    def test_empty_partition_all_hard_routes_nothing(self) -> None:
        """Edge case: every baseline call is HARD (difficulty >= threshold) —
        the partition's easy bucket is empty.

        No candidate can ever route anything in this case, regardless of how
        cheap it is: ``routed_count`` must be 0 and the blended cost must
        equal the baseline cost exactly (every call stays kept). This is the
        degenerate case the O(easy_count) fast path must still get right —
        an empty ``easy_calls`` list should make
        :func:`compute_split_from_partition` a no-op loop that falls straight
        through to "everything kept," never raise, and never invent a routed
        call.
        """
        # Large prompt AND completion AND deep multi-turn — saturates every
        # difficulty signal well above EASY_THRESHOLD (0.35), so every record
        # classifies HARD.  (0.5*1 + 0.35*1 + 0.15*1 == 1.0 >> 0.35.)
        hard_records = [
            LogRecord(
                model="baseline-model",
                messages=[
                    {"role": "user" if j % 2 == 0 else "assistant", "content": "x" * 50}
                    for j in range(10)
                ],
                completion_text="y" * 2000,
                prompt_tokens=5000,
                completion_tokens=2000,
                timestamp=None,
            )
            for _ in range(20)
        ]
        baseline_price = ModelPrice(
            "baseline-model", Decimal("0.00001"), Decimal("0.00003"), "test", None
        )
        call_costs = [self._call_cost(r, baseline_price) for r in hard_records]

        partition = partition_by_difficulty(call_costs)
        assert partition.easy_calls == [], (
            "precondition: every record must classify HARD for this to be a "
            "genuine empty-partition test"
        )
        assert partition.hard_count == len(call_costs)

        baseline_cost = sum((cc.total_cost for cc in call_costs), Decimal("0"))
        assert partition.baseline_cost == baseline_cost
        assert partition.hard_cost == baseline_cost

        # Even a dramatically cheaper candidate cannot route anything — there
        # are no easy calls to consider.
        very_cheap_price = ModelPrice(
            "candidate", Decimal("0.0000000001"), Decimal("0.0000000001"), "test", None
        )
        split = compute_split_from_partition(
            baseline_model="baseline-model",
            candidate_model="candidate",
            partition=partition,
            candidate_price=very_cheap_price,
        )
        assert split.routed_count == 0
        assert split.routed_cost == Decimal("0")
        assert split.kept_count == len(call_costs)
        assert split.blended_cost == split.baseline_cost == baseline_cost, (
            "an all-hard partition must leave blended cost == baseline cost "
            "== the sum of kept costs — nothing is ever routed"
        )

        # Cross-check against the un-optimized path for the SAME call set,
        # proving the fast path and the original path agree even at this
        # degenerate boundary, not just in the common (mixed easy/hard) case.
        legacy = compute_split(
            baseline_model="baseline-model",
            candidate_model="candidate",
            baseline_call_costs=call_costs,
            candidate_price=very_cheap_price,
        )
        _assert_splits_identical(legacy, split, context="empty-partition (all hard)")
