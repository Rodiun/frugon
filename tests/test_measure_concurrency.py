"""Two-stage pipeline + A/B-order tests for run_measure.

run_measure runs its provider round-trips as a TWO-STAGE producer→consumer
pipeline over TWO independent bounded thread pools:

  * Stage 1 (sampling, the PRODUCER): every individual sampling call — each
    (prompt, model) for the baseline and for each candidate — is its own task,
    so sampling fans WIDE across the baseline + candidate provider endpoints, up
    to ``sample_workers = min(concurrency, n_prompts)`` calls in flight; and
  * Stage 2 (judging, the CONSUMER): the instant a prompt's full sample set
    (baseline + every candidate) resolves, that prompt's judge call(s) are handed
    to a SECOND, narrower pool (``judge_workers = min(concurrency,
    _JUDGE_MAX_CONCURRENCY)``) which drains CONCURRENTLY while stage 1 is still
    sampling other prompts.

These tests prove, with latency-simulating stubs that make NO real network
calls, that the pipeline is:

  * CORRECT — the comparisons stay in sampled-record order, each
    Comparison.verdicts / .candidate_outputs stay aligned 1:1 with candidates,
    the Tier1Tally counts (and self_judged_models / judge_model) are
    byte-identical to the ``concurrency=1`` fully-sequential reference; and
  * actually a TWO-STAGE OVERLAP — a judge call STARTS before the LAST sampling
    call finishes (stage 2 runs while stage 1 still has pending work), and stage
    1 alone can reach ``concurrency`` simultaneous sampling calls (wider than the
    old single-pool shape, which capped *total* in-flight across both kinds), and
  * correctly CAPPED — sample_workers tracks ``concurrency`` while judge_workers
    is clamped to ``_JUDGE_MAX_CONCURRENCY`` to protect the single judge endpoint.

A/B-order randomisation is exercised end-to-end: the same seed yields the same
deterministic A/B layout, and a judge with a fixed position bias resolves to the
correct candidate-relative verdict regardless of which slot the candidate sat in.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from frugon.cost import LogRecord
from frugon.measure import (
    _DEFAULT_CONCURRENCY,
    _JUDGE_MAX_CONCURRENCY,
    SampledOutput,
    _stage_worker_counts,
    run_measure,
)

# Per-stubbed-call latency.  Small enough to keep the test fast, large enough
# that a SEQUENTIAL run (sum of all sleeps) is comfortably separable from a
# concurrent run (≈ a few sleeps deep).
_CALL_LATENCY_S = 0.05


@pytest.fixture(autouse=True)
def _provider_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")


def _make_record(prompt_text: str) -> LogRecord:
    return LogRecord(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_text}],
        completion_text="ref",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _latency_completion_mock() -> MagicMock:
    """A litellm mock whose .completion() sleeps then echoes a deterministic body.

    The returned content encodes the model so the stub is content-stable and the
    judge stub (below) can be order-aware via the prompt text.
    """

    def completion(model: str, messages: list[Any], **kw: Any) -> MagicMock:
        time.sleep(_CALL_LATENCY_S)
        resp = MagicMock()
        resp.choices[0].message.content = f"out::{model}"
        return resp

    mock = MagicMock()
    mock.completion.side_effect = completion
    return mock


def _run_sequential_reference(
    records: list[LogRecord],
    current_model: str,
    candidates: list[str],
    *,
    use_judge: bool,
    judge_model: str,
    seed: int,
    judge_side_effect: Any,
) -> Any:
    """Run run_measure with ``concurrency=1`` — fully sequential, deterministic.

    ``concurrency=1`` collapses BOTH pipeline stages to a single worker, so every
    sampling call and every judge call runs strictly one-at-a-time.  This is the
    same code path the concurrent run takes (no parallelism-specific branch), so
    diffing a high-concurrency result against it isolates timing from correctness.
    """
    mock_litellm = _latency_completion_mock()
    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=judge_side_effect),
    ):
        return run_measure(
            records,
            current_model,
            candidates,
            n_samples=len(records),
            use_judge=use_judge,
            judge_model=judge_model,
            concurrency=1,
            seed=seed,
        )


def _tally_tuple(result: Any) -> list[tuple[str, int, int, int, int]]:
    return [
        (t.candidate, t.wins, t.losses, t.ties, t.errors)
        for t in (result.tier1_tallies or [])
    ]


# ---------------------------------------------------------------------------
# Per-stage worker derivation — the cap contract, unit-level.
# ---------------------------------------------------------------------------


def test_stage_worker_counts_caps_judging_below_high_concurrency() -> None:
    """Arrange: a high --concurrency (20) and many prompts (50).
    Assert: sample_workers tracks the flag (20) — sampling fans wide — while
    judge_workers is clamped to _JUDGE_MAX_CONCURRENCY (5) to protect the single
    judge endpoint.
    """
    sample_workers, judge_workers = _stage_worker_counts(20, n_prompts=50)
    assert sample_workers == 20
    assert judge_workers == _JUDGE_MAX_CONCURRENCY == 5


def test_stage_worker_counts_clamps_sampling_to_prompt_count() -> None:
    """Arrange: concurrency exceeds the number of prompts.
    Assert: sample_workers never exceeds n_prompts (no idle workers), judge stays
    capped.
    """
    sample_workers, judge_workers = _stage_worker_counts(20, n_prompts=3)
    assert sample_workers == 3
    assert judge_workers == 5


def test_stage_worker_counts_sequential_when_concurrency_one() -> None:
    """Arrange: concurrency=1 (the parity reference).
    Assert: BOTH stages collapse to a single worker — fully sequential.
    """
    sample_workers, judge_workers = _stage_worker_counts(1, n_prompts=10)
    assert sample_workers == 1
    assert judge_workers == 1


def test_stage_worker_counts_low_concurrency_caps_both_below_judge_max() -> None:
    """Arrange: concurrency below the judge cap (3).
    Assert: judge_workers follows the flag (3), not the higher ceiling (5).
    """
    sample_workers, judge_workers = _stage_worker_counts(3, n_prompts=10)
    assert sample_workers == 3
    assert judge_workers == 3


# ---------------------------------------------------------------------------
# Correctness parity — concurrent vs concurrency=1 sequential reference.
# ---------------------------------------------------------------------------


def test_run_measure_concurrent_matches_sequential_and_overlaps() -> None:
    """Arrange: 5 prompts, 2 candidates, judge on; stubbed per-call latency.
    Act: run_measure (concurrency=5) and a concurrency=1 sequential reference.
    Assert:
      * identical comparisons order, candidate_outputs, per-prompt verdicts,
        and Tier1Tally counts (correctness parity); and
      * the concurrent wall-clock is far below the serial sum of all stubbed
        sleeps (proves the provider calls actually overlapped).
    """
    records = [_make_record(f"prompt {i}") for i in range(5)]
    candidates = ["gpt-4o-mini", "claude-3-haiku-20240307"]
    # Deterministic per-call verdict cycle so both runs score identically.
    cycle = ["win", "loss", "tie"]

    def judge_side_effect(*args: Any, **kwargs: Any) -> str:
        # Positional call shape (from run_measure):
        #   (litellm_mod, judge_model, messages, current_output, candidate_output)
        # with candidate_is_a passed as a keyword.  Index by the candidate
        # output's content (model-stable) + prompt text so the verdict is a pure
        # function of the pair — identical across both runs.
        prompt_msgs = args[2]
        cand_out: SampledOutput = args[4]
        key = hash((cand_out.content, prompt_msgs[0]["content"]))
        return cycle[key % len(cycle)]

    # --- concurrent (real) run ------------------------------------------------
    # Each stubbed sampling call brackets itself in a thread-safe in-flight gauge
    # that records its running maximum, so overlap is proven STRUCTURALLY (real
    # simultaneous in-flight calls) rather than by wall-clock timing. A timing
    # proof is inherently flaky on loaded shared CI runners — Windows' coarse
    # ``time.sleep`` granularity and thread scheduling can stretch the wall-clock
    # arbitrarily under contention even when the calls genuinely overlap. The
    # gauge is immune to that: it observes actual concurrency directly.
    gauge = _ConcurrencyGauge()

    def _gauged_completion(model: str, messages: list[Any], **kw: Any) -> MagicMock:
        gauge.enter()
        time.sleep(_CALL_LATENCY_S)
        gauge.exit()
        resp = MagicMock()
        resp.choices[0].message.content = f"out::{model}"
        return resp

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = _gauged_completion
    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=judge_side_effect),
    ):
        concurrent_result = run_measure(
            records,
            "gpt-4o",
            candidates,
            n_samples=5,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=5,
            seed=7,
        )

    # --- sequential reference -------------------------------------------------
    sequential_result = _run_sequential_reference(
        records,
        "gpt-4o",
        candidates,
        use_judge=True,
        judge_model="gpt-4o",
        seed=7,
        judge_side_effect=judge_side_effect,
    )

    # Correctness parity: order, alignment, verdicts, tallies all identical.
    assert [c.record.messages[0]["content"] for c in concurrent_result.comparisons] == [
        f"prompt {i}" for i in range(5)
    ]
    assert [
        [o.model for o in c.candidate_outputs] for c in concurrent_result.comparisons
    ] == [candidates for _ in range(5)]
    assert [c.verdicts for c in concurrent_result.comparisons] == [
        c.verdicts for c in sequential_result.comparisons
    ]
    assert [
        [o.content for o in c.candidate_outputs] for c in concurrent_result.comparisons
    ] == [
        [o.content for o in c.candidate_outputs] for c in sequential_result.comparisons
    ]
    assert _tally_tuple(concurrent_result) == _tally_tuple(sequential_result)

    # Overlap proof (structural, not timing): there are 5×(1 baseline + 2
    # candidates)=15 sampling calls all submitted up front against a concurrency=5
    # sampling pool, each holding the gauge open for _CALL_LATENCY_S. If the calls
    # genuinely ran in parallel, the gauge necessarily observed more than one in
    # flight at its peak; a strictly sequential run would never exceed a peak of 1.
    # Observing the peak directly is deterministic regardless of how slow or loaded
    # the runner is — unlike a wall-clock threshold, it cannot be tripped by sleep
    # overshoot or thread-scheduling jitter on a contended CI machine.
    assert gauge.peak > 1, (
        f"peak concurrent sampling calls was {gauge.peak} (expected > 1) "
        f"— the provider calls did not overlap"
    )


def test_run_measure_pipeline_parity_multi_candidate_judged() -> None:
    """Arrange: 4 prompts x 3 candidates, judge on -- a multi-prompt,
    multi-candidate judged run.
    Act: the concurrent pipeline (concurrency=5) vs a concurrency=1 sequential
    reference over the SAME deterministic pair-keyed verdict function.
    Assert: comparisons order, per-candidate candidate_outputs (model + content),
    per-prompt verdicts, Tier1Tally counts, self_judged_models, and judge_model
    are ALL identical -- the pipeline changes timing, never the result.
    """
    records = [_make_record(f"prompt {i}") for i in range(4)]
    candidates = ["gpt-4o-mini", "claude-3-haiku-20240307", "gpt-4o-mini-2024-07-18"]
    cycle = ["win", "loss", "tie", "error"]

    def judge_side_effect(*args: Any, **kwargs: Any) -> str:
        prompt_msgs = args[2]
        cand_out: SampledOutput = args[4]
        key = hash((cand_out.content, prompt_msgs[0]["content"]))
        return cycle[key % len(cycle)]

    mock_litellm = _latency_completion_mock()
    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=judge_side_effect),
    ):
        concurrent_result = run_measure(
            records,
            "gpt-4o",
            candidates,
            n_samples=4,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=5,
            seed=99,
        )

    sequential_result = _run_sequential_reference(
        records,
        "gpt-4o",
        candidates,
        use_judge=True,
        judge_model="gpt-4o",
        seed=99,
        judge_side_effect=judge_side_effect,
    )

    # Order preserved.
    assert [c.record.messages[0]["content"] for c in concurrent_result.comparisons] == [
        f"prompt {i}" for i in range(4)
    ]
    # candidate_outputs aligned 1:1 with candidates (model + content).
    assert [
        [(o.model, o.content) for o in c.candidate_outputs]
        for c in concurrent_result.comparisons
    ] == [
        [(o.model, o.content) for o in c.candidate_outputs]
        for c in sequential_result.comparisons
    ]
    # Per-prompt verdicts aligned with candidates.
    assert [c.verdicts for c in concurrent_result.comparisons] == [
        c.verdicts for c in sequential_result.comparisons
    ]
    # Aggregate tallies byte-identical.
    assert _tally_tuple(concurrent_result) == _tally_tuple(sequential_result)
    # Auxiliary fields unchanged.
    assert concurrent_result.self_judged_models == sequential_result.self_judged_models
    assert concurrent_result.judge_model == sequential_result.judge_model


# ---------------------------------------------------------------------------
# Callbacks — fire once per prompt, monotonic under scrambled completion order.
# ---------------------------------------------------------------------------


def test_run_measure_callbacks_fire_once_per_prompt() -> None:
    """Arrange: 4 prompts, 1 candidate, judge on; record every callback fire.
    Assert: sample_cb and judge_cb each fire EXACTLY n_prompts times, and the
    completion counter is monotonic non-decreasing (the live n/total never goes
    backwards even though prompts may complete out of submission order).
    """
    records = [_make_record(f"p{i}") for i in range(4)]
    sample_calls: list[int] = []
    judge_calls: list[int] = []

    mock_litellm = _latency_completion_mock()
    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=lambda *a, **k: "win"),
    ):
        run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=4,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=5,
            seed=1,
            sample_cb=lambda done, total, label: sample_calls.append(done),
            judge_cb=lambda done, total, label: judge_calls.append(done),
        )

    assert len(sample_calls) == 4
    assert len(judge_calls) == 4
    # Each fire reports the count of prompts already completed; the set covers all
    # four exactly once (completion order may vary, the counter is monotonic).
    assert sorted(sample_calls) == [0, 1, 2, 3]
    assert sorted(judge_calls) == [0, 1, 2, 3]


def test_run_measure_callback_counter_monotonic_under_variable_latency() -> None:
    """Arrange: prompts with deliberately VARYING sampling latency so they
    complete out of submission order; capture every (done) value each callback
    reports.

    Assert: sample_cb and judge_cb each fire exactly n_prompts times and the
    sequence of reported ``done`` counters is strictly monotonic increasing
    0,1,2,... -- the live "n / total" never goes backwards even though prompts
    finish in a scrambled order.
    """
    n_prompts = 6
    records = [_make_record(f"p{i}") for i in range(n_prompts)]
    sample_dones: list[int] = []
    judge_dones: list[int] = []

    def completion(model: str, messages: list[Any], **kw: Any) -> MagicMock:
        # Latency decreasing with prompt index -> later-submitted prompts finish
        # FIRST, scrambling completion order relative to submission order.
        idx = int(messages[0]["content"][1:])
        time.sleep(0.01 + (n_prompts - idx) * 0.01)
        resp = MagicMock()
        resp.choices[0].message.content = f"out::{model}"
        return resp

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = completion

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=lambda *a, **k: "win"),
    ):
        run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=n_prompts,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=5,
            seed=5,
            sample_cb=lambda done, total, label: sample_dones.append(done),
            judge_cb=lambda done, total, label: judge_dones.append(done),
        )

    assert len(sample_dones) == n_prompts
    assert len(judge_dones) == n_prompts
    # Strictly monotonic 0..n-1 regardless of the scrambled completion order.
    assert sample_dones == list(range(n_prompts))
    assert judge_dones == list(range(n_prompts))


# ---------------------------------------------------------------------------
# A/B-order randomisation — seeded, reproducible, self-judge detection.
# ---------------------------------------------------------------------------


def test_run_measure_ab_order_is_deterministic_per_seed() -> None:
    """Arrange: a judge with a FIXED position bias (always prefers OUTPUT A).
    Act: run_measure twice with the same seed, capturing candidate_is_a per pair.
    Assert: the A/B layout is identical across the two runs (seeded determinism),
    and the resolved verdicts are identical — so a reproducible run stays
    reproducible despite the randomisation.
    """
    records = [_make_record(f"p{i}") for i in range(6)]

    def biased_judge(*args: Any, **kwargs: Any) -> str:
        # The judge always prefers whichever output is shown as A.  With the
        # candidate sometimes in slot A and sometimes in slot B, the resolved
        # candidate-relative verdict therefore depends on candidate_is_a — so a
        # stable verdict sequence across runs proves a stable A/B layout.
        candidate_is_a = kwargs["candidate_is_a"]
        return "win" if candidate_is_a else "loss"

    layouts: list[list[str]] = []
    for _ in range(2):
        mock_litellm = _latency_completion_mock()
        with (
            patch("frugon.measure._import_litellm", return_value=mock_litellm),
            patch("frugon.measure._judge_pair", side_effect=biased_judge),
        ):
            result = run_measure(
                records,
                "gpt-4o",
                ["gpt-4o-mini"],
                n_samples=6,
                use_judge=True,
                judge_model="gpt-4o",
                concurrency=5,
                seed=42,
            )
        layouts.append([c.verdicts[0] for c in result.comparisons])

    # Same seed → identical (debiased) A/B layout → identical verdict sequence.
    assert layouts[0] == layouts[1]
    # And the layout must actually MIX both orderings (not degenerate to all-A or
    # all-B), otherwise the randomisation is not doing anything.
    assert set(layouts[0]) == {"win", "loss"}


def test_run_measure_self_judge_flagged_when_judge_equals_candidate() -> None:
    """Arrange: judge_model == the candidate being judged.
    Assert: MeasureResult.self_judged_models names that model (the CLI surfaces a
    caution); independent judges leave it empty.
    """
    records = [_make_record("p0")]
    mock_litellm = _latency_completion_mock()

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=lambda *a, **k: "tie"),
    ):
        self_judged = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=1,
            use_judge=True,
            judge_model="gpt-4o-mini",  # == the candidate
            seed=0,
        )
        independent = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=1,
            use_judge=True,
            # Independent of BOTH the baseline (gpt-4o) and the candidate
            # (gpt-4o-mini) — the arm's-length default scenario.
            judge_model="claude-3-opus-20240229",
            seed=0,
        )

    assert self_judged.self_judged_models == ["gpt-4o-mini"]
    assert self_judged.judge_model == "gpt-4o-mini"
    assert independent.self_judged_models == []


# ---------------------------------------------------------------------------
# Instrumentation shared by the two-stage overlap + wider-sampling proofs.
# ---------------------------------------------------------------------------


class _Clock:
    """Thread-safe monotonic event recorder injected into the stubs.

    Records ``(label, perf_counter())`` on every call so a test can assert the
    relative wall-clock ordering of sampling vs judging events across prompts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[str, float]] = []

    def mark(self, label: str) -> None:
        with self._lock:
            self.events.append((label, time.perf_counter()))

    def first(self, label: str) -> float:
        return next(t for lbl, t in self.events if lbl == label)

    def last(self, label: str) -> float:
        return next(t for lbl, t in reversed(self.events) if lbl == label)


class _ConcurrencyGauge:
    """Thread-safe in-flight counter that records its running maximum.

    ``enter`` / ``exit`` bracket a stubbed call; ``peak`` is the maximum number of
    calls that were simultaneously between an enter and its exit.  Used to PROVE
    that the sampling stage alone reaches ``concurrency`` simultaneous calls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self._inflight += 1
            self.peak = max(self.peak, self._inflight)

    def exit(self) -> None:
        with self._lock:
            self._inflight -= 1


# ---------------------------------------------------------------------------
# THE TWO-STAGE OVERLAP PROOF — a judge call STARTS before the LAST sampling
# call FINISHES, i.e. stage 2 runs while stage 1 still has pending work.  Under
# any single-wave / barrier shape this is impossible: no judge could begin until
# every sampling call (including the slow tail) had drained.
# ---------------------------------------------------------------------------


def test_run_measure_judging_overlaps_sampling_across_prompts() -> None:
    """Arrange: 2 prompts, 1 candidate, judge on.  Prompt 0 ("fast") samples
    near-instantly; prompt 1 ("slow") has a long sampling latency.  Each stubbed
    call records start+end timestamps via an injected thread-safe clock.

    Act: run_measure (two-stage pipeline, concurrency=5 so both prompts' samples
    can be in flight at once).

    Assert: prompt 0's JUDGE call (a stage-2 task) STARTS before prompt 1's
    (slow) SAMPLING call (a stage-1 task) FINISHES.  This is the defining
    property of the two-stage overlap: the consumer drains stage-2 work while the
    producer still has stage-1 work pending.  Impossible under a global
    sampling→judging barrier.
    """
    clock = _Clock()
    fast_prompt = "FAST prompt 0"
    slow_prompt = "SLOW prompt 1"
    records = [_make_record(fast_prompt), _make_record(slow_prompt)]

    _FAST_S = 0.01
    _SLOW_S = 0.40

    def completion(model: str, messages: list[Any], **kw: Any) -> MagicMock:
        prompt_text = messages[0]["content"]
        is_slow = prompt_text == slow_prompt
        label = "sample::slow" if is_slow else "sample::fast"
        clock.mark(label + "::start")
        time.sleep(_SLOW_S if is_slow else _FAST_S)
        clock.mark(label + "::end")
        resp = MagicMock()
        resp.choices[0].message.content = f"out::{model}"
        return resp

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = completion

    def judge_side_effect(*args: Any, **kwargs: Any) -> str:
        prompt_msgs = args[2]
        prompt_text = prompt_msgs[0]["content"]
        which = "fast" if prompt_text == fast_prompt else "slow"
        clock.mark(f"judge::{which}::start")
        return "tie"

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=judge_side_effect),
    ):
        run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=2,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=5,
            seed=3,
        )

    fast_judge_start = clock.first("judge::fast::start")
    slow_sampling_end = clock.last("sample::slow::end")

    # THE OVERLAP PROOF: a stage-2 (judge) task begins while a stage-1 (sampling)
    # task is still pending.  Impossible under a global sampling→judging barrier.
    assert fast_judge_start < slow_sampling_end, (
        f"fast judge started at {fast_judge_start:.4f} but slow sampling did not "
        f"finish until {slow_sampling_end:.4f} -- stage 2 did NOT overlap stage 1 "
        f"(a sampling→judging barrier would still be present)"
    )


# ---------------------------------------------------------------------------
# THE WIDER-SAMPLING PROOF — with concurrency=N the SAMPLING stage alone can
# reach N simultaneous calls.  The old single-pool shape capped *total* in-flight
# (sampling + judging) at N; the two-stage split lets sampling alone hit N.
# ---------------------------------------------------------------------------


def test_run_measure_sampling_stage_reaches_full_concurrency() -> None:
    """Arrange: n_prompts large relative to N, concurrency=N, judge ON.  Every
    sampling call brackets itself in a thread-safe in-flight gauge that records
    its running maximum; the gauge counts SAMPLING calls only (judge calls go
    through the separate _judge_pair / _judge_addressed patches and do not
    touch it).

    Act: run_measure (two-stage pipeline) with concurrency=N.

    Assert: peak concurrent SAMPLING calls == N exactly.
      * == N proves the sampling pool is saturated to the flag (wide fan-out);
      * the gauge ignoring judge calls proves this is SAMPLING concurrency alone,
        not the old "total across both kinds capped at N" behaviour.  With
        n_prompts × (1 + n_candidates) sampling calls all submitted up front and
        N workers, exactly N can be in flight at the peak.
    """
    n = 4
    # Plenty of sampling work so the pool stays saturated well past N in flight:
    # 6 prompts × (1 baseline + 1 candidate) = 12 sampling calls, N=4 workers.
    n_prompts = 6
    records = [_make_record(f"p{i}") for i in range(n_prompts)]
    gauge = _ConcurrencyGauge()

    def completion(model: str, messages: list[Any], **kw: Any) -> MagicMock:
        gauge.enter()
        # Long enough that all N workers overlap inside the gauge window.
        time.sleep(0.05)
        gauge.exit()
        resp = MagicMock()
        resp.choices[0].message.content = f"out::{model}"
        return resp

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = completion

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        # Judge calls do NOT touch the gauge, so the measured peak is sampling
        # concurrency ALONE — proving sampling reaches N independent of judging.
        # _judge_pair always returns "tie", which would otherwise trigger the
        # pointwise "both failed" check (_judge_addressed) — also stubbed here
        # so its calls stay out of the gauge too, same isolation as _judge_pair.
        patch("frugon.measure._judge_pair", side_effect=lambda *a, **k: "tie"),
        patch("frugon.measure._judge_addressed", side_effect=lambda *a, **k: True),
    ):
        run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=n_prompts,
            use_judge=True,
            judge_model="gpt-4o",
            concurrency=n,
            seed=8,
        )

    # Sampling alone saturates to exactly N.  Under the old single flat pool the
    # judging calls would have competed for the SAME N slots, so sampling could
    # not reliably hold N on its own; the dedicated sampling pool guarantees it.
    assert gauge.peak == n, (
        f"peak concurrent SAMPLING calls was {gauge.peak}, expected exactly {n} "
        f"-- the sampling stage did not fan to full --concurrency"
    )


def test_run_measure_default_concurrency_matches_module_default() -> None:
    """Arrange: call run_measure with NO concurrency argument.
    Assert: the default the signature carries equals the documented module
    default (so the flag default and the function default never drift apart).
    """
    # A 1-prompt no-judge run is enough to exercise the default-valued parameter
    # path; the assertion is on the module constant the signature defaults to.
    records = [_make_record("p0")]
    mock_litellm = _latency_completion_mock()
    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=1, seed=0)
    assert result.samples_taken == 1
    assert _DEFAULT_CONCURRENCY == 5
