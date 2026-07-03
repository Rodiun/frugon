"""frugon split-routing engine — per-call difficulty classification + blended cost.

Instead of recommending a single wholesale model swap, frugon classifies each
logged call's difficulty *offline* and routes the easy calls to a cheaper
candidate while keeping the hard calls on the premium baseline.  The result is a
routed/kept split with a blended cost and an honest blended saving — the shape a
prospect sees on the landing page (``N routed -> mini, M kept -> premium, -X%``).

Method (RouteLLM-style, but local and heuristic)
--------------------------------------------------------
RouteLLM (https://github.com/lm-sys/RouteLLM, lmsys) trains routers on human
preference data to decide *when a strong model is actually needed*; easy queries
go to a cheap model, hard queries to a strong one.  frugon approximates that
decision **entirely offline** from the shape of each logged call — prompt length,
completion length, and conversation depth — so the recommendation can be produced
on the user's own machine with **zero LLM calls and zero network access**
(privacy invariant).  This is a transparent heuristic, NOT a trained
router: the per-call difficulty *gate* is the quality protection — only calls that
score below the easy threshold are proposed for the cheaper model, and the harder
calls are always kept on the baseline.  The recommendation is a list-price
estimate whose quality is unverified; ``--measure`` samples real outputs before a
user switches (honest-savings policy).

This module makes no network calls and runs no models.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from frugon.pricing import ModelPrice, get_model_price
from frugon.quality import is_unrated as _is_unrated

if TYPE_CHECKING:  # pragma: no cover — typing only, avoids a cost<->routing cycle
    from frugon.cost import CallCost, LogRecord

# ---------------------------------------------------------------------------
# Difficulty heuristic constants
# ---------------------------------------------------------------------------
# A call's difficulty is a weighted blend of three offline signals, each
# normalised to [0, 1] by a saturation scale and then combined with weights that
# sum to exactly 1.0 so the score is itself bounded to [0, 1]:
#   * prompt length      — longer prompts carry more context / harder asks
#   * completion length   — longer generations imply more demanding work
#   * conversation depth  — multi-turn exchanges are harder to route blindly
# The scales are the token/turn counts at which each signal saturates; they are
# deliberately generous so that short classification / Q&A calls score well below
# the easy threshold and long document-analysis / reasoning calls score above it.

_ONE = Decimal("1")
_ZERO = Decimal("0")

_PROMPT_SCALE = Decimal("1000")
_COMPLETION_SCALE = Decimal("600")
_TURNS_SCALE = Decimal("6")

_W_PROMPT = Decimal("0.50")
_W_COMPLETION = Decimal("0.35")
_W_TURNS = Decimal("0.15")

# A call is "easy" (eligible to route to the cheaper candidate) when its
# difficulty score is strictly below this threshold.  A score exactly equal to
# the threshold is treated as hard (kept on baseline) — the conservative choice.
EASY_THRESHOLD = Decimal("0.35")


# ---------------------------------------------------------------------------
# Split result type
# ---------------------------------------------------------------------------


@dataclass
class SplitRouting:
    """A per-call routed/kept split recommendation for one baseline model.

    The split applies to the dominant (highest-cost) baseline model's own calls:
    the easy subset is proposed for ``candidate_model`` (the cheaper "mini"
    target) while the hard subset is kept on ``baseline_model`` (premium).
    ``blended_cost`` is always <= ``baseline_cost`` — routing never costs more
    than staying on the baseline (the easy calls only ever move to a strictly
    cheaper model, and the hard calls are unchanged).
    """

    baseline_model: str
    candidate_model: str
    routed_count: int  # easy calls proposed for candidate_model
    kept_count: int  # hard calls kept on baseline_model
    routed_cost: Decimal  # routed (easy) calls priced at candidate_model
    kept_cost: Decimal  # kept (hard) calls priced at baseline_model (unchanged)
    baseline_cost: Decimal  # all split calls priced at baseline_model (current)
    blended_cost: Decimal  # routed_cost + kept_cost (the recommendation)
    easy_threshold: Decimal
    # Monthly projections — None unless a window/span was supplied to extrapolate.
    monthly_baseline: Decimal | None = None
    monthly_blended: Decimal | None = None

    @property
    def total_count(self) -> int:
        """Total split calls (routed + kept)."""
        return self.routed_count + self.kept_count

    @property
    def saving_pct(self) -> Decimal | None:
        """Blended saving as a percentage, or None when baseline_cost is zero."""
        if self.baseline_cost == _ZERO:
            return None
        return (self.baseline_cost - self.blended_cost) / self.baseline_cost * Decimal("100")


# ---------------------------------------------------------------------------
# Difficulty classification
# ---------------------------------------------------------------------------


def difficulty_score(record: LogRecord) -> Decimal:
    """Return *record*'s offline difficulty score in the closed range [0, 1].

    Higher means harder.  Pure arithmetic over the logged call's token counts and
    message depth — no model, no network.  Monotonic non-decreasing in each of
    prompt tokens, completion tokens, and conversation depth.
    """
    prompt_tokens = max(Decimal(record.prompt_tokens), _ZERO)
    completion_tokens = max(Decimal(record.completion_tokens), _ZERO)
    turns = Decimal(max(len(record.messages) - 1, 0))

    s_prompt = min(prompt_tokens / _PROMPT_SCALE, _ONE)
    s_completion = min(completion_tokens / _COMPLETION_SCALE, _ONE)
    s_turns = min(turns / _TURNS_SCALE, _ONE)

    return _W_PROMPT * s_prompt + _W_COMPLETION * s_completion + _W_TURNS * s_turns


def is_easy(record: LogRecord, threshold: Decimal = EASY_THRESHOLD) -> bool:
    """Return True when *record* is easy enough to route to the cheaper candidate.

    A call is easy iff its difficulty score is strictly below *threshold*.  A
    score exactly equal to the threshold is hard (kept on baseline).
    """
    return difficulty_score(record) < threshold


# ---------------------------------------------------------------------------
# Easy-call candidate selection
# ---------------------------------------------------------------------------


def select_easy_target(baseline_model: str, pool: list[str]) -> str | None:
    """Pick the cheapest rated, priced model in *pool* to route easy calls to.

    The easy-call target is the cheapest candidate that is (a) priced, (b) rated
    in the quality table (so the user knows roughly where it sits — never an
    unknown-quality model), and (c) strictly cheaper than *baseline_model* on a
    blended per-token basis.  Ties break by model name for determinism.

    Unlike the wholesale recommendation in cost.py (which caps the quality-tier
    drop because *every* call would move), the split intentionally allows a
    larger tier gap here: only the calls the difficulty gate marks *easy* are
    routed to this model, and the hard calls stay on the baseline.  That per-call
    gate — not a blanket tier cap — is the quality protection (RouteLLM thesis).

    Returns None when no rated, priced, strictly-cheaper candidate exists.
    """
    baseline_price = get_model_price(baseline_model)
    if baseline_price is None:
        return None
    baseline_blended = _blended_per_token(baseline_price)

    best_model: str | None = None
    best_blended: Decimal | None = None
    for candidate in pool:
        if candidate == baseline_model:
            continue
        if _is_unrated(candidate):
            continue
        price = get_model_price(candidate)
        if price is None:
            continue
        cand_blended = _blended_per_token(price)
        if cand_blended >= baseline_blended:
            continue
        if (
            best_blended is None
            or cand_blended < best_blended
            or (cand_blended == best_blended and candidate < (best_model or ""))
        ):
            best_model = candidate
            best_blended = cand_blended

    return best_model


def _blended_per_token(price: ModelPrice) -> Decimal:
    """Blended (50/50 prompt/completion) per-token price used for comparison."""
    return (price.input_cost_per_token + price.output_cost_per_token) / Decimal("2")


# ---------------------------------------------------------------------------
# Split computation
# ---------------------------------------------------------------------------


def _project_monthly(
    value: Decimal,
    window_days: int | None,
    observed_span_days: float | None,
) -> Decimal | None:
    """Project *value* to a 30-day month using the same disclosure rules as cost.py.

    Prefers an explicit --window; falls back to the observed timestamp span;
    returns None when neither is available (no projection is ever invented).
    """
    if window_days is not None and window_days > 0:
        return value * Decimal("30") / Decimal(window_days)
    if observed_span_days is not None and observed_span_days > 0:
        return value * Decimal("30") / Decimal(str(observed_span_days))
    return None


def compute_split(
    *,
    baseline_model: str,
    candidate_model: str,
    baseline_call_costs: list[CallCost],
    candidate_price: ModelPrice,
    threshold: Decimal = EASY_THRESHOLD,
    window_days: int | None = None,
    observed_span_days: float | None = None,
) -> SplitRouting:
    """Build a SplitRouting over *baseline_call_costs* (the baseline model's calls).

    Each call is classified easy/hard offline.  Easy calls are repriced at
    *candidate_price*; hard calls keep their existing baseline cost.  The blended
    cost is routed_cost + kept_cost.  All arithmetic is exact Decimal.
    """
    partition = partition_by_difficulty(baseline_call_costs, threshold)
    return _split_from_partition(
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        partition=partition,
        candidate_price=candidate_price,
        threshold=threshold,
        window_days=window_days,
        observed_span_days=observed_span_days,
    )


# ---------------------------------------------------------------------------
# Candidate-independent partition (the perf fast path for multi-candidate runs)
# ---------------------------------------------------------------------------


@dataclass
class DifficultyPartition:
    """The easy/hard split of a baseline model's calls, computed ONCE.

    The difficulty classification (:func:`is_easy`) depends only on a call's
    own shape (prompt/completion length, turn depth) and the *threshold* — it
    is entirely independent of which candidate model might receive the easy
    calls.  When a caller is evaluating MANY candidates against the same
    baseline call set (the "Candidates considered" block: up to 23 built-in
    candidates, or an explicit ``--candidates`` list), re-running
    :func:`is_easy` once per candidate repeats the identical Decimal-heavy
    computation for no reason — on the bundled 56,100-record demo this alone
    cost ~4s of wall time across the 22-candidate default pool.

    This partition is computed once and reused for every candidate: each
    candidate's :class:`SplitRouting` is then derived by pricing ONLY the
    already-classified easy subset at that candidate's rate (an O(easy_count)
    pass per candidate, with no difficulty re-classification) and reusing the
    precomputed hard-subset aggregate (an O(1) lookup).  The final per-call
    "route only if actually cheaper for THIS call" gate (§6 never-inflate)
    still runs per (candidate, easy-call) pair — that check is genuinely
    candidate-dependent (a call's own prompt/completion mix can favour a
    cheap-input/expensive-output candidate differently than a cheap-output
    one) and cannot be reduced to aggregates without changing which calls get
    routed, so ranking/selection semantics are byte-for-byte unchanged from
    the pre-optimization implementation.
    """

    threshold: Decimal
    easy_calls: list[CallCost]
    hard_count: int
    hard_cost: Decimal  # sum of total_cost over hard (kept) calls — candidate-independent
    baseline_cost: Decimal  # sum of total_cost over ALL calls (easy + hard)


def partition_by_difficulty(
    baseline_call_costs: list[CallCost],
    threshold: Decimal = EASY_THRESHOLD,
) -> DifficultyPartition:
    """Classify *baseline_call_costs* into easy/hard exactly once.

    Equivalent to calling :func:`is_easy` on every record in
    ``baseline_call_costs`` — same threshold, same per-record semantics — but
    done a single time regardless of how many candidates are later evaluated
    against the result via :func:`compute_split_from_partition`.
    """
    hard_count = 0
    hard_cost = _ZERO
    baseline_cost = _ZERO
    easy_calls: list[CallCost] = []

    for cc in baseline_call_costs:
        baseline_cost += cc.total_cost
        if is_easy(cc.record, threshold):
            easy_calls.append(cc)
        else:
            hard_count += 1
            hard_cost += cc.total_cost

    return DifficultyPartition(
        threshold=threshold,
        easy_calls=easy_calls,
        hard_count=hard_count,
        hard_cost=hard_cost,
        baseline_cost=baseline_cost,
    )


def compute_split_from_partition(
    *,
    baseline_model: str,
    candidate_model: str,
    partition: DifficultyPartition,
    candidate_price: ModelPrice,
    window_days: int | None = None,
    observed_span_days: float | None = None,
) -> SplitRouting:
    """Build a SplitRouting for *candidate_model* from a precomputed partition.

    Reuses the candidate-independent easy/hard classification in *partition*
    (see :func:`partition_by_difficulty`) instead of re-running :func:`is_easy`
    for every call.  Only the easy subset is repriced at *candidate_price* —
    an O(easy_count) pass, versus the O(all_count) classification the naive
    per-candidate loop repeated every time — and the per-call "route only if
    actually cheaper for THIS call" gate is preserved unchanged (§6
    never-inflate), so the resulting :class:`SplitRouting` is numerically
    IDENTICAL, call for call, to what :func:`compute_split` would return for
    the same baseline call set and candidate — proven by
    ``test_partition_equivalence`` (exact match on the bundled demo and on a
    randomized property fixture) before this path is used anywhere.
    """
    return _split_from_partition(
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        partition=partition,
        candidate_price=candidate_price,
        threshold=partition.threshold,
        window_days=window_days,
        observed_span_days=observed_span_days,
    )


def _split_from_partition(
    *,
    baseline_model: str,
    candidate_model: str,
    partition: DifficultyPartition,
    candidate_price: ModelPrice,
    threshold: Decimal,
    window_days: int | None,
    observed_span_days: float | None,
) -> SplitRouting:
    """Shared arithmetic core for both :func:`compute_split` and the
    partition-reusing fast path — one function, so the two entry points can
    never drift onto different arithmetic."""
    routed_count = 0
    routed_cost = _ZERO
    # Easy calls this candidate does NOT actually beat on a per-call basis stay
    # on the baseline — accumulated alongside the precomputed hard bucket so
    # every easy call is accounted exactly once (routed XOR kept).
    not_routed_easy_count = 0
    not_routed_easy_cost = _ZERO

    for cc in partition.easy_calls:
        candidate_call_cost = candidate_price.input_cost_per_token * Decimal(
            cc.record.prompt_tokens
        ) + candidate_price.output_cost_per_token * Decimal(cc.record.completion_tokens)
        # Route an easy call ONLY when the candidate is actually cheaper for
        # THIS call.  select_easy_target compares on a blended 50/50 per-token
        # average, which does not guarantee the candidate is cheaper on the
        # output axis — a completion-heavy easy call could otherwise reprice
        # *higher* than baseline.  This per-call check makes blended_cost <=
        # baseline_cost hold unconditionally: routing never inflates a call
        # (§6 never-inflate), even with an adversarial user --candidates pool.
        if candidate_call_cost < cc.total_cost:
            routed_count += 1
            routed_cost += candidate_call_cost
        else:
            not_routed_easy_count += 1
            not_routed_easy_cost += cc.total_cost

    kept_count = partition.hard_count + not_routed_easy_count
    kept_cost = partition.hard_cost + not_routed_easy_cost
    baseline_cost = partition.baseline_cost
    blended_cost = routed_cost + kept_cost

    return SplitRouting(
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        routed_count=routed_count,
        kept_count=kept_count,
        routed_cost=routed_cost,
        kept_cost=kept_cost,
        baseline_cost=baseline_cost,
        blended_cost=blended_cost,
        easy_threshold=threshold,
        monthly_baseline=_project_monthly(baseline_cost, window_days, observed_span_days),
        monthly_blended=_project_monthly(blended_cost, window_days, observed_span_days),
    )
