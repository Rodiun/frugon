"""frugon cost engine.

Responsibilities:
- Parse JSONL log records (OpenAI-compatible request/response format).
- Count tokens: prefer the ``usage`` block when present; fall back to the
  tokencost tokenizer when absent.
- Apply pricing (see pricing.py for precedence rules).
- Aggregate totals and compute routing projections.
- Monthly extrapolation with honest disclosure:
  * --window DAYS provided  → project monthly using that window, disclose it.
  * timestamps present      → compute REAL span, disclose it.
  * no timestamps, no flag  → DO NOT extrapolate; report observed total only.
  Never multiply by an assumed span. total_cost is always the raw observed total.

Privacy: analysis is 100% local. No network calls are made by this module.
"""

from __future__ import annotations

import functools
import gzip
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import tokencost as _tc  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError("tokencost is required. pip install tokencost") from exc

from frugon.pricing import ModelPrice, get_model_price
from frugon.pricing import pinned_pricing_identity as _pinned_pricing_identity

if TYPE_CHECKING:  # pragma: no cover — typing only, avoids a cost<->routing cycle
    from frugon.routing import SplitRouting
from frugon.quality import UNRATED_TIER as _UNRATED_TIER
from frugon.quality import get_model_tier as _quality_get_tier
from frugon.quality import is_unrated as _quality_is_unrated
from frugon.quality import load_quality_table as _load_quality_table
from frugon.quality import tier_name as _quality_tier_name

# ---------------------------------------------------------------------------
# Log record type
# ---------------------------------------------------------------------------


@dataclass
class LogRecord:
    """A single LLM call parsed from a JSONL log line."""

    model: str
    messages: list[dict[str, str]]
    completion_text: str
    prompt_tokens: int
    completion_tokens: int
    # ISO 8601 timestamp from the log, or None if absent
    timestamp: str | None
    # "usage_block" when token counts came from an explicit usage field;
    # "counted" when derived from the tokencost tokenizer;
    # "approximated" when the tokenizer failed and a rough character heuristic
    # (4 chars ≈ 1 token) was used — a materially less accurate count.
    token_source: str = "usage_block"


# ---------------------------------------------------------------------------
# Per-record cost result
# ---------------------------------------------------------------------------


@dataclass
class CallCost:
    """Cost result for one log record."""

    record: LogRecord
    price: ModelPrice | None
    prompt_cost: Decimal
    completion_cost: Decimal
    total_cost: Decimal
    token_source: str  # "usage_block" or "counted"


# ---------------------------------------------------------------------------
# Aggregate analysis result
# ---------------------------------------------------------------------------


@dataclass
class CandidateProjection:
    """One candidate's projection — shown in the "Candidates considered" block.

    The cost projection's headline picks the SINGLE cheapest candidate that beats
    the baseline. When the user passes multiple ``--candidates``, every one of
    them is surfaced here (recommended, considered, more-expensive, or unpriced),
    with each one's projected monthly cost beside its status — the others were
    considered too, they just lost the cheapest-wins tiebreak. On the DEFAULT
    pool (no explicit ``--candidates``), the same block fires whenever a split
    recommendation exists and the pool has more than one model: real users — and
    the un-pinned demo — should SEE what was considered, not just the winner, so
    the list is capped to the recommended candidate plus the next-4-cheapest
    candidates that also beat the baseline (5 rows max), never the full built-in
    roster. This is rendering metadata: the underlying headline projection math
    is identical to the single-cheapest-candidate path.

    *monthly_cost*, *monthly_saving*, *saving_pct* are None when the candidate
    has no entry in the pricing table (status == "unpriced") or when the
    analysis has no monthly projection (no --window and no parseable log
    timestamps) — they describe the projected MONTHLY figure the user sees in
    the panel, so they only fill in when a monthly figure exists.  When the
    monthly projection is unavailable, the OBSERVED figures live in
    *observed_cost*, *observed_saving*, *observed_saving_pct* so the block still
    has numbers to show (saving% over the analyzed total).

    *status* is the rendering tag the report layer keys off:

      - "recommended"  — cheapest split-routing projection among candidates
                         that beat the baseline; monthly figures match headline.
      - "considered"   — split-routing beats baseline but is not the cheapest.
      - "more_expensive" — split-routing does not improve on the baseline.
      - "unpriced"     — no entry in the pricing table; nothing to project.

    *monthly_cost* and *saving_pct* come from per-candidate split projections
    (route easy baseline calls to this candidate, keep hard calls on baseline)
    — directly comparable to the headline New-spend / saving%.

    *tier_label* is the model's quality tier NAME ("Elite", "Strong",
    "Capable", "Efficient") or ``"Unrated"`` when the model has no entry in the
    quality table — printed as its own column so the display-precision quality
    tie-break (:func:`_select_cheapest_eligible`) is self-evident from the
    table alone: a user can see that the recommended row's tier is at least as
    good as every row it ties with on saving%, without having to trust the
    caption's word for it.
    """

    model: str
    status: str
    monthly_cost: Decimal | None = None
    monthly_saving: Decimal | None = None
    saving_pct: Decimal | None = None
    observed_cost: Decimal | None = None
    observed_saving: Decimal | None = None
    observed_saving_pct: Decimal | None = None
    tier_label: str = "Unrated"


@dataclass
class AnalysisResult:
    """Aggregated cost analysis across all log records."""

    total_calls: int
    priced_calls: int
    unpriced_calls: int
    total_cost: Decimal
    # Keyed by model name
    cost_by_model: dict[str, Decimal] = field(default_factory=dict)
    calls_by_model: dict[str, int] = field(default_factory=dict)
    # Routing projection: cheapest candidate that handles each call
    projected_cost: Decimal = Decimal("0")
    candidate_model: str | None = None
    # Observed window info
    observed_span_days: float | None = None  # None = no timestamps
    # Earliest and latest parseable log timestamps, as ISO dates (YYYY-MM-DD).
    # Populated from the same min/max scan _compute_span_days uses; both
    # None when fewer than 2 timestamps parse (so there is no span to show).
    observed_span_start: str | None = None
    observed_span_end: str | None = None
    window_days: int | None = None  # --window flag value
    # pricing.json last_synced date (may be None)
    pricing_json_last_synced: str | None = None
    # quality.json last_synced date (may be None).  Plumbed from the quality tier
    # table the same way pricing_json_last_synced is plumbed from pricing.json, so
    # the report can disclose how fresh the quality tiers behind the "within
    # tolerance" recommendation are — those tiers are decision-relevant.
    quality_json_last_synced: str | None = None
    # Routing honesty metadata
    # Number of models in the built-in routing candidate pool.  Only meaningful
    # when no explicit --candidates were supplied (used_default_pool=True).
    candidate_pool_size: int = 0
    # True when the recommendation came from the built-in pool (no --candidates).
    used_default_pool: bool = False
    # Quality tier drop from baseline to candidate (None when either model is
    # unrated, or when no candidate was selected).  Only non-None when both
    # models have known tiers.  A value >= 2 means the headline saving assumes
    # a large quality step-down.
    tier_drop: int | None = None
    # True when the dominant baseline model is not in the quality tier map.
    baseline_is_unrated: bool = False
    # True when the chosen candidate model is not in the quality tier map.
    candidate_is_unrated: bool = False
    # Unrated candidates from an EXPLICIT --candidates pool that beat the baseline
    # on their full-dataset split New-spend but were EXCLUDED from the recommended
    # route solely because they have no known quality tier (a rated candidate was
    # recommended instead).  Order: the model order the user passed, de-duplicated.
    # Drives the "could save X%, but it's unrated — excluded from the recommended
    # route until you verify it; run --measure to check" caveat (Change 1b).  This
    # is distinct from candidate_is_unrated (which flags the FALLBACK case where an
    # unrated model IS the recommendation because no rated candidate beat
    # baseline).  Empty for the default-pool path (no --candidates), so the bundled
    # --demo output is unaffected.
    excluded_unrated_models: list[str] = field(default_factory=list)
    # Count of lines that were skipped due to malformed JSON or invalid structure.
    skipped_malformed: int = 0
    # Count of priced calls whose token counts came from the rough character
    # heuristic (token_source="approximated") because the tokenizer failed.
    # Reports disclose this so an estimated figure never looks exact.
    approximated_calls: int = 0
    # Monthly-projected cost for the baseline model (total_cost × 30/window or 30/span).
    # None when no window/span is available to extrapolate.
    # total_cost is always the raw observed total — never multiplied.
    monthly_cost: Decimal | None = None
    # Monthly-projected cost for the candidate model. None when monthly_cost is None
    # or when no candidate was found.
    monthly_projected: Decimal | None = None
    # Per-call split-routing recommendation (route easy calls to a cheaper
    # candidate, keep hard calls on the baseline).  None when split routing is
    # disabled (--wholesale) or no cheaper rated candidate exists.  This is the
    # headline recommendation; the wholesale fields above remain the upper-bound
    # "swap everything" comparison.
    split: SplitRouting | None = None
    # Per-candidate projections — populated when --candidates lists more than one
    # priceable model.  Empty when only one candidate was passed (or none), so the
    # report's "Candidates considered" block fires only on the multi-candidate
    # surface where the honesty is needed.  Order: the model order the user passed
    # on the command line, preserved so the rendered block reads in the user's own
    # sequence (skipping the dominant baseline, which is never a candidate of
    # itself).  Headline projection math is unchanged — each entry is computed
    # alongside the existing cheapest-wins loop, additive only.
    candidate_projections: list[CandidateProjection] = field(default_factory=list)
    # True when the user passed explicit --candidates and NONE of them carry a
    # known list price -- the cost race never ran.  Distinct from
    # ``candidate_model is None`` on its own, which also covers "raced and
    # every candidate lost" (state (a): at least one candidate WAS priced).
    # Always False for the default pool -- every entry in _ROUTING_CANDIDATES
    # is priced (see tests/test_candidate_pool.py for the roster invariant).
    no_priceable_candidates: bool = False
    # The explicit --candidates the user passed that have no known list price,
    # in the order supplied.  Populated alongside no_priceable_candidates so
    # the report can name them without re-deriving pricing lookups.
    unpriced_candidate_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tokenizer memoization
# ---------------------------------------------------------------------------
#
# Tokenizing is pure (same text + model → same count) and, on real assistant
# logs, highly repetitive: prompts are built from a handful of templates reused
# across thousands of calls.  Memoizing by content turns N tokenizations into
# "number of distinct prompts" tokenizations — a large win whenever the log
# reuses templates (the demo fixture, and most production assistant traffic).
# The memo is keyed on the exact ``(model, text)`` / ``(model, messages)`` inputs
# the underlying tokenizer sees, so every cached count is byte-identical to the
# uncached path — only faster.  ``lru_cache`` also serves as the "load the
# encoder once" win: tokencost initialises its tiktoken encoder on first use and
# reuses it, and we now avoid even re-entering it for a prompt already counted.


@functools.lru_cache(maxsize=65536)
def _count_string_tokens_cached(text: str, model: str) -> int | None:
    """Memoized ``tokencost.count_string_tokens``; ``None`` if the tokenizer fails.

    Returning ``None`` (rather than raising) lets callers cache the failure path
    too and apply the identical character-heuristic fallback, so the memoized
    and unmemoized results agree exactly.
    """
    try:
        return int(_tc.count_string_tokens(text, model))  # type: ignore[attr-defined]
    except Exception:
        return None


@functools.lru_cache(maxsize=65536)
def _count_message_tokens_cached(messages_key: tuple[tuple[str, str], ...], model: str) -> int | None:
    """Memoized ``tokencost.count_message_tokens``; ``None`` if the tokenizer fails.

    *messages_key* is the hashable ``((role, content), ...)`` projection of the
    message list — exactly the role/content pairs the tokenizer reads — so two
    requests with identical message content share one tokenization.
    """
    messages = [{"role": role, "content": content} for role, content in messages_key]
    try:
        return int(_tc.count_message_tokens(messages, model))  # type: ignore[attr-defined]
    except Exception:
        return None


def _count_prompt_tokens(messages: list[dict[str, str]], model: str) -> tuple[int, bool]:
    """Count prompt tokens using tokencost, with graceful fallback.

    Returns ``(tokens, approximated)`` where *approximated* is True iff the
    tokenizer failed and a rough character heuristic (4 chars ≈ 1 token) was
    used — a materially-less-accurate count that callers must disclose.

    Counts are memoized by message content (see the cached helpers above), so a
    prompt template reused across many calls is tokenized only once.
    """
    messages_key = tuple((msg.get("role", ""), msg.get("content", "")) for msg in messages)
    counted = _count_message_tokens_cached(messages_key, model)
    if counted is not None:
        return counted, False

    # Fallback: count each content field as a plain string
    total = 0
    approximated = False
    for msg in messages:
        content = msg.get("content", "")
        string_count = _count_string_tokens_cached(content, model)
        if string_count is not None:
            total += string_count
        else:
            # Last resort: rough character approximation (4 chars ≈ 1 token)
            total += max(1, len(content) // 4)
            approximated = True
    return total, approximated


def _count_completion_tokens(text: str, model: str) -> tuple[int, bool]:
    """Count completion tokens using tokencost, with graceful fallback.

    Returns ``(tokens, approximated)`` — see :func:`_count_prompt_tokens`.
    Counts are memoized by ``(text, model)`` so a repeated completion string is
    tokenized only once.
    """
    counted = _count_string_tokens_cached(text, model)
    if counted is not None:
        return counted, False
    return max(1, len(text) // 4), True


# ---------------------------------------------------------------------------
# Safe numeric conversion
# ---------------------------------------------------------------------------


def _safe_int(val: Any) -> int | None:
    """Convert *val* to int, returning None for null/non-numeric values.

    Prevents parse_record from crashing on usage blocks like
    ``{"prompt_tokens": null}`` or ``{"prompt_tokens": "abc"}``.
    """
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def _extract_completion_text(raw: dict[str, Any]) -> str:
    """Extract the assistant completion text from an OpenAI-format response."""
    # OpenAI chat completion response format
    choices = raw.get("choices", [])
    if choices and isinstance(choices, list):
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
            # text completion style
            text = first.get("text", "")
            if isinstance(text, str):
                return text
    return ""


def parse_record(raw: dict[str, Any]) -> LogRecord | None:
    """Parse one JSONL log line into a LogRecord.

    Returns None if the record is malformed or lacks required fields.

    Records with a valid usage block (prompt_tokens + completion_tokens) are
    priced even when no messages are present — the usage block supplies token
    counts directly so the tokenizer fallback is not needed.  Records with
    neither a valid usage block nor messages cannot be tokenized and are
    skipped (return None).

    Malformed token values (null, non-numeric strings) in the usage block are
    handled gracefully via _safe_int: if the usage block is malformed but
    messages are present, the tokenizer fallback is used; if neither is
    available, the record is skipped.
    """
    # Model is required
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        return None

    # Messages from the request body (optional when a valid usage block is present)
    body: dict[str, Any] = raw.get("request", raw)
    messages_raw = body.get("messages", [])
    messages: list[dict[str, str]] = []
    if isinstance(messages_raw, list):
        for m in messages_raw:
            if isinstance(m, dict) and "role" in m and "content" in m:
                messages.append({"role": str(m["role"]), "content": str(m["content"])})

    # Completion text from the response body
    response: dict[str, Any] = raw.get("response", raw)
    completion_text = _extract_completion_text(response)

    # Token counts: prefer explicit usage block; fall back to tokenizer.
    # _safe_int handles null / non-numeric values without crashing.
    usage: dict[str, Any] = raw.get("usage", {})
    token_source: str
    if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
        pt = _safe_int(usage["prompt_tokens"])
        ct = _safe_int(usage["completion_tokens"])
        if pt is not None and ct is not None:
            prompt_tokens = pt
            completion_tokens = ct
            token_source = "usage_block"
        elif messages:
            # Usage block present but values malformed; fall back to tokenizer.
            prompt_tokens, pt_approx = _count_prompt_tokens(messages, model)
            completion_tokens, ct_approx = _count_completion_tokens(completion_text, model)
            token_source = "approximated" if (pt_approx or ct_approx) else "counted"
        else:
            # Malformed usage values and no messages — cannot price.
            return None
    else:
        # No usable usage block: need messages for the tokenizer.
        if not messages:
            return None
        prompt_tokens, pt_approx = _count_prompt_tokens(messages, model)
        completion_tokens, ct_approx = _count_completion_tokens(completion_text, model)
        token_source = "approximated" if (pt_approx or ct_approx) else "counted"

    # Timestamp (optional)
    ts: str | None = None
    raw_ts = raw.get("timestamp")
    if isinstance(raw_ts, str) and raw_ts:
        ts = raw_ts

    return LogRecord(
        model=model,
        messages=messages,
        completion_text=completion_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        timestamp=ts,
        token_source=token_source,
    )


# ---------------------------------------------------------------------------
# Per-call cost computation
# ---------------------------------------------------------------------------


def compute_call_cost(record: LogRecord) -> CallCost:
    """Compute the USD cost for one log record."""
    price = get_model_price(record.model)
    token_source = record.token_source

    if price is None:
        return CallCost(
            record=record,
            price=None,
            prompt_cost=Decimal("0"),
            completion_cost=Decimal("0"),
            total_cost=Decimal("0"),
            token_source=token_source,
        )

    prompt_cost = price.input_cost_per_token * Decimal(record.prompt_tokens)
    completion_cost = price.output_cost_per_token * Decimal(record.completion_tokens)

    return CallCost(
        record=record,
        price=price,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        total_cost=prompt_cost + completion_cost,
        token_source=token_source,
    )


# ---------------------------------------------------------------------------
# Span computation
# ---------------------------------------------------------------------------


def _compute_span_bounds(
    records: list[LogRecord],
) -> tuple[float, str, str] | None:
    """Compute the observed time span from the log timestamps.

    Returns ``(span_days, earliest_iso_date, latest_iso_date)`` where
    ``span_days`` is the fractional-day span between the earliest and latest
    parseable timestamps and the two dates are ``YYYY-MM-DD`` strings.  Returns
    ``None`` when fewer than 2 parseable timestamps are found (no span to
    report).  The earliest/latest are exactly the min/max the span is derived
    from, so the disclosed dates and the disclosed span can never disagree.
    """
    from datetime import datetime

    timestamps: list[datetime] = []
    for rec in records:
        if rec.timestamp:
            # Accept ISO 8601 with or without trailing Z / offset
            ts_str = rec.timestamp.replace("Z", "+00:00")
            try:
                timestamps.append(datetime.fromisoformat(ts_str))
            except ValueError:
                pass

    if len(timestamps) < 2:
        return None

    earliest = min(timestamps)
    latest = max(timestamps)
    span_days = (latest - earliest).total_seconds() / 86400.0
    return span_days, earliest.date().isoformat(), latest.date().isoformat()


def _compute_span_days(records: list[LogRecord]) -> float | None:
    """Compute the observed time span in fractional days.

    Returns None if fewer than 2 parseable timestamps are found.  Thin wrapper
    over :func:`_compute_span_bounds` so existing callers keep the days-only
    contract while the bounds helper additionally exposes the min/max dates.
    """
    bounds = _compute_span_bounds(records)
    return bounds[0] if bounds is not None else None


# ---------------------------------------------------------------------------
# Candidate routing projection
# ---------------------------------------------------------------------------

# Routing candidates in priority order (quality-preserving tiers).
# frugon picks the FIRST candidate in this list that is strictly cheaper
# than the baseline, within the tier constraint.  The list is ordered from
# "minimal quality step-down" to "maximum cost reduction" so the
# recommendation is conservative.  Users can override with --candidates.
#
# 23-model roster (FRG-OSS-034) spanning 11 vendors: OpenAI, Anthropic,
# Google, DeepSeek, Moonshot, xAI, Mistral, Z.ai, MiniMax, Alibaba, Meta (OSS).
# Every entry is priced (src/frugon/data/pricing.json) and rated
# (src/frugon/data/quality.json, some via the effort/date folds in model_id.py)
# — see tests/test_candidate_pool.py for the roster invariant.
_ROUTING_CANDIDATES = [
    # Tier 0 — Elite (minimal quality step-down from typical baselines)
    "gpt-5.5",
    "claude-opus-4-8",
    "gemini-2.5-pro",
    "deepseek-v4-pro",
    "kimi-k2.6",
    # Tier 1 — Strong (meaningful cost reduction, still high quality)
    "o3",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "grok-4",
    "mistral-large-3",
    "glm-4.6",
    "minimax-m3",
    # Tier 2 — Capable (maximum cost reduction)
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4.1-nano",
    "claude-haiku-4-5",
    "grok-3-mini",
    "glm-4.5-air",
    "qwen-max",
    # Reference-host pricing: no first-party per-token price is published for
    # these two OSS Llama-4 checkpoints, so the seed prices them via Groq (a
    # commonly-used inference host) rather than inventing a number (§2a).
    "llama-4-maverick-17b-128e-instruct",
    "llama-4-scout-17b-16e-instruct",
]

# Measure-only candidate pin for ``--demo --measure``.  The demo's RECOMMENDATION
# uses the SAME default _ROUTING_CANDIDATES pool as a real run (demo == production
# — that is the whole point of a demo). This single-model pin exists ONLY so the
# try-out path (`frugon analyze --demo --measure`) needs just an OPENAI_API_KEY —
# no signup, no multi-provider key hunt, to sample one live call and show the
# --measure UX end-to-end. It does NOT affect the recommendation math: cli.py
# passes it to verify_measure_prerequisites/run_measure as the sampled model,
# never as the `candidates=` argument to analyze_records.
_DEMO_MEASURE_CANDIDATE: str = "gpt-4.1-mini"

def _get_model_tier(model: str) -> int:
    """Return the quality tier for *model*, or _UNRATED_TIER if not in the table.

    Delegates to the quality module (LMArena-backed synced table).
    Callers that drive automatic selection MUST check for _UNRATED_TIER and
    exclude those models from auto-recommendation.
    """
    return _quality_get_tier(model)


def _is_unrated(model: str) -> bool:
    """Return True when *model* has no entry in the quality tier table."""
    return _quality_is_unrated(model)


def best_judge_from_log(models: Iterable[str]) -> str | None:
    """Return the highest quality-tier model among *models*, or None if none rated.

    Used to default the ``--judge`` judge to the best model the user ALREADY has
    a key for: the strongest model that actually appears in their own log.  This
    avoids hard-defaulting to an arbitrary external model (e.g. gpt-4.1) the user
    may have no key for.

    Tier semantics (frugon.quality): LOWER integer = BETTER quality (0 = Elite),
    and UNRATED_TIER (-1) is a "no rating" sentinel, NOT a real tier — models
    without a rating are skipped entirely (never chosen as judge).  Ties on tier
    are broken deterministically by model name (ascending) so repeated runs over
    the same log always resolve to the same judge.

    Returns None when *models* is empty or NONE of them carry a known rating —
    the caller then falls back to its own default judge.
    """
    best: str | None = None
    best_tier = _UNRATED_TIER  # placeholder; replaced on first rated model
    for model in models:
        tier = _get_model_tier(model)
        if tier == _UNRATED_TIER:
            continue
        # First rated model seeds the search; thereafter prefer a strictly
        # better (lower) tier, and on an exact tier tie prefer the
        # name-ascending model for deterministic resolution.
        if (
            best is None
            or tier < best_tier
            or (tier == best_tier and model < best)
        ):
            best, best_tier = model, tier
    return best


def _best_candidate(
    baseline_model: str,
    call_costs: list[CallCost],
    max_tier_drop: int = 1,
) -> tuple[str | None, Decimal]:
    """Find the cheapest quality-preserving candidate for *baseline_model*.

    Returns the candidate that is (a) strictly cheaper than the baseline on a
    blended per-token basis, (b) has a known quality tier (not unrated), and
    (c) is within *max_tier_drop* quality tiers of the baseline.  Equal blended
    costs are resolved by model name for deterministic behavior.  Returns
    (None, Decimal('0')) if no candidate qualifies.

    Unrated models (not in _QUALITY_TIERS) are NEVER returned from this
    function — they must be requested explicitly via --candidates.  The
    max_tier_drop=1 default prevents automatic cross-tier recommendations.
    """
    baseline_price = get_model_price(baseline_model)
    if baseline_price is None:
        return None, Decimal("0")

    baseline_tier = _get_model_tier(baseline_model)

    # Blended baseline cost per token (50/50 prompt/completion weight for comparison)
    baseline_blended = (
        baseline_price.input_cost_per_token + baseline_price.output_cost_per_token
    ) / Decimal("2")

    qualifying: list[tuple[Decimal, str, ModelPrice]] = []

    for candidate in _ROUTING_CANDIDATES:
        if candidate == baseline_model:
            continue

        # Exclude unrated candidates from automatic selection.
        candidate_tier = _get_model_tier(candidate)
        if candidate_tier == _UNRATED_TIER:
            continue

        # Enforce quality tier constraint.
        # When the baseline itself is unrated (_UNRATED_TIER = -1), the drop
        # math is baseline_tier = -1.  candidate_tier - (-1) = candidate_tier + 1.
        # A tier-0 candidate gives drop = 1, which passes max_tier_drop=1 —
        # allowing a known-good model to be recommended even when the baseline
        # tier is unknown.  The baseline_is_unrated flag in AnalysisResult
        # surfaces this to the CLI for disclosure.
        if candidate_tier - baseline_tier > max_tier_drop:
            continue

        cand_price = get_model_price(candidate)
        if cand_price is None:
            continue

        cand_blended = (
            cand_price.input_cost_per_token + cand_price.output_cost_per_token
        ) / Decimal("2")

        if cand_blended >= baseline_blended:
            # Not actually cheaper — skip
            continue

        qualifying.append((cand_blended, candidate, cand_price))

    if not qualifying:
        return None, Decimal("0")

    _, candidate, cand_price = min(qualifying, key=lambda item: (item[0], item[1]))

    total = Decimal("0")
    for cc in call_costs:
        total += cand_price.input_cost_per_token * Decimal(cc.record.prompt_tokens)
        total += cand_price.output_cost_per_token * Decimal(cc.record.completion_tokens)

    return candidate, total


# ---------------------------------------------------------------------------
# Escalation ladder — the next rung up when a candidate fails the judge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscalationSuggestion:
    """The next-rung-up model to try when a cheaper candidate failed the judge.

    Returned by :func:`next_rung_up` when a model exists that is BOTH (a) a
    strictly better quality tier than the failed candidate AND (b) still strictly
    cheaper than the baseline the user is paying for today.  It carries everything
    a surface needs to render an actionable next step without re-deriving any of
    the cost or tier maths:

      * ``model``                  — the suggested model name (cyan in the UI).
      * ``tier``                   — its integer quality tier (lower = better).
      * ``tier_label``             — the human tier name ("Strong", "Capable", …).
      * ``pct_cheaper_than_baseline`` — how much cheaper the suggestion is than the
        baseline on a blended per-token basis, as an integer percentage (0-100).
      * ``command``                — the ready-to-run frugon command.

    When NO such model exists (the failed candidate was already the cheapest step
    above nothing, or every cheaper model is the same/worse tier), :func:`next_rung_up`
    returns ``None`` and the surfaces keep the honest "keep these on the baseline"
    dead-end guidance.
    """

    model: str
    tier: int
    tier_label: str
    pct_cheaper_than_baseline: int
    command: str


def _blended_price(model: str) -> Decimal | None:
    """Return the 50/50 blended per-token price for *model*, or None if unpriced.

    Mirrors the blended basis :func:`_best_candidate` compares on, so the
    escalation ladder and the offline routing projection rank cost the same way.
    """
    price = get_model_price(model)
    if price is None:
        return None
    return (price.input_cost_per_token + price.output_cost_per_token) / Decimal("2")


def next_rung_up(
    failed_candidate: str,
    baseline_model: str,
    *,
    pool: list[str] | None = None,
) -> EscalationSuggestion | None:
    """Suggest the cheapest model a quality tier ABOVE *failed_candidate* yet still
    cheaper than *baseline_model*.

    Used when ``--judge`` returns a NOT-confirmed verdict: the cheap candidate was
    materially worse, so rather than shrug ("keep these on the baseline"), point
    the user at the next rung up the quality ladder that still saves money.

    Selection (fully deterministic):

      1. Universe = *pool* (defaults to :data:`_ROUTING_CANDIDATES`).
      2. Keep models that are priced AND rated (a known quality tier).
      3. Keep models whose tier is strictly BETTER than the failed candidate's
         (``tier < failed_tier`` — lower int = higher quality).  The failed
         candidate must itself be rated for "better than it" to be meaningful;
         when it is unrated we cannot reason about stepping up, so return None.
      4. Keep models strictly CHEAPER than the baseline on the blended basis.
      5. Among survivors choose the CHEAPEST (max remaining saving while stepping
         quality up); ties broken by model name for determinism.

    Returns an :class:`EscalationSuggestion` for the winner, or ``None`` when no
    model satisfies every constraint (an honest dead-end — keep the baseline).
    """
    failed_tier = _get_model_tier(failed_candidate)
    if failed_tier == _UNRATED_TIER:
        # Cannot reason about "a tier above" an unrated candidate.
        return None

    baseline_blended = _blended_price(baseline_model)
    if baseline_blended is None:
        return None

    universe = pool if pool is not None else _ROUTING_CANDIDATES

    qualifying: list[tuple[Decimal, str, int]] = []
    for model in universe:
        if model == failed_candidate or model == baseline_model:
            continue
        tier = _get_model_tier(model)
        if tier == _UNRATED_TIER:
            continue
        # Strictly better quality tier than the failed candidate.
        if tier >= failed_tier:
            continue
        blended = _blended_price(model)
        if blended is None:
            continue
        # Strictly cheaper than the baseline the user pays today.
        if blended >= baseline_blended:
            continue
        qualifying.append((blended, model, tier))

    if not qualifying:
        return None

    blended, model, tier = min(qualifying, key=lambda item: (item[0], item[1]))

    # Real cheaper-than-baseline percentage, floored to an honest integer so the
    # surfaces never round a 49.9% saving up to "~50%".
    saving_fraction = (baseline_blended - blended) / baseline_blended
    pct = int((saving_fraction * Decimal("100")).to_integral_value(rounding="ROUND_FLOOR"))

    label = _quality_tier_name(tier) or "better"
    command = f"frugon analyze --measure --candidates {model}"

    return EscalationSuggestion(
        model=model,
        tier=tier,
        tier_label=label,
        pct_cheaper_than_baseline=pct,
        command=command,
    )


# ---------------------------------------------------------------------------
# Full-dataset split New-spend (one basis for headline + candidate block)
# ---------------------------------------------------------------------------


def _full_dataset_split_newspend(
    *,
    total_cost: Decimal,
    split: SplitRouting,
    monthly_factor: Decimal | None,
) -> Decimal:
    """Return what total spend becomes if easy calls route to *split*'s candidate.

    This is the single quantity every candidate surface quotes: the FULL analyzed
    dataset's spend after routing the dominant baseline model's easy calls to the
    candidate, keeping the hard calls on the baseline, and leaving every
    already-on-a-cheaper-model call untouched.  It is computed as the current
    total minus the routing reduction on the baseline's easy calls — the exact
    arithmetic the headline panel uses (:func:`report._split_current_and_blended`),
    so a candidate's block row and the headline New-spend reconcile to the cent
    when that candidate is the chosen routing target.

    When a monthly projection basis exists (``monthly_factor`` is not None) the
    figure is the projected monthly New-spend; otherwise it is the observed
    New-spend over the log's own span.  The reduction is always taken on the same
    basis as the returned total so the two never mix observed and projected math.
    """
    if monthly_factor is not None and (
        split.monthly_baseline is not None and split.monthly_blended is not None
    ):
        current = total_cost * monthly_factor
        reduction = split.monthly_baseline - split.monthly_blended
    else:
        current = total_cost
        reduction = split.baseline_cost - split.blended_cost
    return current - reduction


_DISPLAY_PCT_QUANTUM = Decimal("0.1")


def _display_pct(pct: Decimal) -> Decimal:
    """Quantize *pct* to the ONE decimal place every surface actually prints.

    ``_fmt_candidate_saving`` (report.py) and the headline panel both render
    saving% at 1dp.  Selection must compare candidates on this SAME rounded
    value — not the raw, effectively-never-exactly-equal Decimal — or a
    caption that reads "the cheapest split is the headline recommendation"
    would be provably false the moment two candidates print the identical
    percent but differ in the 5th decimal place (the exact defect this
    function exists to close).
    """
    return pct.quantize(_DISPLAY_PCT_QUANTUM, rounding=ROUND_HALF_UP)


def _select_cheapest_eligible(
    candidates: Iterable[str],
    *,
    cand_split_newspend: dict[str, Decimal],
    baseline_newspend: Decimal,
    rated_only: bool,
) -> str | None:
    """Pick the winning candidate on the FULL-DATASET split New-spend basis.

    THE SELECTION RULE (binding across the explicit-``--candidates`` path and
    the default-pool path — one function, one rule, so headline and block can
    never again name different candidates or justify a false caption):

      1. Eligibility: the candidate must have an entry in *cand_split_newspend*
         (i.e. it was priced) and its full-dataset New-spend must strictly beat
         *baseline_newspend* — the SAME quantity the "Candidates considered"
         block ranks and displays, never a different per-token proxy.  When
         *rated_only* is True, unrated candidates are excluded entirely (the
         default-pool path — every pool entry is rated, see
         ``test_candidate_pool.py``, but the guard is defensive).  When False,
         both rated and unrated eligible candidates are ranked together (the
         explicit ``--candidates`` fallback semantics: prefer rated, but the
         caller applies that preference by calling this twice — once
         rated-only, once unrated-only — and prefers the rated result; see the
         explicit-path call site).
      2. Primary rank: ascending full-dataset New-spend (cheapest wins) —
         except candidates whose saving% renders IDENTICALLY at the 1dp the
         report actually prints (:func:`_display_pct`) are a DISPLAY TIE, not a
         real difference a user can see on the page.
      3. Quality tie-break (PD-ratified 2026-07-02): among candidates tied at
         display precision, the HIGHER quality tier wins — i.e. the LOWER
         ``get_model_tier`` integer (0 = Elite is best).  This makes the
         recommendation's caption provable from the printed values alone: two
         rows can print the same percent, but the winner is never the
         lower-quality one.
      4. Final tie-break: lexicographic model name, for full determinism when
         percent AND tier both tie.

    Returns None when no candidate is eligible.
    """
    best_model: str | None = None
    best_key: tuple[Decimal, int, str] | None = None
    for cand in candidates:
        if cand not in cand_split_newspend:
            continue
        ns = cand_split_newspend[cand]
        if ns >= baseline_newspend:
            continue
        is_unrated = _is_unrated(cand)
        if rated_only and is_unrated:
            continue
        tier = _get_model_tier(cand)
        # Display-tie rank: cheaper (more negative) percent sorts first; the
        # 1dp-quantized percent is the ACTUAL comparison key (step 2 above), so
        # two candidates whose raw New-spend differs but whose printed percent
        # is identical share the same rank position and fall through to tier.
        pct = (
            (baseline_newspend - ns) / baseline_newspend * Decimal("100")
            if baseline_newspend != Decimal("0")
            else Decimal("0")
        )
        rank_pct = -_display_pct(pct)  # negate: ascending sort == best-first
        # Unrated candidates never win a display-tie against a rated one; give
        # them a tier sentinel strictly worse than any real tier.
        tier_key = tier if not is_unrated else 10_000
        key = (rank_pct, tier_key, cand)
        if best_key is None or key < best_key:
            best_key = key
            best_model = cand
    return best_model


# ---------------------------------------------------------------------------
# Core aggregation (accepts pre-parsed records)
# ---------------------------------------------------------------------------


def analyze_records(
    records: list[LogRecord],
    window_days: int | None = None,
    candidates: list[str] | None = None,
    skipped_malformed: int = 0,
    split_routing: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    """Analyze pre-parsed log records and return an AnalysisResult.

    Accepts a list of already-parsed LogRecord objects so callers that
    produce records via another path (e.g. --measure) can avoid re-reading
    the source file.  Use analyze_logs() for the common file-based path.

    Args:
        records:          Pre-parsed LogRecord objects.
        window_days:      If provided, project monthly cost using this window.
        candidates:       Optional explicit candidate model list.
        skipped_malformed: Count of lines dropped before reaching this call
                           (JSON errors + parse_record failures), propagated
                           into the returned AnalysisResult unchanged.
        split_routing:    When True (default), also compute a per-call split
                           recommendation (route easy calls to a cheaper
                           candidate, keep hard calls on the baseline).  Set
                           False for the wholesale-only path (--wholesale).
        progress_cb:      Optional ``(done, total)`` callback invoked once per
                           record as the pricing/tokenizing pass advances — the
                           slow part on a large log.  Used by the CLI to drive a
                           live progress bar on stderr; ``None`` (the default)
                           leaves the hot loop allocation-free and side-effect
                           free, so non-interactive callers pay nothing.
    """
    total_records = len(records)
    # Pin the pricing-table identity (one stat instead of one per record) for
    # this pass — see frugon.pricing.pinned_pricing_identity for why this is
    # safe: nothing within a single ``analyze`` invocation rewrites the
    # pricing file mid-pass, only a separate ``frugon pricing update`` process
    # does, and that always precedes (never overlaps) an analysis run.
    with _pinned_pricing_identity():
        if progress_cb is None:
            call_costs = [compute_call_cost(r) for r in records]
        else:
            call_costs = []
            for index, r in enumerate(records, start=1):
                call_costs.append(compute_call_cost(r))
                progress_cb(index, total_records)
    priced = [cc for cc in call_costs if cc.price is not None]
    unpriced = [cc for cc in call_costs if cc.price is None]

    total_cost = sum((cc.total_cost for cc in priced), Decimal("0"))
    approximated_calls = sum(1 for cc in priced if cc.token_source == "approximated")

    # Guard: no priced calls → report explicit "no priced calls" (P3-4)
    if not priced:
        return AnalysisResult(
            total_calls=len(call_costs),
            priced_calls=0,
            unpriced_calls=len(unpriced),
            total_cost=Decimal("0"),
            projected_cost=Decimal("0"),
            candidate_model=None,
            skipped_malformed=skipped_malformed,
        )

    # Per-model aggregation
    cost_by_model: dict[str, Decimal] = {}
    calls_by_model: dict[str, int] = {}
    for cc in priced:
        model = cc.record.model
        cost_by_model[model] = cost_by_model.get(model, Decimal("0")) + cc.total_cost
        calls_by_model[model] = calls_by_model.get(model, 0) + 1

    # Dominant model for routing comparison
    dominant_model = max(cost_by_model, key=lambda m: cost_by_model[m])

    # Span computation for projection disclosure.  The bounds helper returns
    # the same min/max it derives the span from, so the disclosed dates and
    # the disclosed span always agree; None when <2 timestamps parse.  Computed
    # HERE (before candidate selection) because the explicit-``--candidates``
    # selection now ranks candidates on their FULL-DATASET split New-spend, which
    # is a *projected* quantity — the same monthly basis the headline panel
    # shows — so the span/window factor must be known before the winner is picked.
    _span_bounds = _compute_span_bounds(records)
    observed_span = _span_bounds[0] if _span_bounds is not None else None
    observed_span_start = _span_bounds[1] if _span_bounds is not None else None
    observed_span_end = _span_bounds[2] if _span_bounds is not None else None

    # The monthly extrapolation factor (30 / window-or-span), or None when no
    # projection basis exists.  Shared by the candidate ranking, the headline
    # panel, and the candidate block so EVERY New-spend figure rests on one
    # factor — no surface can drift onto a different basis.
    monthly_factor: Decimal | None = None
    if window_days is not None and window_days > 0:
        monthly_factor = Decimal("30") / Decimal(window_days)
    elif observed_span is not None and observed_span > 0:
        # observed_span is a float (fractional days). Converting via str()
        # preserves repr precision; for spans < ~0.001 day the float→Decimal
        # step can carry sub-millisecond error into the monthly projection —
        # negligible at the disclosed projection granularity, and the observed
        # total_cost it scales is itself exact.
        monthly_factor = Decimal("30") / Decimal(str(observed_span))

    # Routing projection: use caller-supplied candidates or auto-detect
    used_default_pool = not candidates
    # Per-candidate observed totals — populated in the same pass that picks the
    # cheapest candidate below, so the headline projection math is UNCHANGED
    # (the chosen winner + its total are identical to the prior implementation;
    # the additional bookkeeping just lets the report layer surface every
    # candidate the user passed, not only the silent winner).  Maps user-order
    # candidate name -> (observed_total, beats_baseline) for priced candidates;
    # candidates with no entry in the pricing table are tracked separately so the
    # report can tag them "unpriced" without inventing a number.
    cand_observed_totals: dict[str, Decimal] = {}
    cand_unpriced: list[str] = []
    # Per-candidate SPLIT projections, keyed by candidate name.  Computed ONCE
    # here in the explicit-``--candidates`` path so that the headline routing
    # target, the headline panel figures, and the "Candidates considered" block
    # all read from the SAME splits — there is exactly one split per candidate
    # and one New-spend per candidate, so the headline and the block can never
    # name different models or quote different numbers (the self-contradiction
    # this path used to produce).  Maps candidate -> (split, full-dataset
    # New-spend) where the New-spend is the projected (monthly) figure when a
    # projection basis exists, else the observed total.
    cand_splits: dict[str, SplitRouting] = {}
    cand_split_newspend: dict[str, Decimal] = {}
    # Unrated candidates that beat baseline on split but are held out of the
    # recommended route for being unrated (Change 1b).  Populated in the selection
    # pass below; empty for the default-pool path.
    unrated_beats_baseline: list[str] = []
    if candidates:
        from frugon.pricing import get_model_price as _gmp
        from frugon.routing import compute_split_from_partition as _compute_split
        from frugon.routing import partition_by_difficulty as _partition_by_difficulty

        # The dominant baseline model's own calls — the only routable set (the
        # easy subset moves to the candidate, the hard subset stays on the
        # baseline; calls already on another model are untouched).  This is the
        # SAME call set the headline split uses, so reusing it keeps the basis
        # identical between selection, headline, and block.
        baseline_call_costs = [cc for cc in priced if cc.record.model == dominant_model]

        # Candidate-independent work, done ONCE regardless of how many
        # candidates follow (see frugon.routing.DifficultyPartition +
        # partition_by_difficulty docstrings for why this is safe and
        # numerically identical to re-classifying per candidate):
        #   (a) the easy/hard difficulty partition of the baseline's own calls
        #   (b) the full-dataset token sums, so each candidate's full-SWAP
        #       "Upper bound" total is an O(1) multiply instead of an
        #       O(all priced calls) re-walk.
        _difficulty_partition = _partition_by_difficulty(baseline_call_costs)
        _total_prompt_tokens = sum((cc.record.prompt_tokens for cc in priced), 0)
        _total_completion_tokens = sum(
            (cc.record.completion_tokens for cc in priced), 0
        )

        for cand in candidates:
            if cand == dominant_model:
                continue
            cand_price = _gmp(cand)
            if cand_price is None:
                # Track unpriced so the multi-candidate block can honestly show
                # the user that we considered this name and had nothing to project.
                if cand not in cand_observed_totals and cand not in cand_unpriced:
                    cand_unpriced.append(cand)
                continue
            # (a) Full-SWAP total — every call repriced at the candidate.  Retained
            # only for the aggressive "Upper bound" line; it never drives the
            # headline routing target or any candidate-block row (those are
            # split-only now).  Derived from the dataset-wide token sums above
            # (candidate-independent aggregates) rather than re-summing every
            # priced call per candidate — identical arithmetic, O(1) per candidate.
            cand_total = (
                cand_price.input_cost_per_token * Decimal(_total_prompt_tokens)
                + cand_price.output_cost_per_token * Decimal(_total_completion_tokens)
            )
            cand_observed_totals[cand] = cand_total

            # (b) Full-DATASET split New-spend — the quality-preserving quantity
            # that drives EVERYTHING: route this candidate the dominant model's
            # easy calls, keep the hard calls on the baseline, leave already-on-a-
            # cheaper-model calls untouched, measured over the WHOLE analyzed
            # dataset.  This is exactly what the headline panel's New-spend
            # represents for the chosen candidate, so selecting on it guarantees
            # the headline and the block agree.  Reuses the precomputed
            # difficulty partition instead of re-classifying every baseline call.
            cand_split = _compute_split(
                baseline_model=dominant_model,
                candidate_model=cand,
                partition=_difficulty_partition,
                candidate_price=cand_price,
                window_days=window_days,
                observed_span_days=observed_span,
            )
            cand_splits[cand] = cand_split
            cand_split_newspend[cand] = _full_dataset_split_newspend(
                total_cost=total_cost,
                split=cand_split,
                monthly_factor=monthly_factor,
            )

        # The headline routing target is the candidate that wins
        # :func:`_select_cheapest_eligible` on the full-dataset split New-spend
        # basis — cheapest, with a display-precision quality tie-break (see that
        # function's docstring for the full rule) — BUT only RATED candidates
        # are eligible for the headline.  An unrated candidate has no known
        # quality tier, so routing the dominant model's easy calls onto it would
        # silently trade unknown quality for price; such a candidate is
        # surfaced as "considered" in the block, never silently recommended.
        # Only when NO rated candidate beats the baseline do we fall back to the
        # cheapest unrated one that beats it (the user explicitly asked for
        # these candidates and there is no rated alternative) — carried with the
        # unrated caveat so the quality gap is disclosed.  The candidate's split
        # IS the headline split and its full-swap total IS the Upper-bound
        # projection, so the routing target, the panel figures, the block's
        # "recommended"/"considered" tag, and the Upper bound all stay
        # internally consistent.
        baseline_newspend = (
            total_cost * monthly_factor if monthly_factor is not None else total_cost
        )

        # Unrated candidates that beat the baseline on the split but are held out
        # of the recommended route purely for being unrated.  Drives the
        # "excluded because unrated — measure to unlock" caveat (Change 1b).
        # User command-line order, de-duplicated.
        _seen_excluded: set[str] = set()
        for cand in candidates:
            if cand in cand_split_newspend and _is_unrated(cand):
                ns = cand_split_newspend[cand]
                if ns < baseline_newspend and cand not in _seen_excluded:
                    unrated_beats_baseline.append(cand)
                    _seen_excluded.add(cand)

        best_model = _select_cheapest_eligible(
            candidates,
            cand_split_newspend=cand_split_newspend,
            baseline_newspend=baseline_newspend,
            rated_only=True,
        )
        if best_model is None:
            # No rated candidate beats the baseline — fall back to the cheapest
            # unrated one (same selection rule, restricted to the unrated set
            # via a second pass rather than a rated_only=False call, so a rated
            # candidate can never lose a display-tie to an unrated one here).
            best_model = _select_cheapest_eligible(
                (c for c in candidates if _is_unrated(c)),
                cand_split_newspend=cand_split_newspend,
                baseline_newspend=baseline_newspend,
                rated_only=False,
            )
        candidate_model = best_model
        # ``projected_cost`` carries the aggressive full-SWAP total of the chosen
        # candidate, used solely for the "Upper bound" line so it stays internally
        # consistent with the newly-selected routing target.  The headline panel's
        # New-spend comes from the split (via ``result.split``), NOT from here.
        projected_cost = (
            cand_observed_totals[best_model]
            if best_model is not None
            else Decimal("0")
        )
    else:
        candidate_model, projected_cost = _best_candidate(dominant_model, priced)

        # Default-pool candidate splits — the SAME full-dataset split New-spend
        # basis the explicit-``--candidates`` path computes above, run over the
        # built-in ``_ROUTING_CANDIDATES`` roster instead of a user-supplied list.
        # Populates ``cand_splits``/``cand_split_newspend`` so the split-routing
        # selection below (when ``split_routing`` is True) and the "Candidates
        # considered" block both rank on the IDENTICAL basis — one function, one
        # rule, so the headline split target and the block's recommended row
        # can never again name different candidates (see
        # :func:`_select_cheapest_eligible`).  When ``split_routing`` is False
        # (``--wholesale``), this dict is still populated but unused: the
        # wholesale headline keeps the tier-capped ``_best_candidate`` pick
        # above, because a wholesale swap moves EVERY call and so needs the
        # tighter tier-drop cap that only applies to a full swap.
        from frugon.pricing import get_model_price as _gmp_default
        from frugon.routing import (
            compute_split_from_partition as _compute_split_default,
        )
        from frugon.routing import partition_by_difficulty as _partition_by_difficulty_default

        baseline_call_costs_default = [
            cc for cc in priced if cc.record.model == dominant_model
        ]
        # Candidate-independent difficulty classification, computed ONCE for
        # the whole built-in pool instead of once per candidate — see
        # frugon.routing.DifficultyPartition.  On the bundled 56,100-record
        # demo (23-candidate default pool) this closes the ~4s of redundant
        # Decimal difficulty-scoring that dominated the post-pricing pass.
        _difficulty_partition_default = _partition_by_difficulty_default(
            baseline_call_costs_default
        )
        for cand in _ROUTING_CANDIDATES:
            if cand == dominant_model:
                continue
            cand_price = _gmp_default(cand)
            if cand_price is None:
                continue
            cand_split = _compute_split_default(
                baseline_model=dominant_model,
                candidate_model=cand,
                partition=_difficulty_partition_default,
                candidate_price=cand_price,
                window_days=window_days,
                observed_span_days=observed_span,
            )
            cand_splits[cand] = cand_split
            cand_split_newspend[cand] = _full_dataset_split_newspend(
                total_cost=total_cost,
                split=cand_split,
                monthly_factor=monthly_factor,
            )

        if split_routing:
            # The split headline's own routing target is the winner of the SAME
            # :func:`_select_cheapest_eligible` rule the explicit-``--candidates``
            # path uses — full-dataset New-spend, display-precision quality
            # tie-break, name tie-break — run over the built-in pool. Every
            # entry in ``_ROUTING_CANDIDATES`` is rated (``test_candidate_pool``
            # pins this), so ``rated_only=True`` never actually excludes a
            # candidate here; it is passed for consistency with the explicit
            # path's call and as a defensive guard if the roster ever changes.
            # Overriding ``candidate_model``/``projected_cost`` here (rather
            # than leaving the ``_best_candidate`` wholesale pick in place) is
            # what makes ``result.candidate_model`` and ``result.split.candidate_model``
            # PROVABLY the same value — the two could previously coincide only
            # by accident, since ``_best_candidate`` is tier-capped and ranks on
            # blended per-token price while the split needs the uncapped,
            # New-spend-ranked pick.
            baseline_newspend_default = (
                total_cost * monthly_factor if monthly_factor is not None else total_cost
            )
            split_target = _select_cheapest_eligible(
                _ROUTING_CANDIDATES,
                cand_split_newspend=cand_split_newspend,
                baseline_newspend=baseline_newspend_default,
                rated_only=True,
            )
            if split_target is not None:
                candidate_model = split_target
                # ``projected_cost`` carries the full-SWAP total (every call
                # repriced at the candidate) — used solely for the "Upper
                # bound" line — recomputed here for the split's OWN target so
                # it stays internally consistent with the just-overridden
                # ``candidate_model``, exactly as the explicit-candidates path
                # does for its own selection.
                _target_price = _gmp_default(split_target)
                if _target_price is not None:
                    _swap_total = Decimal("0")
                    for cc in priced:
                        _swap_total += _target_price.input_cost_per_token * Decimal(
                            cc.record.prompt_tokens
                        )
                        _swap_total += _target_price.output_cost_per_token * Decimal(
                            cc.record.completion_tokens
                        )
                    projected_cost = _swap_total

    # State (b) of the priceable-pool distinction (report._render_wholesale_panel
    # and its md/html counterparts): the user passed explicit --candidates and
    # none of them had a known list price, so the cost race never ran.  Scoped
    # to the explicit-candidates branch only -- cand_unpriced is never
    # populated on the default-pool branch above, whose every entry is priced.
    no_priceable_candidates = not cand_splits and bool(cand_unpriced)

    # Compute tier_drop: only defined when both baseline and candidate have known tiers.
    tier_drop: int | None = None
    if candidate_model is not None:
        b_tier = _get_model_tier(dominant_model)
        c_tier = _get_model_tier(candidate_model)
        if b_tier != _UNRATED_TIER and c_tier != _UNRATED_TIER:
            tier_drop = c_tier - b_tier

    # Monthly projection: total_cost is always the raw observed total.
    # monthly_cost and monthly_projected are computed separately and stored as
    # distinct fields so renderers can show both observed and projected rows.
    # Both rest on ``monthly_factor`` (30 / window-or-span, computed once above)
    # so the panel, the candidate ranking, and the candidate block can never
    # drift onto different projection bases.
    monthly_cost: Decimal | None = None
    monthly_projected: Decimal | None = None
    if monthly_factor is not None:
        monthly_cost = total_cost * monthly_factor
        if candidate_model is not None:
            monthly_projected = projected_cost * monthly_factor

    # Collect pricing_json_last_synced from any priced call
    pjls: str | None = None
    for cc in priced:
        if cc.price and cc.price.pricing_json_last_synced:
            pjls = cc.price.pricing_json_last_synced
            break

    # Collect quality.json last_synced from the quality tier table — the same way
    # pricing freshness is collected above.  The "within tolerance" recommendation
    # rests on these tiers, so their freshness is decision-relevant and surfaced in
    # the report's Accounting block alongside the Prices row.
    _, quality_last_synced, _ = _load_quality_table()

    # Per-call split-routing recommendation (the headline feature).  Operates on
    # the dominant baseline model's own calls: route the easy subset to a cheaper
    # rated candidate, keep the hard subset on the baseline.  Computed offline —
    # no LLM, no network.  Disabled by --wholesale (split_routing=False).
    split: SplitRouting | None = None
    if split_routing:
        # Both the explicit-``--candidates`` path and the default-pool path
        # picked ``candidate_model`` via the SAME :func:`_select_cheapest_eligible`
        # rule above (full-dataset New-spend, display-precision quality
        # tie-break, name tie-break), and both populated ``cand_splits`` for
        # every priced candidate they considered — so reusing
        # ``cand_splits[candidate_model]`` here, for EITHER path, guarantees the
        # headline panel, the routing-target name, and the block's
        # "recommended" row are always one and the same split, on one and the
        # same basis.  This is what closes the caption-truth gap: the
        # recommended row can never print a cheaper number than the row it is
        # tagged against, because both come from the same selection.
        if candidate_model is not None and candidate_model in cand_splits:
            split = cand_splits[candidate_model]

    # ----- Per-candidate projections (multi-candidate transparency) ----------
    # When --candidates lists more than one model, surface ALL of them in the
    # report (recommended | considered | more_expensive | unpriced).
    # Each candidate is projected on a SPLIT basis (route easy baseline calls to
    # that candidate, keep hard calls on baseline) so every row is directly
    # comparable to the headline New-spend / saving%.
    # Empty list == no extra rendering (single-candidate paths unchanged).
    #
    # Baseline reference (full dataset) for every row's saving%, shared by both
    # the explicit-``--candidates`` branch and the default-pool branch below.
    # Monthly when a projection basis exists, else the observed total —
    # matching the unit the New-spend figure is quoted in.
    full_current_monthly = monthly_cost  # full-dataset monthly current (or None)
    full_current_observed = total_cost  # full-dataset observed current

    def _build_candidate_projection(cand: str, status: str) -> CandidateProjection:
        """Build one candidate's row from its precomputed split (see cand_splits).

        Shared by the explicit-``--candidates`` path and the default-pool path so
        every row's money/saving% arithmetic is computed exactly once, on the
        SAME full-dataset split New-spend basis the headline reads from
        (``cand_splits`` / ``cand_split_newspend``), regardless of which pool the
        candidate came from.
        """
        cand_split = cand_splits[cand]

        # ----- Monthly figures (full-dataset New-spend) ----------------------
        # New-spend = full current monthly - routing reduction on the baseline's
        # easy calls.  saving = reduction; pct = reduction / full current
        # monthly — the exact denominator the headline panel uses (saved /
        # current over the WHOLE dataset), so the recommended row reconciles
        # with the headline to the cent.
        monthly_c: Decimal | None
        monthly_saving_val: Decimal | None
        saving_pct_val: Decimal | None
        if (
            full_current_monthly is not None
            and cand_split.monthly_baseline is not None
            and cand_split.monthly_blended is not None
        ):
            reduction_m = cand_split.monthly_baseline - cand_split.monthly_blended
            monthly_c = full_current_monthly - reduction_m
            monthly_saving_val = reduction_m
            saving_pct_val = (
                (reduction_m / full_current_monthly) * Decimal("100")
                if full_current_monthly > Decimal("0")
                else None
            )
        else:
            monthly_c = None
            monthly_saving_val = None
            saving_pct_val = None

        # ----- Observed figures (full-dataset New-spend) ----------------------
        # Always populated so the block still has numbers when there is no
        # monthly projection basis.  Same full-dataset arithmetic on the
        # observed (un-extrapolated) totals.
        reduction_o = cand_split.baseline_cost - cand_split.blended_cost
        obs_cost = full_current_observed - reduction_o
        obs_saving = reduction_o
        obs_pct = (
            (reduction_o / full_current_observed) * Decimal("100")
            if full_current_observed > Decimal("0")
            else None
        )
        cand_tier = _get_model_tier(cand)
        tier_label = _quality_tier_name(cand_tier) or "Unrated"
        return CandidateProjection(
            model=cand,
            status=status,
            monthly_cost=monthly_c,
            monthly_saving=monthly_saving_val,
            saving_pct=saving_pct_val,
            observed_cost=obs_cost,
            observed_saving=obs_saving,
            observed_saving_pct=obs_pct,
            tier_label=tier_label,
        )

    def _candidate_status(cand: str) -> str:
        """recommended / considered / more_expensive for *cand*, on the split basis."""
        cand_newspend = cand_split_newspend[cand]
        cand_baseline_ref = (
            full_current_monthly
            if full_current_monthly is not None
            else full_current_observed
        )
        beats_split = cand_newspend < cand_baseline_ref
        if cand == candidate_model:
            return "recommended"
        return "considered" if beats_split else "more_expensive"

    candidate_projections: list[CandidateProjection] = []
    if candidates and len(candidates) > 1:
        # Reuse the per-candidate splits computed once in the selection pass —
        # the SAME ``cand_splits`` the headline routing target was chosen from —
        # so every block row is on the identical basis to the headline.  We do
        # NOT recompute a second set of splits here (the old code did, which is
        # how the block's numbers drifted from the headline's).
        #
        # The "recommended" row is the chosen headline routing target
        # (``candidate_model``, the cheapest full-dataset New-spend).  Every row's
        # money column is that candidate's FULL-DATASET split New-spend — route
        # the dominant model's easy calls to this candidate, keep its hard calls
        # on the baseline, leave already-cheaper calls untouched, over the WHOLE
        # dataset — so the recommended row equals the headline New-spend to the
        # cent and gpt-4o-style "two different numbers" can never appear.
        for cand in candidates:
            if cand == dominant_model:
                continue
            if cand in cand_splits:
                # "beats baseline" is decided on the SAME full-dataset New-spend
                # basis the headline ranking uses — never on the dominant-only
                # blended figure — so a candidate's tag and its row number agree.
                candidate_projections.append(
                    _build_candidate_projection(cand, _candidate_status(cand))
                )
            elif cand in cand_unpriced:
                candidate_projections.append(
                    CandidateProjection(model=cand, status="unpriced")
                )
            # Else: cand == dominant_model already skipped; defensive no-op.
    elif used_default_pool and split is not None:
        # Default pool, no explicit --candidates: real users — and the un-pinned
        # demo — should SEE what was considered, not just the winner.  Cap to the
        # recommended candidate (the split's own routing target) plus the
        # next-4-cheapest candidates that ALSO beat the baseline on the same
        # full-dataset split New-spend basis (5 rows max) — never the more-
        # expensive or unpriced tail of the built-in roster, which would just be
        # noise on the default view.  Ranking uses the SAME cand_split_newspend
        # populated above (identical arithmetic to the explicit-candidates path),
        # so this is a populate+render change, not new math.
        recommended_model = split.candidate_model
        baseline_ref = (
            full_current_monthly
            if full_current_monthly is not None
            else full_current_observed
        )
        beating_others = sorted(
            (
                cand
                for cand, ns in cand_split_newspend.items()
                if cand != recommended_model and ns < baseline_ref
            ),
            key=lambda cand: (cand_split_newspend[cand], cand),
        )
        ranked_models = [recommended_model, *beating_others[:4]]
        candidate_projections = [
            _build_candidate_projection(cand, _candidate_status(cand))
            for cand in ranked_models
            if cand in cand_splits
        ]

    return AnalysisResult(
        total_calls=len(call_costs),
        priced_calls=len(priced),
        unpriced_calls=len(unpriced),
        total_cost=total_cost,
        cost_by_model=cost_by_model,
        calls_by_model=calls_by_model,
        projected_cost=projected_cost,
        candidate_model=candidate_model,
        observed_span_days=observed_span,
        observed_span_start=observed_span_start,
        observed_span_end=observed_span_end,
        window_days=window_days,
        pricing_json_last_synced=pjls,
        quality_json_last_synced=quality_last_synced,
        candidate_pool_size=len(_ROUTING_CANDIDATES),
        used_default_pool=used_default_pool,
        tier_drop=tier_drop,
        baseline_is_unrated=_is_unrated(dominant_model),
        candidate_is_unrated=(
            _is_unrated(candidate_model) if candidate_model is not None else False
        ),
        # Unrated candidates beating baseline that were NOT recommended (the
        # recommended model is rated, or — in the fallback — is a different
        # unrated model).  Exclude the recommended model itself so the caveat
        # never tells the user a model is "excluded" when it IS the route.
        excluded_unrated_models=[
            m for m in unrated_beats_baseline if m != candidate_model
        ],
        skipped_malformed=skipped_malformed,
        approximated_calls=approximated_calls,
        monthly_cost=monthly_cost,
        monthly_projected=monthly_projected,
        split=split,
        candidate_projections=candidate_projections,
        no_priceable_candidates=no_priceable_candidates,
        unpriced_candidate_names=list(cand_unpriced),
    )


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

# Hard ceiling on the DECOMPRESSED size of a ``.gz`` log, in bytes.  A gzip
# stream's uncompressed size is attacker-controlled independent of its
# on-disk (compressed) size — a small, crafted ``.gz`` can expand to gigabytes
# ("gzip bomb"), which an uncapped `gzip.decompress(path.read_bytes())` would
# happily materialise fully in memory before any other check runs.  Streaming
# the decompression in bounded chunks and aborting once this ceiling is
# crossed keeps a hostile or corrupted ``.gz`` from OOMing the process.
# ~512MB comfortably covers any real frugon workload — the bundled --demo
# fixture (tens of thousands of records) decompresses to a few MB — while
# still catching a bomb long before it exhausts a typical developer machine.
# Overridable via FRUGON_MAX_GZ_DECOMPRESSED_BYTES for tests that need a small
# ceiling to exercise this path without generating a 512MB fixture.
_DEFAULT_MAX_DECOMPRESSED_GZ_BYTES = 512 * 1024 * 1024  # 512 MiB

# Chunk size for the bounded streaming read below.  Small enough that the
# ceiling check fires promptly after being crossed, large enough to keep the
# per-chunk call overhead negligible for a legitimate multi-hundred-MB log.
_GZ_READ_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def _max_decompressed_gz_bytes() -> int:
    """Return the active decompressed-size ceiling for ``.gz`` log reads.

    Reads ``FRUGON_MAX_GZ_DECOMPRESSED_BYTES`` from the environment on every
    call (not cached at import time) so tests can monkeypatch the environment
    per-test.  Falls back to the 512MB default on an absent, empty, or
    unparseable value — a malformed override must never silently DISABLE the
    cap.
    """
    raw = os.environ.get("FRUGON_MAX_GZ_DECOMPRESSED_BYTES")
    if not raw:
        return _DEFAULT_MAX_DECOMPRESSED_GZ_BYTES
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_DECOMPRESSED_GZ_BYTES
    return value if value > 0 else _DEFAULT_MAX_DECOMPRESSED_GZ_BYTES


class LogReadError(OSError):
    """Raised when a log file cannot be read: corrupt/truncated gzip, or a
    decompressed ``.gz`` payload that exceeds the safety ceiling.

    Subclasses ``OSError`` (not a bespoke base) so every existing call site
    that already does ``except OSError as exc: ...  {exc.strerror or exc}``
    (frugon.cli's analyze / --measure pre-flight paths) handles this new
    failure mode with zero changes — the clean Rich error panel fires
    automatically instead of a raw traceback.
    """


def _read_log_text(path: Path) -> str:
    """Read a JSONL log file as UTF-8 text, transparently decompressing ``.gz``.

    A gzip-compressed log (``*.jsonl.gz``) is decoded on the fly so callers never
    need a temp file.  This is what lets the bundled ``--demo`` fixture ship a
    realistic, several-thousand-call workload in a small package: a repetitive
    production log (the same prompt template reused thousands of times) compresses
    by ~60x, so a multi-megabyte log is a sub-megabyte artifact.  All reading
    stays 100% local — gzip decompression makes no network call (privacy
    invariant).  A UTF-8 decode error is raised to the caller unchanged so the
    CLI's friendly "not valid UTF-8" message still fires (§4 fail-loud).

    The ``.gz`` path streams the decompression in bounded chunks (never loading
    the whole payload via a single unbounded ``gzip.decompress()`` call) and
    raises :class:`LogReadError` — an ``OSError`` subclass — if: the stream is
    not valid gzip (``BadGzipFile``), the stream is truncated mid-record
    (``EOFError``), or the decompressed size crosses
    :func:`_max_decompressed_gz_bytes` (gzip-bomb guard).  Each of those is a
    clean, actionable failure for the CLI's existing ``except OSError`` panel
    rather than an unbounded memory allocation or an unhandled traceback.
    """
    if path.suffix == ".gz":
        ceiling = _max_decompressed_gz_bytes()
        chunks: list[bytes] = []
        total = 0
        try:
            with gzip.open(path, "rb") as gz_file:
                while True:
                    chunk = gz_file.read(_GZ_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > ceiling:
                        raise LogReadError(
                            f"{path}: decompressed size exceeds the "
                            f"{ceiling:,}-byte safety limit. This looks like a "
                            "corrupted or maliciously crafted .gz file — frugon "
                            "will not decompress it fully into memory."
                        )
                    chunks.append(chunk)
        except gzip.BadGzipFile as exc:
            raise LogReadError(f"{path}: not a valid gzip file ({exc}).") from exc
        except EOFError as exc:
            raise LogReadError(
                f"{path}: the gzip stream is truncated or corrupted ({exc})."
            ) from exc
        return b"".join(chunks).decode("utf-8")
    return path.read_text(encoding="utf-8")


def iter_records(path: Path) -> tuple[list[LogRecord], int]:
    """Parse a JSONL log file into (records, skipped_malformed).

    The single source of truth for turning a log file into LogRecords.  Both
    analyze_logs() and the --measure path call this so they agree on exactly
    which lines were dropped — no path silently loses records.  A line is
    counted in *skipped_malformed* when it is non-blank but
    either fails JSON decoding or yields no LogRecord (missing model, no usable
    tokens, etc.).  Blank lines are ignored and never counted.
    """
    records: list[LogRecord] = []
    skipped_malformed = 0
    for line in _read_log_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError:
            skipped_malformed += 1
            continue
        if not isinstance(parsed, dict):
            skipped_malformed += 1
            continue
        raw: dict[str, Any] = parsed
        rec = parse_record(raw)
        if rec is not None:
            records.append(rec)
        else:
            skipped_malformed += 1
    return records, skipped_malformed


def scan_models(path: Path) -> tuple[list[str], str | None]:
    """Cheaply enumerate the distinct model names present in a JSONL log.

    This is the lightweight counterpart to :func:`iter_records`: it reads the
    log once and extracts only the ``model`` field from each line — it does NOT
    tokenize, price, or build :class:`LogRecord` objects.  It exists so the
    ``--measure`` / ``--judge`` paths can run a fail-fast prerequisite check
    (LiteLLM importable + provider keys present) BEFORE the expensive full cost
    analysis, instead of after.

    Returns:
        A ``(distinct_models, dominant_model)`` tuple where:
          * ``distinct_models`` is the list of unique model names in first-seen
            order, and
          * ``dominant_model`` is the model appearing on the most lines (the
            cheap proxy for the baseline the full analysis will pick), or
            ``None`` when the log contains no usable model field.

    Reading stays 100% local (no network call); a UTF-8 decode error is raised
    to the caller unchanged so the CLI's friendly message still fires.
    """
    counts: dict[str, int] = {}
    order: list[str] = []
    for line in _read_log_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        raw: dict[str, Any] = parsed
        model = raw.get("model")
        if not isinstance(model, str) or not model:
            continue
        if model not in counts:
            counts[model] = 0
            order.append(model)
        counts[model] += 1
    dominant = max(order, key=lambda m: counts[m]) if order else None
    return order, dominant


def analyze_logs(
    path: Path,
    window_days: int | None = None,
    candidates: list[str] | None = None,
    split_routing: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    """Analyze a JSONL log file and return an AnalysisResult.

    Args:
        path:        Path to the JSONL log file.
        window_days: If provided, project monthly cost using this window size.
                     Never invented — only used when explicitly passed.
        candidates:  Optional list of candidate model names to evaluate.
                     If None, frugon picks from its internal routing candidates.
        split_routing: When True (default), compute the per-call split
                     recommendation; False for the wholesale-only path.
        progress_cb: Optional ``(done, total)`` per-record callback forwarded to
                     :func:`analyze_records` to drive a live pricing progress
                     bar.  ``None`` (default) keeps the pass side-effect free.

    The monthly projection behaviour:
      - window_days given          → project monthly, disclose "N-day window"
      - timestamps present         → compute real span, project monthly, disclose span
      - no timestamps, no window   → report observed total only, no projection

    total_cost is always the raw observed total — never multiplied.
    monthly_cost and monthly_projected are the extrapolated figures.
    """
    records, skipped_malformed = iter_records(path)
    return analyze_records(
        records,
        window_days,
        candidates,
        skipped_malformed,
        split_routing=split_routing,
        progress_cb=progress_cb,
    )


# ---------------------------------------------------------------------------
# Saving % computation (P3-4 guard included)
# ---------------------------------------------------------------------------


def compute_saving_pct(current: Decimal, projected: Decimal) -> Decimal | None:
    """Compute (current - projected) / current as a percentage.

    Returns None when current is zero (no priced calls) to avoid
    ZeroDivisionError or NaN in reports (P3-4 fix).
    """
    if current == Decimal("0"):
        return None
    return ((current - projected) / current) * Decimal("100")


def window_contradicts_span(
    window_days: int | None,
    observed_span_days: float | None,
    *,
    ratio_threshold: float = 1.5,
) -> bool:
    """True when a ``--window`` override materially disagrees with the real span.

    ``--window N`` overrides the monthly-projection basis (``total_cost × 30/N``),
    so a window that is much shorter or much longer than the log's actual observed
    span silently inflates or deflates the monthly figure — e.g. ``--window 7`` on a
    ~30-day log projects as if the data covered only a quarter of the time, a ~4.3×
    overstatement.  This predicate detects that contradiction so the report can warn.

    Returns True only when BOTH values are present (the user passed ``--window`` AND
    the log carried enough timestamps to observe a span) and they differ by at least
    *ratio_threshold* in either direction: ``max(w, s) / min(w, s) >= ratio_threshold``.
    A 7-vs-30 mismatch fires (ratio ~4.3); a 28-vs-30 near-match does not (ratio ~1.07).

    Returns False when either value is missing (``--window`` absent, or no timestamps
    to observe a span) or when either is non-positive, so a degenerate zero/negative
    span can never raise or warn spuriously.
    """
    if window_days is None or observed_span_days is None:
        return False
    if window_days <= 0 or observed_span_days <= 0:
        return False
    larger = max(float(window_days), observed_span_days)
    smaller = min(float(window_days), observed_span_days)
    return larger / smaller >= ratio_threshold
