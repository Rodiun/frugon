"""frugon measure engine — sample real prompts through candidate models.

Privacy invariant:
  Sampled prompts go ONLY to the user's own providers via their own API keys.
  Keys are read from the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
  by the LiteLLM library.  Keys are never logged, persisted, or forwarded to
  any Rodiun or Frugon host.  The only outbound calls are to the provider
  endpoints implied by the model strings the user supplies.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from frugon.cost import LogRecord
from frugon.model_id import canonicalize

if TYPE_CHECKING:
    from decimal import Decimal

    from frugon.pricing import ModelPrice

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingProviderKeyError(RuntimeError):
    """Raised when a provider API key is absent from the environment.

    Carries the structured list of missing environment-variable names in
    ``missing_vars`` (e.g. ``["OPENAI_API_KEY"]``) so the CLI can render a
    clean, provider-named fix message without re-parsing the error string.
    """

    def __init__(
        self,
        message: str,
        missing_vars: list[str] | None = None,
        suggestions: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing_vars: list[str] = missing_vars or []
        # Map of expected-var -> a currently-set env var that looks like a typo
        # of it (e.g. {"OPENAI_API_KEY": "OPEN_API_KEY"}) so the CLI can nudge
        # "did you mean…?". Variable NAMES only — never any value.
        self.suggestions: dict[str, str] = suggestions or {}


class UnknownModelError(RuntimeError):
    """Raised when a model name cannot be resolved before any provider call.

    Catches typo'd or made-up model names (e.g. ``gpt-5.3``) at the same
    fail-fast pre-flight that verifies provider keys, BEFORE any sampling call
    is made — so a bad ``--candidates`` / ``--judge-model`` / log-baseline name
    never burns a single provider request.

    Carries the structured per-model suggestion list in ``unknown_models`` so
    the CLI can render a clean, framed "did you mean…?" panel without re-parsing
    the error string.  Each entry is ``(bad_model_name, [closest_suggestions])``
    in input order; suggestions are nearest-edit-distance neighbours from the
    pricing table (empty list when no model in the table is within the cap).
    """

    def __init__(
        self,
        message: str,
        unknown_models: list[tuple[str, list[str]]] | None = None,
    ) -> None:
        super().__init__(message)
        self.unknown_models: list[tuple[str, list[str]]] = unknown_models or []


# Default model used to judge candidate outputs when --judge is set.  Exposed
# as a module constant so the CLI's fail-fast pre-check can include it in the
# required-key scan without re-hardcoding the name.
#
# This is the LAST-RESORT arm's-length fallback, not the primary judge choice.
# _resolve_judge_model's precedence is: (1) an explicit --judge-model flag —
# the user's stated intent; (2) the highest quality-tier model already present
# in the user's OWN log (best_judge_from_log) — they already hold a key for
# it; (3) the highest quality-tier rated+priced model whose provider key IS
# present in the environment (best_judge_for_available_keys) — so a user
# without an OpenAI key still gets a reachable judge; (4) THIS constant, and
# only when OPENAI_API_KEY is actually set.  gpt-4.1 is not in the 23-candidate
# roster, is not the demo baseline, and is not the sample-log pin — it exists
# solely as a strong, current, independent OpenAI model to judge against, so a
# judge never grades a candidate that is itself (a self-evaluation bias that
# would systematically flatter the candidate's verdict).  The user can always
# override with --judge-model.
DEFAULT_JUDGE_MODEL = "gpt-4.1"

# Default for the public ``--concurrency`` flag (and ``run_measure``'s
# ``concurrency=`` parameter): the per-stage worker ceiling when the caller does
# not override it.  Used only as the default value; the live caps are derived
# from the flag inside run_measure (see _stage_worker_counts).
_DEFAULT_CONCURRENCY = 5

# Hard upper bound on the SECOND-STAGE (judging) pool, independent of the flag.
# The judging stage fires every (prompt × candidate) pair at ONE provider — the
# single judge model — so its in-flight count concentrates on a lone endpoint
# rather than spreading across the baseline + candidate providers like the
# sampling stage.  8 simultaneous judge calls were a meaningful share of the
# observed transient rate-limit errors; capping the judge pool at 5 keeps good
# latency overlap with materially less pressure on a single provider's per-minute
# limit, EVEN WHEN the user raises --concurrency high to fan sampling wide.  The
# one-shot retry in _judge_pair is the primary resilience fix; this ceiling is the
# complementary, cheaper half (fewer faults to retry in the first place).
#
# The sampling stage is intentionally NOT capped here: it spreads across multiple
# provider endpoints (baseline + every candidate), so a high --concurrency simply
# fans those independent endpoints wider.  Because the two stages run
# CONCURRENTLY (a fast prompt judges while slow prompts still sample), peak
# provider round-trips ≈ sample_workers + judge_workers BY DESIGN — that is the
# whole point of the two-stage overlap, and it is safe precisely because sampling
# is multi-endpoint while judging is the lone judge endpoint guarded by this cap.
_JUDGE_MAX_CONCURRENCY = 5


def _stage_worker_counts(concurrency: int, n_prompts: int) -> tuple[int, int]:
    """Derive (sample_workers, judge_workers) from the --concurrency flag.

    The sampling stage gets the full ``concurrency`` (clamped to the number of
    prompts — never more workers than there is work), spreading WIDE across the
    baseline + candidate provider endpoints.  The judging stage gets
    ``min(concurrency, _JUDGE_MAX_CONCURRENCY)`` so a high flag fans sampling out
    without overwhelming the single judge endpoint.

    ``concurrency=1`` collapses BOTH stages to a single worker — a fully
    sequential, deterministic degenerate path that serves as the parity
    reference.  ``concurrency`` is validated ``>= 1`` at the CLI boundary; the
    ``max(1, …)`` guards keep this helper total even if called directly.
    """
    sample_workers = max(1, min(concurrency, n_prompts)) if n_prompts else 1
    judge_workers = max(1, min(concurrency, _JUDGE_MAX_CONCURRENCY))
    return sample_workers, judge_workers


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MeasureCallUsage:
    """Token usage for ONE provider call made by the measure run itself.

    Records what the measure run COST the user — captured from the provider's
    own ``usage`` block on each ``completion()`` response (sampling calls AND
    judge calls).  ``prompt_tokens`` / ``completion_tokens`` are 0 when the call
    failed before returning usage (e.g. a hard error), but the call is still
    recorded so the displayed call count is honest (errors that consumed tokens
    are counted; failed calls with no usage count as zero-token calls).
    """

    model: str
    prompt_tokens: int
    completion_tokens: int


def _extract_usage(response: Any) -> tuple[int, int]:
    """Pull ``(prompt_tokens, completion_tokens)`` from a LiteLLM response.

    LiteLLM mirrors the OpenAI response shape: ``response.usage.prompt_tokens``
    and ``response.usage.completion_tokens``.  Any missing / non-integer field
    degrades to 0 (never raises) so usage capture can NEVER break a measure run
    — a call that returned content but no parseable usage is still counted, just
    with zero tokens, which is the honest floor.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    def _coerce(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    pt = _coerce(getattr(usage, "prompt_tokens", None))
    ct = _coerce(getattr(usage, "completion_tokens", None))
    return pt, ct


# Typed classification of a sampling failure (see :func:`_classify_failure`).
# Carried on SampledOutput alongside the rendered display cell so downstream
# synthesis consumes the cause and never re-parses display strings.
FailureCause = Literal["quota", "auth", "rate_limit", "other"]


@dataclass
class SampledOutput:
    """Output from one model for one sampled record.

    ``usage`` carries the provider-reported token counts for THIS call (the cost
    the measure run incurred), or ``None`` when the call failed before returning
    a usage block.  It is display-only metadata — it never affects the content
    or the verdict — so existing constructors that omit it are unaffected.

    ``error_cause`` is the typed failure classification, set by the sampling
    path whenever ``error`` is set; synthesis consumes it instead of
    re-parsing the display cell.  Constructors that omit it (tests, older
    fixtures) read as an unclassified failure — the generic fallback.
    """

    model: str
    content: str
    error: str | None = None
    usage: MeasureCallUsage | None = None
    error_cause: FailureCause | None = None


@dataclass
class Comparison:
    """Side-by-side outputs for one sampled record (Tier-0 result).

    When a judge ran (``--judge``), ``verdicts`` holds the per-candidate verdict
    for THIS prompt, aligned 1:1 with ``candidate_outputs`` — each entry is one
    of "win" / "loss" / "tie" / "error" (the same tokens the aggregate tally
    counts).  It lets a verbose view show WHICH sampled prompts the candidate
    lost on, not just how many.  Empty when no judge ran, so non-judge callers
    and their tests are unaffected.

    ``both_failed`` is aligned 1:1 with ``verdicts`` too, and is meaningful ONLY
    where the corresponding verdict is "tie": the pairwise judge's TIE means "no
    MATERIAL difference", which is silent about whether both sides tied because
    they were equally GOOD or because they equally FAILED to address the prompt
    (see :func:`_judge_addressed`).  ``True`` means a pointwise check found
    NEITHER the baseline nor this candidate addressed the prompt — a shared
    failure the pairwise TIE alone would hide.  Elements for a non-"tie" verdict
    are always ``False`` (the pointwise check never runs for win/loss/error).
    Empty when no judge ran, matching ``verdicts``.
    """

    record: LogRecord
    current_output: SampledOutput
    candidate_outputs: list[SampledOutput]
    verdicts: list[str] = field(default_factory=list)
    both_failed: list[bool] = field(default_factory=list)


@dataclass
class Tier1Tally:
    """Win/loss/tie counts for one candidate model from judge evaluation.

    ``both_failed_ties`` counts the subset of ``ties`` flagged both-failed (see
    :attr:`Comparison.both_failed`) — a tie where NEITHER side addressed the
    prompt, not a genuine equally-good result.  It is a strict subset of
    ``ties`` (``both_failed_ties <= ties`` always), tallied from the same
    verdict pass so it never drifts from the printed Win/Loss/Tie/Error counts.

    ``check_errors`` counts ties whose pointwise both-failed check hit a
    transient fault and exhausted its retries (:func:`_judge_addressed`
    defaults such a fault to "addressed" so it never falsely flags a shared
    failure) — a non-zero count means the judged-success rate for this
    candidate may be optimistic, since one or more shared failures could have
    gone undetected.
    """

    candidate: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    errors: int = 0
    both_failed_ties: int = 0
    check_errors: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.ties + self.errors

    @property
    def verdict_count(self) -> int:
        """Comparisons that reached a scored verdict (errors excluded)."""
        return self.wins + self.losses + self.ties

    @property
    def judged_success_count(self) -> int:
        """wins + (ties NOT flagged both-failed) — never counts a shared failure.

        A pairwise TIE alone is silent about whether both sides tied on being
        equally good or equally failing the prompt; ``both_failed_ties`` is the
        FRG-OSS-068 correction that keeps this count honest (a both-failed tie
        is a shared failure, not a judged success).
        """
        return self.wins + self.ties - self.both_failed_ties


@dataclass
class MeasureResult:
    """Aggregated result of a --measure run."""

    samples_requested: int
    samples_taken: int
    current_model: str
    candidates: list[str]
    comparisons: list[Comparison]
    tier1_tallies: list[Tier1Tally] | None = None
    # Count of UNIQUE PROMPTS discovered in the input log (post-dedup by
    # content key — see :func:`_dedup_key`).  When this is less than
    # ``samples_requested`` the renderer surfaces an honest note: the log
    # carried fewer distinct prompts than the user asked for, and every
    # available unique prompt was measured (no duplicate prompts were
    # measured to pad the count).  Defaults to 0 so directly-constructed
    # MeasureResult objects in tests stay compatible — the renderer treats
    # 0 as "unknown" and suppresses the note in that case.
    unique_prompts_available: int = 0
    # The resolved judge model used when --judge ran (None for a Tier-0 sample).
    # Carried so the renderer can name the judge in the methodology note without
    # re-deriving it.
    judge_model: str | None = None
    # Models the judge IS grading (judge == a candidate, or judge == baseline) —
    # a self-evaluation that may bias the verdict.  Empty when the judge is
    # independent of everything it scored.  The CLI surfaces a dim caution naming
    # these so the user knows the verdict is not arm's-length.
    self_judged_models: list[str] = field(default_factory=list)
    # True when the resolved judge was auto-selected as the highest quality-tier
    # model PRESENT IN THE USER'S OWN LOG (so the user already has a key for it),
    # rather than supplied via --judge-model or fallen back to
    # DEFAULT_JUDGE_MODEL.  The renderer uses this to honestly describe the judge
    # as "your highest-tier model" in that case.  Set by run_measure's caller via
    # judge_from_log=; defaults False so existing callers/tests are unaffected.
    judge_from_log: bool = False
    # Per-call token usage for EVERY provider call the measure run made — sampling
    # AND judge calls.  Used to report what the measure run itself cost the user
    # (Frugon's thesis is cost transparency, so it discloses its own footprint).
    # Empty for callers/tests that construct a MeasureResult directly; the cost
    # line is simply not rendered in that case.
    measure_calls: list[MeasureCallUsage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider key map — infer required env var from model name prefix
# ---------------------------------------------------------------------------

# Maps a lowercase model-name prefix to the environment variable that must
# be present.  Checked before any network call is made.
#
# The seven ``deepseek-``/``grok-``/``kimi-``/``glm-``/``minimax-``/``qwen-``
# entries (FRG-OSS-034 23-model roster, added 2026-07-02) were added after a
# gap: the roster's default candidate pool grew from 10 to 23 models across
# 11 vendors, but this map was never extended to match, so ``--measure`` /
# ``--judge`` against any of these bare names silently skipped the provider-key
# precheck entirely (no key required -> no key checked) and then failed at the
# actual sample call with a bare "bad request" cell (see
# :data:`_LITELLM_ROUTE_PREFIX` below for why the bare names need a routing
# prefix too, not just a key).  Every env var name below is verified against
# the LiteLLM source in this repo's own ``.venv`` (``litellm/llms/<vendor>/
# chat/transformation.py``, ``get_secret_str(...)`` call) — not guessed:
#   deepseek/chat/transformation.py   -> DEEPSEEK_API_KEY
#   xai/chat/transformation.py        -> XAI_API_KEY
#   moonshot/chat/transformation.py   -> MOONSHOT_API_KEY
#   zai/chat/transformation.py        -> ZAI_API_KEY   (NOT "zhipu" — Z.ai's
#                                         LiteLLM integration module is named
#                                         "zai", and that is the var it reads)
#   minimax/chat/transformation.py    -> MINIMAX_API_KEY
#   dashscope/chat/transformation.py  -> DASHSCOPE_API_KEY (Qwen's LiteLLM
#                                         provider is "dashscope" — Alibaba
#                                         Cloud's model API, not a "qwen" var)
# The two Llama-4 reference-host models (llama-4-maverick-17b-128e-instruct,
# llama-4-scout-17b-16e-instruct) already route via Groq (GROQ_API_KEY) — see
# _LITELLM_ROUTE_PREFIX; the "groq/" prefix entry above already covers them
# once _route_for_measure prepends it, so no new key entry is needed for them.
_PROVIDER_KEY_MAP: dict[str, str] = {
    "gpt-": "OPENAI_API_KEY",
    "o1": "OPENAI_API_KEY",
    "o3": "OPENAI_API_KEY",
    "o4": "OPENAI_API_KEY",
    "text-davinci": "OPENAI_API_KEY",
    "claude-": "ANTHROPIC_API_KEY",
    "gemini/": "GEMINI_API_KEY",
    "gemini-": "GEMINI_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
    "mistral-": "MISTRAL_API_KEY",
    "cohere/": "COHERE_API_KEY",
    "command-": "COHERE_API_KEY",
    "groq/": "GROQ_API_KEY",
    "together_ai/": "TOGETHERAI_API_KEY",
    "together-": "TOGETHERAI_API_KEY",
    "openrouter/": "OPENROUTER_API_KEY",
    "azure/": "AZURE_API_KEY",
    "bedrock/": "AWS_ACCESS_KEY_ID",
    "vertex_ai/": "VERTEXAI_PROJECT",
    "deepseek-": "DEEPSEEK_API_KEY",
    "grok-": "XAI_API_KEY",
    "kimi-": "MOONSHOT_API_KEY",
    "glm-": "ZAI_API_KEY",
    "minimax-": "MINIMAX_API_KEY",
    "qwen-": "DASHSCOPE_API_KEY",
    # The two reference-host Llama-4 checkpoints route via Groq (see
    # _LITELLM_ROUTE_PREFIX below) — GROQ_API_KEY, same var the "groq/" prefix
    # entry above already maps, added explicitly for the BARE name since the
    # roster's pricing/quality/CLI surfaces only ever use the bare form.
    "llama-4-maverick-17b-128e-instruct": "GROQ_API_KEY",
    "llama-4-scout-17b-16e-instruct": "GROQ_API_KEY",
}


# ---------------------------------------------------------------------------
# Bare-name -> LiteLLM routing prefix (new-vendor gap, FRG-OSS-034 follow-up)
# ---------------------------------------------------------------------------
#
# frugon's pricing/quality tables key every model on its BARE name (no
# provider prefix) — that is the name shown throughout the CLI, the
# "Candidates considered" block, and passed to ``--candidates``.  LiteLLM,
# however, only infers a provider from a bare name for a handful of
# "default-namespace" vendors (OpenAI, Anthropic, a few others); every other
# vendor requires an explicit ``<provider>/`` prefix on the wire, or
# ``litellm.completion`` raises ``BadRequestError: LLM Provider NOT provided``
# — confirmed directly against this repo's own ``.venv`` LiteLLM install via
# ``litellm.get_llm_provider(<bare-name>)`` for every entry below.  Without
# this map, a real ``--measure``/``--judge`` run against any of these bare
# names would reach ``_call_model``/``_judge_prompt``, make a doomed call, and
# report a generic "[bad request — check model name and parameters]" cell —
# not a crash (the existing try/except already catches it), but not truthful
# about the actual, fixable cause either.  ``_route_for_measure`` prepends the
# mapped prefix ONLY for the LiteLLM call; every other surface (reports,
# pricing/quality lookups, ``--candidates`` matching) keeps using the bare
# name unchanged.
#
# The two Llama-4 checkpoints are reference-host-priced via Groq (see
# ``cost.py``'s ``_ROUTING_CANDIDATES`` comment) — Groq's own wire form nests
# the "meta-llama/" org segment, which frugon's ``canonicalize()`` already
# knows how to fold away in the other direction (see
# ``model_id._GROQ_META_LLAMA_RE``); this map produces exactly that nested
# form so measurement uses the SAME route the pricing seed already labels.
# Invariant (enforced by tests/test_measure_precheck.py::TestRoutePrefixNoOverlap):
# no key here is a proper prefix of another key — this dict mixes vendor
# PREFIXES ("deepseek-") with FULL model names ("llama-4-scout-..."), and
# _route_for_measure's startswith() match means a shorter key that happened to
# prefix a longer one would make the match nondeterministic (dependent on dict
# iteration order) — every entry added here must keep this true.
_LITELLM_ROUTE_PREFIX: dict[str, str] = {
    "deepseek-": "deepseek/",
    "grok-": "xai/",
    "kimi-": "moonshot/",
    "glm-": "zai/",
    "minimax-": "minimax/",
    "qwen-": "dashscope/",
    # FRG-OSS-036: the roster's "mistral-large-3" entry had a key mapping
    # (_PROVIDER_KEY_MAP already has "mistral-" -> MISTRAL_API_KEY) but no
    # routing prefix — the exact same bare-name gap the deepseek/grok/etc.
    # entries above closed for their vendors. Confirmed against this repo's
    # own .venv LiteLLM install: litellm.get_llm_provider("mistral-large-3")
    # raises BadRequestError ("LLM Provider NOT provided"); the routed form
    # litellm.get_llm_provider("mistral/mistral-large-3") resolves cleanly to
    # the "mistral" provider. The bare "mistral-" prefix covers the whole
    # family (mistral-large-3 is the only roster member today, but any future
    # bare "mistral-*" addition needs the same prefix, not a per-model entry).
    "mistral-": "mistral/",
    "llama-4-maverick-17b-128e-instruct": "groq/meta-llama/",
    "llama-4-scout-17b-16e-instruct": "groq/meta-llama/",
}


def _route_for_measure(model: str) -> str:
    """Return the model string to actually pass to ``litellm.completion``.

    Bare names for the new-vendor roster entries (see
    :data:`_LITELLM_ROUTE_PREFIX`) are not routable by LiteLLM on their own;
    this prepends the vendor's required prefix so the ACTUAL sample/judge call
    resolves to the right provider.  Every other model name (already-routable
    bare defaults, or a name the user passed WITH its own gateway/provider
    prefix already) is returned unchanged — this never adds a prefix twice and
    never touches a name outside the mapped set.
    """
    lower = model.lower()
    for bare, prefix in _LITELLM_ROUTE_PREFIX.items():
        if lower == bare or lower.startswith(bare):
            return prefix + model
    return model


def _required_key_for_model(model: str) -> str | None:
    """Return the environment variable name required for *model*, or None if unknown.

    Tries the original model name first (preserves correct routing for bedrock/,
    azure/, vertex_ai/ etc.), then falls back to the canonicalized form so that
    gateway-prefixed names like ``openrouter/openai/gpt-4o`` resolve correctly.
    """
    lower = model.lower()
    for prefix, env_var in _PROVIDER_KEY_MAP.items():
        if lower.startswith(prefix):
            return env_var
    # Fallback: try after stripping gateway prefixes so e.g. openrouter/openai/gpt-4o
    # resolves to gpt-4o which then matches the "gpt-" entry.
    canon = canonicalize(model)
    if canon != model:
        lower_canon = canon.lower()
        for prefix, env_var in _PROVIDER_KEY_MAP.items():
            if lower_canon.startswith(prefix):
                return env_var
    return None


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (no third-party dependency)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# Name tokens that mark an environment variable as credential-shaped.  Shared
# by _nearest_env_var (which reads only NAMES, for typo hints) and
# _env_key_values (which reads VALUES, to redact them from provider messages).
_CREDENTIAL_NAME_TOKENS: tuple[str, ...] = ("KEY", "API", "TOKEN", "SECRET")


def _nearest_env_var(expected: str, *, max_distance: int = 3) -> str | None:
    """Return a CURRENTLY-SET env var whose name is a likely typo of *expected*.

    Helps the user who set, say, ``OPEN_API_KEY`` instead of ``OPENAI_API_KEY``
    (a one-token slip).  Only key-shaped names are considered, and only the
    variable NAME is ever read or returned — never any value.
    """
    best: str | None = None
    best_dist = max_distance + 1
    for name in os.environ:
        if name == expected:
            continue
        upper = name.upper()
        if not any(tok in upper for tok in _CREDENTIAL_NAME_TOKENS):
            continue
        dist = _levenshtein(expected, name)
        if dist < best_dist:
            best, best_dist = name, dist
    return best if best is not None and best_dist <= max_distance else None


def _check_provider_keys(models: list[str]) -> None:
    """Raise MissingProviderKeyError if any model's required key is absent.

    All missing keys are reported in a single message so the user can fix
    them all at once.  No network call is made before this check passes.
    """
    missing: list[str] = []
    missing_vars: list[str] = []
    suggestions: dict[str, str] = {}
    seen_vars: set[str] = set()
    for model in models:
        env_var = _required_key_for_model(model)
        if env_var is None or env_var in seen_vars:
            continue
        seen_vars.add(env_var)
        if not os.environ.get(env_var):
            missing.append(f"{env_var} (required for {model!r})")
            missing_vars.append(env_var)
            near = _nearest_env_var(env_var)
            if near is not None:
                suggestions[env_var] = near
    if missing:
        joined = ", ".join(missing)
        raise MissingProviderKeyError(
            f"Missing provider API key(s): {joined}. "
            "Set the environment variable(s) and retry.",
            missing_vars=missing_vars,
            suggestions=suggestions,
        )


# Maximum edit distance considered a useful "did you mean…?" suggestion.  Beyond
# this the candidate is too far from the typo to be a meaningful nudge — better
# to say "(no close match)" than mislead with an unrelated model name.
_UNKNOWN_MODEL_SUGGESTION_MAX_DISTANCE = 4

# How many ranked suggestions to surface per unknown model.  More than 3 is
# noise in a one-shot terminal panel; 1-3 is the same shape as the env-var
# "did you mean…?" hint already in this module.
_UNKNOWN_MODEL_SUGGESTION_LIMIT = 3


def _suggest_models(bad_model: str, known_models: list[str]) -> list[str]:
    """Return the closest 1-N model names from *known_models* for a typo.

    Ranks by Levenshtein distance (capped at
    :data:`_UNKNOWN_MODEL_SUGGESTION_MAX_DISTANCE`) and then alphabetically by
    name so ties are deterministic — important for the test suite and for any
    snapshot reviewer reading the rendered panel.  Returns at most
    :data:`_UNKNOWN_MODEL_SUGGESTION_LIMIT` names; empty when no model in the
    table is within the cap (the CLI then surfaces "no close match…").
    """
    bad_lower = bad_model.lower()
    ranked: list[tuple[int, str]] = []
    for name in known_models:
        dist = _levenshtein(bad_lower, name.lower())
        if dist <= _UNKNOWN_MODEL_SUGGESTION_MAX_DISTANCE:
            ranked.append((dist, name))
    ranked.sort(key=lambda pair: (pair[0], pair[1]))
    return [name for _dist, name in ranked[:_UNKNOWN_MODEL_SUGGESTION_LIMIT]]


def _check_known_models(models: list[str]) -> None:
    """Raise :class:`UnknownModelError` for any unresolvable model name.

    A model is KNOWN when at least one of the following resolves it:

      * :func:`frugon.pricing.get_model_price` returns an entry (our
        LiteLLM-sourced pricing snapshot, ~2k models), OR
      * LiteLLM's own registry knows it — either ``litellm.model_cost`` carries
        the key (mirrors the upstream JSON) OR ``litellm.utils.get_model_info``
        returns info without raising (covers a handful of alias / provider-
        routed forms LiteLLM resolves dynamically).

    When ALL three lookups draw blank the model is UNKNOWN.  Every unknown name
    is collected (so a user mis-typing two ``--candidates`` sees both in one
    pass), with deterministic nearest-neighbour suggestions from the pricing
    table.  Raises a single :class:`UnknownModelError` carrying the full list
    so the CLI can render one clean amber panel — no provider call has yet been
    made when this raises.

    Duplicate names in *models* are checked once (and reported once, in input
    order); the empty / whitespace name is skipped (the caller's responsibility
    to filter, but the pre-check is permissive about it).
    """
    # Lazy LiteLLM import: this helper is part of the --measure pre-flight and
    # must NEVER run on plain `frugon analyze --demo` (which does not import
    # LiteLLM and stays byte-identical to the prior release).
    litellm = _import_litellm()

    from frugon.pricing import get_model_price, load_pricing_override

    known_table, _last_synced = load_pricing_override()
    known_names = list(known_table)

    unknown: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for model in models:
        if not model or model in seen:
            continue
        seen.add(model)

        if get_model_price(model) is not None:
            continue
        # Belt-and-suspenders against LiteLLM resolving names our pricing
        # snapshot misses (a handful of dynamic aliases / provider-routed
        # forms).  Defensive: when `litellm` is a stub (the test fixtures
        # patch `_import_litellm` to return a bare `object()`), neither
        # `model_cost` nor `utils.get_model_info` exists — and the right
        # answer is then "trust the pricing table", not "everything is
        # unknown".  The pricing table covers ~2.3k names including all
        # defaults, so a stub fixture only weakens the secondary check, not
        # the primary one.
        try:
            model_cost = litellm.model_cost  # type: ignore[union-attr]
        except AttributeError:
            model_cost = None
        if isinstance(model_cost, dict) and model in model_cost:
            continue
        # LiteLLM's get_model_info raises for unmapped models — that raise IS
        # the "unknown" signal we want.  Any other failure (no `utils` on a
        # stub, etc.) is treated as "lookup unavailable, fall back to the
        # pricing-table verdict already computed above".
        info: object | None = None
        try:
            info = litellm.utils.get_model_info(model)  # type: ignore[union-attr]
        except Exception:
            info = None
        if info:
            continue

        unknown.append((model, _suggest_models(model, known_names)))

    if unknown:
        joined = ", ".join(repr(name) for name, _ in unknown)
        raise UnknownModelError(
            f"Unknown model name(s): {joined}. "
            "Pass a known model name or run `frugon pricing` to see what's available.",
            unknown_models=unknown,
        )


def _present_provider_key_vars() -> set[str]:
    """Return the set of provider-key env-var NAMES currently present & non-empty.

    Built from the values of :data:`_PROVIDER_KEY_MAP` (several prefixes share one
    env var — e.g. ``gemini/`` and ``gemini-`` both map to ``GEMINI_API_KEY`` —
    so the value set is de-duplicated).  Only the variable NAME is read here; the
    secret value is never returned or logged.  Used by the key-aware judge
    fallback to learn which providers the user can actually reach.
    """
    present: set[str] = set()
    for env_var in set(_PROVIDER_KEY_MAP.values()):
        if os.environ.get(env_var):
            present.add(env_var)
    return present


def _priceable_form(model: str) -> str | None:
    """Resolve a rated family name to a concrete model frugon can PRICE, or None.

    The quality table stores ``base_family``-normalised names (e.g.
    ``claude-3-5-sonnet``), but several providers — Anthropic especially — only
    publish pricing under DATED or ``-latest`` snapshot keys
    (``claude-3-5-sonnet-latest``).  So a bare family name often fails a direct
    price lookup even though a concrete, callable, priceable form exists.  Try the
    family name as-is first, then its ``-latest`` snapshot; return the first form
    :func:`frugon.pricing.get_model_price` can resolve, else None.  The returned
    string is both a valid LiteLLM model id AND one the pricing table covers, so
    the judge call routes and the measurement-cost line stays exact.
    """
    from frugon.pricing import get_model_price

    for candidate in (model, f"{model}-latest"):
        if get_model_price(candidate) is not None:
            return candidate
    return None


def best_judge_for_available_keys() -> str | None:
    """Pick the best rated, priceable judge model the user's PRESENT keys can reach.

    The final fallback for ``--judge`` when NO model in the user's own log carries
    a quality rating: instead of hard-defaulting to an OpenAI model (which demands
    an OpenAI key the user may not hold), scan the environment for present
    provider keys and choose the highest quality-tier rated model whose provider
    key IS present — resolved to a concrete priceable form so the measurement-cost
    line stays exact.

    Selection:
      * universe = the rated-model table (``frugon.quality.load_quality_table``),
        filtered to models whose required key (``_required_key_for_model``) is one
        of the present provider keys AND which resolve to a priceable concrete form
        (``_priceable_form`` — handles family→``-latest`` snapshot resolution so
        Anthropic models, priced only under dated keys, are not wrongly excluded);
      * choose the LOWEST tier integer (0 = Elite is best); ties broken by the
        rated NAME ascending, so the result is deterministic regardless of the
        concrete form returned.

    Returns the concrete priceable model string, or ``None`` when no rated,
    priceable, key-reachable model exists at all (the caller then renders the
    clean fail-fast panel directing the user to set a provider key or pass
    ``--judge-model``).  Pure local read — no network call.
    """
    from frugon.quality import load_quality_table

    present_keys = _present_provider_key_vars()
    if not present_keys:
        return None

    tier_map, _last_synced, _attribution = load_quality_table()
    best_name: str | None = None  # the rated family name (tie-break key)
    best_concrete: str | None = None  # the priceable form actually returned
    best_tier = 0  # placeholder; replaced on first eligible model
    for model in sorted(tier_map):  # deterministic name-ascending scan
        tier = tier_map[model]
        required = _required_key_for_model(model)
        if required is None or required not in present_keys:
            continue
        concrete = _priceable_form(model)
        if concrete is None:
            continue  # rated but not priceable in any concrete form — exclude
        if best_name is None or tier < best_tier:
            best_name, best_concrete, best_tier = model, concrete, tier
    return best_concrete


# ---------------------------------------------------------------------------
# LiteLLM import (lazy — only when --measure is actually invoked)
# ---------------------------------------------------------------------------


def _import_litellm() -> Any:
    """Import litellm; raise a helpful error when the [measure] extra is absent."""
    try:
        # Suppress import-time WARNING chatter before the import executes so
        # that module-level log calls in LiteLLM and botocore are already
        # silenced when they fire.
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("litellm").setLevel(logging.ERROR)
        logging.getLogger("botocore").setLevel(logging.ERROR)
        logging.getLogger("boto3").setLevel(logging.ERROR)

        import litellm  # type: ignore[import-untyped]

        # Suppress LiteLLM's "Give Feedback / LiteLLM.Info" stderr banners.
        litellm.suppress_debug_info = True

        return litellm
    except ImportError as _exc:
        raise ImportError(
            "The --measure flag requires LiteLLM. "
            "Install it with:  uv tool install 'frugon[measure]' --force  "
            "(or, if you installed frugon with pip:  pip install 'frugon[measure]')"
        ) from _exc


def verify_measure_prerequisites(models: list[str]) -> None:
    """Fail-fast pre-check for a ``--measure`` / ``--judge`` run.

    Verifies, in order and BEFORE any expensive cost analysis, that:
      1. LiteLLM (the ``[measure]`` extra) is importable,
      2. every model name in *models* is resolvable in our pricing snapshot
         or LiteLLM's own registry (catches typo'd / made-up names like
         ``gpt-5.3`` at the same fail-fast gate), and
      3. every provider API key required by *models* is present in the
         environment.

    *models* should be the set of model names that the run will actually
    measure — the auto-detected baseline plus the candidate models.

    The unknown-model check fires BEFORE the key check on purpose: if the
    user typo'd ``--candidates gpt-5.3``, the right answer is "did you mean
    gpt-5?", not "set OPENAI_API_KEY".  Surfacing the typo first avoids
    leading them through a useless key-export ritual.

    Raises:
        ImportError:             when the ``[measure]`` extra is not installed.
        UnknownModelError:       when any *models* entry resolves to no known
                                 model (pricing table miss AND LiteLLM miss).
        MissingProviderKeyError: when a required provider API key is absent.

    All three exceptions are EXPECTED, actionable, user-facing conditions: the
    CLI catches them at its boundary and renders a clean, framed message
    (never a Python traceback).  No network call is made by this check.
    """
    _import_litellm()
    _check_known_models(models)
    _check_provider_keys(models)


# ---------------------------------------------------------------------------
# LiteLLM error → friendly cell message
# ---------------------------------------------------------------------------

# Maps substrings of the exception type name to a short human-readable cell.
# Failures with a distinguished cause (quota / auth / rate limit — see
# :func:`_classify_failure`) are handled in :func:`_friendly_cell` with
# redacted provider-message detail and never reach this table; note that a
# BadRequestError carrying Anthropic's credit-exhaustion signal classifies as
# quota, so only genuine bad requests hit the row below.
_LITELLM_ERROR_CELLS: list[tuple[str, str]] = [
    ("NotFoundError", "[unavailable — project lacks access to this model]"),
    ("InternalServerError", "[provider error — retry later]"),
    ("BadRequestError", "[bad request — check model name and parameters]"),
    ("ServiceUnavailableError", "[service unavailable — retry later]"),
    ("Timeout", "[request timed out — retry later]"),
    ("ContextWindowExceeded", "[context too long for this model]"),
]

# Provider JSON error codes that mean billing/quota exhaustion rather than a
# transient requests-per-minute throttle.  Matched against structured response
# bodies before any message-text heuristic.
_QUOTA_ERROR_CODES = frozenset({"insufficient_quota", "billing_not_active"})

# Anthropic signals credit exhaustion as HTTP 400 (BadRequestError), not 429:
# "Your credit balance is too low to access the Anthropic API."  The phrase is
# the only stable signal — the structured error type on that response is the
# generic ``invalid_request_error``.
_ANTHROPIC_CREDIT_PHRASE = "credit balance is too low"

# Strip LiteLLM's own exception framing from a provider message line.
_LITELLM_MSG_PREFIX_RE = re.compile(r"^litellm\.\w+Error:\s*")

# §5 privacy: shapes that must never reach a rendered surface (terminal cell,
# markdown report, HTML report).  Keys first — known LLM-provider prefixes,
# including provider-masked forms that keep a visible prefix and tail (OpenAI
# masks like ``sk-proj-Ab3x***mZ7Q``) — then org/project/account identifiers,
# then URLs (noise, and can embed identifiers in paths or query strings).
# These patterns are the SECOND line: keys carry no reliable shape, so the
# primary defence is exact-value redaction of every credential in the
# environment (:func:`_env_key_values`, applied before these).  The patterns
# still earn their place for values NOT sourced from this process's env — a
# provider echoing a different account's identifier, or a key pasted into a
# prompt — and for org/account ids and URLs, which are not env-sourced at all.
_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9*_-]{4,}"),  # OpenAI sk-/sk-proj-, Anthropic sk-ant-
    re.compile(r"\bAIza[A-Za-z0-9*_-]{4,}"),  # Google
    re.compile(r"\bhf_[A-Za-z0-9*]{4,}"),  # Hugging Face
    re.compile(r"\bgsk_[A-Za-z0-9*]{4,}"),  # Groq
    re.compile(r"\bxai-[A-Za-z0-9*_-]{4,}"),  # xAI
    re.compile(r"\borg[-_][A-Za-z0-9*_-]{2,}"),  # OpenAI organization ids
    re.compile(r"\b(?:proj|project|acct|account)[-_][A-Za-z0-9*_-]{2,}"),
    re.compile(r"https?://\S+"),
)
_REDACTED = "[redacted]"

# Hard cap on the provider detail attached to a friendly cell — "short" is
# enforced by construction, not by provider goodwill.
_PROVIDER_MESSAGE_MAX_CHARS = 160


def _env_key_values() -> tuple[str, ...]:
    """Return every credential-shaped value in the environment.

    Bare random-string keys have no matchable shape, so the redaction floor
    is the VALUE, not a pattern.  Enumerating providers is the trap here:
    ``_PROVIDER_KEY_MAP`` is frugon's PREREQ-CHECK map, not the set of vars
    that authenticate a call.  ``--candidates`` accepts any LiteLLM route
    (``perplexity/sonar``, ``fireworks_ai/…``), and for an unmapped model
    LiteLLM reads its OWN env var, which no provider map of ours would ever
    list.  So scan the UNION: the mapped vars (``VERTEXAI_PROJECT`` carries
    none of the tokens below and would regress on a token scan alone) plus
    every environment name that looks like a credential, using the same
    token test :func:`_nearest_env_var` applies.  No map to maintain, and a
    provider we have never heard of is covered on the day it ships.

    Read per call (never cached) so tests and mid-process env changes are
    honoured.  Longest first, so a value that is a substring of another
    cannot pre-empt the longer match; values under 8 chars are skipped (too
    short to be a real secret, and substring-replacing them would mangle
    ordinary words).  Only VALUES are read — a name is never rendered.
    """
    names = set(_PROVIDER_KEY_MAP.values())
    names.update(
        name
        for name in os.environ
        if any(tok in name.upper() for tok in _CREDENTIAL_NAME_TOKENS)
    )
    seen: set[str] = set()
    for var in names:
        val = os.environ.get(var, "").strip()
        if len(val) >= 8:
            seen.add(val)
    return tuple(sorted(seen, key=len, reverse=True))


def _raw_provider_message(exc: Exception) -> str:
    """Return the provider's message text for CLASSIFICATION only.

    Unredacted and uncapped, so cause detection never misses a signal that
    redaction or truncation would remove.  Never render this — every display
    path goes through :func:`_provider_message_line`.
    """
    raw = getattr(exc, "message", None)
    if raw is None:
        raw = str(exc)
    text = str(raw).strip()
    return _LITELLM_MSG_PREFIX_RE.sub("", text).strip()


def _provider_message_line(exc: Exception) -> str:
    """Return the provider's error line, safe to render on every surface.

    For typed LiteLLM exceptions the ``message`` attribute carries the
    provider text; unknown exception types fall back to ``str(exc)``.  The
    returned line is collapsed to its first line, then redacted in two
    layers — every credential VALUE in this process's environment
    (:func:`_env_key_values`, which needs no shape and no provider list),
    then key shapes, org/project/account identifiers, and URLs
    (``_REDACTION_PATTERNS``, which catch what the environment scan cannot
    source) — and finally capped at ``_PROVIDER_MESSAGE_MAX_CHARS``.
    Together these are the §5 guarantee that a cell never leaks a key, an
    account identifier, or a link: enforced here by construction, because
    the cell renders verbatim into the terminal table and the
    markdown/HTML report files users share.
    """
    text = _raw_provider_message(exc)
    text = text.splitlines()[0].strip() if text else ""
    # Exact env-key values FIRST: they need no shape, and redacting them
    # before the patterns run means a provider echoing the user's own key
    # verbatim can never survive, whatever the key looks like.
    for secret in _env_key_values():
        text = text.replace(secret, _REDACTED)
    for pattern in _REDACTION_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    if len(text) > _PROVIDER_MESSAGE_MAX_CHARS:
        text = text[: _PROVIDER_MESSAGE_MAX_CHARS - 1].rstrip() + "…"
    return text


def _response_error_code(exc: Exception) -> str | None:
    """Extract a provider error ``code`` / ``type`` from a LiteLLM response."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        json_fn = getattr(response, "json", None)
        data = json_fn() if callable(json_fn) else None
    except (ValueError, TypeError, AttributeError):
        # Unparseable / absent body — fall back to the message heuristic.
        return None
    if not isinstance(data, dict):
        return None
    err = data.get("error", data)
    if not isinstance(err, dict):
        return None
    for key in ("code", "type"):
        value = err.get(key)
        if value:
            return str(value).lower()
    return None


def _signals_quota(exc: Exception) -> bool:
    """True when a 429's structured code or message signals quota/billing.

    Secondary signals only — the caller (:func:`_classify_failure`) gates on
    the exception TYPE first.  The structured response code is preferred;
    message text is the last resort, and reads the RAW message so redaction
    or truncation can never hide the signal.
    """
    code = _response_error_code(exc)
    if code is not None and (code in _QUOTA_ERROR_CODES or "quota" in code):
        return True
    msg = _raw_provider_message(exc).lower()
    return "quota" in msg or _ANTHROPIC_CREDIT_PHRASE in msg


def _classify_failure(exc: Exception) -> FailureCause:
    """Classify a sampling failure by cause — the single source of truth.

    Layered detection discipline: exception TYPE first, structured provider
    response code second, provider message text last.  The result travels on
    :attr:`SampledOutput.error_cause`, so downstream synthesis consumes the
    TYPED cause and never re-parses rendered display strings — the appended
    provider text is arbitrary and can contain the very phrases a string
    scan would key on (a throttle message mentioning "quota" must not turn
    into a billing verdict downstream of a rate-limit classification).
    """
    type_name = type(exc).__name__
    if "AuthenticationError" in type_name:
        return "auth"
    if "RateLimitError" in type_name:
        return "quota" if _signals_quota(exc) else "rate_limit"
    if "BadRequestError" in type_name:
        # Anthropic signals credit exhaustion as HTTP 400, not 429.  Gate on
        # the narrow signals only (structured quota code, or Anthropic's
        # credit phrase) so a genuine bad request never reads as billing.
        code = _response_error_code(exc)
        if code in _QUOTA_ERROR_CODES:
            return "quota"
        if _ANTHROPIC_CREDIT_PHRASE in _raw_provider_message(exc).lower():
            return "quota"
    return "other"


def _friendly_cell_with_detail(headline: str, exc: Exception) -> str:
    """Attach the redacted provider message line after a friendly headline."""
    detail = _provider_message_line(exc)
    if detail:
        return f"{headline} — {detail}"
    return headline


# Headline per distinguished cause; "other" falls through to the type-name
# lookup table (_LITELLM_ERROR_CELLS) in _friendly_cell.
_CAUSE_HEADLINES: dict[FailureCause, str] = {
    "quota": "[quota exceeded — check billing]",
    "auth": "[auth failed — check your API key]",
    "rate_limit": "[rate limited — retry later]",
}


def _friendly_cell(exc: Exception) -> str:
    """Map a LiteLLM exception to a short, human-readable cell string.

    Quota, auth, and rate-limit failures append the provider's own one-line
    message — redacted and length-capped by :func:`_provider_message_line` —
    so a drained account is distinguishable from a transient throttle
    without leaking keys, account identifiers, or URLs.  Other known types
    keep headline-only cells; unknown types show only the exception class
    (never a raw debug string).
    """
    cause = _classify_failure(exc)
    if cause != "other":
        return _friendly_cell_with_detail(_CAUSE_HEADLINES[cause], exc)
    type_name = type(exc).__name__
    for fragment, cell in _LITELLM_ERROR_CELLS:
        if fragment in type_name:
            return cell
    return f"[error: {type_name}]"


# Synthesis phrase per distinguished cause, in priority order when a run
# carries mixed causes: quota beats auth beats rate limit (the user should
# fix billing before chasing throttles).
_CAUSE_PHRASES: tuple[tuple[FailureCause, str], ...] = (
    ("quota", "quota exceeded (check billing)"),
    ("auth", "auth failure (check your API key)"),
    ("rate_limit", "rate limited (retry later)"),
)

# The one place the generic fallback wording lives.  Report surfaces import
# this — never restate the string — so the vocabulary of the failure verdict
# has a single source, like the cause phrases above.
GENERIC_SAMPLING_FAILURE_PHRASE = "rate limit or API error"


def summarize_sampling_failures(causes: Iterable[FailureCause | None]) -> str:
    """Return the most specific synthesis phrase for a set of typed causes.

    Consumes :attr:`SampledOutput.error_cause` values — NEVER rendered cell
    strings, whose appended provider text is arbitrary and can contain the
    very phrases a string scan would match.  Priority when causes are mixed:
    quota > auth > rate limit > generic fallback.
    """
    seen = {cause for cause in causes if cause is not None}
    for cause, phrase in _CAUSE_PHRASES:
        if cause in seen:
            return phrase
    return GENERIC_SAMPLING_FAILURE_PHRASE


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _dedup_key(record: LogRecord) -> str:
    """Content-hash key used to dedup records by UNIQUE PROMPT.

    Frugon's quality measurement asks "does the candidate answer THIS prompt as
    well as the baseline?" — a question about prompts, not log lines.  A log
    typically calls the same prompt many times; without dedup, sampling N
    records may pick the same content multiple times and waste judge budget on
    duplicates.  The key is therefore a hash of the comparison-relevant
    content: the FIRST system message (steers the response) and the FINAL
    user message (the most recent thing the model was asked).  Earlier user
    turns in a multi-turn log are deliberately ignored: the response is
    determined by the latest user turn given the system instruction, so two
    records that share (system, last_user) will exercise the candidate
    identically and one of them is enough.

    Uses ``hashlib.sha256`` over a canonical delimited string rather than the
    builtin ``hash()``.  Builtin ``hash()`` on strings is salted per-process by
    ``PYTHONHASHSEED`` (randomized by default since Python 3.3), so the SAME
    prompt would dedup to a DIFFERENT key across two runs of the CLI — silently
    breaking any reproducibility expectation across processes.  sha256 is
    stable across processes, interpreters, and platforms.
    """
    system_msg = next(
        (
            m["content"]
            for m in record.messages
            if m.get("role") == "system" and m.get("content")
        ),
        "",
    )
    last_user = ""
    for m in record.messages:
        if m.get("role") == "user" and m.get("content"):
            last_user = m["content"]
    # Length-prefixing the first component makes the join truly injective —
    # JSON string content can legally carry a NUL (the "\u0000" escape), so a
    # bare NUL join alone could collide two distinct (system, user) pairs.
    canonical = f"{len(system_msg)}\0{system_msg}\0{last_user}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sample_records(
    records: list[LogRecord], n: int, seed: int | None = None
) -> tuple[list[LogRecord], int]:
    """Return up to *n* records sampled by UNIQUE PROMPT CONTENT.

    Returns ``(picked, unique_prompts_available)``:

      * ``picked`` — up to *n* records, each representing a DISTINCT prompt
        (system + final user message).  No two picks carry the same content.
      * ``unique_prompts_available`` — the number of distinct prompts found in
        *records* before sampling, so a caller / renderer can honestly disclose
        when the log has fewer unique prompts than the user asked for.

    Algorithm:
      1. Group ``records`` by ``_dedup_key`` (preserves first-seen group order).
      2. From each group keep the FIRST record in input order — deterministic
         (no first/last ambiguity, reproducible across reruns).
      3. If ``len(unique_groups) <= n``, return ALL unique representatives in
         their first-seen order.  Otherwise sample ``n`` groups uniformly via
         ``random.Random(seed)`` so the same ``(records, n, seed)`` always
         picks the same set of prompts.

    The reproducibility guarantee is preserved: a seeded RNG draws over a
    deterministic candidate list (the unique-representatives list, ordered by
    first appearance in *records*).
    """
    if not records:
        return [], 0
    # Group by content key, preserving first-seen order.  The first record of
    # each group is the representative — deterministic across reruns regardless
    # of how many duplicates trail it.
    representatives: list[LogRecord] = []
    seen_keys: set[str] = set()
    for rec in records:
        key = _dedup_key(rec)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        representatives.append(rec)
    unique_available = len(representatives)
    if unique_available <= n:
        return list(representatives), unique_available
    rng = random.Random(seed)
    return rng.sample(representatives, n), unique_available


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


def _call_model(
    litellm_mod: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    is_baseline: bool = False,
) -> SampledOutput:
    """Call *model* via LiteLLM using the user's own environment keys.

    On failure a friendly cell string is stored in SampledOutput.error so the
    caller can render it in the comparison table.  When *is_baseline* is True
    the cell is prefixed with "baseline unavailable" so it is never blank.

    Pre-flight key checks are done in run_measure before any call reaches here,
    so authentication failures arriving here (e.g. key present but invalid) are
    treated as friendly cells rather than hard exceptions.

    The wire call uses :func:`_route_for_measure` to prepend a provider prefix
    for the new-vendor roster entries that LiteLLM cannot route from a bare
    name — every returned/stored field keeps the ORIGINAL bare *model* name so
    reports and ``--candidates`` matching are unaffected; only the string
    handed to ``litellm.completion`` changes.
    """
    try:
        response = litellm_mod.completion(
            model=_route_for_measure(model), messages=messages
        )
        content: str = response.choices[0].message.content or ""
        pt, ct = _extract_usage(response)
        return SampledOutput(
            model=model,
            content=content,
            usage=MeasureCallUsage(
                model=model, prompt_tokens=pt, completion_tokens=ct
            ),
        )
    except Exception as exc:
        cell = _friendly_cell(exc)
        if is_baseline:
            inner = cell.lstrip("[").rstrip("]")
            # Drop a redundant leading "unavailable — " so the baseline prefix
            # reads cleanly: "[baseline unavailable — ...]" not
            # "[baseline unavailable — unavailable — ...]".
            _REDUNDANT_PREFIX = "unavailable — "
            if inner.startswith(_REDUNDANT_PREFIX):
                inner = inner[len(_REDUNDANT_PREFIX):]
            cell = f"[baseline unavailable — {inner}]"
        return SampledOutput(
            model=model,
            content="",
            error=cell,
            error_cause=_classify_failure(exc),
        )


# ---------------------------------------------------------------------------
# Tier-1 judge
# ---------------------------------------------------------------------------

# The judge is shown two anonymous outputs, OUTPUT A and OUTPUT B, with NO label
# revealing which is the current model and which is the candidate.  That anonymity
# is deliberate: naming one "candidate" primes the judge, and LLMs additionally
# carry a position bias (a mild systematic preference for whichever output is
# shown first/last).  run_measure randomises, per judged pair, which physical
# output occupies slot A vs slot B (see _judge_pair's *candidate_is_a*), then maps
# the A/B verdict back to a candidate-relative win/loss/tie.  Removing both the
# label and the fixed ordering makes the verdict a property of the OUTPUTS, not of
# their presentation.
JUDGE_PROMPT_TEMPLATE = (
    "You are a strict, neutral judge comparing two answers to the same user prompt.\n\n"
    "USER PROMPT:\n{prompt}\n\n"
    "OUTPUT A:\n{output_a}\n\n"
    "OUTPUT B:\n{output_b}\n\n"
    "Judge ONLY on whether each answer is CORRECT and FOLLOWS THE INSTRUCTION.\n"
    "Default to TIE. Choose A or B ONLY when there is a CLEAR, MATERIAL difference:\n"
    "one answer has a factual error, omits key information the prompt required, or\n"
    "fails to follow the instruction. Differences in length, wording, style,\n"
    "formatting, or amount of detail are NOT quality differences — those are a TIE.\n\n"
    'Answer with ONE line and nothing else: "VERDICT: A", "VERDICT: B", or "VERDICT: TIE".\n'
    "No explanation. No parentheses."
)

# Raw A/B/TIE token → the verdict from the candidate's point of view, for each of
# the two presentation orderings.  When the candidate is shown as OUTPUT A, a
# "VERDICT: A" means the candidate won; when it is shown as OUTPUT B, the same
# preference for A means the candidate LOST.  Encoding both orderings in one table
# keeps the inversion explicit and unit-testable rather than buried in branches.
_VERDICT_MAP_CANDIDATE_A = {"A": "win", "B": "loss", "TIE": "tie"}
_VERDICT_MAP_CANDIDATE_B = {"A": "loss", "B": "win", "TIE": "tie"}

# Robust verdict extractor.  The judge is asked for a bare "VERDICT: A|B|TIE"
# line, but real models wrap it: a trailing parenthetical echoed from an older
# prompt ("VERDICT: A (output a is better)"), markdown emphasis
# ("**VERDICT: B**"), or a lowercase reply ("verdict: tie").  A strict
# "token == 'A'" match treated all of those as unparseable and collapsed the
# pair to 'error' (the observed 9/50 error rate).  Instead, scan the WHOLE
# response for the first "VERDICT:" followed by the first alpha token TIE/A/B —
# punctuation, parentheses, and emphasis around it are ignored.  TIE is listed
# first in the alternation so "TIE" is never shortened to a bare "T"/matched as
# the letter before the full word is considered.  Case-insensitive.
_VERDICT_RE = re.compile(r"VERDICT:\s*(TIE|A|B)\b", re.IGNORECASE)

# Number of EXTRA attempts a transient judge-call failure (e.g. a rate-limit
# from the concurrent wave) earns before the pair collapses to 'error'.  One
# retry is the primary resilience fix; a genuine, repeated failure still stays
# neutral (never counted as a loss).  Injectable via _judge_pair(max_retries=…)
# so tests stay fast and deterministic — no wall-clock sleep in the test path.
_JUDGE_MAX_RETRIES = 1
# Fixed backoff (seconds) between judge-call retries.  Tiny and constant — not
# derived from the clock — so it adds no nondeterminism; tests pass 0.0 (or rely
# on max_retries gating) to keep the suite instant.
_JUDGE_RETRY_BACKOFF_S = 0.5


def _parse_verdict(text: str, verdict_map: dict[str, str]) -> str | None:
    """Extract a candidate-relative verdict from a raw judge *text*.

    Returns 'win' / 'loss' / 'tie' (mapped through *verdict_map* for the active
    A/B ordering) when a "VERDICT: A|B|TIE" appears anywhere in *text*, else
    None so the caller can distinguish a genuinely verdict-free reply (→ 'error')
    from a successfully parsed one.  Robust to trailing parentheticals, markdown
    emphasis, and casing — see _VERDICT_RE.
    """
    match = _VERDICT_RE.search(text)
    if match is None:
        return None
    token = match.group(1).upper()
    return verdict_map.get(token)


def _judge_pair(
    litellm_mod: Any,
    judge_model: str,
    messages: list[dict[str, str]],
    current_output: SampledOutput,
    candidate_output: SampledOutput,
    *,
    candidate_is_a: bool = False,
    max_retries: int = _JUDGE_MAX_RETRIES,
    backoff_s: float = _JUDGE_RETRY_BACKOFF_S,
    usage_sink: list[MeasureCallUsage] | None = None,
) -> str:
    """Ask *judge_model* which of two outputs is better, A/B-order randomised.

    *candidate_is_a* controls which physical slot the candidate output occupies:
    when True the candidate is shown as OUTPUT A and the current model as OUTPUT B;
    when False (the default) the current model is OUTPUT A and the candidate is
    OUTPUT B.  The judge sees neither label — only "OUTPUT A"/"OUTPUT B" — so its
    raw A/B preference is free of position-revealing priming.  The returned verdict
    is ALWAYS candidate-relative ('win' = candidate better than current),
    regardless of which slot the candidate occupied: the A/B token is mapped back
    through the ordering-specific table so a WIN shown as A and a WIN shown as B
    both resolve correctly.

    Returns one of: 'win', 'loss', 'tie', 'error'.
    Errors (network failure, unparseable response) collapse to 'error' so a
    single judge failure does not abort the run.

    Transient resilience: a completion call that RAISES (e.g. a provider
    rate-limit triggered by the concurrent judging wave) is retried up to
    *max_retries* extra times with a short fixed *backoff_s* pause between
    attempts before the pair collapses to 'error'.  The retry count and backoff
    are injectable so the test suite can drive both branches (succeed-after-one,
    fail-after-all) instantly and deterministically — there is no clock-derived
    timing.  A successfully-returned but verdict-free reply is NOT retried (it is
    a deterministic parse 'error', not a transient fault); only exceptions retry.

    *usage_sink*, when provided, accumulates one :class:`MeasureCallUsage` per
    judge ``completion()`` that RETURNED (the cost the judge call incurred) — so
    the run can report what the measurement itself cost.  A raised attempt that
    consumed no usage is not recorded; a returned-but-unparseable reply IS
    recorded (it still cost tokens).  Capture never affects the verdict.
    """
    prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    if candidate_is_a:
        output_a, output_b = candidate_output.content, current_output.content
        verdict_map = _VERDICT_MAP_CANDIDATE_A
    else:
        output_a, output_b = current_output.content, candidate_output.content
        verdict_map = _VERDICT_MAP_CANDIDATE_B
    judge_messages = [
        {
            "role": "user",
            "content": JUDGE_PROMPT_TEMPLATE.format(
                prompt=prompt_text,
                output_a=output_a,
                output_b=output_b,
            ),
        }
    ]
    # One call + up to *max_retries* extra attempts on a RAISED transient fault.
    # ``_route_for_measure`` prepends a provider prefix for the new-vendor
    # roster entries (see :data:`_LITELLM_ROUTE_PREFIX`) so a user-chosen
    # ``--judge-model`` from that roster routes correctly; every other judge
    # model (the OpenAI/Anthropic defaults) is returned unchanged.
    attempts = max(0, max_retries) + 1
    for attempt in range(attempts):
        try:
            response = litellm_mod.completion(
                model=_route_for_measure(judge_model), messages=judge_messages
            )
        except Exception:
            # Transient (rate-limit, network blip): back off and retry while
            # attempts remain; collapse to neutral 'error' only once exhausted.
            if attempt + 1 < attempts:
                if backoff_s > 0:
                    time.sleep(backoff_s)
                continue
            return "error"
        # The judge call returned — record its token usage (the cost it incurred),
        # even if the reply turns out to be unparseable below: it still cost tokens.
        if usage_sink is not None:
            pt, ct = _extract_usage(response)
            usage_sink.append(
                MeasureCallUsage(
                    model=judge_model, prompt_tokens=pt, completion_tokens=ct
                )
            )
        text: str = response.choices[0].message.content or ""
        # A returned-but-unparseable reply is a deterministic parse failure, not a
        # transient fault — do NOT retry it; collapse straight to neutral 'error'.
        verdict = _parse_verdict(text, verdict_map)
        return verdict if verdict is not None else "error"
    # Unreachable: the loop always returns.  Present for exhaustiveness.
    return "error"


# ---------------------------------------------------------------------------
# Pointwise "did this answer address the prompt at all" check
# ---------------------------------------------------------------------------
#
# The pairwise judge (_judge_pair) is instructed to "Default to TIE" whenever
# there is no MATERIAL difference between two outputs.  That instruction is
# silent about WHY there was no difference: a TIE covers both "both outputs are
# equally good" and "both outputs equally failed to address the prompt" — the
# latter is a shared failure the pairwise verdict alone hides.  This is a
# SEPARATE, single-response, absolute (non-comparative) check run ONLY when the
# pairwise verdict comes back "tie", to close that blind spot without touching
# the calibrated pairwise preference signal itself.

ADDRESS_PROMPT_TEMPLATE = (
    "You are checking whether an answer engages with a user's prompt at all —\n"
    "NOT how good the answer is, only whether it attempts to address what was\n"
    "asked.\n\n"
    "USER PROMPT:\n{prompt}\n\n"
    "ANSWER:\n{output}\n\n"
    "Does the answer attempt to address the prompt? Answer NO only when the\n"
    "answer is blank, refuses without attempting the task, or is entirely\n"
    "unrelated to the prompt. When in doubt, answer YES.\n\n"
    'Answer with ONE line and nothing else: "ADDRESSED: YES" or "ADDRESSED: NO".\n'
    "No explanation. No parentheses."
)

_ADDRESSED_RE = re.compile(r"ADDRESSED:\s*(YES|NO)\b", re.IGNORECASE)


def _parse_addressed(text: str) -> bool | None:
    """Extract the YES/NO answer from a raw pointwise-check reply.

    Returns True/False when an "ADDRESSED: YES|NO" line appears anywhere in
    *text*, else None so the caller can apply its own honest default for a
    genuinely unparseable reply (see :func:`_judge_addressed`).
    """
    match = _ADDRESSED_RE.search(text)
    if match is None:
        return None
    return match.group(1).upper() == "YES"


def _judge_addressed(
    litellm_mod: Any,
    judge_model: str,
    messages: list[dict[str, str]],
    output_content: str,
    *,
    max_retries: int = _JUDGE_MAX_RETRIES,
    backoff_s: float = _JUDGE_RETRY_BACKOFF_S,
    usage_sink: list[MeasureCallUsage] | None = None,
    fault_sink: list[bool] | None = None,
) -> bool:
    """Ask *judge_model* whether *output_content* attempts to address the prompt.

    Independent of the pairwise judge — used only to check a candidate TIE for
    a SHARED failure (see the module comment above).  Ambiguous/unparseable
    replies and a transient fault that exhausts retries default to True
    (addressed): the check exists to catch an UNAMBIGUOUS double-failure, not
    to second-guess a plausible attempt, so the honest default errs toward NOT
    flagging a false "both failed" that the raw outputs would not support.

    *usage_sink*, when provided, accumulates one :class:`MeasureCallUsage` per
    call that RETURNED — mirroring :func:`_judge_pair`'s cost-disclosure
    contract exactly, so a pointwise check's cost is never silently dropped
    from the run's reported spend.

    *fault_sink*, when provided, gets ``True`` appended ONLY when retries were
    exhausted by a transient fault — never on the honest ambiguous-parse
    default. A silent fault-default could mask a real shared failure as
    "addressed", inflating the judged-success rate with no signal to the
    reader; callers use this to flag :attr:`Tier1Tally.check_errors` so that
    silence is fail-loud instead.
    """
    prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    judge_messages = [
        {
            "role": "user",
            "content": ADDRESS_PROMPT_TEMPLATE.format(
                prompt=prompt_text, output=output_content
            ),
        }
    ]
    attempts = max(0, max_retries) + 1
    for attempt in range(attempts):
        try:
            response = litellm_mod.completion(
                model=_route_for_measure(judge_model), messages=judge_messages
            )
        except Exception:
            if attempt + 1 < attempts:
                if backoff_s > 0:
                    time.sleep(backoff_s)
                continue
            if fault_sink is not None:
                fault_sink.append(True)
            return True  # exhausted retries: honest default, never flag on a fault
        if usage_sink is not None:
            pt, ct = _extract_usage(response)
            usage_sink.append(
                MeasureCallUsage(
                    model=judge_model, prompt_tokens=pt, completion_tokens=ct
                )
            )
        text: str = response.choices[0].message.content or ""
        addressed = _parse_addressed(text)
        return addressed if addressed is not None else True
    # Unreachable: the loop always returns.  Present for exhaustiveness.
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_measure(
    records: list[LogRecord],
    current_model: str,
    candidates: list[str],
    n_samples: int = 5,
    use_judge: bool = False,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_from_log: bool = False,
    seed: int | None = None,
    concurrency: int = _DEFAULT_CONCURRENCY,
    sample_cb: Callable[[int, int, str], None] | None = None,
    judge_cb: Callable[[int, int, str], None] | None = None,
) -> MeasureResult:
    """Sample *n_samples* records and compare outputs across *current_model* + *candidates*.

    Privacy contract: all network egress goes exclusively through LiteLLM to
    the user's own provider endpoints.  No data is forwarded to any Rodiun or
    Frugon host.  API keys are read from the user's environment by LiteLLM.

    Args:
        records:       Parsed log records to sample from.
        current_model: The model that produced the existing logs.
        candidates:    Model names to evaluate as routing candidates.
        n_samples:     Number of prompts to sample (default 5).
        use_judge:     When True, a judge model scores each comparison (Tier-1).
        judge_model:   Model to use as judge when use_judge=True.  Defaults to
                       DEFAULT_JUDGE_MODEL (gpt-4.1) — the last-resort arm's-length
                       fallback (see the constant's docstring for the full
                       precedence), independent of the typical candidate, so the
                       judge does not grade its own output.  When the resolved
                       judge IS one of the models
                       it scores, the offending names are surfaced in
                       MeasureResult.self_judged_models (a non-blocking caution).
        judge_from_log: True when *judge_model* was auto-selected as the highest
                       quality-tier model present in the user's own log (the CLI's
                       best_judge_from_log path), as opposed to an explicit
                       --judge-model or the DEFAULT_JUDGE_MODEL fallback.  Recorded
                       on MeasureResult.judge_from_log so the renderer can honestly
                       call it "your highest-tier model" rather than "(independent)".
        seed:          RNG seed for reproducible sampling.  Also seeds the
                       per-pair A/B order randomisation the judge sees, so the
                       same seed yields the same (debiased) A/B layout each run.
        concurrency:   Max concurrent provider calls PER STAGE.  Drives the two
                       independent pools: the sampling stage runs
                       ``min(concurrency, n_prompts)`` workers (fanning WIDE
                       across the baseline + candidate provider endpoints), and
                       the judging stage runs
                       ``min(concurrency, _JUDGE_MAX_CONCURRENCY)`` workers
                       (NARROWER, to protect the single judge endpoint).  Because
                       the two stages overlap, peak provider round-trips ≈
                       sample_workers + judge_workers by design.  ``concurrency=1``
                       collapses both stages to one worker — a fully sequential,
                       deterministic path (the parity reference).  Defaults to 5.
        sample_cb:     Optional ``(prompt_done, n_prompts, label)`` callback
                       fired ONCE PER SAMPLED PROMPT, as that prompt's sampling
                       begins, where *prompt_done* counts prompts already
                       completed (so it advances 0, 1, 2, ... as each prompt
                       starts), *n_prompts* is the number of prompts sampled
                       (matching the "N prompt(s)" header), and *label* names the
                       candidate model(s) being compared.  The underlying work is
                       unchanged -- each prompt still hits the baseline plus every
                       candidate -- but the unit shown to the user is PROMPTS,
                       uniform with the header.  ``None`` is the default.
        judge_cb:      Optional ``(prompt_done, n_prompts, label)`` callback fired
                       ONCE PER SAMPLED PROMPT when *use_judge* is True, as that
                       prompt's judging begins -- the same prompt-counting unit as
                       *sample_cb*.

    Raises:
        MissingProviderKeyError: when a required API key is absent.
        ImportError:             when the [measure] extra is not installed.
    """
    litellm_mod = _import_litellm()

    # Pre-flight: verify all required API keys are present before any network call.
    all_models = [current_model, *candidates]
    if use_judge:
        all_models.append(judge_model)
    _check_provider_keys(all_models)

    sampled, unique_prompts_available = sample_records(records, n_samples, seed=seed)

    # Self-judge detection: when the judge IS one of the models it scores (a
    # candidate, or the baseline), its verdict is partly a self-assessment and
    # may be biased.  Recorded (not blocking) so the CLI can warn.  Order:
    # baseline first, then candidates in their given order, de-duplicated.
    self_judged: list[str] = []
    if use_judge:
        for m in [current_model, *candidates]:
            if m == judge_model and m not in self_judged:
                self_judged.append(m)

    # Progress is reported in PROMPTS — the same unit as the "N prompt(s)"
    # header — not in provider calls.  Each prompt still hits the baseline plus
    # every candidate underneath; only the unit shown to the user is the prompt.
    n_prompts = len(sampled)
    # A single label naming the candidate(s) under comparison, e.g.
    # "gpt-4o-mini" or "gpt-4o-mini, claude-3-haiku-20240307".
    candidate_label = ", ".join(candidates)
    n_candidates = len(candidates)

    # ---- Two-stage producer→consumer pipeline -------------------------------
    #
    # Stage 1 (sampling pool) is the PRODUCER: every individual sampling call —
    # each (prompt, model) for the baseline and for each candidate — is submitted
    # as its OWN task, so all sampling calls compete for one bounded pool.  This
    # is the key difference from the old per-prompt shape: a prompt's baseline and
    # its candidates no longer run back-to-back inside one task, so sampling fans
    # WIDE across the multiple provider endpoints (baseline + candidates) up to
    # ``sample_workers`` calls in flight — not throttled by per-prompt chaining.
    #
    # Stage 2 (judging pool) is the CONSUMER: the MOMENT a prompt's full sample
    # set (baseline + ALL candidates) has resolved, that prompt's judge call(s)
    # are submitted to a SEPARATE, narrower pool, which drains CONCURRENTLY while
    # stage 1 is still sampling other prompts — true overlap.
    #
    # Hand-off: we iterate stage-1 completions with as_completed and keep a
    # per-prompt outstanding-sample counter; when a prompt's counter reaches 0 we
    # immediately submit its judge future(s) to stage 2.  After stage 1 fully
    # drains we await every stage-2 future.
    #
    # Deadlock-free by construction: the two stages are SEPARATE pools, and a
    # stage-1 task never blocks waiting on a stage-2 task (it only does a sampling
    # call and returns).  The hand-off submit happens on the main thread as it
    # consumes stage-1 completions, so no stage-1 worker is ever parked behind a
    # stage-2 worker.  Worst case every worker runs an independent provider call
    # — they all make progress.
    #
    # Correctness is identical to the old shapes: results are written into
    # per-prompt slots keyed by sampled index and assembled in sampled-record
    # order regardless of completion order; the per-pair A/B layout is drawn from
    # the same seeded RNG in the same (prompt-major, candidate-minor) order, so a
    # given seed yields byte-identical orderings and therefore verdicts.
    current_outs: list[SampledOutput | None] = [None] * n_prompts
    candidate_outs_by_prompt: list[list[SampledOutput | None]] = [
        [None] * n_candidates for _ in range(n_prompts)
    ]
    verdicts_by_prompt: list[list[str]] = [[] for _ in range(n_prompts)]
    # Parallel to verdicts_by_prompt; set True ONLY for a "tie" verdict whose
    # pointwise check found neither side addressed the prompt (see the module
    # comment above _judge_addressed).  Stays False for every other verdict.
    both_failed_by_prompt: list[list[bool]] = [[] for _ in range(n_prompts)]
    # Parallel to both_failed_by_prompt; set True ONLY for a "tie" whose
    # pointwise both-failed determination relied on a _judge_addressed call
    # that exhausted its retries (fault-defaulted to "addressed" rather than
    # a genuine parsed answer) -- see Tier1Tally.check_errors.
    check_fault_by_prompt: list[list[bool]] = [[] for _ in range(n_prompts)]

    # Pre-draw the full A/B layout BEFORE any task runs, from a seeded RNG that is
    # independent of (but derived from) the sampling seed.  Drawing it up front in
    # the original prompt-major / candidate-minor order keeps the layout identical
    # to the previous code for a given seed — the concurrent tasks only READ their
    # own slot, they never advance the RNG — so reproducibility is preserved.
    candidate_is_a: list[list[bool]] = []
    if use_judge and n_prompts:
        order_rng = random.Random(None if seed is None else seed + 1)
        candidate_is_a = [
            [order_rng.random() < 0.5 for _ in range(n_candidates)]
            for _ in range(n_prompts)
        ]
        for row in verdicts_by_prompt:
            row.extend([""] * n_candidates)
        for bf_row in both_failed_by_prompt:
            bf_row.extend([False] * n_candidates)
        for cf_row in check_fault_by_prompt:
            cf_row.extend([False] * n_candidates)

    # Progress callbacks fire as each prompt finishes its sampling / judging.  A
    # lock makes the fire atomic; the reported counter is taken under the lock so
    # it is monotonic non-decreasing even though prompts complete out of order.
    # Each callback still fires exactly n_prompts times (once per prompt).
    _cb_lock = threading.Lock()
    _sample_done_count = 0
    _judge_done_count = 0

    def _fire_sample_cb() -> None:
        if sample_cb is None:
            return
        nonlocal _sample_done_count
        with _cb_lock:
            done = _sample_done_count
            _sample_done_count += 1
            sample_cb(done, n_prompts, candidate_label)

    def _fire_judge_cb() -> None:
        if judge_cb is None:
            return
        nonlocal _judge_done_count
        with _cb_lock:
            done = _judge_done_count
            _judge_done_count += 1
            judge_cb(done, n_prompts, candidate_label)

    # A stage-1 task does ONE sampling call and writes it into its slot.  c_idx is
    # None for the baseline, else the candidate index.  Returns p_idx so the
    # hand-off loop can decrement that prompt's outstanding-sample counter.
    def _sample_one(p_idx: int, c_idx: int | None, model: str, messages: list[dict[str, str]]) -> int:
        out = _call_model(litellm_mod, model, messages, is_baseline=(c_idx is None))
        if c_idx is None:
            current_outs[p_idx] = out
        else:
            candidate_outs_by_prompt[p_idx][c_idx] = out
        return p_idx

    # Judge-call usage is accumulated across the concurrent judge pool.  Each
    # _judge_prompt collects into a thread-LOCAL list (no contention on the hot
    # path), then merges into this shared list once under _cb_lock — so the final
    # measure_calls total is complete and race-free regardless of completion order.
    judge_usage: list[MeasureCallUsage] = []

    # A stage-2 task judges EVERY candidate of ONE prompt (its full sample set is
    # guaranteed resolved before this is submitted) and fires the judge callback.
    def _judge_prompt(p_idx: int) -> None:
        cur_out = current_outs[p_idx]
        assert cur_out is not None  # the prompt's baseline resolved before submit
        local_usage: list[MeasureCallUsage] = []
        # A sampling error on EITHER side is neutral — never a candidate-quality
        # loss.  The errored slot has no real content (only a friendly cell
        # like '[unavailable — project lacks access to this model]'); sending
        # that string to the judge as OUTPUT would have it correctly pick the
        # other side, miscounting a sampling failure as a candidate LOSS.  We
        # short-circuit to the neutral 'error' verdict and skip the judge
        # call entirely (saves the API call + its cost).  A baseline-errored
        # prompt marks EVERY candidate 'error' since the comparison is
        # impossible — not a per-candidate quality signal.  The neutral
        # 'error' token is the same one _judge_pair returns when the judge
        # itself fails after retry, and the downstream tally already routes it
        # to the errors column, so the unified treatment is correct.
        baseline_errored = cur_out.error is not None
        # Lazily computed and cached across this prompt's candidates: the
        # baseline output is the SAME for every candidate on this prompt, so a
        # baseline that ties against more than one candidate only needs ONE
        # pointwise check, not one per tie.  None means "not yet computed" —
        # distinct from a real False/True result.
        baseline_addressed: bool | None = None
        baseline_check_fault = False
        for c_idx in range(n_candidates):
            cand_out = candidate_outs_by_prompt[p_idx][c_idx]
            assert cand_out is not None  # every candidate resolved before submit
            if baseline_errored or cand_out.error is not None:
                verdicts_by_prompt[p_idx][c_idx] = "error"
                continue
            verdict = _judge_pair(
                litellm_mod,
                judge_model,
                sampled[p_idx].messages,
                cur_out,
                cand_out,
                candidate_is_a=candidate_is_a[p_idx][c_idx],
                usage_sink=local_usage,
            )
            verdicts_by_prompt[p_idx][c_idx] = verdict
            # Pointwise "both failed" check — ONLY on a TIE, where the pairwise
            # judge's verdict is silent about whether the tie is two equally
            # GOOD outputs or two equally FAILED ones (see the module comment
            # above _judge_addressed).
            if verdict == "tie":
                if baseline_addressed is None:
                    baseline_fault_sink: list[bool] = []
                    baseline_addressed = _judge_addressed(
                        litellm_mod,
                        judge_model,
                        sampled[p_idx].messages,
                        cur_out.content,
                        usage_sink=local_usage,
                        fault_sink=baseline_fault_sink,
                    )
                    baseline_check_fault = bool(baseline_fault_sink)
                if baseline_addressed:
                    # The baseline already addressed the prompt, so "both
                    # failed" is impossible regardless of the candidate --
                    # skip the candidate check entirely (halves the
                    # pointwise-check calls whenever the baseline holds up).
                    both_failed_by_prompt[p_idx][c_idx] = False
                    if baseline_check_fault:
                        # The baseline's "addressed" came from a fault
                        # default, not a genuine parsed answer -- the skip
                        # above may be masking a real shared failure.
                        check_fault_by_prompt[p_idx][c_idx] = True
                else:
                    candidate_fault_sink: list[bool] = []
                    candidate_addressed = _judge_addressed(
                        litellm_mod,
                        judge_model,
                        sampled[p_idx].messages,
                        cand_out.content,
                        usage_sink=local_usage,
                        fault_sink=candidate_fault_sink,
                    )
                    both_failed_by_prompt[p_idx][c_idx] = not candidate_addressed
                    if candidate_fault_sink:
                        check_fault_by_prompt[p_idx][c_idx] = True
        if local_usage:
            with _cb_lock:
                judge_usage.extend(local_usage)
        _fire_judge_cb()

    if n_prompts:
        sample_workers, judge_workers = _stage_worker_counts(concurrency, n_prompts)
        # Per-prompt count of sampling calls still outstanding.  A prompt is ready
        # to judge (or to fire its sample_cb) the instant this hits zero.  Guarded
        # by _cb_lock so the decrement-and-test is atomic under concurrent
        # stage-1 completions.  Each prompt has 1 baseline + n_candidates calls.
        samples_per_prompt = 1 + n_candidates
        outstanding = [samples_per_prompt] * n_prompts

        # Two SEPARATE pools — a stage-1 task can never block on a stage-2 task.
        with (
            ThreadPoolExecutor(max_workers=sample_workers) as sample_pool,
            ThreadPoolExecutor(max_workers=judge_workers) as judge_pool,
        ):
            # Stage 1: submit EVERY individual sampling call up front so they all
            # compete for the sampling pool (max sampling fan-out, not gated by
            # per-prompt chaining).  Order of submission is irrelevant to
            # correctness — every result lands in its own indexed slot.
            sample_futures: list[Future[int]] = []
            for p_idx, record in enumerate(sampled):
                sample_futures.append(
                    sample_pool.submit(_sample_one, p_idx, None, current_model, record.messages)
                )
                for c_idx, cand in enumerate(candidates):
                    sample_futures.append(
                        sample_pool.submit(_sample_one, p_idx, c_idx, cand, record.messages)
                    )

            # Consume stage-1 completions as they land.  When a prompt's last
            # sampling call resolves, fire its sample_cb and — if judging — hand
            # its judge task to stage 2 IMMEDIATELY, so judging overlaps the
            # sampling of still-pending prompts.
            judge_futures: list[Future[None]] = []
            for fut in as_completed(sample_futures):
                # .result() re-raises any genuine programming bug from the task
                # (provider faults are swallowed into friendly cells upstream).
                p_idx = fut.result()
                ready = False
                with _cb_lock:
                    outstanding[p_idx] -= 1
                    if outstanding[p_idx] == 0:
                        ready = True
                if ready:
                    _fire_sample_cb()
                    if use_judge:
                        judge_futures.append(judge_pool.submit(_judge_prompt, p_idx))

            # Stage 1 has fully drained; await stage 2.  Re-raise any bug.
            for jfut in judge_futures:
                jfut.result()

    # ---- Assemble results in sampled-record order ----------------------------
    tallies: dict[str, Tier1Tally] = {c: Tier1Tally(candidate=c) for c in candidates}
    comparisons: list[Comparison] = []
    for p_idx, record in enumerate(sampled):
        cur_out = current_outs[p_idx]
        assert cur_out is not None
        # Wave 1 populated every candidate slot for this prompt; narrow the
        # Optional list for mypy and rebuild a plain list[SampledOutput].
        cand_outs_checked: list[SampledOutput] = [
            o for o in candidate_outs_by_prompt[p_idx] if o is not None
        ]
        verdicts = verdicts_by_prompt[p_idx] if use_judge else []
        both_failed = both_failed_by_prompt[p_idx] if use_judge else []
        check_fault = check_fault_by_prompt[p_idx] if use_judge else []
        if use_judge:
            for c_idx, cand in enumerate(candidates):
                verdict = verdicts[c_idx]
                tally = tallies[cand]
                if verdict == "win":
                    tally.wins += 1
                elif verdict == "loss":
                    tally.losses += 1
                elif verdict == "tie":
                    tally.ties += 1
                    if both_failed[c_idx]:
                        tally.both_failed_ties += 1
                    if check_fault[c_idx]:
                        tally.check_errors += 1
                else:
                    tally.errors += 1
        comparisons.append(
            Comparison(
                record=record,
                current_output=cur_out,
                candidate_outputs=cand_outs_checked,
                verdicts=list(verdicts),
                both_failed=list(both_failed),
            )
        )

    tier1: list[Tier1Tally] | None = list(tallies.values()) if use_judge else None

    # ---- Aggregate what the MEASURE RUN itself cost ---------------------------
    # One MeasureCallUsage per provider call the run made: the baseline + every
    # candidate sampling call (from each SampledOutput's captured usage), then the
    # judge calls.  A sampling call that failed before returning usage
    # (usage is None) still counts as a call with zero tokens — honest about the
    # call count without inventing tokens it did not spend.  Order is deterministic
    # (sampled-record order, baseline-then-candidates, then judge) so a seeded run
    # produces a stable list.
    measure_calls: list[MeasureCallUsage] = []
    for p_idx in range(n_prompts):
        cur_out = current_outs[p_idx]
        assert cur_out is not None
        measure_calls.append(
            cur_out.usage
            or MeasureCallUsage(
                model=cur_out.model, prompt_tokens=0, completion_tokens=0
            )
        )
        for c_idx in range(n_candidates):
            cand_out = candidate_outs_by_prompt[p_idx][c_idx]
            assert cand_out is not None
            measure_calls.append(
                cand_out.usage
                or MeasureCallUsage(
                    model=cand_out.model, prompt_tokens=0, completion_tokens=0
                )
            )
    measure_calls.extend(judge_usage)

    return MeasureResult(
        samples_requested=n_samples,
        samples_taken=len(sampled),
        unique_prompts_available=unique_prompts_available,
        current_model=current_model,
        candidates=candidates,
        comparisons=comparisons,
        tier1_tallies=tier1,
        judge_model=judge_model if use_judge else None,
        self_judged_models=self_judged,
        judge_from_log=judge_from_log if use_judge else False,
        measure_calls=measure_calls,
    )


# ---------------------------------------------------------------------------
# Measurement-cost accounting — what the measure run itself cost the user
# ---------------------------------------------------------------------------


@dataclass
class MeasurementCost:
    """The exact dollar cost of a measure run, priced via frugon's own table.

    ``total_cost`` is summed from each call's ``(prompt_tokens × input price) +
    (completion_tokens × output price)`` using :func:`frugon.pricing.get_model_price`
    — the SAME instrument that prices the user's log, so the figure is exact, not
    guessed.  ``call_count`` is every provider call the run made (priced or not).
    ``unpriced_calls`` counts calls whose model is absent from the pricing table:
    those are flagged, never silently estimated, so the total is honest about
    what it could and could not price.
    """

    total_cost: Decimal
    call_count: int
    unpriced_calls: int


def measurement_cost(measure_result: MeasureResult) -> MeasurementCost | None:
    """Price what *measure_result*'s run cost the user, or ``None`` when unknown.

    Returns ``None`` when no per-call usage was captured (e.g. a MeasureResult
    constructed directly in a test, or a run that made no calls) so the caller
    simply omits the cost line.  Otherwise every captured call is priced through
    :func:`frugon.pricing.get_model_price`; a call whose model is not in the
    table contributes 0 to the total and increments ``unpriced_calls`` (it is
    flagged, never guessed).  Pure local arithmetic — no network, no provider
    call.
    """
    from decimal import Decimal

    from frugon.pricing import get_model_price

    if not measure_result.measure_calls:
        return None

    total = Decimal("0")
    unpriced = 0
    # Memoise per-model lookups across the (often many) calls of one run — the
    # pricing layer already caches, but this avoids repeated dict round-trips.
    price_cache: dict[str, ModelPrice | None] = {}
    for call in measure_result.measure_calls:
        if call.model not in price_cache:
            price_cache[call.model] = get_model_price(call.model)
        price = price_cache[call.model]
        if price is None:
            unpriced += 1
            continue
        total += price.input_cost_per_token * Decimal(call.prompt_tokens)
        total += price.output_cost_per_token * Decimal(call.completion_tokens)

    return MeasurementCost(
        total_cost=total,
        call_count=len(measure_result.measure_calls),
        unpriced_calls=unpriced,
    )


# ---------------------------------------------------------------------------
# Pre-run estimate — what a measure run is ABOUT to cost (before it starts)
# ---------------------------------------------------------------------------

# Call count above which the CLI surfaces the pre-run estimate (and, on a TTY,
# asks to proceed).  Below it, small runs stay frictionless — no estimate, no
# prompt.  30 is a deliberate floor: a default 10-sample single-candidate judge
# run is 10×(1+1) + 10×1 = 30 calls, so the *default* run stays silent; only
# a materially larger run (more samples, more candidates) trips the gate.
_ESTIMATE_CALL_THRESHOLD = 30

# Estimated tokens a single judge REPLY costs — the judge is instructed to answer
# with one short "VERDICT: A|B|TIE" line, so its completion is a handful of
# tokens regardless of input size.  A small fixed constant keeps the estimate
# honest without pretending to know the exact reply length in advance.  The
# pointwise "both failed" check's reply ("ADDRESSED: YES|NO") is the same
# shape — one short line — so this constant is reused for it too.
_JUDGE_REPLY_TOKENS_EST = 8

# Fixed token overhead of ADDRESS_PROMPT_TEMPLATE's own wording — everything in
# the template EXCEPT the interpolated {prompt} and {output}, which are already
# priced from the record's own token counts.  ~479 characters of fixed
# instruction text, rounded up from a 4-chars/token estimate (~120) to a round
# number that stays a genuine upper bound rather than an exact-but-fragile
# count that drifts the moment the template's wording changes.
_ADDRESS_TEMPLATE_OVERHEAD_TOKENS_EST = 130


@dataclass
class MeasureEstimate:
    """A pre-run projection of a measure run's call count and dollar cost.

    ``planned_calls`` is the exact number of provider calls the run will make:
    ``n_prompts × (1 + n_candidates)`` sampling calls, plus
    ``n_prompts × n_candidates`` judge calls when a judge runs.  ``estimated_cost``
    is the projected USD spend, priced from the SAMPLED records' own token counts
    (prompt tokens in, the log's completion tokens as the expected output size)
    against frugon's pricing table for each target model — or ``None`` when none
    of the target models could be priced (the call count is still meaningful).
    ``unpriced_models`` names target models absent from the pricing table, so the
    estimate is honest about which legs it could not price.

    ``n_prompts`` and ``n_candidates`` are the EXACT counts the plan was computed
    from (``n_prompts`` is capped to the available records, so it can be smaller
    than the requested ``--samples``), and ``use_judge`` records whether a judge
    leg is included.  The CLI renders its transparent ``About to make …`` line
    straight from these three numbers, so the arithmetic the user reads always
    reconciles to ``planned_calls`` exactly — never to a requested sample count
    that silently exceeded the records available.

    ``max_check_calls`` is the WORST-CASE count of extra pointwise "both failed"
    calls (see :func:`_judge_addressed`): unlike the sampling/judge legs, these
    calls are conditional on the pairwise judge actually returning a TIE, which
    is data-dependent and unknowable before the run — so this is deliberately an
    upper bound, NOT folded into the exact ``planned_calls`` total.  It is 0
    whenever ``use_judge`` is False (a TIE, and therefore the pointwise check,
    is impossible without a judge).  Defaults to 0 so existing direct-
    construction callers/tests are unaffected.

    ``max_check_cost`` is the matching WORST-CASE EXTRA dollar cost of those
    same ``max_check_calls`` calls — priced at the judge model, the same way
    the exact legs are priced, over the same sampled records.  It is ``None``
    whenever the ceiling cannot be honestly stated: no judge, no judge model,
    or the judge model itself unpriced (never fabricate a figure the pricing
    table cannot support).  When not ``None``, the true worst-case run cost is
    ``estimated_cost + max_check_cost`` — never displayed as a stand-alone
    figure, only ever added to ``estimated_cost`` to form the printed ceiling.
    Defaults to ``None`` so existing direct-construction callers/tests are
    unaffected.
    """

    planned_calls: int
    estimated_cost: Decimal | None
    unpriced_models: list[str]
    n_prompts: int
    n_candidates: int
    use_judge: bool
    max_check_calls: int = 0
    max_check_cost: Decimal | None = None


def planned_call_count(
    n_prompts: int, n_candidates: int, *, use_judge: bool
) -> int:
    """Return the exact number of provider calls a measure run will make.

    ``n_prompts × (1 + n_candidates)`` sampling calls (baseline + each candidate
    per prompt) plus, when *use_judge*, ``n_prompts × n_candidates`` judge calls
    (one per candidate per prompt).  Pure arithmetic — the single source of the
    planned-call formula, shared by the estimate and any caller that needs the
    count alone.
    """
    sampling = n_prompts * (1 + n_candidates)
    judging = n_prompts * n_candidates if use_judge else 0
    return sampling + judging


def max_check_call_count(n_prompts: int, n_candidates: int, *, use_judge: bool) -> int:
    """Return the WORST-CASE count of extra pointwise "both failed" check calls.

    The pointwise check (:func:`_judge_addressed`) only runs when the pairwise
    judge returns "tie" for a given (prompt, candidate) — a data-dependent
    outcome unknowable before the run.  The upper bound assumes EVERY candidate
    on EVERY prompt ties: for one prompt that is 1 baseline check (cached and
    shared across that prompt's candidates — see :func:`run_measure`) plus one
    check per candidate, i.e. ``1 + n_candidates`` — the SAME per-prompt shape
    as a sampling call, summed over ``n_prompts``.  Zero when *use_judge* is
    False (no judge means no TIE, means no pointwise check can ever fire).
    """
    return n_prompts * (1 + n_candidates) if use_judge else 0


def estimate_measure_cost(
    records: list[LogRecord],
    current_model: str,
    candidates: list[str],
    n_samples: int,
    *,
    use_judge: bool = False,
    judge_model: str | None = None,
    seed: int | None = None,
) -> MeasureEstimate:
    """Project what a measure run will cost BEFORE any call is made.

    Samples up to *n_samples* records (the same way :func:`run_measure` does) and
    prices the planned calls from those records' OWN token counts:

      * each sampling call (baseline + each candidate) is priced as the record's
        ``prompt_tokens`` in and its ``completion_tokens`` out, at the target
        model's per-token price;
      * each judge call is priced with an input of ``prompt_tokens`` plus BOTH
        outputs (baseline + candidate, proxied by the record's completion tokens
        for each) and a small fixed reply size (the judge answers one line);
      * each WORST-CASE pointwise "both failed" check call (see
        :func:`max_check_call_count`) is priced at the judge model too, with an
        input of ``prompt_tokens`` plus ONE output plus the fixed
        ``ADDRESS_PROMPT_TEMPLATE`` wording overhead, and the same small fixed
        reply size — accumulated into ``MeasureEstimate.max_check_cost``, the
        EXTRA dollars on top of ``estimated_cost`` if every judged pair ties.

    The completion-token count of the LOG record is used as the expected output
    size for the candidate models too — a proxy, since the candidate's real reply
    length is unknown until the call is made; this is why the figure is an
    ESTIMATE (the CLI marks it ``~``).  A target model missing from the pricing
    table contributes nothing to the total and is named in ``unpriced_models``;
    when EVERY priced leg is unpriced the estimate's ``estimated_cost`` is
    ``None`` (the call count is still returned).  ``max_check_cost`` is ``None``
    whenever the judge model itself is unpriced — the check leg's cost cannot be
    honestly derived without a priced judge, so the ceiling is omitted rather
    than understated.  Pure local arithmetic — no network, no provider call.
    """
    from decimal import Decimal

    from frugon.pricing import get_model_price

    # Estimate samples THE SAME WAY run_measure does (unique-prompt dedup) so
    # the projected call count matches what will actually be made.  The second
    # element (unique_prompts_available) is not surfaced in the estimate today
    # — the estimate is a "what will this cost" figure, not an honesty notice
    # — but threading the same path keeps the two in lockstep.
    sampled, _unique_available = sample_records(records, n_samples, seed=seed)
    n_prompts = len(sampled)
    n_candidates = len(candidates)
    planned = planned_call_count(n_prompts, n_candidates, use_judge=use_judge)
    max_checks = max_check_call_count(n_prompts, n_candidates, use_judge=use_judge)

    # Resolve and cache each target model's price once.  None marks an unpriced
    # model — its legs contribute nothing and the name is surfaced.
    target_models = [current_model, *candidates]
    if use_judge and judge_model is not None:
        target_models.append(judge_model)
    prices: dict[str, ModelPrice | None] = {}
    unpriced: list[str] = []
    seen: set[str] = set()
    for model in target_models:
        if model in seen:
            continue
        seen.add(model)
        price = get_model_price(model)
        prices[model] = price
        if price is None:
            unpriced.append(model)

    def _leg_cost(model: str, in_tokens: int, out_tokens: int) -> Decimal:
        price = prices.get(model)
        if price is None:
            return Decimal("0")
        return (
            price.input_cost_per_token * Decimal(in_tokens)
            + price.output_cost_per_token * Decimal(out_tokens)
        )

    total = Decimal("0")
    max_check_total = Decimal("0")
    for rec in sampled:
        pt = max(0, rec.prompt_tokens)
        ct = max(0, rec.completion_tokens)
        # Baseline sampling call.
        total += _leg_cost(current_model, pt, ct)
        # Candidate sampling calls.
        for cand in candidates:
            total += _leg_cost(cand, pt, ct)
        # Judge calls: input ≈ prompt + both outputs (baseline + candidate); the
        # judge reply is a short fixed line.
        if use_judge and judge_model is not None:
            judge_in = pt + ct + ct  # prompt + baseline output + candidate output
            for _cand in candidates:
                total += _leg_cost(judge_model, judge_in, _JUDGE_REPLY_TOKENS_EST)

            # WORST-CASE pointwise "both failed" check calls (see
            # max_check_call_count): 1 baseline check (shared/cached across this
            # prompt's candidates) + 1 check per candidate, the SAME per-prompt
            # shape max_check_call_count sums over n_prompts — so the priced
            # call count here reconciles with max_checks exactly.  Each check is
            # a single-output pointwise call: input ≈ prompt + that output +
            # the ADDRESS_PROMPT_TEMPLATE's own fixed wording; reply is the
            # same short fixed line as the pairwise judge.
            check_in = pt + ct + _ADDRESS_TEMPLATE_OVERHEAD_TOKENS_EST
            check_call_cost = _leg_cost(judge_model, check_in, _JUDGE_REPLY_TOKENS_EST)
            max_check_total += check_call_cost * (1 + len(candidates))

    # estimated_cost is None only when NOTHING could be priced — then the dollar
    # figure is meaningless and the CLI shows "est. unavailable" while still
    # surfacing the call count.
    all_priced_unknown = all(prices[m] is None for m in prices) if prices else True
    estimated: Decimal | None = None if all_priced_unknown else total

    # max_check_cost is only ever stated when the judge itself is priced — an
    # unpriced judge means the check leg's dollar figure cannot be honestly
    # derived, so the ceiling is OMITTED entirely (never understated, never
    # fabricated) rather than silently excluding just the check leg's cost.
    judge_priced = (
        use_judge and judge_model is not None and prices.get(judge_model) is not None
    )
    max_check_cost: Decimal | None = max_check_total if judge_priced else None

    return MeasureEstimate(
        planned_calls=planned,
        estimated_cost=estimated,
        unpriced_models=unpriced,
        n_prompts=n_prompts,
        n_candidates=n_candidates,
        use_judge=use_judge,
        max_check_calls=max_checks,
        max_check_cost=max_check_cost,
    )
