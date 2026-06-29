"""frugon pricing module.

Pricing precedence rule (deterministic):
  1. Our pricing table wins when the model is present there.
  2. Fall back to tokencost TOKEN_COSTS for every model we don't carry.

This means a user can pin specific prices without being overridden by a
tokencost library update, while getting broad model coverage for free from
tokencost for everything else.

The ``pricing.json`` ``_last_synced`` date is exposed in every lookup result
so a stale snapshot is visible to callers and report renderers.

Writable pricing table location
---------------------------------
The live pricing table is stored in the user data directory
(``~/.local/share/frugon/pricing.json`` on Linux/macOS,
``%LOCALAPPDATA%\\frugon\\pricing.json`` on Windows via platformdirs).

The wheel-bundled file at ``src/frugon/data/pricing.json`` is a read-only
seed used only when no user-dir file exists yet.  ``frugon pricing update``
always writes to the user data directory, so a reinstall or upgrade never
silently reverts prices the user has synced.

First-run behaviour
-------------------
On the first call to ``load_pricing_override()`` after a fresh install (or
after clearing the user data dir), the module copies the bundled seed to
the user data directory so subsequent writes have a sane starting point.
"""

from __future__ import annotations

import functools
import json
import re
import urllib.error
from datetime import date as _date
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple

from frugon import USER_AGENT
from frugon._store import (
    atomic_write_json,
    fetch_url_with_retry,
    load_json_or_empty,
    seed_if_missing,
    validate_fetch_url,
)
from frugon.model_id import base_family, canonicalize

try:
    import platformdirs as _platformdirs  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "platformdirs is required. Install it: pip install platformdirs"
    ) from exc

try:
    import tokencost as _tc  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tokencost is required for pricing. Install it: pip install tokencost"
    ) from exc

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# The wheel-bundled read-only seed.  Never written to by this module.
_BUNDLED_SEED_PATH: Path = Path(__file__).parent / "data" / "pricing.json"

# The writable user-data-dir path.  ``frugon pricing update`` (via cli.py)
# imports ``_PRICING_JSON`` and passes it as ``output_path``; keeping this
# name stable means no CLI changes are required.
_PRICING_JSON: Path = Path(_platformdirs.user_data_dir("frugon")) / "pricing.json"

_LITELLM_REGISTRY_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

_ALLOWED_PRICING_HOSTS: frozenset[str] = frozenset({"raw.githubusercontent.com"})

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024  # 16 MB cap

# Retry parameters for the registry fetch.  raw.githubusercontent.com returns
# sporadic 5xx under load, so a single bad response must not fail the whole
# update — retry on HTTP 429, HTTP 5xx, and transient URLError/OSError with
# exponential backoff.  Matched to the quality fetcher's budget for parity.
_FETCH_MAX_RETRIES: int = 4
_FETCH_BACKOFF_BASE: float = 1.0  # seconds; doubles each attempt: 1, 2, 4, 8


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ModelPrice(NamedTuple):
    """Resolved price for a single model."""

    model: str
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    source: str  # "pricing.json" or "tokencost"
    pricing_json_last_synced: str | None  # ISO date from pricing.json, or None


# ---------------------------------------------------------------------------
# Core read path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Override-table cache
# ---------------------------------------------------------------------------
#
# Parsing pricing.json is pure I/O + JSON decode + a full-table filter pass.
# The cost engine calls ``get_model_price`` once per log record, so on a large
# log (tens of thousands of calls) re-reading and re-parsing the same file every
# time dominates the run — profiling a 56k-record log showed ~140s spent almost
# entirely in ``load_pricing_override`` (JSON decode + the filter loop), versus
# ~2s of actual record parsing.
#
# The table only changes when ``frugon pricing update`` rewrites the file, so we
# cache the parsed result keyed on the resolved path together with its mtime and
# size.  A mid-process update (a new file with a different mtime/size) is picked
# up automatically; within a single analysis run the file is read exactly once.
# ``clear_pricing_cache`` resets all caches for tests that swap the table out
# from under us via monkeypatching.

_OverrideTable = tuple[dict[str, dict[str, object]], str | None]
_CacheKey = tuple[str, str, int, int]  # (override_path, seed_path, mtime_ns, size)

# Cached (key -> parsed table); a single entry is kept for the live path so the
# common case is one dict lookup and zero disk access after the first call.
_override_cache: dict[_CacheKey, _OverrideTable] = {}


def _override_cache_key() -> _CacheKey:
    """Build a cache key from the resolved pricing path's identity + stat.

    The key encodes the override path, the seed path, and whichever file
    ``load_json_or_empty`` would actually read (user-dir file if present, else
    bundled seed) by its ``(mtime_ns, size)``.  A missing file stats as
    ``(0, 0)`` so the absent/error case is itself cacheable and stable.
    """
    if _PRICING_JSON.exists():
        read_path = _PRICING_JSON
    elif _BUNDLED_SEED_PATH.exists():
        read_path = _BUNDLED_SEED_PATH
    else:
        read_path = _PRICING_JSON  # neither exists; stat below yields (0, 0)
    try:
        st = read_path.stat()
        stat_part = (st.st_mtime_ns, st.st_size)
    except OSError:
        stat_part = (0, 0)
    return (str(_PRICING_JSON), str(_BUNDLED_SEED_PATH), stat_part[0], stat_part[1])


def clear_pricing_cache() -> None:
    """Drop all cached pricing state.

    Call this when the on-disk pricing table is replaced out-of-band within the
    same process (e.g. tests that monkeypatch ``_PRICING_JSON`` to a new file, or
    code that updates pricing and re-analyzes in one run without the mtime
    changing).  Production callers never need this — the mtime/size cache key
    already invalidates on a real ``frugon pricing update``.
    """
    global _canonical_tc_index, _canonical_tc_token
    _override_cache.clear()
    _resolve_model_price.cache_clear()
    _canonical_tc_index = None
    _canonical_tc_token = None


def load_pricing_override() -> _OverrideTable:
    """Load the pricing table from the user data directory.

    On first use, the user-data-dir file is seeded from the wheel-bundled
    snapshot if it does not yet exist.

    Returns:
        Tuple of (model_table, last_synced_date_or_None).

    The model_table maps model name -> {"input_cost_per_token": float,
    "output_cost_per_token": float}.

    Does not raise on missing or malformed file -- returns empty dict so the
    tokencost fallback takes over.

    The parsed result is cached keyed on the pricing file's path + mtime + size,
    so repeated calls within an analysis run (one per priced record) do not
    re-read and re-parse the file.  The cache invalidates automatically when the
    file is rewritten by ``frugon pricing update``.
    """
    seed_if_missing(_PRICING_JSON, _BUNDLED_SEED_PATH)

    key = _override_cache_key()
    cached = _override_cache.get(key)
    if cached is not None:
        return cached

    raw = load_json_or_empty(_PRICING_JSON, _BUNDLED_SEED_PATH)

    last_synced: str | None = raw.get("_last_synced")  # type: ignore[assignment]
    models: dict[str, dict[str, object]] = {}

    for table_key, val in raw.items():
        if table_key.startswith("_"):
            continue
        if isinstance(val, dict) and "input_cost_per_token" in val:
            models[table_key] = val  # type: ignore[assignment]

    result: _OverrideTable = (models, last_synced)
    # Keep only the current key — the path rarely changes, and bounding the dict
    # avoids unbounded growth if a long-lived process cycles many pricing files.
    _override_cache.clear()
    _override_cache[key] = result
    return result


def _price_pair(entry: object) -> tuple[Decimal, Decimal] | None:
    """Extract a valid (input, output) per-token price pair from a registry entry.

    Returns None when *entry* is not a dict, or when either cost field is absent
    or None.  Centralises the "is this entry priceable, and what is the pair?"
    decision so every lookup path (override, exact tokencost, newest-dated, the
    canonical bridge) handles missing/null costs identically — fail-safe to None
    rather than crashing on ``Decimal(str(None))`` or a missing key.
    """
    if not isinstance(entry, dict):
        return None
    in_cost = entry.get("input_cost_per_token")
    out_cost = entry.get("output_cost_per_token")
    if in_cost is None or out_cost is None:
        return None
    return (Decimal(str(in_cost)), Decimal(str(out_cost)))


def _agreed_pair(
    pairs: set[tuple[Decimal, Decimal]],
) -> tuple[Decimal, Decimal] | None:
    """The single agreed (input, output) price pair from *pairs*, else None.

    Returns the pair when the set holds exactly one distinct (input, output)
    tuple; returns None when the set is empty or holds two or more (divergent).
    This is the §2a "never guess on divergence" gate, centralised so every caller
    — the newest-dated fallback and the canonical bridge index — refuses divergent
    prices identically; a fix to one gate cannot silently miss the other.
    """
    return next(iter(pairs)) if len(pairs) == 1 else None


# ---------------------------------------------------------------------------
# Newest-dated tokencost fallback helpers
# ---------------------------------------------------------------------------
#
# Purpose: resolve bare family names (e.g. "claude-3-5-sonnet") that exist in
# tokencost only under dated variants (e.g. "claude-3-5-sonnet-20241022").
#
# Condition for use: ONLY when all three existing lookup steps (exact →
# canonicalized → base_family) have already failed.  We then look for tokencost
# keys of the form ``<base>-YYYYMMDD`` (compact date), collect those whose
# price fields are both present and consistent, and pick the newest (max
# lexicographic date).  Consistent = every dated variant of that family carries
# the same (input, output) cost pair.  A price mismatch across dates means the
# provider changed their rate and we cannot safely attribute one price to the
# bare name — we return None rather than guess.
#
# This is deliberately NOT applied when the override-table or exact/canonical
# steps succeed; it is a fallback of last resort, not the primary path.

_COMPACT_DATE_SUFFIX_RE = re.compile(r"-(\d{8})$")


def _newest_dated_tokencost_price(
    base: str,
    model: str,
    tc_costs: dict[str, dict[str, object]],
    last_synced: str | None,
) -> ModelPrice | None:
    """Return a price for bare *base* via the newest consistently-priced dated variant.

    Scans *tc_costs* for keys matching ``<base>-YYYYMMDD``.  If at least one
    such key exists AND all such keys share the same (input, output) cost pair,
    returns the price attributed to the newest key.  Returns None if no dated
    variants exist, any cost field is absent on any variant, or prices differ
    across variants (rate change — cannot safely attribute to the bare name).

    *model* is the original user-supplied name stored in ModelPrice.model.
    *base* is the canonicalized base-family form used to build the search prefix.
    """
    prefix = base + "-"
    dated: list[tuple[str, Decimal, Decimal]] = []  # (key, input_cost, output_cost)
    for key, entry in tc_costs.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not _COMPACT_DATE_SUFFIX_RE.fullmatch(f"-{suffix}"):
            # Suffix is not an 8-digit compact date — skip (could be a suffix like
            # "-latest" or another name that starts with the same prefix).
            continue
        pair = _price_pair(entry)
        if pair is None:
            return None  # any dated variant missing cost → fail safe
        dated.append((key, pair[0], pair[1]))

    if not dated:
        return None

    # Consistency gate: every dated variant must share one (input, output) pair.
    # A mismatch means the provider changed their rate across snapshot dates and we
    # cannot safely attribute a single price to the bare name → refuse (None).
    agreed = _agreed_pair({(row[1], row[2]) for row in dated})
    if agreed is None:
        return None

    # Pick the newest: compact dates sort lexicographically (YYYYMMDD).
    newest_key = max(dated, key=lambda row: row[0])[0]
    return ModelPrice(
        model=model,
        input_cost_per_token=agreed[0],
        output_cost_per_token=agreed[1],
        source=f"tokencost[{newest_key}]",
        pricing_json_last_synced=last_synced,
    )


# ---------------------------------------------------------------------------
# Canonicalized-tokencost bridge
# ---------------------------------------------------------------------------
#
# tokencost stores most models under provider-prefixed keys (e.g.
# "deepseek/deepseek-r1", "mistral/mistral-large-latest", "vertex_ai/...").  The
# exact/canonical steps look a user's canonicalized name up against tokencost's
# RAW keys, so a bare wire form ("deepseek-r1") misses the prefixed key even
# though tokencost carries the price.
#
# The bridge indexes tokencost BY canonical form: canonicalize(raw_key) -> price.
# Consistency gate (§2a — never fabricate a number): a canonical name that
# several raw keys map to is included ONLY when every contributing key agrees on
# the (input, output) price pair; a name with divergent prices across providers
# maps to None so the lookup refuses to guess.  Built once per process
# (tokencost is static); reset by clear_pricing_cache() for tests that swap the
# registry out via monkeypatching.

_canonical_tc_index: dict[str, tuple[Decimal, Decimal] | None] | None = None
# Identity token (id, len) of the tokencost table the index was built from.  When
# it changes — e.g. a test swaps tokencost.TOKEN_COSTS for a different-length dict —
# the index rebuilds on next use.  In production TOKEN_COSTS is never reassigned, so
# the token is constant and this path is dead.  A test that mutates the SAME table
# object in place without changing its length must still call clear_pricing_cache().
_canonical_tc_token: tuple[int, int] | None = None


def _build_canonical_tokencost_index(
    tc_costs: dict[str, dict[str, object]],
) -> dict[str, tuple[Decimal, Decimal] | None]:
    """Map ``canonicalize(tokencost_key)`` to its (input, output) price pair.

    When several raw keys share a canonical form, the entry is the agreed price
    iff every contributing key carries the same pair; otherwise the entry is
    None (divergent → refuse to guess).  Keys missing either cost field are
    skipped (they cannot price a request anyway).
    """
    from collections import defaultdict

    groups: dict[str, set[tuple[Decimal, Decimal]]] = defaultdict(set)
    for key, entry in tc_costs.items():
        pair = _price_pair(entry)
        if pair is None:
            continue
        groups[canonicalize(key)].add(pair)

    index: dict[str, tuple[Decimal, Decimal] | None] = {}
    for canon, pairs in groups.items():
        index[canon] = _agreed_pair(pairs)
    return index


def _canonical_tokencost_price(
    name: str,
    model: str,
    tc_costs: dict[str, dict[str, object]],
    last_synced: str | None,
) -> ModelPrice | None:
    """Resolve *name* via the consistency-gated canonicalized-tokencost index."""
    global _canonical_tc_index, _canonical_tc_token
    token = (id(tc_costs), len(tc_costs))
    if _canonical_tc_index is None or _canonical_tc_token != token:
        _canonical_tc_index = _build_canonical_tokencost_index(tc_costs)
        _canonical_tc_token = token
    pair = _canonical_tc_index.get(name)
    if pair is None:
        return None
    return ModelPrice(
        model=model,
        input_cost_per_token=pair[0],
        output_cost_per_token=pair[1],
        source="tokencost[canonical]",
        pricing_json_last_synced=last_synced,
    )


@functools.lru_cache(maxsize=4096)
def _resolve_model_price(
    model: str, _cache_key: _CacheKey, _tc_token: tuple[int, int]
) -> ModelPrice | None:
    """Resolve and cache the price for *model* under a pricing-table identity.

    The ``_cache_key`` argument is the override table's identity (path + mtime +
    size, from :func:`_override_cache_key`).  It is part of the cache key so a
    ``frugon pricing update`` (which changes the file's mtime/size) transparently
    invalidates every memoized model — without it a long-lived process could
    return prices from a superseded table.

    ``_tc_token`` is the tokencost registry's identity ``(id, len)``.  It is part
    of the cache key purely so that swapping ``tokencost.TOKEN_COSTS`` (which tests
    do via monkeypatch) invalidates the memo without depending on a
    ``clear_pricing_cache()`` call; in production the registry is static so this
    token never changes.  Both extra args are cache-key-only (unused in the body).

    The same key values are reused for every record in one analysis run, so each
    distinct model is resolved exactly once even across tens of thousands of calls.
    """
    override_table, last_synced = load_pricing_override()
    tc_costs: dict[str, dict[str, object]] = _tc.TOKEN_COSTS  # type: ignore[attr-defined]

    def _from_override(name: str) -> ModelPrice | None:
        pair = _price_pair(override_table.get(name))
        if pair is None:
            return None
        return ModelPrice(
            model=model,
            input_cost_per_token=pair[0],
            output_cost_per_token=pair[1],
            source="pricing.json",
            pricing_json_last_synced=last_synced,
        )

    def _from_tokencost(name: str) -> ModelPrice | None:
        pair = _price_pair(tc_costs.get(name))
        if pair is None:
            return None
        return ModelPrice(
            model=model,
            input_cost_per_token=pair[0],
            output_cost_per_token=pair[1],
            source="tokencost",
            pricing_json_last_synced=last_synced,
        )

    def _lookup(name: str) -> ModelPrice | None:
        return _from_override(name) or _from_tokencost(name)

    # 1. Exact match
    result = _lookup(model)
    if result is not None:
        return result

    # 2. Canonicalized form (strips gateway prefixes, normalises Bedrock)
    canon = canonicalize(model)
    if canon != model:
        result = _lookup(canon)
        if result is not None:
            return result

    # 3. Base-family fallback (strips dated snapshot suffix — lookup only)
    base = base_family(canon)
    if base != canon:
        result = _lookup(base)
        if result is not None:
            return result

    # 4. Newest-dated tokencost fallback — ONLY for a genuinely bare family name
    #    (canon == base, i.e. no date / "-latest" / tag suffix was stripped).
    #    Probes tokencost for consistently-priced compact-dated variants of the
    #    family and attributes the newest to the bare name.  This resolves bare
    #    Anthropic families (e.g. "claude-3-5-sonnet") that tokencost only carries
    #    under dated keys (e.g. "claude-3-5-sonnet-20241022"); the consistency gate
    #    refuses when the list rate changed across snapshot dates.
    #
    #    The `canon == base` guard is load-bearing, not cosmetic: without it a
    #    suffixed name like "foo-latest" (canon "foo-latest", base "foo") whose own
    #    `-latest` price is DIVERGENT across providers would be re-priced here from
    #    the consistent "foo" dated family — the identical §2a fabrication the
    #    step-5 gate refuses, one step earlier (see
    #    test_divergent_latest_does_not_fall_through_to_newest_dated).  A suffixed
    #    name is instead routed to the canon-gated step 5, which prices it iff the
    #    name's own canonical form is consistent and refuses it otherwise.
    if canon == base:
        result = _newest_dated_tokencost_price(base, model, tc_costs, last_synced)
        if result is not None:
            return result

    # 5. Canonicalized-tokencost bridge (last resort, after override + exact +
    #    canonical + base + newest-dated all missed): tokencost stores most models
    #    under provider-prefixed keys (e.g. "deepseek/deepseek-r1"); the index maps
    #    canonicalize(raw_key) -> price so a bare wire name matches.  Purely
    #    additive — never overrides an earlier resolution.
    #
    #    CANON-ONLY — deliberately NO base-family arm.  Folding to base_family here
    #    would evaluate the consistency gate over a broader set of raw keys than the
    #    one that prices the requested name, letting a base-family price be
    #    attributed to a `-latest`/dated name the gate already refused as divergent
    #    (a fabricated number — see test_divergent_latest_does_not_fall_through_to_base).
    #    Legitimate bare-family coverage already resolves through this canon step
    #    (canonicalize("mistral-large") == "mistral-large" is itself a clean index
    #    key); provider-qualified names (e.g. "mistral/mistral-large-latest") resolve
    #    via the exact step above.  Consistency-gated: a canonical name priced
    #    divergently across providers resolves to None rather than guessing.
    result = _canonical_tokencost_price(canon, model, tc_costs, last_synced)
    if result is not None:
        return result

    return None


class PricedModelRow(NamedTuple):
    """One row for the ``frugon models`` discovery listing.

    Costs are per-token Decimals, exactly as the pricing table stores them; the
    renderer scales to a per-1M display unit.  *quality_tier* is the integer tier
    from the quality table or :data:`frugon.quality.UNRATED_TIER` when the model
    is unrated.
    """

    model: str
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    quality_tier: int


def list_priced_models(query: str | None = None) -> list[PricedModelRow]:
    """List models in the local pricing table, name-sorted, optionally filtered.

    The source is the local pricing override table — the SAME table
    ``--candidates`` resolves against — so every name returned is exactly a name
    ``--candidates`` accepts.  Pure local read: no network, no tokencost fallback
    (the fallback would surface names ``--candidates`` does not key on directly).

    *query*, when given, filters to models whose name contains it
    (case-insensitive substring).  Each row carries the per-token input/output
    costs as stored and the model's quality tier (UNRATED_TIER when unrated).
    """
    from frugon.quality import get_model_tier

    override_table, _last_synced = load_pricing_override()
    needle = query.lower() if query else None

    rows: list[PricedModelRow] = []
    for name, entry in override_table.items():
        if needle is not None and needle not in name.lower():
            continue
        rows.append(
            PricedModelRow(
                model=name,
                input_cost_per_token=Decimal(str(entry["input_cost_per_token"])),
                output_cost_per_token=Decimal(str(entry["output_cost_per_token"])),
                quality_tier=get_model_tier(name),
            )
        )
    rows.sort(key=lambda r: r.model)
    return rows


def get_model_price(model: str) -> ModelPrice | None:
    """Resolve the price for *model*.

    Lookup order: exact → canonicalize() → base_family() — first hit wins.
    Existing behaviour for bare names is unchanged; gateway-prefixed and
    dated-snapshot names now resolve transparently.

    Precedence within each lookup step: pricing.json > tokencost.TOKEN_COSTS.
    Returns None when the model is unknown in all three steps.

    The resolution is memoized per ``(model, pricing-table-identity)`` so the
    cost engine — which calls this once per log record — resolves each distinct
    model name only once per run.  The memo invalidates automatically when the
    pricing file is rewritten (its mtime/size change the cache key).
    """
    tc_token = (id(_tc.TOKEN_COSTS), len(_tc.TOKEN_COSTS))  # type: ignore[attr-defined]
    return _resolve_model_price(model, _override_cache_key(), tc_token)


def is_model_known(model: str) -> bool:
    """Return True if we can price *model*."""
    return get_model_price(model) is not None


# ---------------------------------------------------------------------------
# Pricing update -- fetch LiteLLM registry, validate, atomic write
# ---------------------------------------------------------------------------


class PricingUpdateError(RuntimeError):
    """Raised when pricing update fails (network error, malformed payload, I/O error)."""


def fetch_and_update_pricing(
    registry_url: str,
    output_path: Path,
    today_date_str: str,
) -> dict[str, int]:
    """Fetch the LiteLLM registry, validate, and atomically update *output_path*.

    *output_path* should be the user data dir path (``_PRICING_JSON``) so
    updates survive reinstalls.  The CLI passes ``_PRICING_JSON`` directly;
    callers that need a custom destination (e.g. tests) pass their own path.

    Returns {"models_synced": N} on success.
    Raises PricingUpdateError on any failure -- *output_path* is never
    modified if an error occurs before the final rename.
    Raises ValueError if *registry_url* is not HTTPS or not in the allowed
    host list.
    """
    validate_fetch_url(registry_url, _ALLOWED_PRICING_HOSTS)

    # 1. Fetch (bounded retry on HTTP 429, HTTP 5xx, and transient network errors;
    #    raw.githubusercontent.com 5xxs sporadically under load).
    def _on_failure(exc: Exception) -> PricingUpdateError:
        if isinstance(exc, urllib.error.HTTPError):
            return PricingUpdateError("pricing registry unavailable")
        return PricingUpdateError(f"Network error fetching registry: {exc}")

    raw_bytes: bytes = fetch_url_with_retry(
        registry_url,
        user_agent=USER_AGENT,
        max_bytes=_MAX_RESPONSE_BYTES,
        timeout=30,
        max_retries=_FETCH_MAX_RETRIES,
        backoff_base=_FETCH_BACKOFF_BASE,
        on_failure=_on_failure,
    )

    # 2. Parse
    try:
        raw: Any = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise PricingUpdateError(f"JSON parse error in pricing registry: {exc}") from exc

    # 3. Validate shape
    if not isinstance(raw, dict):
        raise PricingUpdateError(
            f"Registry has unexpected shape (expected dict, got {type(raw).__name__})"
        )
    registry: dict[str, Any] = raw

    # 4. Extract models with both cost fields
    priced: dict[str, dict[str, object]] = {}
    for model, entry in registry.items():
        if model.startswith("_"):
            continue
        if not isinstance(entry, dict):
            continue
        if "input_cost_per_token" in entry and "output_cost_per_token" in entry:
            priced[model] = {
                "input_cost_per_token": entry["input_cost_per_token"],
                "output_cost_per_token": entry["output_cost_per_token"],
            }

    if not priced:
        raise PricingUpdateError(
            "Registry contains no priced models (no entries with both "
            "input_cost_per_token and output_cost_per_token) -- "
            "refusing to overwrite pricing.json with no priced models"
        )

    # 5. Build output dict (metadata first, then models)
    output: dict[str, object] = {
        "_last_synced": today_date_str,
        "_source": registry_url,
        "_note": (
            "Models listed here take precedence over tokencost for pricing. "
            "Sync with: frugon pricing update"
        ),
    }
    output.update(priced)

    # 6. Atomic write via shared helper
    try:
        atomic_write_json(output_path, output)
    except OSError as exc:
        raise PricingUpdateError(f"Failed to write pricing.json: {exc}") from exc

    return {"models_synced": len(priced)}


def _fetch_registry(registry_url: str) -> dict[str, Any]:
    """Validate *registry_url*, fetch, and parse the LiteLLM registry JSON.

    Used by :func:`refresh_seed_prices`.  Uses the same fetch/retry/parse
    behaviour as :func:`fetch_and_update_pricing`.

    Raises:
        ValueError: if *registry_url* is not HTTPS or not in the allowed list.
        PricingUpdateError: on any network, HTTP, or JSON-parse failure.
    """
    validate_fetch_url(registry_url, _ALLOWED_PRICING_HOSTS)

    def _on_failure(exc: Exception) -> PricingUpdateError:
        if isinstance(exc, urllib.error.HTTPError):
            return PricingUpdateError("pricing registry unavailable")
        return PricingUpdateError(f"Network error fetching registry: {exc}")

    raw_bytes: bytes = fetch_url_with_retry(
        registry_url,
        user_agent=USER_AGENT,
        max_bytes=_MAX_RESPONSE_BYTES,
        timeout=30,
        max_retries=_FETCH_MAX_RETRIES,
        backoff_base=_FETCH_BACKOFF_BASE,
        on_failure=_on_failure,
    )

    try:
        raw: Any = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise PricingUpdateError(f"JSON parse error in pricing registry: {exc}") from exc

    if not isinstance(raw, dict):
        raise PricingUpdateError(
            f"Registry has unexpected shape (expected dict, got {type(raw).__name__})"
        )

    return raw  # type: ignore[return-value]


def refresh_seed_prices(
    registry_url: str,
    seed_path: Path,
    today_date_str: str,
) -> dict[str, int]:
    """Refresh the cost values in the curated seed file without changing its model set.

    Designed for a weekly CI workflow: fetches the LiteLLM registry and updates
    ONLY the ``input_cost_per_token`` / ``output_cost_per_token`` values of seed
    keys that can be matched EXACTLY in the registry.  Never adds a key the seed
    does not already have, never removes a key.

    Exact-key match only — no canonicalization, no prefix stripping — so a seed
    key ``claude-x`` cannot accidentally pick up a registry entry under
    ``bedrock/us-east-1/claude-x`` (a different price point).

    If no price actually changed the seed file is **not** written and
    ``_last_synced`` is **not** bumped, keeping ``git diff`` clean for the CI
    commit gate.  When at least one price changes the file is written atomically
    and ``_last_synced`` is set to *today_date_str*.

    All metadata keys (those starting with ``_``) and all model keys are
    preserved.  Only ``input_cost_per_token`` and ``output_cost_per_token``
    values on already-present model entries may change.

    Args:
        registry_url: URL of the LiteLLM pricing registry.  Must be HTTPS and
            resolve to ``raw.githubusercontent.com`` (validated before fetch).
        seed_path: Path to the seed ``pricing.json`` to update in-place.
        today_date_str: ISO-8601 date string (``YYYY-MM-DD``) written to
            ``_last_synced`` when at least one price changes.

    Returns:
        ``{"checked": N, "updated": M}`` where *N* is the count of non-metadata
        seed keys examined and *M* is the count whose cost values changed.

    Raises:
        ValueError: if *registry_url* fails HTTPS/host validation.
        PricingUpdateError: on network failure, JSON parse error, bad registry
            shape, or inability to read/parse the seed file.
    """
    # 1. Fetch and parse the upstream registry.
    registry: dict[str, Any] = _fetch_registry(registry_url)

    # 2. Load the existing seed.  A missing or unparseable seed is an error —
    #    the caller must supply a valid seed path; we must not silently create
    #    a new one or return an empty baseline.
    try:
        with seed_path.open(encoding="utf-8") as fh:
            raw_seed: Any = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise PricingUpdateError(
            f"Cannot read seed file {seed_path}: {exc}"
        ) from exc

    if not isinstance(raw_seed, dict):
        raise PricingUpdateError(
            f"Seed file {seed_path} has unexpected shape "
            f"(expected dict, got {type(raw_seed).__name__})"
        )
    seed: dict[str, Any] = raw_seed

    # 3. Walk every non-metadata seed key.  Apply the registry price iff:
    #    - the registry contains an entry under the EXACT same key, AND
    #    - that entry carries BOTH input_cost_per_token AND output_cost_per_token.
    #    Otherwise the existing (curated) value is kept unchanged.
    checked = 0
    updated = 0

    # Build the output preserving insertion order: metadata keys first (in their
    # original order), then model keys (in their original order).  We mutate a
    # fresh output dict rather than the seed in-place so that any mid-loop
    # failure leaves *seed_path* untouched.
    output: dict[str, Any] = {}

    for key, value in seed.items():
        if key.startswith("_"):
            # Metadata key — copy verbatim; _last_synced overwritten later if
            # any prices changed.
            output[key] = value
            continue

        checked += 1
        reg_entry = registry.get(key)

        if (
            isinstance(reg_entry, dict)
            and "input_cost_per_token" in reg_entry
            and "output_cost_per_token" in reg_entry
        ):
            new_in = reg_entry["input_cost_per_token"]
            new_out = reg_entry["output_cost_per_token"]

            if isinstance(value, dict):
                old_in = value.get("input_cost_per_token")
                old_out = value.get("output_cost_per_token")
            else:
                old_in = None
                old_out = None

            if new_in != old_in or new_out != old_out:
                # Price changed — patch cost fields; preserve any sibling
                # fields already present on the seed entry (e.g. a manually
                # curated "note" field the seed might carry alongside costs).
                if isinstance(value, dict):
                    patched: dict[str, Any] = dict(value)
                else:
                    patched = {}
                patched["input_cost_per_token"] = new_in
                patched["output_cost_per_token"] = new_out
                output[key] = patched
                updated += 1
            else:
                output[key] = value
        else:
            # Key absent from registry or missing cost fields — keep curated
            # value unchanged.
            output[key] = value

    # 4. If nothing changed, do not write — preserve file mtime so git diff
    #    stays clean and the weekly CI workflow does not produce an empty commit.
    if updated == 0:
        return {"checked": checked, "updated": 0}

    # 5. Bump _last_synced in the output dict.  The key is guaranteed to be
    #    in *output* (it was either already in the seed or implicitly absent
    #    and we insert it now so the file always carries a sync date after a
    #    real update).
    output["_last_synced"] = today_date_str

    # 6. Atomic write — never leaves a partial file on disk.
    #    trailing_newline=True keeps the seed file at its fixed point so that a
    #    one-price change produces a one-line diff (not a whole-file reformat).
    try:
        atomic_write_json(seed_path, output, trailing_newline=True)
    except OSError as exc:
        raise PricingUpdateError(
            f"Failed to write refreshed seed {seed_path}: {exc}"
        ) from exc

    return {"checked": checked, "updated": updated}


def is_pricing_stale(
    last_synced: str | None,
    max_days: int = 30,
    today: str | None = None,
) -> bool:
    """Return True if *last_synced* is at least *max_days* days before *today*.

    Returns False when *last_synced* is None or cannot be parsed, so a missing
    or malformed date never triggers a spurious warning.
    """
    if last_synced is None:
        return False
    try:
        synced = _date.fromisoformat(last_synced)
        today_date = _date.fromisoformat(today) if today else _date.today()
        return (today_date - synced).days >= max_days
    except ValueError:
        return False
