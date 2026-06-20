"""frugon quality module.

Quality tier lookup backed by the LMArena leaderboard dataset.
Source: lmarena-ai/leaderboard-dataset (CC-BY-4.0 — commercial use + redistribution permitted).

Tier scale (lower = better quality):
  0  Elite      Top 10% of the leaderboard field by Arena score
  1  Strong     10–30% of the field
  2  Capable    30–60% of the field
  3  Efficient  Bottom 40% of the field

Tiers are derived by percentile-rank over the full set of models fetched in
each sync, so the bands self-recalibrate as the leaderboard grows and Arena
rebaselines its scoring.  Equal scores receive the same tier (tie-robust).

Storage layout:
  <user-data-dir>/quality.json   (writable; updated by `frugon quality update`)
  src/frugon/data/quality.json   (bundled seed; offline fallback for first run)

When building the table from Arena data, model names are normalised via
canonicalize() + base_family() so that dated snapshots ("gpt-4o-2024-05-13")
and bare API names ("gpt-4o") map to the same storage key ("gpt-4o").  An alias
map covers the handful of Arena names that don't canonicalize cleanly.

Category + date filtering:
  The LMArena dataset contains 20+ categories (overall, coding, creative_writing,
  spanish, etc.).  Only the "overall" category represents a model's general-purpose
  quality tier.  The /filter endpoint pre-filters to category=='overall' at the
  server side so that only ~1,036 rows (11 pages) are transferred instead of the
  full 21,259-row corpus (213 pages) that the /rows endpoint returns.
  fetch_and_update_quality also applies client-side category + date filtering as a
  defense-in-depth safety net (the /filter server query is a no-op safety match;
  the latest-date selection and fail-loud check still matter client-side).
"""

from __future__ import annotations

import json
import urllib.error
from datetime import date as _date
from pathlib import Path
from typing import Any

try:
    import platformdirs as _platformdirs  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "platformdirs is required. Install it: pip install platformdirs"
    ) from exc

from frugon import USER_AGENT
from frugon._store import (
    atomic_write_json,
    fetch_url_with_retry,
    load_json_or_empty,
    seed_if_missing,
    validate_fetch_url,
)
from frugon.model_id import base_family, canonicalize

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_BUNDLED_SEED_PATH: Path = Path(__file__).parent / "data" / "quality.json"
_QUALITY_JSON: Path = Path(_platformdirs.user_data_dir("frugon")) / "quality.json"

# Hugging Face datasets-server endpoint for the LMArena leaderboard.
# Uses the /filter endpoint with a server-side where clause to pre-filter to the
# "overall" category — returns ~1,036 rows (≈11 pages) instead of the full
# 21,259-row corpus (213 pages) that the /rows endpoint would return.
# The where clause uses DOUBLE-quoted column name and SINGLE-quoted value, both
# URL-encoded: "category"='overall' → %22category%22%3D%27overall%27.
# _fetch_rows appends &offset=…&length=… which composes correctly with /filter.
_HF_DATASET = "lmarena-ai%2Fleaderboard-dataset"
_HF_BASE_URL = (
    "https://datasets-server.huggingface.co/filter"
    f"?dataset={_HF_DATASET}&config=text&split=latest"
    "&where=%22category%22%3D%27overall%27"
)
_HF_PAGE_LENGTH = 100

# Retry parameters for _fetch_rows.  On HTTP 429, HTTP 5xx, or transient
# URLError/OSError, retry up to _FETCH_MAX_RETRIES times with exponential backoff.
# A 429 response may include a Retry-After header (integer seconds); that overrides
# the schedule.
_FETCH_MAX_RETRIES: int = 4
_FETCH_BACKOFF_BASE: float = 1.0  # seconds; doubles each attempt: 1, 2, 4, 8

_ALLOWED_QUALITY_HOSTS: frozenset[str] = frozenset({"datasets-server.huggingface.co"})

# Category to keep when the leaderboard dataset contains per-category breakdowns.
# The "overall" category is the only one that reflects general-purpose quality;
# specialty categories (coding, spanish, math, etc.) must not influence the tier
# a model receives — a strong coding model is not necessarily a strong general model.
_OVERALL_CATEGORY = "overall"

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024  # 16 MB cap

# ---------------------------------------------------------------------------
# Percentile-rank tier cut-points (fraction of the field scoring strictly
# higher). Lower fraction = better tier. Calibrated so the current frontier
# lands in Elite and legacy/small models in Efficient; self-recalibrates each
# sync as the leaderboard shifts (robust to Arena rebaselining).
# ---------------------------------------------------------------------------
_TIER_PERCENTILES: tuple[tuple[float, int], ...] = (
    (0.10, 0),  # Elite:     top 10%
    (0.30, 1),  # Strong:    next 20% (10–30%)
    (0.60, 2),  # Capable:   next 30% (30–60%)
)                # Efficient: bottom 40%
_TIER_DEFAULT = 3

# ---------------------------------------------------------------------------
# Arena → provider name alias map
# Keys: Arena model names that don't canonicalize cleanly to a provider API name.
# Values: canonical provider names to use as storage keys.
# ---------------------------------------------------------------------------
_ARENA_ALIASES: dict[str, str] = {
    "chatgpt-4o-latest": "gpt-4o",
    "chatgpt-4o": "gpt-4o",
    "gpt-4-0125": "gpt-4",
    "gpt-4-1106": "gpt-4",
    "o1-preview": "o1",
    "llama-3.1-405b-instruct-fp8": "llama-3.1-405b-instruct",
    "meta-llama-3.1-405b-instruct-fp8": "llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-405b-instruct-fp8": "llama-3.1-405b-instruct",
}

# ---------------------------------------------------------------------------
# Public sentinel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# classify_quality_update — validation + change-magnitude classification
# ---------------------------------------------------------------------------

# Minimum number of non-metadata model keys for a quality.json to be considered
# non-trivially populated.  The bundled seed holds a few hundred models; fewer
# than 50 implies the fetch returned a fragment or an entirely wrong payload.
_CLASSIFY_MIN_MODELS: int = 50

# MAJOR verdict threshold: if the model count moved by more than this fraction
# (relative to the OLD count), the update is flagged for human review.  A 15%
# swing could indicate a rebaseline or a scoring-mechanism change.
_CLASSIFY_MAX_COUNT_DELTA_FRAC: float = 0.15

# MAJOR verdict threshold: if more than this fraction of models that appear in
# BOTH old and new datasets changed tier, the update is flagged for human
# review.  A 20% tier-churn rate signals a distribution-wide rescore.
_CLASSIFY_MAX_TIER_CHURN_FRAC: float = 0.20

# MAJOR verdict threshold: if fewer than this fraction of the OLD roster's
# model keys still appear in the NEW dataset, the update is flagged for human
# review.  A near-total roster swap is the signature of a rebaseline or schema
# change; computing churn over a tiny shared set would be misleading.
# Invariant: reason strings must never embed raw fetched model names so that
# fetched content cannot escape through the reason field.
_CLASSIFY_MIN_ROSTER_OVERLAP_FRAC: float = 0.70

# Verdict constants — used by the workflow and callers; kept as module-level
# strings so they are importable without quoting magic.
VERDICT_INVALID: str = "INVALID"
VERDICT_MINOR: str = "MINOR"
VERDICT_MAJOR: str = "MAJOR"


def _extract_model_tiers(data: dict[str, object]) -> dict[str, int]:
    """Return the model→tier mapping from a raw quality.json dict.

    Skips metadata keys (prefixed with ``_``) and non-integer values.
    Returns an empty dict for an empty or metadata-only input.
    """
    result: dict[str, int] = {}
    for key, val in data.items():
        if key.startswith("_"):
            continue
        if isinstance(val, int):
            result[key] = val
    return result


def classify_quality_update(
    new: dict[str, object],
    old: dict[str, object] | None,
) -> tuple[str, str]:
    """Validate *new* and classify the magnitude of the change vs *old*.

    This function is **pure** — no I/O, no network, no side effects.  Both
    *new* and *old* are expected to be the raw parsed contents of a
    quality.json file (metadata keys prefixed ``_`` plus model→tier entries).

    Parameters
    ----------
    new:
        The freshly fetched quality.json contents.
    old:
        The previously stored quality.json contents, or ``None`` when there is
        no prior version (first run / seed missing).

    Returns
    -------
    (verdict, reason) where *verdict* is one of:

    ``VERDICT_INVALID``
        The new data fails a structural invariant and must **never** be
        committed.  The workflow must discard the file and fail the job.

    ``VERDICT_MAJOR``
        The new data is structurally valid but the change vs *old* is large
        enough to require human review before merging (rebaseline / scoring
        mechanism change).

    ``VERDICT_MINOR``
        The new data is valid and the delta is within expected weekly drift
        bounds — safe to auto-merge after CI.

    Thresholds are the module-level ``_CLASSIFY_*`` constants so they can be
    tuned and are visible in documentation.

    Invariants checked for INVALID (in order):

    1. Fewer than ``_CLASSIFY_MIN_MODELS`` non-metadata keys → payload too
       small to be a real leaderboard sync.
    2. Any tier value outside ``{0, 1, 2, 3}`` → ranking engine produced an
       out-of-spec result.
    3. No tier-0 models present → frontier models are missing, which means the
       percentile ranking broke or the fetch returned an incomplete slice.
    4. Any model key that is NOT already canonical
       (``canonicalize(key) != key``) → an un-normalizable model name slipped
       past the fetch pipeline; landing it would break downstream lookups.

    Notes on ``old is None``
        A missing prior version (first sync, or the seed file was removed)
        is treated as ``MINOR`` when the new data passes all invariant checks,
        because there is no baseline to measure delta against.
    """
    new_tiers = _extract_model_tiers(new)

    # -----------------------------------------------------------------------
    # Invariant 1 — minimum model count
    # -----------------------------------------------------------------------
    if len(new_tiers) < _CLASSIFY_MIN_MODELS:
        return (
            VERDICT_INVALID,
            f"too few models: got {len(new_tiers)}, minimum is {_CLASSIFY_MIN_MODELS}",
        )

    # -----------------------------------------------------------------------
    # Invariant 2 — all tier values in {0, 1, 2, 3}
    # -----------------------------------------------------------------------
    _valid_tiers: frozenset[int] = frozenset({0, 1, 2, 3})
    bad_tiers = {k: v for k, v in new_tiers.items() if v not in _valid_tiers}
    if bad_tiers:
        sample = dict(list(bad_tiers.items())[:3])
        return (
            VERDICT_INVALID,
            f"out-of-range tier values: {sample}",
        )

    # -----------------------------------------------------------------------
    # Invariant 3 — at least one tier-0 (Elite) model must be present
    # -----------------------------------------------------------------------
    if not any(t == 0 for t in new_tiers.values()):
        return (
            VERDICT_INVALID,
            "no tier-0 (Elite) models present — frontier missing, ranking may be broken",
        )

    # -----------------------------------------------------------------------
    # Invariant 4 — all model keys must already be canonical
    # -----------------------------------------------------------------------
    non_canonical: list[str] = [k for k in new_tiers if canonicalize(k) != k]
    if non_canonical:
        nc_sample: list[str] = non_canonical[:3]
        return (
            VERDICT_INVALID,
            f"non-canonical model keys detected: {nc_sample}",
        )

    # -----------------------------------------------------------------------
    # No prior version — valid data with no baseline to compare → MINOR
    # -----------------------------------------------------------------------
    if old is None:
        return (
            VERDICT_MINOR,
            f"no prior version; {len(new_tiers)} models validated — treated as initial sync",
        )

    old_tiers = _extract_model_tiers(old)

    # -----------------------------------------------------------------------
    # MAJOR check 1 — model-count delta > _CLASSIFY_MAX_COUNT_DELTA_FRAC
    # -----------------------------------------------------------------------
    if old_tiers:
        count_delta_frac = abs(len(new_tiers) - len(old_tiers)) / len(old_tiers)
        if count_delta_frac > _CLASSIFY_MAX_COUNT_DELTA_FRAC:
            pct = round(count_delta_frac * 100, 1)
            return (
                VERDICT_MAJOR,
                (
                    f"model count changed by {pct}% "
                    f"(old={len(old_tiers)}, new={len(new_tiers)}); "
                    f"threshold is {round(_CLASSIFY_MAX_COUNT_DELTA_FRAC * 100, 0):.0f}% — "
                    "possible rebaseline or scoring-mechanism change"
                ),
            )

    # -----------------------------------------------------------------------
    # MAJOR check 2 — roster overlap < _CLASSIFY_MIN_ROSTER_OVERLAP_FRAC
    # -----------------------------------------------------------------------
    # A near-total roster swap (disjoint or low-overlap new vs old) is itself
    # the rebaseline / schema-change signature.  Computing tier churn over a
    # tiny shared set would be meaningless and could let the swap slip through
    # as MINOR.  Guard here before entering the churn check.
    if old_tiers:
        overlap_count = len(set(new_tiers) & set(old_tiers))
        overlap_frac = overlap_count / len(old_tiers)
        if overlap_frac < _CLASSIFY_MIN_ROSTER_OVERLAP_FRAC:
            return (
                VERDICT_MAJOR,
                (
                    f"only {round(overlap_frac * 100, 1)}% of the previous roster "
                    f"still present ({overlap_count}/{len(old_tiers)}); "
                    f"threshold {round(_CLASSIFY_MIN_ROSTER_OVERLAP_FRAC * 100):.0f}% — possible "
                    "roster replacement / rebaseline / schema change"
                ),
            )

    # -----------------------------------------------------------------------
    # MAJOR check 3 — tier churn > _CLASSIFY_MAX_TIER_CHURN_FRAC
    # -----------------------------------------------------------------------
    common_keys = set(new_tiers) & set(old_tiers)
    if common_keys:
        changed_count = sum(
            1 for k in common_keys if new_tiers[k] != old_tiers[k]
        )
        tier_churn_frac = changed_count / len(common_keys)
        if tier_churn_frac > _CLASSIFY_MAX_TIER_CHURN_FRAC:
            pct = round(tier_churn_frac * 100, 1)
            return (
                VERDICT_MAJOR,
                (
                    f"{pct}% of shared models changed tier "
                    f"({changed_count}/{len(common_keys)}); "
                    f"threshold is {round(_CLASSIFY_MAX_TIER_CHURN_FRAC * 100, 0):.0f}% — "
                    "possible distribution-wide rescore"
                ),
            )

    # -----------------------------------------------------------------------
    # All checks passed — MINOR
    # -----------------------------------------------------------------------
    new_count = len(new_tiers)
    old_count = len(old_tiers)
    changed_in_common = (
        sum(1 for k in common_keys if new_tiers[k] != old_tiers[k])
        if common_keys
        else 0
    )
    return (
        VERDICT_MINOR,
        (
            f"{new_count} models (was {old_count}); "
            f"{changed_in_common} tier change(s) among {len(common_keys)} shared models"
        ),
    )


UNRATED_TIER: int = -1
"""Returned by get_model_tier() when a model has no entry in the quality table.

Negative so that arithmetic comparisons degrade gracefully — a model with an
unknown tier is treated conservatively (not auto-recommended, not counted as
better or worse than a known tier without explicit user intent).
"""


# Human-readable label for each integer tier — the same names documented in the
# module header and the synced quality.json ``_note``.  Used by surfaces that
# show a tier to a person (e.g. ``frugon models``) so the displayed label and
# the stored tier never drift.
TIER_NAMES: dict[int, str] = {
    0: "Elite",
    1: "Strong",
    2: "Capable",
    3: "Efficient",
}


def tier_name(tier: int) -> str | None:
    """Return the human-readable label for *tier*, or None when unrated/unknown.

    None for UNRATED_TIER (and any value outside the known band) so callers can
    render a blank/"—" rather than inventing a label for a model with no tier.
    """
    return TIER_NAMES.get(tier)


# ---------------------------------------------------------------------------
# Distribution-relative tier assignment
# ---------------------------------------------------------------------------


def _assign_percentile_tiers(scores: dict[str, float]) -> dict[str, int]:
    """Assign tiers to *scores* by percentile rank over the full distribution.

    For each key, its position is the fraction of OTHER keys with a strictly
    higher score:  position = (count of keys with score > this score) / N.

    This is tie-robust: models with equal scores get the same position value
    and therefore the same tier — no arbitrary split by insertion order.

    The first entry in _TIER_PERCENTILES whose ``position < threshold`` wins;
    if none match, the key receives _TIER_DEFAULT (Efficient).

    Returns an empty dict for empty input.
    """
    if not scores:
        return {}

    n = len(scores)
    result: dict[str, int] = {}
    score_values = list(scores.values())

    for key, score in scores.items():
        # Count how many OTHER models score strictly higher than this one.
        # Using the full list (including self) is equivalent because
        # (count strictly > score) / N is independent of whether we count
        # the model against itself for the strict inequality.
        strictly_higher = sum(1 for s in score_values if s > score)
        position = strictly_higher / n

        tier = _TIER_DEFAULT
        for threshold, candidate_tier in _TIER_PERCENTILES:
            if position < threshold:
                tier = candidate_tier
                break
        result[key] = tier

    return result


# ---------------------------------------------------------------------------
# Arena name → canonical storage key
# ---------------------------------------------------------------------------


def _arena_name_to_key(arena_name: str) -> str:
    """Return the canonical storage key for an Arena model name.

    Pipeline: alias map → canonicalize (strip prefixes) → base_family (fold dates).
    This ensures "gpt-4o-2024-05-13" and "gpt-4o" both map to the key "gpt-4o".
    """
    aliased = _ARENA_ALIASES.get(arena_name) or _ARENA_ALIASES.get(arena_name.lower())
    if aliased:
        return aliased
    canon = canonicalize(arena_name)
    return base_family(canon)


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------


def load_quality_table() -> tuple[dict[str, int], str | None, str | None]:
    """Load the quality tier table from the user data dir (or bundled seed).

    Returns:
        (tier_map, last_synced, attribution) where tier_map maps canonical
        model names to integer tiers (0=best).  last_synced is an ISO date
        string or None.  attribution is the CC-BY-4.0 string or None.

    Never raises — returns empty dict on any I/O or parse error so callers
    degrade to UNRATED_TIER gracefully.
    """
    seed_if_missing(_QUALITY_JSON, _BUNDLED_SEED_PATH)
    raw = load_json_or_empty(_QUALITY_JSON, _BUNDLED_SEED_PATH)

    last_synced: str | None = raw.get("_last_synced")  # type: ignore[assignment]
    attribution: str | None = raw.get("_attribution")  # type: ignore[assignment]

    tier_map: dict[str, int] = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(val, int):
            tier_map[key] = val

    return tier_map, last_synced, attribution


# ---------------------------------------------------------------------------
# Public tier lookup
# ---------------------------------------------------------------------------


def get_model_tier(model: str) -> int:
    """Return the quality tier for *model*, or UNRATED_TIER if unknown.

    Lookup order: canonicalize(model) → exact match → base_family fallback.
    The table stores base_family-normalised keys ("gpt-4o"), so versioned
    API names ("gpt-4o-2024-08-06") and gateway-prefixed names
    ("openrouter/gpt-4o") both resolve correctly.
    """
    tier_map, _, _ = load_quality_table()
    if not tier_map:
        return UNRATED_TIER

    canon = canonicalize(model)

    if canon in tier_map:
        return tier_map[canon]

    base = base_family(canon)
    if base != canon and base in tier_map:
        return tier_map[base]

    return UNRATED_TIER


def is_unrated(model: str) -> bool:
    """Return True when *model* has no entry in the quality tier table."""
    return get_model_tier(model) == UNRATED_TIER


def get_attribution() -> str | None:
    """Return the CC-BY-4.0 attribution string from the quality file, or None."""
    _, _, attribution = load_quality_table()
    return attribution


def is_quality_stale(
    last_synced: str | None,
    max_days: int = 60,
    today: str | None = None,
) -> bool:
    """Return True if *last_synced* is at least *max_days* days before *today*.

    Mirrors :func:`frugon.pricing.is_pricing_stale` exactly, with one deliberate
    difference: the default threshold is 60 days, not 30.  Quality tier tables
    drift more slowly than prices — the LMArena leaderboard standings that back
    the tier bands move on the scale of months, whereas provider list prices can
    change in a single billing cycle.  A 60-day window keeps the caution honest
    (a genuinely stale table is still flagged) without nagging the user to refresh
    a table that has not meaningfully changed.

    Returns False when *last_synced* is None or cannot be parsed, so a missing or
    malformed date never triggers a spurious caution.
    """
    if last_synced is None:
        return False
    try:
        synced = _date.fromisoformat(last_synced)
        today_date = _date.fromisoformat(today) if today else _date.today()
        return (today_date - synced).days >= max_days
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Quality update
# ---------------------------------------------------------------------------


class QualityUpdateError(RuntimeError):
    """Raised when quality update fails (network error, bad payload, I/O error)."""


def _fetch_one_page(url: str, timeout: int) -> bytes:
    """Fetch a single page URL, retrying on HTTP 429, HTTP 5xx, and transient
    network errors.

    Retry schedule: up to _FETCH_MAX_RETRIES attempts after the initial request,
    with exponential backoff (_FETCH_BACKOFF_BASE * 2^attempt seconds).  A 429
    response may carry a Retry-After header (integer seconds); when present it
    overrides the computed backoff.  The HF datasets-server returns sporadic 500s
    on individual /filter pages under load, so a single bad page must not fail the
    whole sync.

    Raises QualityUpdateError after retries are exhausted, or immediately on a
    non-retryable HTTP error (4xx other than 429).
    """

    def _on_failure(exc: Exception) -> QualityUpdateError:
        if isinstance(exc, urllib.error.HTTPError):
            return QualityUpdateError("leaderboard unavailable, using bundled tiers")
        return QualityUpdateError(f"Network error fetching leaderboard: {exc}")

    return fetch_url_with_retry(
        url,
        user_agent=USER_AGENT,
        max_bytes=_MAX_RESPONSE_BYTES,
        timeout=timeout,
        max_retries=_FETCH_MAX_RETRIES,
        backoff_base=_FETCH_BACKOFF_BASE,
        on_failure=_on_failure,
    )


def _fetch_rows(base_url: str, page_length: int, timeout: int = 30) -> list[dict[str, Any]]:
    """Paginate through the HF datasets-server /filter (or /rows) endpoint.

    Each page is fetched via _fetch_one_page which handles retry-with-backoff
    for HTTP 429 / HTTP 5xx and transient URLError/OSError.  Raises
    QualityUpdateError when any page cannot be fetched after exhausting retries.
    """
    all_rows: list[dict[str, Any]] = []
    offset = 0
    num_rows_total: int | None = None

    while True:
        url = f"{base_url}&offset={offset}&length={page_length}"
        page_bytes = _fetch_one_page(url, timeout)

        try:
            page: dict[str, Any] = json.loads(page_bytes)
        except json.JSONDecodeError as exc:
            raise QualityUpdateError(
                f"JSON parse error in leaderboard response: {exc}"
            ) from exc

        if not isinstance(page, dict):
            raise QualityUpdateError(
                f"Unexpected leaderboard response shape (got {type(page).__name__})"
            )

        rows: list[Any] = page.get("rows", [])
        if not isinstance(rows, list):
            raise QualityUpdateError("Leaderboard response missing 'rows' list")

        for row_entry in rows:
            if isinstance(row_entry, dict):
                row_data: Any = row_entry.get("row", {})
                if isinstance(row_data, dict):
                    all_rows.append(row_data)

        if num_rows_total is None:
            raw_total: Any = page.get("num_rows_total")
            if isinstance(raw_total, int):
                num_rows_total = raw_total

        offset += len(rows)
        if not rows:
            break
        if num_rows_total is not None and offset >= num_rows_total:
            break

    return all_rows


def _detect_columns(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Detect the model-name and score column names from the first row."""
    if not rows:
        return None, None
    first = rows[0]
    name_candidates = ["key", "model_name", "model", "name"]
    score_candidates = ["rating", "elo_rating", "arena_score", "score", "elo"]
    name_col: str | None = next((c for c in name_candidates if c in first), None)
    score_col: str | None = next((c for c in score_candidates if c in first), None)
    return name_col, score_col


def _detect_category_and_date_columns(
    rows: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """Detect the category and publish-date column names from the first row.

    Returns (category_col, date_col) where either may be None when the column
    is absent.  Absence means the dataset uses an older, simpler schema that
    does not segment by category or date — callers must treat None as
    "not present, fall back to all rows".

    Candidate lists are intentionally ordered: the first matching candidate
    wins so that if the schema adds a new alias we only need to prepend it.
    """
    if not rows:
        return None, None
    first = rows[0]
    category_candidates = ["category"]
    date_candidates = ["leaderboard_publish_date", "publish_date", "date"]
    category_col: str | None = next((c for c in category_candidates if c in first), None)
    date_col: str | None = next((c for c in date_candidates if c in first), None)
    return category_col, date_col


def fetch_and_update_quality(
    hf_base_url: str,
    output_path: Path,
    today_date_str: str,
    page_length: int = _HF_PAGE_LENGTH,
    timeout: int = 30,
) -> dict[str, int]:
    """Fetch the LMArena leaderboard, bin into tiers, and atomically update *output_path*.

    *output_path* should be the user data dir path (_QUALITY_JSON) so updates
    survive reinstalls.  The CLI passes _QUALITY_JSON directly.

    Returns {"models_synced": N} on success.
    Raises QualityUpdateError on any failure — *output_path* is never modified
    if an error occurs before the final rename.
    Raises ValueError if *hf_base_url* is not HTTPS or not in the allowed host list.
    """
    validate_fetch_url(hf_base_url, _ALLOWED_QUALITY_HOSTS)

    rows = _fetch_rows(hf_base_url, page_length, timeout=timeout)

    if not rows:
        raise QualityUpdateError("Leaderboard returned no rows")

    name_col, score_col = _detect_columns(rows)
    if name_col is None:
        first_keys = sorted(rows[0].keys()) if rows else []
        raise QualityUpdateError(
            f"Could not detect model-name column in leaderboard response. "
            f"Expected one of: key, model_name, model, name. "
            f"Available keys: {first_keys}"
        )
    if score_col is None:
        first_keys = sorted(rows[0].keys()) if rows else []
        raise QualityUpdateError(
            f"Could not detect score column in leaderboard response. "
            f"Expected one of: rating, elo_rating, arena_score, score, elo. "
            f"Available keys: {first_keys}"
        )

    # Category + date filter — applied BEFORE the max-score pass so that
    # specialty-category scores (coding, spanish, math, etc.) never pollute
    # overall quality tiers.
    #
    # Algorithm:
    #   1. Detect whether the dataset contains a category column.
    #   2. If detected: find the maximum publish date present, then keep only
    #      rows where category == "overall" AND date == max date.
    #      Zero matching rows after the filter → fail loud (schema surprise).
    #   3. If NOT detected: the dataset uses an older, simpler schema with no
    #      per-category breakdown; fall back to using all rows as before.
    category_col, date_col = _detect_category_and_date_columns(rows)

    if category_col is not None:
        # Step 1: determine the target date (latest publish date in the full set).
        # Use string max — ISO-8601 dates sort lexicographically.
        target_date: str | None = None
        if date_col is not None:
            date_values: list[str] = []
            for row in rows:
                raw_date: Any = row.get(date_col)
                if isinstance(raw_date, str) and raw_date:
                    date_values.append(raw_date)
            target_date = max(date_values) if date_values else None

        # Step 2: filter to overall category (and target date when available).
        # date_col is non-None whenever target_date is non-None (target_date is
        # derived only from date_col values), so the assert narrows the type for mypy.
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if row.get(category_col) != _OVERALL_CATEGORY:
                continue
            if target_date is not None:
                assert date_col is not None  # invariant: target_date set iff date_col detected
                if row.get(date_col) != target_date:
                    continue
            filtered.append(row)

        if not filtered:
            raise QualityUpdateError(
                f"Category column '{category_col}' is present but no rows match "
                f"category='{_OVERALL_CATEGORY}'"
                + (f" AND {date_col}='{target_date}'" if target_date else "")
                + ". The dataset schema may have changed — refusing to bin "
                "data from specialty categories as if it were overall quality."
            )

        rows = filtered
    # else: no category column detected; backward-compatible — use all rows.

    # Pass 1 — collect max score per canonical key across all rows.
    # Dedup to base-family: versioned snapshots ("gpt-4o-2024-05-13") and bare
    # names ("gpt-4o") share the same key; we keep the highest score seen.
    max_scores: dict[str, float] = {}
    for row in rows:
        arena_name: Any = row.get(name_col)
        score_raw: Any = row.get(score_col)
        if not isinstance(arena_name, str) or not arena_name:
            continue
        if score_raw is None:
            continue
        try:
            score = float(score_raw)
        except (ValueError, TypeError):
            continue

        storage_key = _arena_name_to_key(arena_name)
        if not storage_key:
            continue

        # Keep the maximum score when multiple rows map to the same key,
        # so the best-performing snapshot wins before percentile ranking.
        if storage_key not in max_scores or score > max_scores[storage_key]:
            max_scores[storage_key] = score

    if not max_scores:
        raise QualityUpdateError(
            "No valid models extracted from leaderboard — "
            "refusing to overwrite quality.json with empty data"
        )

    # Pass 2 — assign percentile-rank tiers over the full distribution.
    tier_map: dict[str, int] = _assign_percentile_tiers(max_scores)

    attribution = (
        f"Quality tiers from LMArena (lmarena-ai/leaderboard-dataset, CC-BY-4.0), "
        f"snapshot {today_date_str}"
    )
    output: dict[str, Any] = {
        "_last_synced": today_date_str,
        "_source": "lmarena-ai/leaderboard-dataset",
        "_attribution": attribution,
        "_note": (
            "Tiers are percentile-rank bands over the full leaderboard field: "
            "0=Elite(top 10%), 1=Strong(10–30%), 2=Capable(30–60%), 3=Efficient(bottom 40%). "
            "Self-recalibrates each sync as the leaderboard shifts. "
            "Run: frugon quality update"
        ),
    }
    output.update(tier_map)

    try:
        atomic_write_json(output_path, output, sort_keys=True)
    except OSError as exc:
        raise QualityUpdateError(f"Failed to write quality.json: {exc}") from exc

    return {"models_synced": len(tier_map)}
