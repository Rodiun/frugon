"""Tests for frugon.quality — LMArena-backed quality tier module.

Covers:
  - _assign_percentile_tiers: synthetic 100-score golden vectors, tie handling,
    small-N, monotonicity, empty input
  - Canonicalize-backed tier lookup (exact, base_family fallback)
  - Unrated sentinel for unknown models
  - Unrated models excluded from auto-selection but allowed via --candidates
  - Attribution string present in synced file
  - Offline seed fallback (no network)
  - Mocked HTTP fetch for fetch_and_update_quality (percentile binning, dedup-to-max)
  - Atomic write and error handling
  - Privacy invariant: fetch_and_update_quality makes no call to Rodiun endpoints
  - /filter endpoint URL shape and validate_fetch_url acceptance
  - _fetch_rows retry-with-backoff on HTTP 429 / HTTP 5xx (then-200 succeeds; exhausted raises)
  - _fetch_rows retry-with-backoff on URLError (URLError-then-200 succeeds; exhausted raises)
  - TestFetchResilience: User-Agent always sent; 5xx retried (incl. exhausted); 4xx not retried
"""

from __future__ import annotations

import json
import urllib.error
from http.client import HTTPMessage
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from frugon import USER_AGENT
from frugon.quality import (
    _BUNDLED_SEED_PATH,
    _CLASSIFY_MAX_COUNT_DELTA_FRAC,
    _CLASSIFY_MAX_TIER_CHURN_FRAC,
    _CLASSIFY_MIN_MODELS,
    _CLASSIFY_MIN_ROSTER_OVERLAP_FRAC,
    _FETCH_BACKOFF_BASE,
    _FETCH_MAX_RETRIES,
    _HF_BASE_URL,
    _MAX_RESPONSE_BYTES,
    _OVERALL_CATEGORY,
    UNRATED_TIER,
    VERDICT_INVALID,
    VERDICT_MAJOR,
    VERDICT_MINOR,
    QualityUpdateError,
    _assign_percentile_tiers,
    _build_folded_index,
    _detect_category_and_date_columns,
    _fetch_one_page,
    classify_quality_update,
    fetch_and_update_quality,
    get_attribution,
    get_model_tier,
    is_unrated,
    load_quality_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hf_page(
    rows: list[dict[str, Any]],
    num_rows_total: int | None = None,
) -> bytes:
    """Build a mock HF datasets-server JSON page."""
    return json.dumps(
        {
            "features": [
                {"name": "key", "dtype": "string"},
                {"name": "rating", "dtype": "float64"},
            ],
            "rows": [
                {"row_idx": i, "row": row, "truncated_cells": []}
                for i, row in enumerate(rows)
            ],
            "num_rows_total": num_rows_total if num_rows_total is not None else len(rows),
        }
    ).encode("utf-8")


def _build_quality_json(
    tiers: dict[str, int],
    last_synced: str = "2026-01-01",
    attribution: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "_last_synced": last_synced,
        "_source": "lmarena-ai/leaderboard-dataset",
        "_attribution": attribution
        or f"Quality tiers from LMArena (lmarena-ai/leaderboard-dataset, CC-BY-4.0), snapshot {last_synced}",
        "_note": "Tier 0=Elite, 1=Strong, 2=Capable, 3=Efficient",
    }
    result.update(tiers)
    return result


# ---------------------------------------------------------------------------
# _assign_percentile_tiers — golden-vector tests
# ---------------------------------------------------------------------------


def _make_scores(n: int, top: int = 0) -> dict[str, float]:
    """Build a dict of *n* models with descending distinct scores starting at
    1000 + n*10 so model-0 has the highest score.  *top* is unused but kept
    for parametrize clarity.
    """
    return {f"model-{i}": float(1000 + (n - i) * 10) for i in range(n)}


class TestAssignPercentileTiers:
    """Golden-vector tests for _assign_percentile_tiers.

    Tier bands (from _TIER_PERCENTILES):
      position < 0.10 → tier 0  (Elite)
      position < 0.30 → tier 1  (Strong)
      position < 0.60 → tier 2  (Capable)
      else            → tier 3  (Efficient)

    position = (count of models with strictly higher score) / N
    """

    # -----------------------------------------------------------------------
    # Synthetic 100-score distribution: exact tier-count golden vectors
    # -----------------------------------------------------------------------

    def test_100_models_tier_counts_match_percentile_bands(self) -> None:
        """Arrange: 100 models with distinct descending scores.
        Assert: exactly 10 in tier 0, 20 in tier 1, 30 in tier 2, 40 in tier 3.

        With N=100 and distinct scores:
          model-i has position = i/100 (i models score strictly higher).
          i  0..9  → position 0.00..0.09 < 0.10 → tier 0  (10 models)
          i 10..29 → position 0.10..0.29 < 0.30 → tier 1  (20 models)
          i 30..59 → position 0.30..0.59 < 0.60 → tier 2  (30 models)
          i 60..99 → position 0.60..0.99 ≥ 0.60 → tier 3  (40 models)
        """
        scores = _make_scores(100)
        result = _assign_percentile_tiers(scores)

        counts = dict.fromkeys(range(4), 0)
        for tier in result.values():
            counts[tier] += 1

        assert counts[0] == 10, f"Expected 10 Elite, got {counts[0]}"
        assert counts[1] == 20, f"Expected 20 Strong, got {counts[1]}"
        assert counts[2] == 30, f"Expected 30 Capable, got {counts[2]}"
        assert counts[3] == 40, f"Expected 40 Efficient, got {counts[3]}"

    def test_100_models_top_model_is_tier_0(self) -> None:
        """The single highest-scoring model in a 100-model set is always tier 0."""
        scores = _make_scores(100)
        result = _assign_percentile_tiers(scores)
        # model-0 has the highest score (position=0.00 < 0.10)
        assert result["model-0"] == 0

    def test_100_models_bottom_model_is_tier_3(self) -> None:
        """The single lowest-scoring model in a 100-model set is always tier 3."""
        scores = _make_scores(100)
        result = _assign_percentile_tiers(scores)
        # model-99 has the lowest score (position=0.99 ≥ 0.60)
        assert result["model-99"] == 3

    # -----------------------------------------------------------------------
    # Tie handling: equal scores get the same tier
    # -----------------------------------------------------------------------

    def test_tie_at_top_all_get_tier_0(self) -> None:
        """Arrange: 5 models all sharing the top score + 95 lower-scoring models.
        Assert: all 5 top-tied models land in tier 0.

        With N=100: tied top models each have position = 0/100 = 0.00 < 0.10.
        """
        scores: dict[str, float] = {}
        for i in range(5):
            scores[f"top-{i}"] = 2000.0  # tied top score
        for i in range(95):
            scores[f"other-{i}"] = float(1000 - i)  # distinct lower scores

        result = _assign_percentile_tiers(scores)
        for i in range(5):
            assert result[f"top-{i}"] == 0, (
                f"top-{i} should be tier 0 (tied top), got {result[f'top-{i}']}"
            )

    def test_tie_splits_do_not_vary_by_insertion_order(self) -> None:
        """Two dicts with the same keys/scores but different insertion order
        must produce identical tier assignments (no key-order dependency).
        """
        scores_a: dict[str, float] = {"alpha": 900.0, "beta": 900.0, "gamma": 800.0}
        scores_b: dict[str, float] = {"gamma": 800.0, "alpha": 900.0, "beta": 900.0}
        result_a = _assign_percentile_tiers(scores_a)
        result_b = _assign_percentile_tiers(scores_b)
        assert result_a == result_b

    def test_all_models_same_score_all_tier_0(self) -> None:
        """When every model has the same score, position=0.00 for all → tier 0."""
        scores = {f"model-{i}": 1000.0 for i in range(10)}
        result = _assign_percentile_tiers(scores)
        assert all(t == 0 for t in result.values()), (
            f"All tied top scores should land in tier 0, got: {result}"
        )

    # -----------------------------------------------------------------------
    # Small-N edge cases
    # -----------------------------------------------------------------------

    def test_single_model_is_tier_0(self) -> None:
        """N=1: only one model, position=0.00 < 0.10 → tier 0."""
        result = _assign_percentile_tiers({"only-model": 1000.0})
        assert result == {"only-model": 0}

    def test_two_models_top_tier_0_bottom_tier_3(self) -> None:
        """N=2: top model position=0.00 → tier 0; bottom position=0.50 < 0.60 → tier 2.

        With N=2: bottom has 1 model strictly higher → position=1/2=0.50 < 0.60 → tier 2.
        """
        result = _assign_percentile_tiers({"high": 1500.0, "low": 900.0})
        assert result["high"] == 0
        assert result["low"] == 2  # 0.50 < 0.60 → Capable

    def test_three_models_distinct_scores(self) -> None:
        """N=3: distinct scores.

        positions: top=0/3≈0.00→tier 0; mid=1/3≈0.33→tier 2; bot=2/3≈0.67→tier 3.
        0.00 < 0.10 → Elite
        0.33 < 0.60 → Capable  (not Strong: 0.33 ≥ 0.30)
        0.67 ≥ 0.60 → Efficient
        """
        result = _assign_percentile_tiers(
            {"top": 1500.0, "mid": 1200.0, "bot": 900.0}
        )
        assert result["top"] == 0
        assert result["mid"] == 2
        assert result["bot"] == 3

    # -----------------------------------------------------------------------
    # Monotonicity: a higher score must never produce a numerically worse tier
    # -----------------------------------------------------------------------

    def test_monotonicity_higher_score_never_worse_tier(self) -> None:
        """For any two models A and B where score(A) > score(B), tier(A) ≤ tier(B).

        Tests the monotonicity property exhaustively over the 100-model set.
        """
        scores = _make_scores(100)
        result = _assign_percentile_tiers(scores)
        pairs_checked = 0
        for key_a, score_a in scores.items():
            for key_b, score_b in scores.items():
                if score_a > score_b:
                    assert result[key_a] <= result[key_b], (
                        f"Monotonicity violated: {key_a}(score={score_a}, tier={result[key_a]}) "
                        f"has worse tier than {key_b}(score={score_b}, tier={result[key_b]})"
                    )
                    pairs_checked += 1
        assert pairs_checked > 0, "No pairs checked — test logic error"

    # -----------------------------------------------------------------------
    # Empty input
    # -----------------------------------------------------------------------

    def test_empty_dict_returns_empty(self) -> None:
        """Empty input → empty output, no exception."""
        result = _assign_percentile_tiers({})
        assert result == {}


# ---------------------------------------------------------------------------
# Offline seed — load_quality_table and get_model_tier from bundled seed
# ---------------------------------------------------------------------------


class TestOfflineSeed:
    """The bundled seed quality.json must provide correct tiers without network."""

    def test_bundled_seed_exists(self) -> None:
        """Arrange/Assert: bundled seed file is present."""
        assert _BUNDLED_SEED_PATH.exists(), (
            f"Bundled quality seed not found at {_BUNDLED_SEED_PATH}"
        )

    def test_bundled_seed_is_valid_json(self) -> None:
        """Arrange: read bundled seed.  Assert: valid JSON dict."""
        with _BUNDLED_SEED_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_bundled_seed_has_required_metadata_keys(self) -> None:
        """Arrange: bundled seed.  Assert: required metadata keys present."""
        with _BUNDLED_SEED_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert "_last_synced" in data
        assert "_attribution" in data
        assert "_source" in data

    def test_bundled_seed_attribution_contains_lmarena(self) -> None:
        """Assert: attribution string references LMArena and CC-BY-4.0."""
        with _BUNDLED_SEED_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        attribution: str = data["_attribution"]
        assert "lmarena" in attribution.lower()
        assert "cc-by" in attribution.lower()

    def test_bundled_seed_has_minimum_model_coverage(self) -> None:
        """Assert: seed covers at least 4 canonical routing models (claude-3-5-haiku dropped
        from the 2026-06-10 LMArena overall leaderboard; the other 4 remain present).
        """
        tier_map, _, _ = load_quality_table()
        required = {
            "gpt-4o",
            "claude-3-5-sonnet",
            "gpt-4o-mini",
            "claude-3-haiku",
        }
        missing = required - set(tier_map.keys())
        assert not missing, f"Seed is missing required models: {missing}"

    def test_get_model_tier_gpt4o_is_rated(self) -> None:
        """gpt-4o must be present and within the valid tier range [0, 3].

        Absolute tier depends on the leaderboard snapshot: in 2024 gpt-4o was
        frontier (tier 0); in the 2026-06-10 snapshot the frontier has moved
        to Claude Opus 4.x / GPT-5.x so gpt-4o is now mid-tier (tier 2).
        The invariant that matters: it must be rated, within range, and no worse
        than gpt-4o-mini (see TestSeedOrderingConsistency).
        """
        tier = get_model_tier("gpt-4o")
        assert tier in {0, 1, 2, 3}, f"gpt-4o tier out of range: {tier}"

    def test_get_model_tier_claude35_sonnet_is_rated(self) -> None:
        """claude-3-5-sonnet-20241022 must be present and within valid tier range.

        Resolves via base_family to 'claude-3-5-sonnet'.  In the 2026-06-10
        snapshot it is tier 2 (outpaced by Claude Opus 4.x); the contract is
        that it is rated and no worse than claude-3-haiku.
        """
        tier = get_model_tier("claude-3-5-sonnet-20241022")
        assert tier in {0, 1, 2, 3}, f"claude-3-5-sonnet tier out of range: {tier}"

    def test_get_model_tier_gpt4o_mini_is_rated(self) -> None:
        """gpt-4o-mini must be present and within valid tier range."""
        tier = get_model_tier("gpt-4o-mini")
        assert tier in {0, 1, 2, 3}, f"gpt-4o-mini tier out of range: {tier}"

    def test_get_model_tier_claude35_haiku_is_rated_or_unrated(self) -> None:
        """claude-3-5-haiku: present if the seed has it, or UNRATED_TIER if absent.

        The 2026-06-10 overall leaderboard does not include claude-3-5-haiku;
        it may reappear in future snapshots.  The constraint is: if rated,
        the tier is in [0, 3].
        """
        tier = get_model_tier("claude-3-5-haiku-20241022")
        assert tier in {UNRATED_TIER, 0, 1, 2, 3}, f"Unexpected tier value: {tier}"

    def test_get_model_tier_claude3_haiku_is_tier_3(self) -> None:
        """Golden vector: claude-3-haiku-20240307 → tier 3 (via base_family).

        claude-3-haiku is a legacy model; it consistently lands in tier 3
        regardless of leaderboard snapshot because newer models always outrank it.
        """
        assert get_model_tier("claude-3-haiku-20240307") == 3


# ---------------------------------------------------------------------------
# Canonicalize-backed tier lookup
# ---------------------------------------------------------------------------


class TestCanonicalizeLookup:
    """Tier lookup must resolve gateway-prefixed and versioned model names."""

    def test_openrouter_prefix_resolves(self) -> None:
        """openrouter/gpt-4o → strip prefix → gpt-4o → same tier as the base name.

        Asserts the *canonicalization* (gateway-prefix strip), not a fixed band:
        the prefixed name must resolve to whatever tier the bare name currently
        holds, and that tier must be rated. Drift-proof across re-anchors — the
        leaderboard moving gpt-4o between bands never breaks this.
        """
        base = get_model_tier("gpt-4o")
        assert base != UNRATED_TIER
        assert get_model_tier("openrouter/gpt-4o") == base

    def test_anthropic_prefix_resolves(self) -> None:
        """anthropic/claude-3-5-sonnet-20241022 → strip prefix + base_family → base tier."""
        base = get_model_tier("claude-3-5-sonnet")
        assert base != UNRATED_TIER
        assert get_model_tier("anthropic/claude-3-5-sonnet-20241022") == base

    def test_openai_prefix_resolves(self) -> None:
        """openai/gpt-4o-mini → strip prefix → same tier as the base name."""
        base = get_model_tier("gpt-4o-mini")
        assert base != UNRATED_TIER
        assert get_model_tier("openai/gpt-4o-mini") == base

    def test_dated_snapshot_resolves_via_base_family(self) -> None:
        """gpt-4o-2024-08-06 → base_family → gpt-4o → same tier as the base name."""
        base = get_model_tier("gpt-4o")
        assert base != UNRATED_TIER
        assert get_model_tier("gpt-4o-2024-08-06") == base

    def test_compact_date_snapshot_resolves(self) -> None:
        """claude-3-haiku-20240307 → base_family strips compact date → tier 3."""
        assert get_model_tier("claude-3-haiku-20240307") == 3

    def test_unknown_model_returns_unrated(self) -> None:
        """No-such-model-xyz → UNRATED_TIER (-1)."""
        assert get_model_tier("no-such-model-xyz-9999") == UNRATED_TIER

    def test_unknown_model_is_unrated(self) -> None:
        """is_unrated returns True for unknown models."""
        assert is_unrated("no-such-model-xyz-9999") is True

    def test_known_model_is_not_unrated(self) -> None:
        """is_unrated returns False for gpt-4o."""
        assert is_unrated("gpt-4o") is False


# ---------------------------------------------------------------------------
# Regression pins — existing resolutions must stay byte-identical after the
# effort-fold + date/version-fold extension (no behaviour change for any
# model name that resolved before this change).
# ---------------------------------------------------------------------------


class TestGetModelTierNoRegression:
    """Pin three existing resolution paths against the bundled seed."""

    def test_gpt4o_still_rated_same_as_before(self) -> None:
        """gpt-4o resolves via direct canonical match, unaffected by new folds."""
        tier = get_model_tier("gpt-4o")
        assert tier in {0, 1, 2, 3}

    def test_openrouter_gpt4o_still_matches_base(self) -> None:
        """openrouter/gpt-4o still resolves to the same tier as the bare name."""
        assert get_model_tier("openrouter/gpt-4o") == get_model_tier("gpt-4o")

    def test_claude3_haiku_still_tier_3_via_base_family(self) -> None:
        """claude-3-haiku-20240307 still resolves to tier 3 via the pre-existing
        ISO/compact date base_family fold (unrelated to the new fold forms)."""
        assert get_model_tier("claude-3-haiku-20240307") == 3


# ---------------------------------------------------------------------------
# Effort-fold + date/version-fold recovery — get_model_tier
# ---------------------------------------------------------------------------


class TestGetModelTierEffortFold:
    """get_model_tier must recover a quality signal for a bare reasoning-model
    name whose only rated seed entry carries a trailing effort suffix, a
    trailing date/version pin, or both -- see model_id.effort_family and
    quality._build_folded_index.
    """

    def test_bare_gpt5_resolves_via_high_variant(self) -> None:
        """The seed rates 'gpt-5-high' but not bare 'gpt-5'; get_model_tier
        must recover the '-high' variant's tier for the bare name.
        """
        tier_map, _, _ = load_quality_table()
        assert "gpt-5-high" in tier_map, "fixture assumption: seed must carry gpt-5-high"
        assert "gpt-5" not in tier_map, "fixture assumption: bare gpt-5 must be absent"

        assert get_model_tier("gpt-5") == tier_map["gpt-5-high"]

    def test_bare_gpt5_mini_resolves_via_high_variant(self) -> None:
        """'gpt-5-mini-high' rated but bare 'gpt-5-mini' is not -- mini is a
        SKU (never folded), high is the effort suffix that IS folded.
        """
        tier_map, _, _ = load_quality_table()
        assert "gpt-5-mini-high" in tier_map
        assert "gpt-5-mini" not in tier_map

        assert get_model_tier("gpt-5-mini") == tier_map["gpt-5-mini-high"]

    def test_direct_key_shadows_folded_index(self) -> None:
        """A bare key that exists directly in the table always wins over any
        folded-index recovery -- verified against a REAL bundled-seed
        collision where the direct tier and the index-recovered tier
        actually differ (deepseek-r1 is rated directly at one tier, while
        its dated variant deepseek-r1-0528 folds to a different tier).
        """
        tier_map, _, _ = load_quality_table()
        assert "deepseek-r1" in tier_map, "fixture assumption: seed must rate bare deepseek-r1"
        assert "deepseek-r1-0528" in tier_map, "fixture assumption: seed must rate deepseek-r1-0528"

        folded_index = _build_folded_index(tier_map)
        assert folded_index.get("deepseek-r1") != tier_map["deepseek-r1"], (
            "fixture assumption: the direct tier and folded-index tier must "
            "genuinely differ for this test to prove shadowing, not coincide"
        )

        assert get_model_tier("deepseek-r1") == tier_map["deepseek-r1"]

    def test_grok4_resolves_via_dated_seed_key(self) -> None:
        """Real-seed integration: the bundled seed rates 'grok-4-0709' (a
        compact-MMDD dated key) but not bare 'grok-4'. The folded index folds
        the KEY through base_family + effort_family, not the query, so this
        can only resolve once Part B's -MMDD folding is in place.
        """
        tier_map, _, _ = load_quality_table()
        assert "grok-4-0709" in tier_map, "fixture assumption: seed must carry grok-4-0709"
        assert "grok-4" not in tier_map, "fixture assumption: bare grok-4 must be absent"

        assert get_model_tier("grok-4") == tier_map["grok-4-0709"]

    def test_qwen_max_resolves_via_dated_seed_key(self) -> None:
        """Real-seed integration: the bundled seed rates 'qwen-max-0919' but
        not bare 'qwen-max'.
        """
        tier_map, _, _ = load_quality_table()
        assert "qwen-max-0919" in tier_map, "fixture assumption: seed must carry qwen-max-0919"
        assert "qwen-max" not in tier_map, "fixture assumption: bare qwen-max must be absent"

        assert get_model_tier("qwen-max") == tier_map["qwen-max-0919"]

    def test_precedence_high_beats_thinking_on_collision(self) -> None:
        """When two folded variants collide, '-high' outranks '-thinking' per
        the deterministic precedence order.
        """
        custom = {
            "widget-high": 0,
            "widget-thinking": 2,
        }
        index = _build_folded_index(custom)
        assert index["widget"] == 0

    def test_precedence_thinking_beats_medium_on_collision(self) -> None:
        """'-thinking' outranks '-medium' per the precedence order."""
        custom = {
            "widget-medium": 3,
            "widget-thinking": 1,
        }
        index = _build_folded_index(custom)
        assert index["widget"] == 1

    def test_precedence_dated_variants_prefer_lexicographically_last_key(self) -> None:
        """Two dated-only variants (no effort suffix) folding to the same base
        -- the lexicographically LAST original key wins. For year-carrying
        forms like these (an implicit shared year), lexicographic order
        happens to track chronological order too, so this also reads as
        "newest snapshot wins" -- but the tie-break itself is a deterministic
        ordering rule, not a calendar computation (see
        test_precedence_year_less_forms_tie_break_can_invert_across_a_year for
        the case where that coincidence does NOT hold).
        """
        custom = {
            "grok-4-0709": 2,
            "grok-4-1105": 0,
        }
        index = _build_folded_index(custom)
        # "grok-4-1105" > "grok-4-0709" lexicographically -> its tier wins.
        assert index["grok-4"] == 0

    def test_precedence_year_less_forms_tie_break_can_invert_across_a_year(self) -> None:
        """Year-less -MMDD / -MM-DD forms carry no year information, so the
        lexicographic tie-break is a deterministic, stable choice -- NOT a
        genuine "newest snapshot" computation. This is the documented
        counter-example: 'model-1215' (December of some OLDER year) sorts
        lexicographically AFTER, and therefore wins over, 'model-0110'
        (January of a NEWER year), even though 0110 is chronologically the
        more recent snapshot. No such cross-year collision exists in the
        bundled seed today; this test pins the documented, deterministic
        (if not chronologically "correct") outcome so the behaviour is
        load-bearing rather than aspirational.
        """
        custom = {
            "model-1215": 1,  # December, presumed an OLDER year
            "model-0110": 3,  # January, presumed a NEWER year -- chronologically later
        }
        index = _build_folded_index(custom)
        # "model-1215" > "model-0110" lexicographically -> it wins, even
        # though it is the chronologically OLDER snapshot of the two.
        assert index["model"] == 1

    def test_folded_index_invalidates_on_table_reload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: quality.json rating only 'widget-high'; get_model_tier
        recovers 'widget' via the folded index.
        Act: rewrite the file so 'widget-high' becomes 'widget-low' with a
        different tier, monkeypatching the same _QUALITY_JSON path.
        Assert: the SECOND get_model_tier call reflects the NEW file content
        -- the folded-index cache must invalidate on table reload, not stay
        pinned to the first-seen tier_map.
        """
        import frugon.quality as q

        path = tmp_path / "quality.json"
        path.write_text(
            json.dumps(_build_quality_json({"widget-high": 0}, last_synced="2026-01-01")),
            encoding="utf-8",
        )
        monkeypatch.setattr(q, "_QUALITY_JSON", path)

        first = get_model_tier("widget")
        assert first == 0

        path.write_text(
            json.dumps(_build_quality_json({"widget-low": 3}, last_synced="2026-01-02")),
            encoding="utf-8",
        )

        second = get_model_tier("widget")
        assert second == 3, (
            "folded-index cache did not invalidate after quality.json changed"
        )

    def test_folded_index_reused_when_table_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: a stable quality.json across two lookups.
        Act: call get_model_tier twice for two different bare names that both
        recover via the same folded index.
        Assert: both resolve correctly (index survives being reused, not just
        rebuilt-and-discarded per call).
        """
        import frugon.quality as q

        path = tmp_path / "quality.json"
        path.write_text(
            json.dumps(
                _build_quality_json(
                    {"widget-high": 0, "gadget-thinking": 1}, last_synced="2026-01-01"
                )
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(q, "_QUALITY_JSON", path)

        assert get_model_tier("widget") == 0
        assert get_model_tier("gadget") == 1


# ---------------------------------------------------------------------------
# load_quality_table — custom file injection
# ---------------------------------------------------------------------------


class TestLoadQualityTable:
    """load_quality_table must read from the user data dir or fallback to seed."""

    def test_load_from_custom_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Arrange: custom quality.json at user data path.
        Act: load_quality_table.
        Assert: returns the custom tiers.
        """
        custom = tmp_path / "quality.json"
        custom.write_text(
            json.dumps(
                _build_quality_json({"test-model": 1}, last_synced="2026-01-15")
            ),
            encoding="utf-8",
        )
        import frugon.quality as q

        monkeypatch.setattr(q, "_QUALITY_JSON", custom)

        tier_map, last_synced, attribution = load_quality_table()

        assert tier_map.get("test-model") == 1
        assert last_synced == "2026-01-15"
        assert attribution is not None

    def test_load_falls_back_to_seed_when_no_user_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: user data path does not exist.
        Act: load_quality_table.
        Assert: falls back to bundled seed and returns non-empty tier_map.
        """
        import frugon.quality as q

        # Point to a non-existent file to force seed fallback
        monkeypatch.setattr(q, "_QUALITY_JSON", tmp_path / "quality.json")

        tier_map, _, _ = load_quality_table()

        assert tier_map  # seed has entries
        assert "gpt-4o" in tier_map

    def test_malformed_json_returns_empty_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: malformed quality.json.
        Act: load_quality_table.
        Assert: returns empty tier_map (no exception).
        """
        bad = tmp_path / "quality.json"
        bad.write_text("{this is not json}", encoding="utf-8")

        import frugon.quality as q

        monkeypatch.setattr(q, "_QUALITY_JSON", bad)

        tier_map, last_synced, attribution = load_quality_table()

        assert tier_map == {}
        assert last_synced is None
        assert attribution is None

    def test_attribution_exposed_by_get_attribution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: quality.json with known attribution string.
        Act: get_attribution().
        Assert: returns the attribution string.
        """
        expected_attr = "Quality tiers from LMArena (test), snapshot 2026-01-15"
        custom = tmp_path / "quality.json"
        custom.write_text(
            json.dumps(
                _build_quality_json(
                    {"gpt-4o": 0},
                    attribution=expected_attr,
                    last_synced="2026-01-15",
                )
            ),
            encoding="utf-8",
        )
        import frugon.quality as q

        monkeypatch.setattr(q, "_QUALITY_JSON", custom)

        result = get_attribution()
        assert result == expected_attr


# ---------------------------------------------------------------------------
# fetch_and_update_quality — mocked HTTP
# ---------------------------------------------------------------------------


class TestFetchAndUpdateQuality:
    """fetch_and_update_quality: mocked network, validate output shape."""

    def _mock_urlopen(self, pages: list[bytes]) -> Any:
        """Return a context-manager mock that yields pages in sequence."""
        call_count = 0

        class FakeResponse:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self, *args: object) -> bytes:
                return self._data

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        page_iter = iter(pages)

        def fake_urlopen(*args: object, **kwargs: object) -> FakeResponse:
            nonlocal call_count
            call_count += 1
            return FakeResponse(next(page_iter))

        return fake_urlopen

    def test_fetch_single_page_writes_quality_json(self, tmp_path: Path) -> None:
        """Arrange: single-page HF response with 3 distinct-score models.
        Act: fetch_and_update_quality with mocked urlopen.
        Assert: quality.json written with correct percentile-rank tiers.

        N=3 distinct scores → positions: top=0/3≈0.00, mid=1/3≈0.33, bot=2/3≈0.67.
          0.00 < 0.10 → tier 0 (Elite)
          0.33 < 0.60 → tier 2 (Capable)
          0.67 ≥ 0.60 → tier 3 (Efficient)
        """
        rows = [
            {"key": "gpt-4o-2024-05-13", "rating": 1285.0},
            {"key": "gpt-4o-mini-2024-07-18", "rating": 1130.0},
            {"key": "claude-3-haiku-20240307", "rating": 1068.0},
        ]
        page = _make_hf_page(rows, num_rows_total=3)
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page])):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
            )

        assert result["models_synced"] == 3
        data = json.loads(output.read_text(encoding="utf-8"))

        # Verify percentile-rank tiers for N=3 distinct scores
        assert data["gpt-4o"] == 0          # top scorer → position 0.00 < 0.10 → Elite
        assert data["gpt-4o-mini"] == 2     # middle → position 0.33 < 0.60 → Capable
        assert data["claude-3-haiku"] == 3  # bottom → position 0.67 ≥ 0.60 → Efficient

    def test_fetch_stores_attribution_and_last_synced(self, tmp_path: Path) -> None:
        """Assert: output JSON contains _attribution and _last_synced."""
        rows = [{"key": "gpt-4o", "rating": 1285.0}]
        page = _make_hf_page(rows)
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page])):
            fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
            )

        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["_last_synced"] == "2026-06-04"
        assert "lmarena" in data["_attribution"].lower()
        assert "cc-by" in data["_attribution"].lower()
        assert "2026-06-04" in data["_attribution"]

    def test_fetch_note_describes_percentile_bands(self, tmp_path: Path) -> None:
        """Assert: output _note describes percentile-rank bands, not absolute scores."""
        rows = [{"key": "gpt-4o", "rating": 1285.0}]
        page = _make_hf_page(rows)
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page])):
            fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
            )

        data = json.loads(output.read_text(encoding="utf-8"))
        note: str = data["_note"]
        assert "percentile" in note.lower(), (
            f"_note must describe percentile bands, got: {note!r}"
        )
        assert "frugon quality update" in note, (
            f"_note must include the update command, got: {note!r}"
        )

    def test_fetch_paginated_merges_all_pages(self, tmp_path: Path) -> None:
        """Arrange: two pages of 2 models each (4 total, distinct scores).
        Act: fetch_and_update_quality.
        Assert: all 4 models present; percentile tiers computed over full distribution.

        N=4 distinct scores → positions: 0/4=0.00, 1/4=0.25, 2/4=0.50, 3/4=0.75.
          0.00 < 0.10 → tier 0 (Elite)
          0.25 < 0.30 → tier 1 (Strong)
          0.50 < 0.60 → tier 2 (Capable)
          0.75 ≥ 0.60 → tier 3 (Efficient)
        """
        page1 = _make_hf_page(
            [
                {"key": "model-a", "rating": 1300.0},
                {"key": "model-b", "rating": 1250.0},
            ],
            num_rows_total=4,
        )
        page2 = _make_hf_page(
            [
                {"key": "model-c", "rating": 1150.0},
                {"key": "model-d", "rating": 1050.0},
            ],
            num_rows_total=4,
        )
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page1, page2])):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
                page_length=2,
            )

        assert result["models_synced"] == 4
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["model-a"] == 0  # position 0.00 < 0.10 → Elite
        assert data["model-b"] == 1  # position 0.25 < 0.30 → Strong
        assert data["model-c"] == 2  # position 0.50 < 0.60 → Capable
        assert data["model-d"] == 3  # position 0.75 ≥ 0.60 → Efficient

    def test_fetch_network_error_raises_quality_update_error(self, tmp_path: Path) -> None:
        """Arrange: urlopen raises URLError.
        Act: fetch_and_update_quality.
        Assert: QualityUpdateError raised; output file NOT written.
        """
        output = tmp_path / "quality.json"

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("no network in test"),
        ):
            with pytest.raises(QualityUpdateError, match="Network error"):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=output,
                    today_date_str="2026-06-04",
                )

        assert not output.exists(), "output file must NOT be written on network error"

    def test_fetch_empty_response_raises_quality_update_error(self, tmp_path: Path) -> None:
        """Arrange: HF returns page with no rows.
        Act: fetch_and_update_quality.
        Assert: QualityUpdateError raised.
        """
        page = _make_hf_page([])
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page])):
            with pytest.raises(QualityUpdateError):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=output,
                    today_date_str="2026-06-04",
                )

    def test_fetch_unknown_columns_raises_quality_update_error(self, tmp_path: Path) -> None:
        """Arrange: rows with unrecognised column names.
        Act: fetch_and_update_quality.
        Assert: QualityUpdateError raised with helpful message.
        """
        # Columns that don't match known name_candidates or score_candidates
        page = json.dumps(
            {
                "features": [{"name": "foo", "dtype": "string"}, {"name": "bar", "dtype": "float64"}],
                "rows": [{"row_idx": 0, "row": {"foo": "gpt-4o", "bar": 1285.0}, "truncated_cells": []}],
                "num_rows_total": 1,
            }
        ).encode("utf-8")
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page])):
            with pytest.raises(QualityUpdateError, match="detect"):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=output,
                    today_date_str="2026-06-04",
                )

    def test_fetch_dedup_to_max_score_for_duplicate_base_families(self, tmp_path: Path) -> None:
        """Arrange: two Arena snapshots of the same model family with different scores,
        plus two additional models to make the distribution meaningful (N=3 unique keys).
        Act: fetch_and_update_quality.
        Assert: deduplicated to a single key using the MAX score; percentile tier applied.

        N=3 unique keys (gpt-4o, model-mid, model-low) with scores 1300, 1100, 900.
        Positions: gpt-4o=0/3≈0.00→tier 0, model-mid=1/3≈0.33→tier 2, model-low=2/3≈0.67→tier 3.
        The two gpt-4o snapshots fold to "gpt-4o"; max(1285, 1300)=1300 is used for ranking.
        """
        rows = [
            {"key": "gpt-4o-2024-05-13", "rating": 1285.0},  # lower snapshot
            {"key": "gpt-4o-2024-08-06", "rating": 1300.0},  # higher snapshot — max wins
            {"key": "model-mid", "rating": 1100.0},
            {"key": "model-low", "rating": 900.0},
        ]
        page = _make_hf_page(rows, num_rows_total=4)
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page])):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
            )

        data = json.loads(output.read_text(encoding="utf-8"))
        # Both snapshots fold to "gpt-4o"; max score = 1300 → top of N=3 → tier 0
        assert data.get("gpt-4o") == 0
        # models_synced counts unique keys (3, not 4, because gpt-4o-* deduped)
        assert result["models_synced"] == 3

    def test_fetch_does_not_call_rodiun_endpoint(self, tmp_path: Path) -> None:
        """Privacy invariant: fetch_and_update_quality must never call any Rodiun endpoint.

        This test asserts that the only outbound calls go to the HF datasets server.
        """
        called_urls: list[str] = []

        class TrackingMock:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self, *args: object) -> bytes:
                return self._data

            def __enter__(self) -> TrackingMock:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def tracking_urlopen(req: Any, *args: object, **kwargs: object) -> TrackingMock:
            url: str = req.full_url if hasattr(req, "full_url") else str(req)
            called_urls.append(url)
            # Return valid data so the call completes
            return TrackingMock(
                _make_hf_page([{"key": "gpt-4o", "rating": 1285.0}])
            )

        output = tmp_path / "quality.json"
        with patch("urllib.request.urlopen", tracking_urlopen):
            fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
            )

        for url in called_urls:
            assert "rodiun" not in url.lower(), (
                f"Privacy violation: fetch_and_update_quality called Rodiun endpoint: {url}"
            )
            assert "frugon.io" not in url.lower(), (
                f"Privacy violation: fetch_and_update_quality called frugon.io: {url}"
            )

    def test_fetch_atomic_write_does_not_leave_tmp_file(self, tmp_path: Path) -> None:
        """Assert: no .tmp file remains after successful fetch."""
        rows = [{"key": "gpt-4o", "rating": 1285.0}]
        page = _make_hf_page(rows)
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page])):
            fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
            )

        assert not (tmp_path / "quality.tmp").exists()
        assert output.exists()


# ---------------------------------------------------------------------------
# Unrated model gating — integration with cost module
# ---------------------------------------------------------------------------


class TestUnratedModelGating:
    """Unrated models must be excluded from auto-selection but allowed via --candidates."""

    def test_unrated_model_returns_unrated_tier(self) -> None:
        """Unknown model → UNRATED_TIER sentinel (-1)."""
        assert get_model_tier("absolutely-unknown-model-12345") == UNRATED_TIER

    def test_unrated_tier_constant_is_negative(self) -> None:
        """UNRATED_TIER must be negative so arithmetic comparisons degrade gracefully."""
        assert UNRATED_TIER < 0

    def test_unrated_excluded_from_auto_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: routing candidate pool contains only an unrated model.
        Act: analyze_logs (no explicit candidates).
        Assert: candidate_model is None — unrated models must not be auto-recommended.
        """
        import json as _json
        from decimal import Decimal

        from frugon.cost import analyze_logs
        from frugon.pricing import ModelPrice

        fake_prices = {
            "gpt-4-turbo": ModelPrice(
                "gpt-4-turbo", Decimal("0.00001"), Decimal("0.00003"), "test", None
            ),
            "unlisted-cheap": ModelPrice(
                "unlisted-cheap", Decimal("0.000001"), Decimal("0.000003"), "test", None
            ),
        }
        monkeypatch.setattr("frugon.cost._ROUTING_CANDIDATES", ["unlisted-cheap"])
        monkeypatch.setattr("frugon.cost.get_model_price", fake_prices.get)

        records = [
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "test"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        ]
        log_file = tmp_path / "logs.jsonl"
        with log_file.open("w") as fh:
            for r in records:
                fh.write(_json.dumps(r) + "\n")

        result = analyze_logs(log_file)

        assert result.candidate_model is None, (
            "Unrated 'unlisted-cheap' must not be auto-recommended as candidate. "
            f"Got candidate_model={result.candidate_model!r}"
        )

    def test_unrated_allowed_via_explicit_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: explicit --candidates includes an unrated-but-priced model.
        Act: analyze_logs with candidates=['unlisted-cheap'].
        Assert: candidate_model is 'unlisted-cheap' — explicit list bypasses gating.

        The explicit-candidates path imports get_model_price directly from
        frugon.pricing, so both the module-level alias and the source must be patched.
        """
        import json as _json
        from decimal import Decimal

        from frugon.cost import analyze_logs
        from frugon.pricing import ModelPrice

        fake_prices = {
            "gpt-4o": ModelPrice(
                "gpt-4o", Decimal("0.0000025"), Decimal("0.00001"), "test", None
            ),
            "unlisted-cheap": ModelPrice(
                "unlisted-cheap", Decimal("0.0000005"), Decimal("0.000001"), "test", None
            ),
        }
        # Patch both the module-level alias (used in _best_candidate/compute_call_cost)
        # and the source in frugon.pricing (used in the explicit-candidates block).
        monkeypatch.setattr("frugon.cost.get_model_price", fake_prices.get)
        monkeypatch.setattr("frugon.pricing.get_model_price", fake_prices.get)

        records = [
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "test"}]},
                "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }
        ]
        log_file = tmp_path / "logs.jsonl"
        with log_file.open("w") as fh:
            for r in records:
                fh.write(_json.dumps(r) + "\n")

        result = analyze_logs(log_file, candidates=["unlisted-cheap"])

        assert result.candidate_model == "unlisted-cheap", (
            "Explicit --candidates must allow any priced model, including unrated ones. "
            f"Got candidate_model={result.candidate_model!r}"
        )


# ---------------------------------------------------------------------------
# HTTPS pin + response cap + 404 handling
# ---------------------------------------------------------------------------


class TestHttpsPinAndFetchSafety:
    """fetch_and_update_quality: HTTPS-only, host allowlist, 16 MB cap, 404 handling."""

    def test_http_url_raises_value_error(self, tmp_path: Path) -> None:
        """HTTP URL rejected before any network call."""
        with pytest.raises(ValueError, match="HTTPS"):
            fetch_and_update_quality(
                hf_base_url="http://datasets-server.huggingface.co/rows",
                output_path=tmp_path / "quality.json",
                today_date_str="2026-01-01",
            )

    def test_unknown_host_raises_value_error(self, tmp_path: Path) -> None:
        """HTTPS URL with disallowed host rejected before any network call."""
        with pytest.raises(ValueError, match="allowed"):
            fetch_and_update_quality(
                hf_base_url="https://evil.example.com/rows",
                output_path=tmp_path / "quality.json",
                today_date_str="2026-01-01",
            )

    def test_allowed_host_passes_validation(self, tmp_path: Path) -> None:
        """datasets-server.huggingface.co passes validation and fetch proceeds."""
        rows = [{"key": "gpt-4o", "rating": 1285.0}]
        page = _make_hf_page(rows)
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen([page])):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
            )
        assert result["models_synced"] >= 1

    def test_http_404_raises_with_friendly_message(self, tmp_path: Path) -> None:
        """HTTP 404 from the leaderboard raises QualityUpdateError with friendly message."""
        import urllib.error as _ue

        output = tmp_path / "quality.json"
        with patch(
            "urllib.request.urlopen",
            side_effect=_ue.HTTPError(
                url=_HF_BASE_URL, code=404, msg="Not Found", hdrs={}, fp=None
            ),
        ):
            with pytest.raises(QualityUpdateError, match="leaderboard unavailable"):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=output,
                    today_date_str="2026-06-04",
                )
        assert not output.exists()

    def test_response_read_called_with_16mb_cap(self, tmp_path: Path) -> None:
        """resp.read() is called with the 16 MB limit."""
        from unittest.mock import MagicMock

        from frugon.quality import _MAX_RESPONSE_BYTES

        rows = [{"key": "gpt-4o", "rating": 1285.0}]
        page_bytes = _make_hf_page(rows, num_rows_total=1)

        resp = MagicMock()
        resp.read.return_value = page_bytes
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=resp)
        ctx.__exit__ = MagicMock(return_value=False)

        output = tmp_path / "quality.json"
        with patch("urllib.request.urlopen", return_value=ctx):
            fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-04",
            )

        resp.read.assert_called_once_with(_MAX_RESPONSE_BYTES)

    def test_fetch_oversized_page_raises(self, tmp_path: Path) -> None:
        """Arrange: urlopen returns a response whose .read() yields a byte string
        that is larger than _MAX_RESPONSE_BYTES but is truncated mid-JSON (as
        .read(limit) would do in the real implementation).  The truncated JSON
        causes a json.JSONDecodeError inside _fetch_rows, which must propagate
        as QualityUpdateError — confirming the 16 MB cap fails loud rather than
        silently accepting a truncated payload.

        We mock resp.read(n) to return exactly n bytes of a JSON string that is
        only syntactically valid if read in full, simulating a mid-JSON cut-off.
        """
        # Build a byte string that is >16 MB so that .read(_MAX_RESPONSE_BYTES)
        # would cut it mid-JSON.  We construct a valid JSON prefix that becomes
        # invalid when truncated at _MAX_RESPONSE_BYTES bytes.
        oversized_prefix = b'{"rows": [' + b'{"row_idx": 0, "row": {}} ' * 1
        # Pad to slightly over the cap so the caller receives exactly
        # _MAX_RESPONSE_BYTES bytes (simulating the OS-level truncation).
        # The truncated slice will not be valid JSON (no closing }]).
        padding = b"x" * (_MAX_RESPONSE_BYTES + 1)
        truncated_at_cap = (oversized_prefix + padding)[:_MAX_RESPONSE_BYTES]

        class OversizedFakeResponse:
            def read(self, n: int = -1) -> bytes:
                # Simulate resp.read(n) returning exactly n bytes of a
                # larger body — the same behaviour as urllib3 with a limit.
                return truncated_at_cap[:n] if n >= 0 else truncated_at_cap

            def __enter__(self) -> OversizedFakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def fake_urlopen(*args: object, **kwargs: object) -> OversizedFakeResponse:
            return OversizedFakeResponse()

        output = tmp_path / "quality.json"
        with patch("urllib.request.urlopen", fake_urlopen):
            with pytest.raises(QualityUpdateError):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=output,
                    today_date_str="2026-06-19",
                )

        assert not output.exists(), (
            "output must NOT be written when the response is truncated mid-JSON"
        )

    def _mock_urlopen(self, pages: list[bytes]) -> Any:
        """Reuse the helper pattern from TestFetchAndUpdateQuality."""
        class FakeResponse:
            def __init__(self, data: bytes) -> None:
                self._data = data
            def read(self, *args: object) -> bytes:
                return self._data
            def __enter__(self) -> FakeResponse:
                return self
            def __exit__(self, *args: object) -> None:
                pass

        page_iter = iter(pages)

        def fake_urlopen(*args: object, **kwargs: object) -> FakeResponse:
            return FakeResponse(next(page_iter))

        return fake_urlopen


# ---------------------------------------------------------------------------
# Seed ordering consistency — frontier models must be better-tiered than legacy
# ---------------------------------------------------------------------------


class TestSeedOrderingConsistency:
    """Bundled seed tiers must reflect relative quality: frontier < legacy (lower = better).

    Absolute tier values from the seed depend on the full leaderboard distribution
    at sync time and are not hardcoded here.  Instead we assert that the relative
    ordering between known-better and known-worse model pairs is preserved, and that
    every model present is within the valid tier range [0, 3].
    """

    def _seed_tier_map(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, int]:
        """Return the bundled seed's tier_map, bypassing any user-synced copy."""
        import frugon.quality as q_mod

        monkeypatch.setattr(q_mod, "_QUALITY_JSON", tmp_path / "quality.json")
        tier_map, _, _ = load_quality_table()
        return tier_map

    def test_all_seed_tiers_in_valid_range(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Every tier in the seed must be 0, 1, 2, or 3."""
        tier_map = self._seed_tier_map(monkeypatch, tmp_path)
        assert tier_map, "Seed tier_map must not be empty"
        invalid = {k: v for k, v in tier_map.items() if v not in {0, 1, 2, 3}}
        assert not invalid, f"Seed contains out-of-range tier values: {invalid}"

    def test_required_models_present_in_seed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Four core routing models must be in the seed.

        claude-3-5-haiku is excluded: absent from the 2026-06-10 LMArena
        overall leaderboard snapshot.  The 4 remaining models are present.
        """
        tier_map = self._seed_tier_map(monkeypatch, tmp_path)
        required = {
            "gpt-4o",
            "claude-3-5-sonnet",
            "gpt-4o-mini",
            "claude-3-haiku",
        }
        missing = required - set(tier_map.keys())
        assert not missing, f"Required models missing from seed: {missing}"

    @pytest.mark.parametrize(
        ("better_model", "worse_model"),
        [
            # gpt-4o is mid-tier in 2026; gemini-pro (2023 era) is legacy tier 3
            ("gpt-4o", "gemini-pro"),
            # claude-3-5-sonnet is better than claude-3-haiku
            ("claude-3-5-sonnet", "claude-3-haiku"),
            # gpt-4o is better than or equal to gpt-4o-mini (both mid-tier in 2026)
            ("gpt-4o", "gpt-4o-mini"),
        ],
    )
    def test_seed_frontier_better_than_legacy(
        self,
        better_model: str,
        worse_model: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Better model tier must be <= worse model tier (lower = better).

        Uses model pairs whose relative ordering is stable across leaderboard
        snapshots.  Absolute tier values change as the field advances; the
        invariant that must hold is the ordering, not the absolute tier.

        Reads the bundled seed directly so the test validates the shipped seed,
        not any locally-synced copy.
        """
        tier_map = self._seed_tier_map(monkeypatch, tmp_path)

        assert better_model in tier_map, f"{better_model!r} missing from seed"
        assert worse_model in tier_map, f"{worse_model!r} missing from seed"

        better_tier = tier_map[better_model]
        worse_tier = tier_map[worse_model]
        assert better_tier <= worse_tier, (
            f"Ordering violated: {better_model!r}(tier={better_tier}) should be "
            f"≤ {worse_model!r}(tier={worse_tier}) in the seed"
        )


class TestQualityStale:
    """is_quality_stale threshold logic — mirrors is_pricing_stale, 60-day default."""

    def test_fresh_quality_not_stale(self) -> None:
        from frugon.quality import is_quality_stale

        assert is_quality_stale("2026-06-01", today="2026-06-10") is False

    def test_quality_fresh_under_60_days(self) -> None:
        from frugon.quality import is_quality_stale

        # 45 days old — stale for pricing (30) but fresh for quality (60).
        assert is_quality_stale("2026-04-16", today="2026-06-01") is False

    def test_quality_stale_over_60_days(self) -> None:
        from frugon.quality import is_quality_stale

        assert is_quality_stale("2026-03-01", today="2026-06-01") is True

    def test_exactly_at_60_day_threshold_is_stale(self) -> None:
        from frugon.quality import is_quality_stale

        assert is_quality_stale("2026-04-02", today="2026-06-01") is True

    def test_one_day_before_threshold_not_stale(self) -> None:
        from frugon.quality import is_quality_stale

        assert is_quality_stale("2026-04-03", today="2026-06-01") is False

    def test_custom_max_days_respected(self) -> None:
        from frugon.quality import is_quality_stale

        assert is_quality_stale("2026-05-02", max_days=30, today="2026-06-01") is True

    def test_none_last_synced_not_stale(self) -> None:
        from frugon.quality import is_quality_stale

        assert is_quality_stale(None, today="2026-06-01") is False

    def test_invalid_date_format_not_stale(self) -> None:
        from frugon.quality import is_quality_stale

        assert is_quality_stale("not-a-date", today="2026-06-01") is False


# ---------------------------------------------------------------------------
# Category + date filter — the core bug fix
# ---------------------------------------------------------------------------


class TestCategoryAndDateFilter:
    """fetch_and_update_quality must filter to overall category + latest date
    before percentile-binning.  All tests use mocked _fetch_rows to avoid
    any network call; rows are injected with the real LMArena column names.

    Column naming mirrors the live dataset:
      model_name, rating, category, leaderboard_publish_date
    """

    # ------------------------------------------------------------------
    # Helper — build a HF-server-style page with the multi-category schema
    # ------------------------------------------------------------------

    @staticmethod
    def _make_multi_cat_page(
        rows: list[dict[str, Any]],
        num_rows_total: int | None = None,
    ) -> bytes:
        """Build a mock page whose rows have the full leaderboard schema
        (model_name, rating, category, leaderboard_publish_date).
        """
        return json.dumps(
            {
                "features": [
                    {"name": "model_name", "dtype": "string"},
                    {"name": "rating", "dtype": "float64"},
                    {"name": "category", "dtype": "string"},
                    {"name": "leaderboard_publish_date", "dtype": "string"},
                ],
                "rows": [
                    {"row_idx": i, "row": row, "truncated_cells": []}
                    for i, row in enumerate(rows)
                ],
                "num_rows_total": num_rows_total
                if num_rows_total is not None
                else len(rows),
            }
        ).encode("utf-8")

    @staticmethod
    def _mock_urlopen(page: bytes) -> Any:
        """Return a urlopen-compatible callable that always returns *page*."""

        class FakeResponse:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self, *args: object) -> bytes:
                return self._data

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def fake_urlopen(*args: object, **kwargs: object) -> FakeResponse:
            return FakeResponse(page)

        return fake_urlopen

    # ------------------------------------------------------------------
    # _detect_category_and_date_columns unit tests
    # ------------------------------------------------------------------

    def test_detect_category_col_present(self) -> None:
        """Arrange: rows with the live LMArena schema.
        Assert: category column detected as 'category'.
        """
        rows = [{"model_name": "gpt-4o", "rating": 1285.0, "category": "overall",
                 "leaderboard_publish_date": "2026-06-10"}]
        cat_col, date_col = _detect_category_and_date_columns(rows)
        assert cat_col == "category"
        assert date_col == "leaderboard_publish_date"

    def test_detect_no_category_col_returns_none(self) -> None:
        """Arrange: rows with the OLD simpler schema (no category column).
        Assert: category_col is None (backward-compat path).
        """
        rows = [{"key": "gpt-4o", "rating": 1285.0}]
        cat_col, date_col = _detect_category_and_date_columns(rows)
        assert cat_col is None
        assert date_col is None

    def test_detect_publish_date_alt_column_name(self) -> None:
        """Arrange: rows with 'publish_date' (second candidate) instead of
        'leaderboard_publish_date'.  Assert: date_col resolved correctly.
        """
        rows = [{"model_name": "gpt-4o", "rating": 1285.0, "category": "overall",
                 "publish_date": "2026-06-10"}]
        cat_col, date_col = _detect_category_and_date_columns(rows)
        assert cat_col == "category"
        assert date_col == "publish_date"

    def test_detect_empty_rows_returns_none_none(self) -> None:
        """Empty row list → (None, None) — no exception."""
        cat_col, date_col = _detect_category_and_date_columns([])
        assert cat_col is None
        assert date_col is None

    # ------------------------------------------------------------------
    # Overall-category constant
    # ------------------------------------------------------------------

    def test_overall_category_constant_value(self) -> None:
        """_OVERALL_CATEGORY must be the string 'overall'."""
        assert _OVERALL_CATEGORY == "overall"

    # ------------------------------------------------------------------
    # fetch_and_update_quality: multi-category filter scenarios
    # ------------------------------------------------------------------

    def test_only_overall_rows_binned_spanish_score_ignored(
        self, tmp_path: Path
    ) -> None:
        """Arrange: dataset with 3 categories (overall, spanish, coding).
        A model has a high SPANISH score but a low OVERALL score.
        Assert: that model is tiered by its OVERALL score, not its Spanish score.

        Row distribution (all same date 2026-06-10):
          model-frontier: overall=1400, spanish=1000
          model-legacy:   overall=1100, spanish=1450 ← high spanish, low overall

        N=2 overall rows → positions: frontier=0/2=0.00→tier 0, legacy=1/2=0.50→tier 2.
        If spanish leaked in: legacy would have score 1450 → rank higher → wrong tier.
        """
        rows = [
            # overall rows
            {"model_name": "model-frontier", "rating": 1400.0,
             "category": "overall", "leaderboard_publish_date": "2026-06-10"},
            {"model_name": "model-legacy", "rating": 1100.0,
             "category": "overall", "leaderboard_publish_date": "2026-06-10"},
            # spanish rows — must be excluded
            {"model_name": "model-frontier", "rating": 1000.0,
             "category": "spanish", "leaderboard_publish_date": "2026-06-10"},
            {"model_name": "model-legacy", "rating": 1450.0,
             "category": "spanish", "leaderboard_publish_date": "2026-06-10"},
            # coding rows — must be excluded
            {"model_name": "model-frontier", "rating": 1350.0,
             "category": "coding", "leaderboard_publish_date": "2026-06-10"},
        ]
        page = self._make_multi_cat_page(rows, num_rows_total=len(rows))
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen(page)):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-19",
            )

        data = json.loads(output.read_text(encoding="utf-8"))
        # Only 2 unique keys after dedup (from the 2 overall rows)
        assert result["models_synced"] == 2

        # model-frontier: overall=1400 → top of 2 → position 0/2=0.00 → tier 0
        assert data.get("model-frontier") == 0, (
            f"model-frontier should be tier 0 (best overall), got {data.get('model-frontier')}"
        )
        # model-legacy: overall=1100 → position 1/2=0.50 < 0.60 → tier 2
        # (NOT tier 0 which it would be if spanish score 1450 were used)
        assert data.get("model-legacy") == 2, (
            f"model-legacy should be tier 2 (overall score), got {data.get('model-legacy')}; "
            "a tier 0 here means spanish/coding scores leaked into overall binning"
        )

    def test_only_latest_date_rows_used_older_date_excluded(
        self, tmp_path: Path
    ) -> None:
        """Arrange: dataset with two publish dates (2026-05-01 and 2026-06-10).
        A model has a very high score on 2026-05-01 but a low score on 2026-06-10.
        Assert: only the LATEST date (2026-06-10) is used; old scores are excluded.

        Row distribution:
          model-a: 2026-06-10 overall=1400, 2026-05-01 overall=900
          model-b: 2026-06-10 overall=1100, 2026-05-01 overall=1500

        Using only latest date → model-a tier 0, model-b tier 2.
        If old date leaked → model-b would dominate → wrong.
        """
        rows = [
            # Latest date rows
            {"model_name": "model-a", "rating": 1400.0,
             "category": "overall", "leaderboard_publish_date": "2026-06-10"},
            {"model_name": "model-b", "rating": 1100.0,
             "category": "overall", "leaderboard_publish_date": "2026-06-10"},
            # Older date rows — must be excluded
            {"model_name": "model-a", "rating": 900.0,
             "category": "overall", "leaderboard_publish_date": "2026-05-01"},
            {"model_name": "model-b", "rating": 1500.0,
             "category": "overall", "leaderboard_publish_date": "2026-05-01"},
        ]
        page = self._make_multi_cat_page(rows, num_rows_total=len(rows))
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen(page)):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-19",
            )

        data = json.loads(output.read_text(encoding="utf-8"))
        assert result["models_synced"] == 2

        # model-a: 1400 at latest date → tier 0
        assert data.get("model-a") == 0, (
            f"model-a (score 1400 at 2026-06-10) should be tier 0, got {data.get('model-a')}"
        )
        # model-b: 1100 at latest date → tier 2 (NOT the 1500 from old date)
        assert data.get("model-b") == 2, (
            f"model-b (score 1100 at 2026-06-10) should be tier 2, got {data.get('model-b')}; "
            "a tier 0 here means old-date rows leaked into binning"
        )

    def test_multiple_rows_per_model_in_overall_max_score_wins(
        self, tmp_path: Path
    ) -> None:
        """Arrange: multiple rows for the same model within overall + same date
        (bootstrap rating samples in the real dataset share the same rank).
        Assert: the MAX score is used for that model.

        model-a has 3 rows with scores 1400, 1380, 1350 → max 1400 used.
        model-b has 1 row with score 1200.

        N=2 unique keys → model-a tier 0, model-b tier 2.
        """
        rows = [
            {"model_name": "model-a", "rating": 1380.0,
             "category": "overall", "leaderboard_publish_date": "2026-06-10"},
            {"model_name": "model-a", "rating": 1400.0,
             "category": "overall", "leaderboard_publish_date": "2026-06-10"},
            {"model_name": "model-a", "rating": 1350.0,
             "category": "overall", "leaderboard_publish_date": "2026-06-10"},
            {"model_name": "model-b", "rating": 1200.0,
             "category": "overall", "leaderboard_publish_date": "2026-06-10"},
        ]
        page = self._make_multi_cat_page(rows, num_rows_total=len(rows))
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen(page)):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-19",
            )

        data = json.loads(output.read_text(encoding="utf-8"))
        # 4 input rows but only 2 unique canonical keys after dedup
        assert result["models_synced"] == 2
        # model-a max=1400 > model-b 1200 → tier 0
        assert data.get("model-a") == 0
        # model-b 1200 → position 1/2=0.50 < 0.60 → tier 2
        assert data.get("model-b") == 2

    def test_category_col_present_no_overall_rows_raises_quality_update_error(
        self, tmp_path: Path
    ) -> None:
        """Arrange: dataset has a category column but NO rows with category=='overall'.
        Assert: QualityUpdateError raised — fail loud, do not bin specialty data.

        This is the schema-changed / unknown-category scenario.  Silent binning of
        specialty-only data would produce wrong tiers with no warning to the user.
        """
        rows = [
            {"model_name": "model-a", "rating": 1400.0,
             "category": "coding", "leaderboard_publish_date": "2026-06-10"},
            {"model_name": "model-b", "rating": 1200.0,
             "category": "math", "leaderboard_publish_date": "2026-06-10"},
        ]
        page = self._make_multi_cat_page(rows, num_rows_total=len(rows))
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen(page)):
            with pytest.raises(QualityUpdateError, match="overall"):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=output,
                    today_date_str="2026-06-19",
                )

        assert not output.exists(), (
            "output file must NOT be written when fail-loud guard triggers"
        )

    def test_backward_compat_no_category_col_uses_all_rows(
        self, tmp_path: Path
    ) -> None:
        """Arrange: dataset with the OLD schema — no category column.
        Assert: all rows are used for binning (backward-compatible path).

        This ensures the filter does not break on older/simpler leaderboard
        snapshots that lack per-category breakdowns.
        """
        # Old schema: just key + rating (from the existing _make_hf_page helper)
        rows = [
            {"key": "model-top", "rating": 1400.0},
            {"key": "model-mid", "rating": 1200.0},
            {"key": "model-low", "rating": 1000.0},
        ]
        # Build a page WITHOUT category / leaderboard_publish_date columns
        page = json.dumps(
            {
                "features": [
                    {"name": "key", "dtype": "string"},
                    {"name": "rating", "dtype": "float64"},
                ],
                "rows": [
                    {"row_idx": i, "row": row, "truncated_cells": []}
                    for i, row in enumerate(rows)
                ],
                "num_rows_total": len(rows),
            }
        ).encode("utf-8")
        output = tmp_path / "quality.json"

        with patch("urllib.request.urlopen", self._mock_urlopen(page)):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=output,
                today_date_str="2026-06-19",
            )

        data = json.loads(output.read_text(encoding="utf-8"))
        # All 3 rows used; N=3 distinct scores
        assert result["models_synced"] == 3
        # model-top: position 0/3=0.00 < 0.10 → tier 0
        assert data.get("model-top") == 0
        # model-mid: position 1/3≈0.33 < 0.60 → tier 2
        assert data.get("model-mid") == 2
        # model-low: position 2/3≈0.67 ≥ 0.60 → tier 3
        assert data.get("model-low") == 3


# ---------------------------------------------------------------------------
# /filter endpoint URL shape + validate_fetch_url acceptance
# ---------------------------------------------------------------------------


class TestFilterEndpointUrl:
    """/filter endpoint URL must be well-formed and accepted by validate_fetch_url."""

    def test_hf_base_url_uses_filter_endpoint(self) -> None:
        """_HF_BASE_URL must point to /filter, not /rows."""
        assert "/filter?" in _HF_BASE_URL, (
            f"_HF_BASE_URL must use the /filter endpoint; got: {_HF_BASE_URL!r}"
        )

    def test_hf_base_url_contains_overall_where_clause(self) -> None:
        """_HF_BASE_URL must include the URL-encoded where clause for category='overall'."""
        from urllib.parse import parse_qs, unquote, urlsplit

        # The URL-encoded form: "category"='overall' → %22category%22%3D%27overall%27
        assert "where=" in _HF_BASE_URL, (
            f"_HF_BASE_URL must contain a where= clause; got: {_HF_BASE_URL!r}"
        )
        # Verify the decoded intent: double-quoted column, single-quoted value.
        parsed = urlsplit(_HF_BASE_URL)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        where_values = qs.get("where", [])
        assert where_values, "where= parameter missing from _HF_BASE_URL query string"
        where_decoded = unquote(where_values[0])
        assert "overall" in where_decoded, (
            f"where= clause must reference 'overall'; decoded: {where_decoded!r}"
        )

    def test_hf_base_url_retains_config_and_split(self) -> None:
        """_HF_BASE_URL must still include config=text and split=latest."""
        assert "config=text" in _HF_BASE_URL, "_HF_BASE_URL missing config=text"
        assert "split=latest" in _HF_BASE_URL, "_HF_BASE_URL missing split=latest"

    def test_hf_base_url_is_https(self) -> None:
        """_HF_BASE_URL must be HTTPS."""
        assert _HF_BASE_URL.startswith("https://"), (
            f"_HF_BASE_URL must be HTTPS; got: {_HF_BASE_URL!r}"
        )

    def test_validate_fetch_url_accepts_filter_endpoint(self, tmp_path: Path) -> None:
        """validate_fetch_url must accept _HF_BASE_URL (same host, HTTPS)."""
        from frugon._store import validate_fetch_url
        from frugon.quality import _ALLOWED_QUALITY_HOSTS

        # Must not raise.
        validate_fetch_url(_HF_BASE_URL, _ALLOWED_QUALITY_HOSTS)

    def test_filter_url_offset_length_composes_correctly(self) -> None:
        """_fetch_rows appends &offset=…&length=… — confirm the composition is valid."""
        from urllib.parse import parse_qs, urlsplit

        composed = f"{_HF_BASE_URL}&offset=0&length=100"
        # Must remain a valid URL with all expected params.
        parsed = urlsplit(composed)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        assert "offset" in qs
        assert "length" in qs
        assert "where" in qs
        assert "dataset" in qs


# ---------------------------------------------------------------------------
# _fetch_one_page — retry-with-exponential-backoff
# ---------------------------------------------------------------------------


class TestFetchOnePageRetry:
    """_fetch_one_page retries on HTTP 429 and transient URLError/OSError."""

    @staticmethod
    def _make_fake_resp(data: bytes) -> Any:
        """Return a context-manager response returning *data*."""
        class FakeResp:
            def __init__(self, d: bytes) -> None:
                self._d = d
            def read(self, *a: object) -> bytes:
                return self._d
            def __enter__(self) -> FakeResp:
                return self
            def __exit__(self, *a: object) -> None:
                pass
        return FakeResp(data)

    @staticmethod
    def _make_429_error(retry_after: str | None = None) -> urllib.error.HTTPError:
        """Build an HTTPError with code 429, optionally with Retry-After header."""
        hdrs: Any = MagicMock(spec=HTTPMessage)
        hdrs.get.return_value = retry_after
        return urllib.error.HTTPError(
            url="https://datasets-server.huggingface.co/filter",
            code=429,
            msg="Too Many Requests",
            hdrs=hdrs,
            fp=None,
        )

    def test_429_then_200_succeeds(self) -> None:
        """Arrange: first call raises HTTPError 429; second returns 200 data.
        Act: _fetch_one_page.
        Assert: returns the 200 data; time.sleep called once with backoff.
        """
        ok_data = b'{"rows":[],"num_rows_total":0}'
        effects = [self._make_429_error(), self._make_fake_resp(ok_data)]
        call_n = 0

        def fake_urlopen(*args: object, **kwargs: object) -> Any:
            nonlocal call_n
            e = effects[call_n]
            call_n += 1
            if isinstance(e, Exception):
                raise e
            return e

        with patch("urllib.request.urlopen", fake_urlopen), \
             patch("time.sleep") as mock_sleep:
            result = _fetch_one_page("https://datasets-server.huggingface.co/filter?x=1", 30)

        assert result == ok_data
        # time.sleep must have been called once for the 429 backoff.
        mock_sleep.assert_called_once()
        wait_arg = mock_sleep.call_args[0][0]
        assert wait_arg == _FETCH_BACKOFF_BASE * (2**0), (
            f"First retry backoff should be {_FETCH_BACKOFF_BASE}s; got {wait_arg}"
        )

    def test_429_respects_retry_after_header(self) -> None:
        """Arrange: 429 response with Retry-After: 7.
        Assert: time.sleep called with 7.0 (header overrides backoff).
        """
        ok_data = b'{"rows":[],"num_rows_total":0}'
        effects = [self._make_429_error(retry_after="7"), self._make_fake_resp(ok_data)]
        call_n = 0

        def fake_urlopen(*args: object, **kwargs: object) -> Any:
            nonlocal call_n
            e = effects[call_n]
            call_n += 1
            if isinstance(e, Exception):
                raise e
            return e

        with patch("urllib.request.urlopen", fake_urlopen), \
             patch("time.sleep") as mock_sleep:
            _fetch_one_page("https://datasets-server.huggingface.co/filter?x=1", 30)

        mock_sleep.assert_called_once_with(7.0)

    def test_exhausted_429_retries_raises_quality_update_error(self) -> None:
        """Arrange: all calls raise HTTPError 429 (1 initial + _FETCH_MAX_RETRIES retries).
        Assert: QualityUpdateError raised after all retries exhausted.
        Assert: time.sleep called _FETCH_MAX_RETRIES times.
        """
        def always_429(*args: object, **kwargs: object) -> Any:
            raise self._make_429_error()

        with patch("urllib.request.urlopen", always_429), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(QualityUpdateError, match="leaderboard unavailable"):
                _fetch_one_page("https://datasets-server.huggingface.co/filter?x=1", 30)

        assert mock_sleep.call_count == _FETCH_MAX_RETRIES, (
            f"Expected {_FETCH_MAX_RETRIES} sleep calls; got {mock_sleep.call_count}"
        )

    def test_exhausted_429_backoff_schedule_correct(self) -> None:
        """Assert the sleep durations follow the exponential schedule: 1, 2, 4, 8 s."""
        def always_429(*args: object, **kwargs: object) -> Any:
            raise self._make_429_error()

        with patch("urllib.request.urlopen", always_429), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(QualityUpdateError):
                _fetch_one_page("https://datasets-server.huggingface.co/filter?x=1", 30)

        sleep_args = [c[0][0] for c in mock_sleep.call_args_list]
        expected = [_FETCH_BACKOFF_BASE * (2**i) for i in range(_FETCH_MAX_RETRIES)]
        assert sleep_args == expected, (
            f"Backoff schedule mismatch: expected {expected}, got {sleep_args}"
        )

    def test_url_error_then_200_succeeds(self) -> None:
        """Arrange: first call raises URLError; second returns 200 data.
        Assert: returns the 200 data; time.sleep called once.
        """
        ok_data = b'{"rows":[],"num_rows_total":0}'
        call_n = 0

        def fake_urlopen(*args: object, **kwargs: object) -> Any:
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                raise urllib.error.URLError("connection refused")
            return self._make_fake_resp(ok_data)

        with patch("urllib.request.urlopen", fake_urlopen), \
             patch("time.sleep") as mock_sleep:
            result = _fetch_one_page("https://datasets-server.huggingface.co/filter?x=1", 30)

        assert result == ok_data
        mock_sleep.assert_called_once()

    def test_exhausted_url_error_raises_quality_update_error(self) -> None:
        """Arrange: all calls raise URLError.
        Assert: QualityUpdateError raised with 'Network error' message.
        Assert: time.sleep called _FETCH_MAX_RETRIES times.
        """
        def always_err(*args: object, **kwargs: object) -> Any:
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", always_err), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(QualityUpdateError, match="Network error"):
                _fetch_one_page("https://datasets-server.huggingface.co/filter?x=1", 30)

        assert mock_sleep.call_count == _FETCH_MAX_RETRIES

    def test_non_429_http_error_not_retried(self) -> None:
        """Arrange: first call raises HTTPError 404 (not 429).
        Assert: QualityUpdateError raised immediately; time.sleep NOT called.
        """
        http_404 = urllib.error.HTTPError(
            url="https://datasets-server.huggingface.co/filter?x=1",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

        def always_404(*args: object, **kwargs: object) -> Any:
            raise http_404

        with patch("urllib.request.urlopen", always_404), \
             patch("time.sleep") as mock_sleep:
            with pytest.raises(QualityUpdateError, match="leaderboard unavailable"):
                _fetch_one_page("https://datasets-server.huggingface.co/filter?x=1", 30)

        mock_sleep.assert_not_called()

    def test_os_error_retried(self) -> None:
        """Arrange: first call raises OSError; second succeeds.
        Assert: returns data; sleep called once.
        """
        ok_data = b'{"rows":[],"num_rows_total":0}'
        call_n = 0

        def fake_urlopen(*args: object, **kwargs: object) -> Any:
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                raise OSError("connection reset")
            return self._make_fake_resp(ok_data)

        with patch("urllib.request.urlopen", fake_urlopen), \
             patch("time.sleep") as mock_sleep:
            result = _fetch_one_page("https://datasets-server.huggingface.co/filter?x=1", 30)

        assert result == ok_data
        mock_sleep.assert_called_once()

    def test_fetch_rows_propagates_exhausted_retry(self, tmp_path: Path) -> None:
        """Arrange: all page fetches raise 429 until retries exhausted.
        Act: fetch_and_update_quality.
        Assert: QualityUpdateError raised; output file NOT written.
        """
        def always_429(*args: object, **kwargs: object) -> Any:
            raise urllib.error.HTTPError(
                url="https://x", code=429, msg="rate limit", hdrs={}, fp=None
            )

        output = tmp_path / "quality.json"
        with patch("urllib.request.urlopen", always_429), \
             patch("time.sleep"):
            with pytest.raises(QualityUpdateError, match="leaderboard unavailable"):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=output,
                    today_date_str="2026-06-19",
                )

        assert not output.exists(), "output must NOT be written when retries exhausted"


# ---------------------------------------------------------------------------
# classify_quality_update — validation + change-magnitude classification
# ---------------------------------------------------------------------------


def _make_valid_quality_dict(
    n: int,
    tier_override: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build a syntactically valid quality.json payload with *n* model entries.

    Models are keyed ``model-000`` … ``model-{n-1}`` (already canonical —
    ``canonicalize("model-NNN") == "model-NNN"`` because they have no
    provider prefix).  Tiers cycle through 0→1→2→3 so there is always at
    least one tier-0 model as long as n >= 1.

    *tier_override* lets individual keys be set to specific tier values for
    negative tests (e.g. setting a key to tier 5 to trigger INVALID).
    """
    data: dict[str, object] = {
        "_last_synced": "2026-06-19",
        "_source": "lmarena-ai/leaderboard-dataset",
        "_attribution": "Quality tiers from LMArena, snapshot 2026-06-19",
        "_note": "Tier 0=Elite, 1=Strong, 2=Capable, 3=Efficient",
    }
    for i in range(n):
        key = f"model-{i:03d}"
        data[key] = i % 4  # cycles 0,1,2,3 → always includes tier-0
    if tier_override:
        data.update(tier_override)  # type: ignore[arg-type]
    return data


class TestClassifyQualityUpdate:
    """classify_quality_update — all verdict paths, deterministic, no I/O."""

    # ------------------------------------------------------------------
    # MINOR — sane small drift
    # ------------------------------------------------------------------

    def test_minor_small_drift_returns_minor(self) -> None:
        """Arrange: old and new are both valid with N=200 models; only 2 tiers changed.
        Assert: verdict is MINOR (well under 20% tier-churn threshold).
        """
        old = _make_valid_quality_dict(200)
        new = _make_valid_quality_dict(200)
        # Flip 2 models' tiers (tier 0 → 1)
        for key in ["model-000", "model-004"]:
            new[key] = 1  # was 0

        verdict, reason = classify_quality_update(new, old)

        assert verdict == VERDICT_MINOR, (
            f"Expected MINOR for 2/{200} tier changes (1% churn); got {verdict!r}: {reason}"
        )
        assert "2" in reason, f"reason should mention 2 tier changes; got: {reason!r}"

    def test_minor_old_is_none_valid_new_returns_minor(self) -> None:
        """Arrange: old=None (first run); new is valid with N=200 models.
        Assert: verdict is MINOR — treated as initial sync, no baseline to compare.
        """
        new = _make_valid_quality_dict(200)

        verdict, reason = classify_quality_update(new, None)

        assert verdict == VERDICT_MINOR, (
            f"Expected MINOR for valid new with old=None; got {verdict!r}: {reason}"
        )
        assert "initial" in reason.lower() or "no prior" in reason.lower(), (
            f"reason should mention initial sync; got: {reason!r}"
        )

    def test_minor_constants_exported_correctly(self) -> None:
        """Module-level VERDICT_MINOR constant must equal the string 'MINOR'."""
        assert VERDICT_MINOR == "MINOR"

    def test_minor_identical_old_and_new_returns_minor(self) -> None:
        """Arrange: old and new are identical (no sync change this week).
        Assert: verdict is MINOR.
        """
        data = _make_valid_quality_dict(200)
        verdict, _reason = classify_quality_update(data, data)
        assert verdict == VERDICT_MINOR

    # ------------------------------------------------------------------
    # MAJOR — model-count cliff
    # ------------------------------------------------------------------

    def test_major_model_count_cliff_above_threshold(self) -> None:
        """Arrange: old has 200 models; new has 100 (50% reduction > 15% threshold).
        Assert: verdict is MAJOR and reason names the count delta.
        """
        old = _make_valid_quality_dict(200)
        new = _make_valid_quality_dict(100)

        verdict, reason = classify_quality_update(new, old)

        assert verdict == VERDICT_MAJOR, (
            f"Expected MAJOR for 50% model-count reduction; got {verdict!r}: {reason}"
        )
        assert "count" in reason.lower() or "model" in reason.lower(), (
            f"reason should mention model count; got: {reason!r}"
        )

    def test_major_model_count_increase_above_threshold(self) -> None:
        """Arrange: old has 100 models; new has 200 (100% increase > 15% threshold).
        Assert: verdict is MAJOR.
        """
        old = _make_valid_quality_dict(100)
        new = _make_valid_quality_dict(200)

        verdict, reason = classify_quality_update(new, old)

        assert verdict == VERDICT_MAJOR, (
            f"Expected MAJOR for 100% model-count increase; got {verdict!r}: {reason}"
        )

    def test_major_count_threshold_exactly_at_boundary_is_major(self) -> None:
        """Count delta at _CLASSIFY_MAX_COUNT_DELTA_FRAC + 1pp → MAJOR.

        old=100, new=116 → delta_frac = 16/100 = 0.16 > _CLASSIFY_MAX_COUNT_DELTA_FRAC (0.15).
        """
        # Derive n_new so the count delta is strictly above the threshold.
        n_old = 100
        n_new = n_old + int(n_old * _CLASSIFY_MAX_COUNT_DELTA_FRAC) + 1  # 116
        old = _make_valid_quality_dict(n_old)
        new = _make_valid_quality_dict(n_new)

        verdict, _reason = classify_quality_update(new, old)

        assert verdict == VERDICT_MAJOR

    def test_minor_count_delta_just_under_threshold(self) -> None:
        """Count delta exactly at _CLASSIFY_MAX_COUNT_DELTA_FRAC → MINOR.

        The check is STRICT (> not >=), so delta == threshold is NOT MAJOR.
        old=100, new=115 → delta_frac = 0.15 == _CLASSIFY_MAX_COUNT_DELTA_FRAC → not strictly >.
        """
        n_old = 100
        n_new = n_old + int(n_old * _CLASSIFY_MAX_COUNT_DELTA_FRAC)  # 115, exactly 15%
        old = _make_valid_quality_dict(n_old)
        new = _make_valid_quality_dict(n_new)

        verdict, _reason = classify_quality_update(new, old)

        # 0.15 > 0.15 is False → not MAJOR on this axis; tier churn should also be minor
        assert verdict == VERDICT_MINOR

    # ------------------------------------------------------------------
    # MAJOR — tier churn
    # ------------------------------------------------------------------

    def test_major_tier_churn_above_threshold(self) -> None:
        """Arrange: 200 shared models; 50 (25%) changed tier → above 20% threshold.
        Assert: verdict is MAJOR and reason names the churn percentage.
        """
        old = _make_valid_quality_dict(200)
        new = dict(old)  # shallow copy to mutate
        # Flip tier on 50 models (25% churn)
        for i in range(50):
            key = f"model-{i:03d}"
            old_tier = int(old[key])  # type: ignore[arg-type]
            new[key] = (old_tier + 1) % 4

        verdict, reason = classify_quality_update(new, old)

        assert verdict == VERDICT_MAJOR, (
            f"Expected MAJOR for 25% tier churn; got {verdict!r}: {reason}"
        )
        assert "%" in reason, f"reason should include percentage; got: {reason!r}"

    def test_major_tier_churn_threshold_exact_boundary_is_major(self) -> None:
        """Tier churn above _CLASSIFY_MAX_TIER_CHURN_FRAC → MAJOR.

        n_flip = int(100 * _CLASSIFY_MAX_TIER_CHURN_FRAC) + 1 → 21 of 100 → 21% > 20%.
        """
        n = 100
        n_flip = int(n * _CLASSIFY_MAX_TIER_CHURN_FRAC) + 1  # 21 → 21% > 20%
        old = _make_valid_quality_dict(n)
        new = dict(old)
        for i in range(n_flip):
            key = f"model-{i:03d}"
            old_tier = int(old[key])  # type: ignore[arg-type]
            new[key] = (old_tier + 1) % 4

        verdict, _reason = classify_quality_update(new, old)

        assert verdict == VERDICT_MAJOR

    def test_minor_tier_churn_just_under_threshold(self) -> None:
        """Tier churn exactly at _CLASSIFY_MAX_TIER_CHURN_FRAC → MINOR.

        The check is STRICT (> not >=): n_flip = int(100 * threshold) = 20 of 100 → 20% == threshold → MINOR.
        """
        n = 100
        n_flip = int(n * _CLASSIFY_MAX_TIER_CHURN_FRAC)  # 20 → 20% == threshold
        old = _make_valid_quality_dict(n)
        new = dict(old)
        for i in range(n_flip):
            key = f"model-{i:03d}"
            old_tier = int(old[key])  # type: ignore[arg-type]
            new[key] = (old_tier + 1) % 4

        verdict, _reason = classify_quality_update(new, old)

        assert verdict == VERDICT_MINOR

    # ------------------------------------------------------------------
    # INVALID — too few models
    # ------------------------------------------------------------------

    def test_invalid_too_few_models_returns_invalid(self) -> None:
        """Arrange: new has 49 models (< _CLASSIFY_MIN_MODELS = 50).
        Assert: verdict is INVALID and reason names the count.
        """
        new = _make_valid_quality_dict(49)
        old = _make_valid_quality_dict(200)

        verdict, reason = classify_quality_update(new, old)

        assert verdict == VERDICT_INVALID, (
            f"Expected INVALID for 49 models; got {verdict!r}: {reason}"
        )
        assert "49" in reason, f"reason should include the model count; got: {reason!r}"

    def test_invalid_too_few_models_with_old_none(self) -> None:
        """Arrange: new has 1 model; old=None.  Assert: INVALID regardless of old.

        The minimum-model floor is an absolute structural invariant; the absence
        of a prior version must not suppress it.
        """
        new = _make_valid_quality_dict(1)

        verdict, reason = classify_quality_update(new, None)

        assert verdict == VERDICT_INVALID
        assert "1" in reason

    def test_invalid_min_models_constant_is_50(self) -> None:
        """Module-level constant must be 50."""
        assert _CLASSIFY_MIN_MODELS == 50

    # ------------------------------------------------------------------
    # INVALID — out-of-range tier value
    # ------------------------------------------------------------------

    def test_invalid_out_of_range_tier_value_returns_invalid(self) -> None:
        """Arrange: new has a model with tier=5 (not in {0,1,2,3}).
        Assert: verdict is INVALID and reason mentions the bad tier.
        """
        new = _make_valid_quality_dict(200, tier_override={"model-099": 5})
        old = _make_valid_quality_dict(200)

        verdict, reason = classify_quality_update(new, old)

        assert verdict == VERDICT_INVALID, (
            f"Expected INVALID for tier=5; got {verdict!r}: {reason}"
        )
        # reason must include either the key or the value
        assert "model-099" in reason or "5" in reason, (
            f"reason should identify the bad entry; got: {reason!r}"
        )

    def test_invalid_negative_tier_value_returns_invalid(self) -> None:
        """Arrange: model with tier=-1 (UNRATED_TIER).  Assert: INVALID.

        UNRATED_TIER is a sentinel for callers, not a valid stored value.
        """
        new = _make_valid_quality_dict(200, tier_override={"model-010": -1})
        old = _make_valid_quality_dict(200)

        verdict, reason = classify_quality_update(new, old)

        assert verdict == VERDICT_INVALID

    def test_invalid_tier_4_returns_invalid(self) -> None:
        """Tier 4 is out of range (valid range is 0–3).  Assert: INVALID."""
        new = _make_valid_quality_dict(200, tier_override={"model-020": 4})

        verdict, _reason = classify_quality_update(new, None)

        assert verdict == VERDICT_INVALID

    def test_reason_is_shell_and_markdown_inert(self) -> None:
        """A reason derived from fetched model keys must be free of shell and
        markdown metacharacters before it reaches the git-commit / PR-body sinks.

        quality-sync.yml surfaces the reason in a ``git commit -m`` argument
        (MINOR path) and a PR body (MAJOR path).  _sanitize_reason makes that
        safety *structural* — not an accident of which verdict branch ran — so a
        hostile model key fetched from the upstream dataset cannot carry shell
        or markdown metacharacters through to either sink.
        """
        new = _make_valid_quality_dict(60)
        # A fetched key laced with shell + markdown metacharacters, given an
        # out-of-range tier so it surfaces verbatim in the INVALID reason.
        evil_key = "evil`$(rm -rf /)`;&& cat /etc/passwd | sh <x> '\"\\"
        new[evil_key] = 99

        verdict, reason = classify_quality_update(new, None)

        assert verdict == VERDICT_INVALID, (
            f"Expected INVALID for out-of-range tier; got {verdict!r}: {reason}"
        )
        for metachar in ["`", "$", ";", "&", "|", "<", ">", '"', "'", "\\"]:
            assert metachar not in reason, (
                f"reason must be metachar-inert; found {metachar!r} in: {reason!r}"
            )

    # ------------------------------------------------------------------
    # INVALID — missing tier-0 (frontier absent)
    # ------------------------------------------------------------------

    def test_invalid_no_tier0_models_returns_invalid(self) -> None:
        """Arrange: all 200 models have tier >= 1 (no tier-0 Elite models).
        Assert: verdict is INVALID with reason mentioning tier-0 / Elite.
        """
        new = _make_valid_quality_dict(200)
        # Overwrite every tier-0 model to tier-1
        tier_0_keys = [k for k, v in new.items() if not k.startswith("_") and v == 0]
        for k in tier_0_keys:
            new[k] = 1

        verdict, reason = classify_quality_update(new, None)

        assert verdict == VERDICT_INVALID, (
            f"Expected INVALID when no tier-0 models; got {verdict!r}: {reason}"
        )
        assert "tier-0" in reason or "elite" in reason.lower() or "frontier" in reason.lower(), (
            f"reason should mention missing frontier; got: {reason!r}"
        )

    # ------------------------------------------------------------------
    # INVALID — non-canonical model key
    # ------------------------------------------------------------------

    def test_invalid_non_canonical_key_openrouter_prefix_returns_invalid(self) -> None:
        """Arrange: new contains 'openrouter/gpt-4o' (has prefix → not canonical).
        Assert: verdict is INVALID and reason names the key.
        """
        new = _make_valid_quality_dict(200)
        new["openrouter/gpt-4o"] = 0  # type: ignore[assignment]

        verdict, reason = classify_quality_update(new, None)

        assert verdict == VERDICT_INVALID, (
            f"Expected INVALID for non-canonical key 'openrouter/gpt-4o'; got {verdict!r}"
        )
        assert "openrouter/gpt-4o" in reason, (
            f"reason should name the non-canonical key; got: {reason!r}"
        )

    def test_invalid_non_canonical_key_anthropic_prefix_returns_invalid(self) -> None:
        """Arrange: new contains 'anthropic/claude-3-5-sonnet' (provider prefix).
        Assert: INVALID.
        """
        new = _make_valid_quality_dict(200)
        new["anthropic/claude-3-5-sonnet"] = 1  # type: ignore[assignment]

        verdict, reason = classify_quality_update(new, None)

        assert verdict == VERDICT_INVALID
        assert "anthropic/claude-3-5-sonnet" in reason

    def test_valid_canonical_key_is_not_flagged(self) -> None:
        """Arrange: new contains 'gpt-4o' (canonical, no prefix).
        Assert: NOT flagged as non-canonical — verdict is MINOR.
        """
        new = _make_valid_quality_dict(200)
        new["gpt-4o"] = 0  # type: ignore[assignment]  # canonical key

        verdict, _reason = classify_quality_update(new, None)

        assert verdict == VERDICT_MINOR

    # ------------------------------------------------------------------
    # MAJOR — roster overlap below threshold
    # ------------------------------------------------------------------

    def test_major_disjoint_roster_returns_major(self) -> None:
        """Arrange: old has 100 models; new has 100 completely different models.
        All new models are valid (tiers cycle 0→1→2→3, canonical keys).
        Assert: verdict is MAJOR — disjoint roster (0% overlap < 70% threshold).

        Without the roster-overlap guard this would pass all invariant checks
        and slip through as MINOR because common_keys is empty and the churn
        check is skipped.
        """
        # Old roster: model-old-000 … model-old-099
        old_data: dict[str, object] = {
            "_last_synced": "2026-06-12",
            "_source": "lmarena-ai/leaderboard-dataset",
            "_attribution": "test",
            "_note": "test",
        }
        for i in range(100):
            old_data[f"model-old-{i:03d}"] = i % 4

        # New roster: model-new-000 … model-new-099 (completely disjoint)
        new_data: dict[str, object] = {
            "_last_synced": "2026-06-19",
            "_source": "lmarena-ai/leaderboard-dataset",
            "_attribution": "test",
            "_note": "test",
        }
        for i in range(100):
            new_data[f"model-new-{i:03d}"] = i % 4

        verdict, reason = classify_quality_update(new_data, old_data)

        assert verdict == VERDICT_MAJOR, (
            f"Expected MAJOR for 0% roster overlap (disjoint rosters); got {verdict!r}: {reason}"
        )
        assert "0.0%" in reason or "roster" in reason.lower(), (
            f"reason should mention the overlap percentage or roster; got: {reason!r}"
        )
        # Reason must not contain raw model names from the fetched data
        assert "model-new-" not in reason, (
            f"reason must not embed raw fetched model names; got: {reason!r}"
        )

    def test_major_low_overlap_roster_returns_major(self) -> None:
        """Arrange: old has 100 models; new shares only 50 of them (50% overlap < 70%).
        New roster is padded with 50 entirely different models to keep count stable.
        Assert: verdict is MAJOR — roster overlap below threshold.
        """
        # Old roster: model-000 … model-099
        old_data: dict[str, object] = {
            "_last_synced": "2026-06-12",
            "_source": "lmarena-ai/leaderboard-dataset",
            "_attribution": "test",
            "_note": "test",
        }
        for i in range(100):
            old_data[f"model-{i:03d}"] = i % 4

        # New roster: model-000 … model-049 (50 shared) + model-alt-050 … model-alt-099 (50 new)
        new_data: dict[str, object] = {
            "_last_synced": "2026-06-19",
            "_source": "lmarena-ai/leaderboard-dataset",
            "_attribution": "test",
            "_note": "test",
        }
        for i in range(50):
            new_data[f"model-{i:03d}"] = i % 4  # shared with old
        for i in range(50):
            new_data[f"model-alt-{i:03d}"] = i % 4  # not in old

        verdict, reason = classify_quality_update(new_data, old_data)

        assert verdict == VERDICT_MAJOR, (
            f"Expected MAJOR for 50% roster overlap (< 70% threshold); got {verdict!r}: {reason}"
        )
        assert "50.0%" in reason or "roster" in reason.lower(), (
            f"reason should mention the overlap percentage; got: {reason!r}"
        )

    def test_major_roster_overlap_at_threshold_not_major(self) -> None:
        """Arrange: old has 100 models; new shares exactly 70 of them (70% overlap).
        The check is STRICT (< not <=): 70% == threshold is NOT MAJOR.
        The remaining 30 new keys are different but keep total count within 15%.

        Note: new count = 70 shared + 30 new = 100 → count delta 0% → not MAJOR on count.
        Tier churn: 0 shared keys changed tier → 0% churn → not MAJOR.
        Assert: verdict is MINOR.
        """
        old_data: dict[str, object] = {
            "_last_synced": "2026-06-12",
            "_source": "lmarena-ai/leaderboard-dataset",
            "_attribution": "test",
            "_note": "test",
        }
        for i in range(100):
            old_data[f"model-{i:03d}"] = i % 4

        new_data: dict[str, object] = {
            "_last_synced": "2026-06-19",
            "_source": "lmarena-ai/leaderboard-dataset",
            "_attribution": "test",
            "_note": "test",
        }
        for i in range(70):
            new_data[f"model-{i:03d}"] = i % 4  # 70 shared, same tiers
        for i in range(30):
            new_data[f"model-extra-{i:03d}"] = i % 4  # 30 new keys

        verdict, _reason = classify_quality_update(new_data, old_data)

        assert verdict == VERDICT_MINOR, (
            f"Expected MINOR for exactly 70% overlap (at threshold, not below); got {verdict!r}"
        )

    def test_minor_roster_overlap_constant_is_70_pct(self) -> None:
        """Module-level constant must be 0.70."""
        assert _CLASSIFY_MIN_ROSTER_OVERLAP_FRAC == 0.70

    # ------------------------------------------------------------------
    # Verdict constants
    # ------------------------------------------------------------------

    def test_verdict_constants_are_correct_strings(self) -> None:
        """All three verdict constants must be the expected strings."""
        assert VERDICT_INVALID == "INVALID"
        assert VERDICT_MINOR == "MINOR"
        assert VERDICT_MAJOR == "MAJOR"

    def test_classify_returns_tuple_of_two_strings(self) -> None:
        """classify_quality_update must always return a (str, str) tuple."""
        new = _make_valid_quality_dict(200)
        result = classify_quality_update(new, None)
        assert isinstance(result, tuple)
        assert len(result) == 2
        verdict, reason = result
        assert isinstance(verdict, str)
        assert isinstance(reason, str)

    def test_reason_is_non_empty_string(self) -> None:
        """The reason string must be non-empty for all verdict paths."""
        cases = [
            # MINOR — valid with old=None
            (_make_valid_quality_dict(200), None),
            # MINOR — valid with same old
            (_make_valid_quality_dict(200), _make_valid_quality_dict(200)),
            # INVALID — too few
            (_make_valid_quality_dict(10), None),
            # MAJOR — count cliff
            (_make_valid_quality_dict(100), _make_valid_quality_dict(200)),
        ]
        for new, old in cases:
            _verdict, reason = classify_quality_update(new, old)
            assert reason, f"reason must be non-empty; new has {len(new)} keys, old={'None' if old is None else len(old)}"


# ---------------------------------------------------------------------------
# Fetch resilience — User-Agent header + 5xx retry
#
# The HF datasets-server rejects the default ``Python-urllib`` User-Agent with
# HTTP 500, and returns sporadic 500s on individual /filter pages under load.
# A header-less fetch with no 5xx retry is why a live ``quality update`` failed
# even though the dataset was up. These tests pin both fixes so the regression
# (which the mocked happy-path tests above could never catch) cannot return.
# ---------------------------------------------------------------------------


class TestFetchResilience:
    """User-Agent header is always sent; 5xx is retried; 4xx is not."""

    @staticmethod
    def _fake_response(data: bytes) -> Any:
        class FakeResponse:
            def read(self, *args: object) -> bytes:
                return data

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        return FakeResponse()

    def test_fetch_sends_identifying_user_agent(self, tmp_path: Path) -> None:
        """Every leaderboard request carries an explicit frugon User-Agent — the
        datasets-server 500s a header-less request."""
        page = _make_hf_page([{"key": "gpt-4o", "rating": 1285.0}], num_rows_total=1)
        captured: list[Any] = []

        def capturing_urlopen(req: Any, *args: object, **kwargs: object) -> Any:
            captured.append(req)
            return self._fake_response(page)

        with patch("urllib.request.urlopen", capturing_urlopen):
            fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=tmp_path / "quality.json",
                today_date_str="2026-06-04",
            )

        assert captured, "expected at least one request"
        ua = captured[0].get_header("User-agent")
        assert ua is not None, "request sent without a User-Agent header"
        assert ua == USER_AGENT
        assert "frugon/" in ua

    def test_fetch_retries_on_transient_5xx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sporadic 5xx on a page is retried (HF returns intermittent 500s),
        not treated as a fatal error."""
        page = _make_hf_page([{"key": "gpt-4o", "rating": 1285.0}], num_rows_total=1)
        calls = {"n": 0}

        def flaky_urlopen(req: Any, *args: object, **kwargs: object) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(req.full_url, 500, "Server Error", None, None)  # type: ignore[arg-type]
            return self._fake_response(page)

        monkeypatch.setattr("time.sleep", lambda *args: None)  # skip real backoff
        with patch("urllib.request.urlopen", flaky_urlopen):
            result = fetch_and_update_quality(
                hf_base_url=_HF_BASE_URL,
                output_path=tmp_path / "quality.json",
                today_date_str="2026-06-04",
            )

        assert calls["n"] >= 2, "a transient 500 must be retried, not fatal"
        assert result["models_synced"] == 1

    def test_fetch_does_not_retry_on_4xx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 4xx client error is a permanent failure — not retried."""
        calls = {"n": 0}

        def always_404(req: Any, *args: object, **kwargs: object) -> Any:
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)  # type: ignore[arg-type]

        monkeypatch.setattr("time.sleep", lambda *args: None)
        with patch("urllib.request.urlopen", always_404):
            with pytest.raises(QualityUpdateError):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=tmp_path / "quality.json",
                    today_date_str="2026-06-04",
                )

        assert calls["n"] == 1, "4xx must not be retried"

    def test_exhausted_5xx_retries_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A persistent 5xx exhausts the bounded retry budget and then raises,
        rather than retrying forever — the budget is exactly _FETCH_MAX_RETRIES + 1
        total attempts (1 initial + the retries)."""
        calls = {"n": 0}

        def always_500(req: Any, *args: object, **kwargs: object) -> Any:
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 500, "Server Error", None, None)  # type: ignore[arg-type]

        monkeypatch.setattr("time.sleep", lambda *args: None)
        output = tmp_path / "quality.json"
        with patch("urllib.request.urlopen", always_500):
            with pytest.raises(QualityUpdateError, match="leaderboard unavailable"):
                fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=output,
                    today_date_str="2026-06-04",
                )

        assert calls["n"] == _FETCH_MAX_RETRIES + 1, (
            f"Expected {_FETCH_MAX_RETRIES + 1} attempts; got {calls['n']}"
        )
        assert not output.exists(), "output must NOT be written when retries exhausted"
