"""Asserts that the default routing candidate pool is fully quality-rated.

Every model in _ROUTING_CANDIDATES must have a known quality tier so frugon
never auto-routes to an unrated model.  The test also asserts that the
bundled seed quality.json contains the demo log's dominant baseline model
(gpt-5.5), because an unrated baseline suppresses the tier-drop
disclosure that the report relies on.

The quality-tier checks use the BUNDLED seed (src/frugon/data/quality.json)
rather than the runtime table so the assertion holds in a fresh install (where
the user has not yet run `frugon quality update`) and is not skipped by a
user-level quality.json that predates a bundled-seed addition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import frugon
import frugon.pricing as _pricing_module
import frugon.quality as _quality_module
from frugon.cost import _ROUTING_CANDIDATES
from frugon.model_id import base_family, canonicalize, effort_family
from frugon.quality import UNRATED_TIER, _build_folded_index

assert frugon.__file__ is not None
_BUNDLED_SEED: Path = Path(frugon.__file__).parent / "data" / "quality.json"


def _load_bundled_tier_map() -> dict[str, int]:
    """Return the tier map from the bundled seed (not the user-data-dir file)."""
    raw = json.loads(_BUNDLED_SEED.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, int)}


def _is_rated_in_bundled_seed(model: str, tier_map: dict[str, int]) -> bool:
    """Mirror the quality.get_model_tier() lookup against the bundled tier map."""
    canon = canonicalize(model)
    if canon in tier_map:
        return True
    base = base_family(canon)
    return base in tier_map


def _tier_in_bundled_seed(model: str, tier_map: dict[str, int]) -> int:
    """Return *model*'s tier against the BUNDLED quality seed, mirroring the
    full ``quality.get_model_tier()`` resolution order (exact -> effort ->
    base -> effort(base) -> folded index) so this test exercises the SAME
    lookup path production uses -- not a simplified subset of it.  Returns
    UNRATED_TIER when no step resolves.
    """
    canon = canonicalize(model)
    if canon in tier_map:
        return tier_map[canon]

    effort_canon = effort_family(canon)
    if effort_canon != canon and effort_canon in tier_map:
        return tier_map[effort_canon]

    base = base_family(canon)
    if base != canon and base in tier_map:
        return tier_map[base]

    effort_base = effort_family(base)
    if effort_base != base and effort_base in tier_map:
        return tier_map[effort_base]

    folded_index = _build_folded_index(tier_map)
    if canon in folded_index:
        return folded_index[canon]
    if base != canon and base in folded_index:
        return folded_index[base]

    return UNRATED_TIER


_BUNDLED_PRICING: Path = Path(frugon.__file__).parent / "data" / "pricing.json"


def _load_bundled_pricing_keys() -> set[str]:
    """Return the set of model keys in the bundled pricing seed."""
    raw = json.loads(_BUNDLED_PRICING.read_text(encoding="utf-8"))
    return {k for k in raw if not k.startswith("_")}


class TestDefaultCandidatePoolIsRated:
    def test_all_pool_members_are_rated_in_bundled_seed(self) -> None:
        tier_map = _load_bundled_tier_map()
        unrated = [
            m
            for m in _ROUTING_CANDIDATES
            if _tier_in_bundled_seed(m, tier_map) == UNRATED_TIER
        ]
        assert unrated == [], (
            f"Default pool contains model(s) not in the bundled quality seed: {unrated}. "
            "Every candidate must appear in src/frugon/data/quality.json so "
            "frugon never silently routes to an unknown-quality model."
        )

    def test_demo_baseline_gpt_5_5_is_rated_in_bundled_seed(self) -> None:
        # The bundled demo log (sample_logs.jsonl.gz) is dominated by gpt-5.5
        # calls; an unrated baseline suppresses the tier-drop disclosure the report
        # surfaces to the user.
        tier_map = _load_bundled_tier_map()
        assert _is_rated_in_bundled_seed("gpt-5.5", tier_map), (
            "gpt-5.5 (the demo log's dominant baseline) is not in the "
            "bundled quality seed (src/frugon/data/quality.json). "
            "Add it so the report can show an honest tier-drop disclosure."
        )

    def test_routing_candidates_has_23_members(self) -> None:
        assert len(_ROUTING_CANDIDATES) == 23, (
            f"Expected 23 routing candidates, got {len(_ROUTING_CANDIDATES)}: "
            f"{_ROUTING_CANDIDATES}"
        )

    def test_all_routing_candidates_are_priced(self) -> None:
        priced_keys = _load_bundled_pricing_keys()
        unpriced = [m for m in _ROUTING_CANDIDATES if m not in priced_keys]
        assert unpriced == [], (
            f"Routing candidate(s) not in bundled pricing seed: {unpriced}. "
            "Every candidate must be priced so frugon can project costs."
        )


class TestDemoCandidatePool:
    def test_demo_candidates_pool_is_subset_of_priced_and_rated(self) -> None:
        from frugon.cost import _DEMO_CANDIDATES

        tier_map = _load_bundled_tier_map()
        priced_keys = _load_bundled_pricing_keys()
        unrated = [m for m in _DEMO_CANDIDATES if not _is_rated_in_bundled_seed(m, tier_map)]
        unpriced = [m for m in _DEMO_CANDIDATES if m not in priced_keys]
        assert unrated == [], (
            f"Demo pool contains unrated model(s): {unrated}. "
            "All demo candidates must be rated in the bundled quality seed."
        )
        assert unpriced == [], (
            f"Demo pool contains unpriced model(s): {unpriced}. "
            "All demo candidates must be priced in the bundled pricing seed."
        )

    def test_demo_candidates_unchanged(self) -> None:
        from frugon.cost import _DEMO_CANDIDATES

        assert _DEMO_CANDIDATES == [
            "claude-sonnet-4-5",
            "gpt-4.1",
            "claude-haiku-4-5",
            "gemini-2.5-flash",
            "gpt-4.1-mini",
        ], (
            "Pinning test: _DEMO_CANDIDATES changed unexpectedly. "
            "Changing the demo pool will alter committed demo numbers. "
            "Update this test and regenerate the demo GIF if this is intentional."
        )


class TestRosterInvariant:
    """FRG-OSS-034 Part 4: every _ROUTING_CANDIDATES entry is BOTH priced and
    rated through the REAL production lookup paths (frugon.pricing.get_model_price
    and frugon.quality.get_model_tier) — not a hand-rolled approximation of them.

    Both modules are pointed at the bundled seed for the duration of this test
    (via monkeypatch on _PRICING_JSON / _QUALITY_JSON) so the assertion is
    hermetic against a fresh install and is not shadowed by a stale or
    ahead-of-seed user-data-dir file left over from local dev/sync runs.
    """

    @pytest.fixture(autouse=True)
    def _point_at_bundled_seeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            _pricing_module, "_PRICING_JSON", _pricing_module._BUNDLED_SEED_PATH
        )
        monkeypatch.setattr(
            _quality_module, "_QUALITY_JSON", _quality_module._BUNDLED_SEED_PATH
        )
        _pricing_module.clear_pricing_cache()

    def test_every_roster_entry_is_priced_and_rated(self) -> None:
        from frugon.pricing import get_model_price
        from frugon.quality import get_model_tier

        unpriced: list[str] = []
        unrated: list[str] = []
        for model in _ROUTING_CANDIDATES:
            if get_model_price(model) is None:
                unpriced.append(model)
            if get_model_tier(model) == UNRATED_TIER:
                unrated.append(model)

        assert unpriced == [], (
            f"_ROUTING_CANDIDATES contains unpriced model(s): {unpriced}. "
            "Every routing candidate must resolve through get_model_price() "
            "against the bundled pricing seed."
        )
        assert unrated == [], (
            f"_ROUTING_CANDIDATES contains unrated model(s): {unrated}. "
            "Every routing candidate must resolve through get_model_tier() "
            "against the bundled quality seed (directly or via the "
            "effort/date folded index)."
        )

    def test_folded_index_resolves_grok4_grok3mini_qwenmax(self) -> None:
        """Load-bearing check for the Phase-1 fold dependency (FRG-OSS-034 §B):
        grok-4, grok-3-mini, and qwen-max carry no BARE entry in the bundled
        quality seed — they are recovered ONLY through the effort/date-fold
        reverse index (grok-4 <- grok-4-0709 date-fold; grok-3-mini <-
        grok-3-mini-high effort-fold; qwen-max <- qwen-max-0919 date-fold).
        Asserting them explicitly, by name, makes this roster's dependency on
        the Phase-1 folding machinery an enforced invariant rather than an
        implicit assumption.
        """
        from frugon.quality import get_model_tier

        for model in ("grok-4", "grok-3-mini", "qwen-max"):
            tier = get_model_tier(model)
            assert tier != UNRATED_TIER, (
                f"{model!r} must resolve to a real tier via the folded index "
                "(effort_family/base_family fold of a dated or effort-tagged "
                "seed entry) — got UNRATED_TIER. This is the load-bearing "
                "Phase-1 dependency FRG-OSS-034 relies on."
            )
