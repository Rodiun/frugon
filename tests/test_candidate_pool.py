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

import frugon
from frugon.cost import _ROUTING_CANDIDATES
from frugon.model_id import base_family, canonicalize

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


_BUNDLED_PRICING: Path = Path(frugon.__file__).parent / "data" / "pricing.json"


def _load_bundled_pricing_keys() -> set[str]:
    """Return the set of model keys in the bundled pricing seed."""
    raw = json.loads(_BUNDLED_PRICING.read_text(encoding="utf-8"))
    return {k for k in raw if not k.startswith("_")}


class TestDefaultCandidatePoolIsRated:
    def test_all_pool_members_are_rated_in_bundled_seed(self) -> None:
        tier_map = _load_bundled_tier_map()
        unrated = [m for m in _ROUTING_CANDIDATES if not _is_rated_in_bundled_seed(m, tier_map)]
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

    def test_routing_candidates_has_10_members(self) -> None:
        assert len(_ROUTING_CANDIDATES) == 10, (
            f"Expected 10 routing candidates, got {len(_ROUTING_CANDIDATES)}: "
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
