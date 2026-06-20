"""Tests for the escalation-ladder pick logic (``frugon.cost.next_rung_up``).

When ``--judge`` returns a NOT-confirmed verdict, the tool points the user at the
next-cheapest model that is a quality tier ABOVE the failed candidate yet still
cheaper than the baseline.  These tests pin the selection contract so the pick is
deterministic and honest:

  * the winner is strictly cheaper than the baseline AND a strictly better tier
    than the failed candidate;
  * among qualifying models the CHEAPEST wins (max remaining saving while
    stepping quality up), ties broken by name;
  * no qualifying model → ``None`` (the honest dead-end);
  * the cheaper-than-baseline percentage is the real figure, floored.

All tests inject a small explicit ``pool`` and monkeypatch the tier/price lookups
so they assert the ALGORITHM, never the shipped pricing/quality snapshots.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from frugon import cost


@pytest.fixture
def stub_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch tier + blended-price lookups with a small deterministic world.

    Tiers (lower = better): cheap_tier3=3, mid_tier2=2, strong_tier1=1,
    elite_tier0a/elite_tier0b=0, baseline=unrated.  Prices are blended per-token.
    """
    tiers = {
        "baseline": cost._UNRATED_TIER,
        "cheap_tier3": 3,
        "mid_tier2": 2,
        "strong_tier1": 1,
        "elite_tier0a": 0,
        "elite_tier0b": 0,
        "unrated_cheap": cost._UNRATED_TIER,
    }
    prices = {
        "baseline": Decimal("0.00002"),
        "cheap_tier3": Decimal("0.0000005"),
        "mid_tier2": Decimal("0.000001"),
        "strong_tier1": Decimal("0.000004"),
        "elite_tier0a": Decimal("0.000006"),
        "elite_tier0b": Decimal("0.000009"),
        "unrated_cheap": Decimal("0.0000001"),
    }
    monkeypatch.setattr(cost, "_get_model_tier", lambda m: tiers.get(m, cost._UNRATED_TIER))
    monkeypatch.setattr(cost, "_blended_price", lambda m: prices.get(m))
    monkeypatch.setattr(cost, "_quality_tier_name", lambda t: {0: "Elite", 1: "Strong", 2: "Capable", 3: "Efficient"}.get(t))


_POOL = [
    "elite_tier0a",
    "elite_tier0b",
    "strong_tier1",
    "mid_tier2",
    "cheap_tier3",
    "unrated_cheap",
]


def test_next_rung_up_picks_cheapest_higher_tier_than_failed(stub_tables: None) -> None:
    # Failed candidate is tier-2; baseline unrated but priced at 0.00002.
    # Candidates a tier above 2 AND cheaper than baseline: strong_tier1 (0.000004),
    # elite_tier0a (0.000006), elite_tier0b (0.000009).  Cheapest = strong_tier1.
    s = cost.next_rung_up("mid_tier2", "baseline", pool=_POOL)
    assert s is not None
    assert s.model == "strong_tier1"
    assert s.tier == 1
    assert s.tier_label == "Strong"


def test_next_rung_up_winner_is_cheaper_than_baseline_and_better_tier(
    stub_tables: None,
) -> None:
    s = cost.next_rung_up("cheap_tier3", "baseline", pool=_POOL)
    assert s is not None
    # Strictly better tier than the failed tier-3 candidate.
    assert s.tier < 3
    # Strictly cheaper than the baseline blended price.
    assert cost._blended_price(s.model) < cost._blended_price("baseline")
    # Cheapest qualifying tier<3 model is mid_tier2 (0.000001).
    assert s.model == "mid_tier2"


def test_next_rung_up_tie_broken_by_name(
    stub_tables: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Make the two elite models cost EXACTLY the same and the only qualifiers, so
    # the deterministic name tie-break decides — elite_tier0a sorts before 0b.
    prices = {
        "baseline": Decimal("0.00002"),
        "elite_tier0a": Decimal("0.000006"),
        "elite_tier0b": Decimal("0.000006"),
        "strong_tier1": Decimal("0.00005"),  # pricier than baseline → excluded
        "mid_tier2": Decimal("0.00005"),
    }
    monkeypatch.setattr(cost, "_blended_price", lambda m: prices.get(m))
    s = cost.next_rung_up("mid_tier2", "baseline", pool=_POOL)
    assert s is not None
    assert s.model == "elite_tier0a"


def test_next_rung_up_no_cheaper_higher_tier_returns_none(stub_tables: None) -> None:
    # Failed candidate is already the BEST tier in the pool (tier-0): nothing is a
    # tier above it → honest dead-end.
    assert cost.next_rung_up("elite_tier0a", "baseline", pool=_POOL) is None


def test_next_rung_up_higher_tier_but_not_cheaper_returns_none(
    stub_tables: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every higher-tier model is MORE expensive than the (cheap) baseline → no rung
    # that also saves money.
    monkeypatch.setattr(cost, "_blended_price", lambda m: {
        "baseline": Decimal("0.0000001"),  # baseline is already very cheap
        "cheap_tier3": Decimal("0.0000005"),
        "mid_tier2": Decimal("0.000001"),
        "strong_tier1": Decimal("0.000004"),
        "elite_tier0a": Decimal("0.000006"),
        "elite_tier0b": Decimal("0.000009"),
        "unrated_cheap": Decimal("0.00000005"),
    }.get(m))
    assert cost.next_rung_up("mid_tier2", "baseline", pool=_POOL) is None


def test_next_rung_up_excludes_unrated_candidates(
    stub_tables: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # unrated_cheap is the cheapest model overall, but unrated → never auto-picked.
    # The winner must be the cheapest RATED higher-tier model instead.
    s = cost.next_rung_up("mid_tier2", "baseline", pool=_POOL)
    assert s is not None
    assert s.model != "unrated_cheap"


def test_next_rung_up_unrated_failed_candidate_returns_none(stub_tables: None) -> None:
    # Cannot reason about "a tier above" an unrated failed candidate.
    assert cost.next_rung_up("unrated_cheap", "baseline", pool=_POOL) is None


def test_next_rung_up_unpriced_baseline_returns_none(
    stub_tables: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cost, "_blended_price", lambda m: None if m == "baseline" else Decimal("0.000001"))
    assert cost.next_rung_up("mid_tier2", "baseline", pool=_POOL) is None


def test_next_rung_up_percentage_is_real_and_floored(stub_tables: None) -> None:
    # baseline 0.00002, winner strong_tier1 0.000004 → saving (0.00002-0.000004)/0.00002
    # = 0.8 → 80%.
    s = cost.next_rung_up("mid_tier2", "baseline", pool=_POOL)
    assert s is not None
    assert s.pct_cheaper_than_baseline == 80


def test_next_rung_up_percentage_floors_not_rounds(
    stub_tables: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # saving fraction 0.499 → must FLOOR to 49, never round up to 50.
    monkeypatch.setattr(cost, "_blended_price", lambda m: {
        "baseline": Decimal("1.0"),
        "strong_tier1": Decimal("0.501"),  # 49.9% cheaper
        "mid_tier2": Decimal("2.0"),  # pricier than baseline → excluded
        "elite_tier0a": Decimal("0.9"),  # higher tier but pricier than strong
        "elite_tier0b": Decimal("0.95"),
        "cheap_tier3": Decimal("0.4"),
        "unrated_cheap": Decimal("0.1"),
    }.get(m))
    s = cost.next_rung_up("mid_tier2", "baseline", pool=_POOL)
    assert s is not None
    assert s.model == "strong_tier1"
    assert s.pct_cheaper_than_baseline == 49


def test_next_rung_up_command_names_the_model(stub_tables: None) -> None:
    s = cost.next_rung_up("mid_tier2", "baseline", pool=_POOL)
    assert s is not None
    assert s.command == f"frugon analyze --measure --candidates {s.model}"


def test_next_rung_up_default_pool_is_routing_candidates(stub_tables: None) -> None:
    # Calling without an explicit pool falls back to _ROUTING_CANDIDATES; with the
    # stubbed tables those names are all unrated, so the result is None — proving
    # the default universe is consulted (not a crash, not the test pool).
    assert cost.next_rung_up("mid_tier2", "baseline") is None
