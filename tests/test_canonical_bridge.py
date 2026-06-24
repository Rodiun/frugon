"""§2a golden tests: the canonicalized-tokencost bridge.

tokencost stores most models under provider-prefixed keys (e.g.
``deepseek/deepseek-r1``).  The exact/canonical/base/newest-dated steps look a
user's name up against tokencost's *raw* keys, so a bare wire form
(``deepseek-r1``) misses the prefixed key even though tokencost has the price.

The bridge indexes tokencost **by canonical form** so the bare name matches.
Consistency gate (§2a — never fabricate a number): a canonical name that several
raw keys map to resolves ONLY when every contributing key agrees on the
(input, output) price; divergent names resolve to None.

The real-registry test is drift-proof: it computes the gate's expected answer
from the live tokencost data, so an upstream price/key change updates both the
production path and the assertion together.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest


def _empty_override(monkeypatch: pytest.MonkeyPatch, tmp_path, pricing_module) -> None:
    """Point the override table at an empty file so resolution falls through to
    the tokencost fallback + bridge (isolates the bridge from the bundled seed)."""
    f = tmp_path / "pricing.json"
    f.write_text(json.dumps({"_last_synced": "2026-01-01"}), encoding="utf-8")
    monkeypatch.setattr(pricing_module, "_PRICING_JSON", f)
    monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", f)


def _bridge_expected(bare: str) -> tuple[Decimal, Decimal] | None:
    """The price step-5 should yield for *bare*: the consistency-gated CANONICAL
    price — canon-only, because the resolver has NO base-family arm by design
    (folding to base would bypass the gate; see C1).  None when the canonical name
    is divergent or absent.  An independent reimplementation of the spec (not the
    production helper), so the assertion genuinely checks the impl."""
    from collections import defaultdict

    import tokencost as tc

    from frugon.model_id import canonicalize

    groups: dict[str, set[tuple[Decimal, Decimal]]] = defaultdict(set)
    for key, entry in tc.TOKEN_COSTS.items():
        if not isinstance(entry, dict):
            continue
        in_c = entry.get("input_cost_per_token")
        out_c = entry.get("output_cost_per_token")
        if in_c is None or out_c is None:
            continue
        groups[canonicalize(key)].add((Decimal(str(in_c)), Decimal(str(out_c))))

    pairs = groups.get(canonicalize(bare))
    return next(iter(pairs)) if pairs and len(pairs) == 1 else None


class TestBridgeResolvesProviderPrefixedModels:
    """A bare wire name resolves to the consistency-gated price the bridge derives."""

    @pytest.mark.parametrize("bare", ["deepseek-r1", "mistral-large-latest"])
    def test_bare_matches_gated_bridge_price(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, bare: str
    ) -> None:
        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        expected = _bridge_expected(bare)
        _empty_override(monkeypatch, tmp_path, pricing_module)
        clear_pricing_cache()
        price = get_model_price(bare)
        if expected is None:
            # Registry has this name only divergently/absent — gate must refuse.
            assert price is None, (
                f"{bare!r}: registry is divergent/absent, bridge must not guess"
            )
        else:
            assert price is not None, f"{bare!r} should resolve via the canonical bridge"
            assert price.input_cost_per_token == expected[0]
            assert price.output_cost_per_token == expected[1]
            assert "tokencost" in price.source
        clear_pricing_cache()

    def test_bridge_resolves_at_least_one_real_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Non-triviality: with an empty seed, the bridge must price at least one
        common provider-prefixed-only model (else it adds nothing on real data)."""
        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        _empty_override(monkeypatch, tmp_path, pricing_module)
        clear_pricing_cache()
        candidates = ["deepseek-r1", "mistral-large-latest", "deepseek-chat", "deepseek-reasoner"]
        resolved = [c for c in candidates if get_model_price(c) is not None]
        clear_pricing_cache()
        assert resolved, (
            "the bridge resolved none of the common provider-prefixed models — "
            "it is adding no real coverage"
        )


class TestBridgeConsistencyGate:
    """Divergent prices across provider keys for one canonical name → None; agreeing → resolve."""

    def test_divergent_prefixed_prices_return_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import tokencost as tc

        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        costs = dict(tc.TOKEN_COSTS)
        costs["groq/frugon-synthetic-open-42b"] = {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
        }
        costs["together_ai/frugon-synthetic-open-42b"] = {
            "input_cost_per_token": 0.000009,  # divergent — open model hosted at 2 prices
            "output_cost_per_token": 0.000018,
        }
        monkeypatch.setattr(tc, "TOKEN_COSTS", costs)
        _empty_override(monkeypatch, tmp_path, pricing_module)
        clear_pricing_cache()
        assert get_model_price("frugon-synthetic-open-42b") is None, (
            "divergent provider prices must yield None — the bridge must not guess"
        )
        clear_pricing_cache()

    def test_agreeing_prefixed_prices_resolve(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import tokencost as tc

        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        costs = dict(tc.TOKEN_COSTS)
        for prefix in ("groq/", "together_ai/"):
            costs[f"{prefix}frugon-synthetic-agree-7b"] = {
                "input_cost_per_token": 0.0000005,  # identical across providers
                "output_cost_per_token": 0.0000010,
            }
        monkeypatch.setattr(tc, "TOKEN_COSTS", costs)
        _empty_override(monkeypatch, tmp_path, pricing_module)
        clear_pricing_cache()
        price = get_model_price("frugon-synthetic-agree-7b")
        assert price is not None
        assert price.input_cost_per_token == Decimal("0.0000005")
        assert price.output_cost_per_token == Decimal("0.0000010")
        clear_pricing_cache()

    def test_divergent_latest_does_not_fall_through_to_base(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """C1 regression (fails-before / passes-after): a ``-latest`` name whose
        canonical form is divergent must NOT be re-priced via its base family.

        Before the fix, the bridge's base-family arm folded the refused
        ``-latest`` canon to its base and returned the base price — a number that
        matches NEITHER divergent ``-latest`` variant.  The bridge is now
        canon-only, so a name the gate refused stays unpriced."""
        import tokencost as tc

        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        costs = dict(tc.TOKEN_COSTS)
        # `-latest` canon is divergent across providers → the gate must refuse it.
        costs["groq/frugon-synth-fam-latest"] = {
            "input_cost_per_token": 0.000010,
            "output_cost_per_token": 0.000020,
        }
        costs["together_ai/frugon-synth-fam-latest"] = {
            "input_cost_per_token": 0.000030,
            "output_cost_per_token": 0.000060,
        }
        # The base family resolves to a THIRD value — the trap the base-arm sprang.
        costs["groq/frugon-synth-fam"] = {
            "input_cost_per_token": 0.000099,
            "output_cost_per_token": 0.000099,
        }
        monkeypatch.setattr(tc, "TOKEN_COSTS", costs)
        _empty_override(monkeypatch, tmp_path, pricing_module)
        clear_pricing_cache()
        assert get_model_price("frugon-synth-fam-latest") is None, (
            "divergent `-latest` must stay unpriced — the base-family price must "
            "not leak through (C1 gate-bypass regression)"
        )
        clear_pricing_cache()

    def test_divergent_latest_does_not_fall_through_to_newest_dated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Step-4 sibling of the C1 guard: a divergent ``-latest`` must NOT inherit
        a consistent DATED base-family price via the newest-dated fallback.

        Distinct from the step-5 case — here the base family carries a *dated*
        variant ("...-20240101"), which step 4 would scan and attribute to the
        refused ``-latest`` name unless step 4 is gated to ``canon == base``."""
        import tokencost as tc

        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        costs = dict(tc.TOKEN_COSTS)
        # `-latest` canon is divergent across providers → must be refused.
        costs["groq/frugon-synth-dated-latest"] = {
            "input_cost_per_token": 0.000010,
            "output_cost_per_token": 0.000020,
        }
        costs["together_ai/frugon-synth-dated-latest"] = {
            "input_cost_per_token": 0.000030,
            "output_cost_per_token": 0.000060,
        }
        # A consistent DATED base variant — the trap step 4 would otherwise spring.
        costs["frugon-synth-dated-20240101"] = {
            "input_cost_per_token": 0.000077,
            "output_cost_per_token": 0.000077,
        }
        monkeypatch.setattr(tc, "TOKEN_COSTS", costs)
        _empty_override(monkeypatch, tmp_path, pricing_module)
        clear_pricing_cache()
        assert get_model_price("frugon-synth-dated-latest") is None, (
            "divergent `-latest` must not inherit the dated base price via the "
            "newest-dated fallback (step-4 gate-bypass regression)"
        )
        clear_pricing_cache()


class TestBridgePrecedence:
    """The bridge is a last-resort step — pricing.json wins first."""

    def test_override_table_wins_over_bridge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        f = tmp_path / "pricing.json"
        f.write_text(
            json.dumps(
                {
                    "_last_synced": "2026-01-01",
                    "deepseek-r1": {
                        "input_cost_per_token": 0.0000011,  # sentinel
                        "output_cost_per_token": 0.0000022,
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", f)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", f)
        clear_pricing_cache()
        price = get_model_price("deepseek-r1")
        assert price is not None
        assert price.source == "pricing.json", "pricing.json must win over the bridge"
        assert price.input_cost_per_token == Decimal("0.0000011")
        clear_pricing_cache()


class TestXaiGrokCoverage:
    """xAI's first-party ``xai/`` prefix is stripped so bare grok names reach the bridge.

    tokencost carries grok under ``xai/grok-2``, ``xai/grok-3`` (first-party) plus
    reseller forms (``azure_ai/grok-3``, ``oci/xai.grok-3``) under prefixes that are
    deliberately NOT stripped — so only the first-party price maps to the bare name
    and the consistency gate sees a single price.
    """

    @pytest.mark.parametrize("bare", ["grok-2", "grok-3"])
    def test_xai_prefix_is_stripped(self, bare: str) -> None:
        from frugon.model_id import canonicalize

        assert canonicalize(f"xai/{bare}") == bare

    @pytest.mark.parametrize("bare", ["grok-2", "grok-3"])
    def test_bare_grok_prices_via_bridge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, bare: str
    ) -> None:
        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        expected = _bridge_expected(bare)
        if expected is None:
            pytest.skip(f"{bare!r} absent/divergent in this tokencost snapshot")
        _empty_override(monkeypatch, tmp_path, pricing_module)
        clear_pricing_cache()
        price = get_model_price(bare)
        assert price is not None, f"{bare!r} should resolve via the xai/ prefix + bridge"
        assert (price.input_cost_per_token, price.output_cost_per_token) == expected
        assert "tokencost" in price.source
        clear_pricing_cache()


class TestBridgeUnknownReturnsNone:
    def test_unknown_model_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        _empty_override(monkeypatch, tmp_path, pricing_module)
        clear_pricing_cache()
        assert get_model_price("frugon-nonexistent-model-zzz-2099") is None
        clear_pricing_cache()
