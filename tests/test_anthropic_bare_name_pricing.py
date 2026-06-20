"""Golden tests: bare and dotted Anthropic model names resolve to real prices.

§2a cost-math carve-out: these tests assert against prices traced directly from
the live tokencost/LiteLLM registry loaded at test time.  They are golden tests
in the sense that they assert price(bare) == price(newest-dated) — i.e. the
fallback produces the same numeric answer as the dated key, not a fabricated
value — and they document the registry source for each price.

Test scope
----------
1. Bare family names (claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus,
   claude-3-7-sonnet) now resolve to a price via the newest-dated fallback.
2. Dotted Anthropic forms (claude-3.5-sonnet, claude-3.7-sonnet) now resolve
   via the dot→hyphen canonicalization in model_id.canonicalize().
3. Non-Anthropic dotted names (gpt-4.1, gemini-2.5-pro) remain priced under
   their dotted key — they must not be redirected by the claude-only rule.
4. The combined effect: get_model_tier() + get_model_price() both succeed for
   all four bare families and the two dotted Claude forms.
5. price(bare) == price(newest-dated) — the fallback is consistent with the
   dated key, not an invented number.
6. The newest-dated fallback does NOT fire for families where the override
   table (pricing.json) already has a direct dated entry — verifies that the
   existing exact/canonical/base_family chain still takes precedence.
7. The fallback returns None for a genuinely unknown family so no phantom price
   is invented.
8. Price-inconsistency gate: when dated variants disagree on price (simulated),
   the fallback returns None rather than picking arbitrarily.
"""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _newest_dated_key(family: str) -> str:
    """Return the newest compact-dated tokencost key for *family*.

    Reads tokencost.TOKEN_COSTS directly — the same source the fallback probes
    at runtime — so any upstream update to tokencost automatically updates both
    the production path and this assertion.
    """
    import re

    import tokencost as tc

    prefix = family + "-"
    _COMPACT = re.compile(r"-(\d{8})$")
    candidates = [
        k
        for k in tc.TOKEN_COSTS
        if k.startswith(prefix)
        and _COMPACT.fullmatch(k[len(family):])
        and tc.TOKEN_COSTS[k].get("input_cost_per_token") is not None
        and tc.TOKEN_COSTS[k].get("output_cost_per_token") is not None
    ]
    assert candidates, f"No dated variants found in tokencost for family {family!r}"
    return max(candidates)  # lexicographic max == latest date for YYYYMMDD


# ---------------------------------------------------------------------------
# 1. Bare family names price to the newest-dated variant
# ---------------------------------------------------------------------------


class TestBareAnthropicFamilyPricing:
    """Bare Anthropic family names resolve via the newest-dated tokencost fallback."""

    @pytest.mark.parametrize(
        "bare_family",
        [
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-3-opus",
            "claude-3-7-sonnet",
        ],
    )
    def test_bare_family_prices_via_newest_dated_fallback(
        self, bare_family: str
    ) -> None:
        """Arrange: bare Anthropic family name absent from pricing.json and
        tokencost bare keys, but present in tokencost as dated variants.
        Act: get_model_price(bare_family).
        Assert: price is not None and equals the price of the newest dated key.

        This is the §2a golden test: price(bare) == price(newest-dated), with
        the actual cost-per-token values traced to the live tokencost registry.
        """
        import tokencost as tc

        from frugon.pricing import clear_pricing_cache, get_model_price

        clear_pricing_cache()
        price = get_model_price(bare_family)

        newest_key = _newest_dated_key(bare_family)
        dated_entry = tc.TOKEN_COSTS[newest_key]
        expected_in = Decimal(str(dated_entry["input_cost_per_token"]))
        expected_out = Decimal(str(dated_entry["output_cost_per_token"]))

        assert price is not None, (
            f"get_model_price({bare_family!r}) returned None. "
            f"Expected price via newest dated key {newest_key!r}: "
            f"in={expected_in}, out={expected_out}."
        )
        assert price.input_cost_per_token == expected_in, (
            f"{bare_family!r}: input cost {price.input_cost_per_token} "
            f"!= {expected_in} (from {newest_key!r} in tokencost)"
        )
        assert price.output_cost_per_token == expected_out, (
            f"{bare_family!r}: output cost {price.output_cost_per_token} "
            f"!= {expected_out} (from {newest_key!r} in tokencost)"
        )
        # Source must reference tokencost and the newest dated key.
        assert "tokencost" in price.source, (
            f"Expected source containing 'tokencost', got {price.source!r}"
        )
        assert newest_key in price.source, (
            f"Expected source to name the newest key {newest_key!r}, got {price.source!r}"
        )
        clear_pricing_cache()

    @pytest.mark.parametrize(
        ("bare_family", "expected_input_per_token", "registry_key"),
        [
            # Prices traced from tokencost (LiteLLM registry) as of 2026-06-20.
            # All variants of each family carry the same price — confirmed by probe.
            # Update these values if Anthropic changes list prices.
            (
                "claude-3-5-sonnet",
                Decimal("0.000003"),       # $3 / 1M input tokens
                "claude-3-5-sonnet-20241022",
            ),
            (
                "claude-3-5-haiku",
                Decimal("0.0000008"),      # $0.80 / 1M input tokens
                "claude-3-5-haiku-20241022",
            ),
            (
                "claude-3-opus",
                Decimal("0.000015"),       # $15 / 1M input tokens
                "claude-3-opus-20240229",
            ),
            (
                "claude-3-7-sonnet",
                Decimal("0.000003"),       # $3 / 1M input tokens
                "claude-3-7-sonnet-20250219",
            ),
        ],
    )
    def test_bare_family_price_absolute_values(
        self,
        bare_family: str,
        expected_input_per_token: Decimal,
        registry_key: str,
    ) -> None:
        """Arrange: bare family name + known absolute price from tokencost registry.
        Act: get_model_price(bare_family).
        Assert: input_cost_per_token matches the registry value.

        This anchors the test to the actual cost-per-token value so a registry
        price change is immediately visible as a test failure — protecting the
        §2a cost-math carve-out invariant.

        Prices sourced from: tokencost TOKEN_COSTS (LiteLLM registry),
        retrieved 2026-06-20 via probe in workspace/frugon.
        """
        import tokencost as tc

        from frugon.pricing import clear_pricing_cache, get_model_price

        # Double-check: the expected value must still match the registry today.
        live_entry = tc.TOKEN_COSTS.get(registry_key)
        assert live_entry is not None, (
            f"Registry key {registry_key!r} no longer in tokencost — update test fixture"
        )
        live_in = Decimal(str(live_entry["input_cost_per_token"]))
        assert live_in == expected_input_per_token, (
            f"Registry price for {registry_key!r} changed: "
            f"expected {expected_input_per_token}, now {live_in}. "
            "Update the test fixture to the current value."
        )

        clear_pricing_cache()
        price = get_model_price(bare_family)
        assert price is not None, (
            f"get_model_price({bare_family!r}) returned None — fallback did not fire"
        )
        assert price.input_cost_per_token == expected_input_per_token, (
            f"{bare_family!r}: expected input {expected_input_per_token} "
            f"(from {registry_key!r}), got {price.input_cost_per_token}"
        )
        clear_pricing_cache()


# ---------------------------------------------------------------------------
# 2. Dotted Claude forms resolve via dot→hyphen canonicalization
# ---------------------------------------------------------------------------


class TestDottedClaudeCanonicalisation:
    """claude-<M>.<N>-* forms are normalised to claude-<M>-<N>-* and priced."""

    @pytest.mark.parametrize(
        ("dotted", "canonical_bare"),
        [
            ("claude-3.5-sonnet", "claude-3-5-sonnet"),
            ("claude-3.7-sonnet", "claude-3-7-sonnet"),
            ("claude-3.5-haiku", "claude-3-5-haiku"),
        ],
    )
    def test_dotted_claude_canonicalize_folds_to_hyphen(
        self, dotted: str, canonical_bare: str
    ) -> None:
        """Arrange: dotted Claude wire form.
        Act: canonicalize(dotted).
        Assert: output is the hyphenated bare form.
        """
        from frugon.model_id import canonicalize

        result = canonicalize(dotted)
        assert result == canonical_bare, (
            f"canonicalize({dotted!r}) -> {result!r}, want {canonical_bare!r}"
        )

    @pytest.mark.parametrize(
        ("dotted", "canonical_bare"),
        [
            ("claude-3.5-sonnet", "claude-3-5-sonnet"),
            ("claude-3.7-sonnet", "claude-3-7-sonnet"),
            ("claude-3.5-haiku", "claude-3-5-haiku"),
        ],
    )
    def test_dotted_claude_idempotent(self, dotted: str, canonical_bare: str) -> None:
        """Arrange: dotted Claude form.
        Act: canonicalize twice.
        Assert: second call is a fixed point (idempotent).
        """
        from frugon.model_id import canonicalize

        first = canonicalize(dotted)
        second = canonicalize(first)
        assert first == canonical_bare, (
            f"first canonicalize({dotted!r}) -> {first!r}, want {canonical_bare!r}"
        )
        assert second == first, (
            f"canonicalize not idempotent: {first!r} -> {second!r}"
        )

    @pytest.mark.parametrize(
        "dotted",
        [
            "claude-3.5-sonnet",
            "claude-3.7-sonnet",
            "claude-3.5-haiku",
        ],
    )
    def test_dotted_claude_prices_correctly(self, dotted: str) -> None:
        """Arrange: dotted Claude form that was previously unpriced.
        Act: get_model_price(dotted).
        Assert: price is not None after the dot→hyphen fix + newest-dated fallback.
        """
        from frugon.pricing import clear_pricing_cache, get_model_price

        clear_pricing_cache()
        price = get_model_price(dotted)
        assert price is not None, (
            f"get_model_price({dotted!r}) still None after dot→hyphen fix. "
            "The canonicalization or newest-dated fallback did not fire."
        )
        assert price.input_cost_per_token > Decimal("0"), (
            f"{dotted!r}: input cost must be positive, got {price.input_cost_per_token}"
        )
        clear_pricing_cache()


# ---------------------------------------------------------------------------
# 3. Non-Anthropic dotted names are unaffected
# ---------------------------------------------------------------------------


class TestNonClaudeDottedNamesUntouched:
    """gpt-4.1, gemini-*.* must NOT be altered by the claude-specific dot rule."""

    @pytest.mark.parametrize(
        "dotted_name",
        [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    )
    def test_non_claude_dotted_names_canonicalize_unchanged(
        self, dotted_name: str
    ) -> None:
        """Arrange: legitimately-dotted non-Claude model name.
        Act: canonicalize(dotted_name).
        Assert: result is identical to the input (no dot replaced).
        """
        from frugon.model_id import canonicalize

        result = canonicalize(dotted_name)
        assert result == dotted_name.lower(), (
            f"canonicalize({dotted_name!r}) -> {result!r}; "
            "non-Claude dotted names must pass through unchanged"
        )

    @pytest.mark.parametrize(
        "dotted_name",
        [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    )
    def test_non_claude_dotted_names_still_price(self, dotted_name: str) -> None:
        """Arrange: known non-Claude dotted model.
        Act: get_model_price(dotted_name).
        Assert: still returns a price (the fix did not break them).
        """
        from frugon.pricing import clear_pricing_cache, get_model_price

        clear_pricing_cache()
        price = get_model_price(dotted_name)
        assert price is not None, (
            f"get_model_price({dotted_name!r}) returned None — non-Claude dotted "
            "name lost its price after the fix"
        )
        clear_pricing_cache()


# ---------------------------------------------------------------------------
# 4. Combined: get_model_tier + get_model_price both succeed
# ---------------------------------------------------------------------------


class TestAnthropicBareNameCombinedResolution:
    """After both fixes: tier + price resolve for bare and dotted Anthropic names."""

    @pytest.mark.parametrize(
        "model",
        [
            # Bare hyphenated family names
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-3-opus",
            "claude-3-7-sonnet",
            # Dotted wire forms (OpenRouter-style)
            "claude-3.5-sonnet",
            "claude-3.7-sonnet",
            "claude-3.5-haiku",
        ],
    )
    def test_bare_and_dotted_anthropic_price_and_tier_resolve(
        self, model: str
    ) -> None:
        """Arrange: bare or dotted Anthropic model name.
        Act: get_model_price + get_model_tier.
        Assert: both return non-None / non-UNRATED values.

        This is the end-to-end regression guard: a stranger's log containing any
        of these API name forms will now see a saving number instead of silence.
        """
        from frugon.pricing import clear_pricing_cache, get_model_price
        from frugon.quality import UNRATED_TIER, get_model_tier

        clear_pricing_cache()
        price = get_model_price(model)
        tier = get_model_tier(model)

        assert price is not None, (
            f"get_model_price({model!r}) returned None after fixes"
        )
        assert price.input_cost_per_token > Decimal("0"), (
            f"{model!r}: input cost must be positive"
        )
        assert tier != UNRATED_TIER, (
            f"get_model_tier({model!r}) returned UNRATED_TIER={UNRATED_TIER} after fixes"
        )
        clear_pricing_cache()


# ---------------------------------------------------------------------------
# 5. Newest-dated fallback does not fire when exact chain already succeeds
# ---------------------------------------------------------------------------


class TestNewestDatedFallbackPrecedence:
    """The existing exact/canonical/base_family chain takes precedence.

    When pricing.json already has a dated key (e.g. claude-3-5-sonnet-20241022),
    the exact match in step 1 fires and the newest-dated fallback (step 4) is
    never reached.  This verifies that the fallback does not interfere with the
    three existing steps.
    """

    def test_dated_key_in_override_table_wins_over_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """Arrange: pricing.json with a dated Claude key at a distinct price.
        Act: get_model_price of that exact dated key.
        Assert: source is 'pricing.json' (override wins; fallback was not used).
        """
        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        pricing_file = tmp_path / "pricing.json"
        pricing_file.write_text(
            json.dumps(
                {
                    "_last_synced": "2026-01-01",
                    "claude-3-5-sonnet-20241022": {
                        "input_cost_per_token": 0.0000031,  # distinct sentinel value
                        "output_cost_per_token": 0.0000151,
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", pricing_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", pricing_file)
        clear_pricing_cache()

        price = get_model_price("claude-3-5-sonnet-20241022")
        assert price is not None
        assert price.source == "pricing.json", (
            f"Expected 'pricing.json' source, got {price.source!r}"
        )
        assert price.input_cost_per_token == Decimal("0.0000031"), (
            "pricing.json sentinel value not returned — override table lost precedence"
        )
        clear_pricing_cache()


# ---------------------------------------------------------------------------
# 6. Fallback returns None for genuinely unknown family
# ---------------------------------------------------------------------------


class TestNewestDatedFallbackRejectsUnknownFamily:
    """The fallback must not invent a price for a family with no dated variants."""

    def test_unknown_family_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """Arrange: a synthetic family name absent from both pricing.json and
        tokencost (no dated variant of that prefix exists).
        Act: get_model_price.
        Assert: returns None — no phantom price invented.
        """
        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        absent = tmp_path / "no_pricing.json"
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", absent)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", absent)
        clear_pricing_cache()

        price = get_model_price("frugon-synthetic-nonexistent-family-xyz-2099")
        assert price is None, (
            "A synthetic family absent from every registry must yield None; "
            f"got {price}"
        )
        clear_pricing_cache()


# ---------------------------------------------------------------------------
# 7. Price-inconsistency gate: differing dated variant prices → None
# ---------------------------------------------------------------------------


class TestNewestDatedFallbackConsistencyGate:
    """When dated variants disagree on price, the fallback returns None."""

    def test_inconsistent_dated_variants_return_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        """Arrange: tokencost patched with two dated variants of a synthetic
        family at DIFFERENT prices (simulating a provider rate change).
        Act: get_model_price of the bare family name.
        Assert: returns None — the inconsistency gate fires; no guess is made.

        Pricing the bare name to either dated key would be wrong when the rates
        differ; the correct answer is to surface no price rather than mis-price
        the user's usage.
        """
        import tokencost as tc

        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        # Use a synthetic family that has no entries in the real registry.
        synthetic_family = "frugon-test-inconsistent-family-v1"
        fake_costs: dict[str, object] = {
            synthetic_family + "-20240101": {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000002,
            },
            synthetic_family + "-20241201": {
                "input_cost_per_token": 0.000009,  # different — rate changed
                "output_cost_per_token": 0.000018,
            },
        }

        absent = tmp_path / "no_pricing.json"
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", absent)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", absent)

        original_costs = dict(tc.TOKEN_COSTS)
        original_costs.update(fake_costs)  # type: ignore[arg-type]
        monkeypatch.setattr(tc, "TOKEN_COSTS", original_costs)

        clear_pricing_cache()
        price = get_model_price(synthetic_family)
        assert price is None, (
            "Inconsistent dated variants must yield None; "
            f"got {price} — the consistency gate did not fire"
        )
        clear_pricing_cache()
