"""Pricing-guard regression tests for the effort-fold feature (FRG-OSS-034).

Reasoning effort (-high, -thinking, etc.) changes how many tokens a model
spends thinking, not its per-token rate, so folding an effort variant to its
base name is honest for QUALITY-tier lookups only. Providers have historically
priced a thinking/non-thinking variant differently (e.g. Gemini 2.5 Flash), so
a PRICING lookup must never effort-fold. These tests assert that guarantee
structurally: pricing.py never imports effort_family, and a synthetic
effort-variant price is never inherited by its base (or vice versa) through
get_model_price.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_pricing_module_never_imports_effort_family() -> None:
    """Arrange: the pricing module source.
    Act: inspect its import list.
    Assert: effort_family is never imported into pricing.py -- the pricing
    lookup path must have no code path capable of effort-folding.
    """
    import frugon.pricing as pricing_module

    assert not hasattr(pricing_module, "effort_family"), (
        "pricing.py must never import model_id.effort_family -- "
        "reasoning effort does not change price, but some providers price "
        "thinking vs non-thinking variants differently, so a pricing lookup "
        "must never fold across an effort suffix"
    )


class TestPricingDoesNotEffortFold:
    """Behavioural guard: a synthetic effort-variant price must not leak to
    its base name (or vice versa) through get_model_price.
    """

    def _write_pricing_json(self, path: Path, data: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_high_variant_price_not_inherited_by_bare_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: pricing.json prices ONLY 'widget-high', at a distinct rate.
        Act: get_model_price('widget') (the bare, un-suffixed name).
        Assert: None -- the bare name must NOT inherit the effort variant's
        price via any fold.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        user_file = tmp_path / "pricing.json"
        self._write_pricing_json(
            user_file,
            {
                "_last_synced": "2026-01-01",
                "widget-high": {
                    "input_cost_per_token": 0.00005,
                    "output_cost_per_token": 0.0002,
                },
            },
        )
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        clear_pricing_cache()

        assert get_model_price("widget") is None
        assert get_model_price("widget-high") is not None

    def test_bare_name_price_not_inherited_by_high_variant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: pricing.json prices ONLY the bare 'widget' name.
        Act: get_model_price('widget-high').
        Assert: None -- the effort variant must NOT inherit the bare name's
        price either (folding must not run in the reverse direction).
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        user_file = tmp_path / "pricing.json"
        self._write_pricing_json(
            user_file,
            {
                "_last_synced": "2026-01-01",
                "widget": {
                    "input_cost_per_token": 0.00001,
                    "output_cost_per_token": 0.00004,
                },
            },
        )
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        clear_pricing_cache()

        assert get_model_price("widget-high") is None
        assert get_model_price("widget") is not None

    def test_thinking_and_non_thinking_variants_price_independently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: both 'widget-thinking' and 'widget' priced at DIFFERENT
        rates (mirrors a provider that genuinely charges more for extended
        thinking, e.g. Gemini 2.5 Flash).
        Act: get_model_price for each.
        Assert: each resolves to its OWN distinct price -- no cross-fold
        collapses the two into a single rate.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        user_file = tmp_path / "pricing.json"
        self._write_pricing_json(
            user_file,
            {
                "_last_synced": "2026-01-01",
                "widget": {
                    "input_cost_per_token": 0.00001,
                    "output_cost_per_token": 0.00004,
                },
                "widget-thinking": {
                    "input_cost_per_token": 0.00005,
                    "output_cost_per_token": 0.0002,
                },
            },
        )
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        clear_pricing_cache()

        bare_price = get_model_price("widget")
        thinking_price = get_model_price("widget-thinking")

        assert bare_price is not None
        assert thinking_price is not None
        assert float(bare_price.input_cost_per_token) == pytest.approx(0.00001)
        assert float(thinking_price.input_cost_per_token) == pytest.approx(0.00005)
        assert bare_price.input_cost_per_token != thinking_price.input_cost_per_token
