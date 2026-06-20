"""Bundled-seed PRICING coverage for current-generation flagship models.

tokencost (the runtime pricing fallback) lags the newest flagships, so the
wheel-bundled override seed (``src/frugon/data/pricing.json``) carries them
explicitly. These tests prove that a *default* install — reading only the
bundled seed, with no ``frugon pricing update`` having run — prices every
flagship frugon advertises coverage for.

Every price traces to a real source: the LiteLLM
``model_prices_and_context_window.json`` registry (bare canonical key), the same
registry ``frugon pricing update`` syncs from.

Quality tiers are intentionally NOT expanded alongside this: the quality.json
Arena cut-points are anchored to a 2024-Q4 snapshot, and binning current models
against them collapses every frontier model into one band (no routing
discrimination). Re-anchoring that scale is a separate change, so these new
flagships are deliberately PRICED-but-(quality-)unexpanded — frugon never
invents a tier it cannot cite.

Each test forces the loader to read the bundled seed by pointing the
user-data-dir path at a non-existent file in ``tmp_path`` (the pattern from
``test_pricing.py``), validating the SHIPPED seed, not a locally-synced copy.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

import frugon.pricing as pricing_module
from frugon.pricing import clear_pricing_cache, get_model_price

# (model, input_per_token, output_per_token) — bare/canonical LiteLLM registry
# costs the seed was built from; kept as strings for exact Decimal comparison.
_FLAGSHIP_PRICES: list[tuple[str, str, str]] = [
    # Anthropic — bare canonical keys
    ("claude-haiku-4-5", "0.000001", "0.000005"),
    ("claude-sonnet-4-5", "0.000003", "0.000015"),
    ("claude-opus-4-1", "0.000015", "0.000075"),
    ("claude-opus-4-5", "0.000005", "0.000025"),
    # Anthropic — dated keys (no bare claude-sonnet-4 / claude-opus-4 exist
    # under the anthropic provider in the registry)
    ("claude-sonnet-4-20250514", "0.000003", "0.000015"),
    ("claude-opus-4-20250514", "0.000015", "0.000075"),
    # OpenAI
    ("gpt-4.1", "0.000002", "0.000008"),
    ("gpt-4.1-mini", "0.0000004", "0.0000016"),
    ("gpt-4.1-nano", "0.0000001", "0.0000004"),
    ("gpt-5", "0.00000125", "0.00001"),
    ("gpt-5-mini", "0.00000025", "0.000002"),
    ("gpt-5-nano", "0.00000005", "0.0000004"),
    ("o3", "0.000002", "0.000008"),
    ("o4-mini", "0.0000011", "0.0000044"),
    ("o3-mini", "0.0000011", "0.0000044"),
    # Google
    ("gemini-2.5-pro", "0.00000125", "0.00001"),
    ("gemini-2.5-flash", "0.0000003", "0.0000025"),
    ("gemini-2.0-flash", "0.0000001", "0.0000004"),
    # DeepSeek — bare provider keys (deepseek-chat=V3, deepseek-reasoner=R1)
    ("deepseek-chat", "0.00000028", "0.00000042"),
    ("deepseek-reasoner", "0.00000028", "0.00000042"),
]


@pytest.fixture
def _bundled_seed_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Force the pricing loader to read only the wheel-bundled seed.

    Points the user-data-dir path at a non-existent file so the loader falls
    back to the bundled seed, isolating the test from any locally-synced
    ``frugon pricing update`` output. Clears the pricing cache around the test.
    """
    monkeypatch.setattr(pricing_module, "_PRICING_JSON", tmp_path / "pricing.json")
    clear_pricing_cache()
    yield
    clear_pricing_cache()


@pytest.mark.usefixtures("_bundled_seed_only")
@pytest.mark.parametrize(("model", "inp", "out"), _FLAGSHIP_PRICES)
def test_flagship_priced_by_bundled_seed(model: str, inp: str, out: str) -> None:
    """Each flagship resolves to its seed price, sourced from pricing.json.

    Asserts ``source == "pricing.json"`` (not the tokencost fallback) so the test
    proves the SEED carries the model, regardless of what tokencost ships today.
    """
    price = get_model_price(model)
    assert price is not None, f"{model!r} must be priced by the bundled seed"
    assert price.source == "pricing.json", (
        f"{model!r} must resolve from the bundled seed, got {price.source!r}"
    )
    assert price.input_cost_per_token == Decimal(inp)
    assert price.output_cost_per_token == Decimal(out)
