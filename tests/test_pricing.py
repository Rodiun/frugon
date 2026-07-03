"""Tests for frugon.pricing — precedence, fallback, last_synced exposure, canonicalization.

Covers P2-3 requirements:
  - pricing.json wins when model present in both sources
  - tokencost fallback for models not in pricing.json
  - _last_synced is exposed on ModelPrice
  - Missing pricing.json is handled gracefully
"""

from __future__ import annotations

import json
import pathlib
import urllib.error
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest


class TestPricingJsonPrecedence:
    """P2-3: pricing.json beats tokencost for overlapping models."""

    def test_gpt4o_price_matches_pricing_json(self) -> None:
        """Arrange: gpt-4o is in both pricing.json and tokencost at different rates.
        Act: get_model_price('gpt-4o').
        Assert: price matches pricing.json rate ($0.0000025 input).

        This is the canonical P2-3 regression test: a model present in both
        sources at different prices MUST resolve to pricing.json.
        """
        from frugon.pricing import get_model_price

        # pricing.json says gpt-4o input = 0.0000025
        price = get_model_price("gpt-4o")
        assert price is not None
        assert price.input_cost_per_token == Decimal("0.0000025"), (
            f"Expected 0.0000025 (from pricing.json), got {price.input_cost_per_token}"
        )
        assert price.source == "pricing.json"

    def test_gpt4o_mini_price_matches_pricing_json(self) -> None:
        """Arrange: gpt-4o-mini is in pricing.json.
        Act: get_model_price.
        Assert: source is pricing.json, rate matches $0.00000015 input.
        """
        from frugon.pricing import get_model_price

        price = get_model_price("gpt-4o-mini")
        assert price is not None
        assert price.input_cost_per_token == Decimal("0.00000015")
        assert price.source == "pricing.json"

    def test_unknown_model_falls_back_through_chain_to_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """Arrange: a synthetic model name guaranteed absent from every registry
        (pricing.json seed, base-family folding, and tokencost), with the
        user-data-dir isolated to tmp_path so a developer's synced pricing.json
        cannot shadow the result.
        Act: get_model_price.
        Assert: the full fallback chain is exercised and yields None.

        Using a synthetic, never-real model name keeps this test deterministic:
        it does not depend on whether any particular real model (e.g. o1-mini)
        happens to be present in tokencost or absent from the seed today — a
        registry assumption that rots as upstream tables change.

        Because the name is absent from tokencost too, the honest result of the
        precedence chain (pricing.json -> base-family fold -> tokencost) is None:
        a truly-unknown model has no price to invent.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import get_model_price

        # Isolate: point both paths at a location that does not exist so that
        # the real user-data-dir pricing.json (which may contain extra models
        # after a 'frugon pricing update') never shadows the test.
        absent = pathlib.Path(tmp_path / "no_pricing.json")
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", absent)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", absent)

        price = get_model_price("frugon-synthetic-absent-model-xyz")
        assert price is None, (
            "A synthetic model absent from every registry must yield None — "
            "no phantom price may be invented for an unknown model"
        )

    def test_completely_unknown_model_returns_none(self) -> None:
        """Arrange: model unknown to both sources.
        Act: get_model_price.
        Assert: returns None.
        """
        from frugon.pricing import get_model_price

        result = get_model_price("imaginary-model-xyz-does-not-exist-2099")
        assert result is None

    def test_is_model_known_true_for_gpt4o(self) -> None:
        from frugon.pricing import is_model_known

        assert is_model_known("gpt-4o") is True

    def test_is_model_known_false_for_fantasy_model(self) -> None:
        from frugon.pricing import is_model_known

        assert is_model_known("fantasy-model-9999-beta") is False


class TestLastSyncedExposure:
    """P2-3: _last_synced from pricing.json is surfaced on ModelPrice."""

    def test_pricing_json_last_synced_is_exposed(self) -> None:
        """Arrange: gpt-4o is in pricing.json which has a _last_synced field.
        Act: get_model_price.
        Assert: pricing_json_last_synced is a non-empty string.
        """
        from frugon.pricing import get_model_price

        price = get_model_price("gpt-4o")
        assert price is not None
        assert price.pricing_json_last_synced is not None
        assert len(price.pricing_json_last_synced) > 0

    def test_tokencost_only_model_last_synced_may_be_none(self) -> None:
        """Arrange: model only in tokencost (pricing.json has no _last_synced for it).
        Act: get_model_price.
        Assert: pricing_json_last_synced may be None for tokencost-sourced models
        when pricing.json has a _last_synced at the top level.

        We just verify it doesn't crash.
        """
        from frugon.pricing import get_model_price

        price = get_model_price("gpt-4o-2024-05-13")
        if price is not None and price.source == "tokencost":
            # No crash expected; last_synced may or may not be set
            _ = price.pricing_json_last_synced


class TestMissingPricingJson:
    """Graceful handling when pricing.json is absent.

    Both the user-data-dir path and the bundled seed path are patched to
    non-existent locations so that no seeding or migration occurs and the
    pure fallback behaviour is exercised.
    """

    def test_missing_pricing_json_falls_back_to_tokencost(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Arrange: neither user-dir pricing.json nor the bundled seed exist.
        Act: get_model_price for a tokencost-known model.
        Assert: returns a price from tokencost, no exception.
        """
        from frugon import pricing as pricing_module

        # tmp_path is an empty, already-existing directory on every OS, so this
        # child file is guaranteed absent without creating any stray dirs.
        fake_path = tmp_path / "does_not_exist_frugon.json"
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", fake_path)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", fake_path)

        from frugon.pricing import get_model_price

        price = get_model_price("gpt-4o")
        # tokencost carries gpt-4o, so we expect a result from tokencost
        assert price is not None
        assert price.source == "tokencost"

    def test_load_pricing_override_returns_empty_for_missing_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Arrange: neither user-dir pricing.json nor the bundled seed exist.
        Act: load_pricing_override.
        Assert: returns empty dict and None.
        """
        from frugon import pricing as pricing_module

        # tmp_path is an empty, already-existing directory on every OS, so this
        # child file is guaranteed absent without creating any stray dirs.
        fake_path = tmp_path / "nope_frugon.json"
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", fake_path)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", fake_path)

        from frugon.pricing import load_pricing_override

        table, last_synced = load_pricing_override()
        assert table == {}
        assert last_synced is None


class TestGatewayPrefixPricing:
    """Canonicalization integration: gateway-prefixed models must price correctly."""

    def test_openai_prefix_gpt4o_prices_same_as_bare(self) -> None:
        """Arrange: 'openai/gpt-4o' is not literally in pricing table.
        Act: get_model_price('openai/gpt-4o').
        Assert: price found via canonicalize(); matches bare 'gpt-4o' rate.
        """
        from frugon.pricing import get_model_price

        bare = get_model_price("gpt-4o")
        gateway = get_model_price("openai/gpt-4o")

        assert gateway is not None, "openai/gpt-4o should resolve via canonicalize()"
        assert bare is not None
        assert gateway.input_cost_per_token == bare.input_cost_per_token
        assert gateway.output_cost_per_token == bare.output_cost_per_token

    def test_openrouter_nested_prefix_prices_correctly(self) -> None:
        """Arrange: 'openrouter/openai/gpt-4o' with double prefix.
        Act: get_model_price.
        Assert: resolves to same price as bare 'gpt-4o'.
        """
        from frugon.pricing import get_model_price

        bare = get_model_price("gpt-4o")
        gateway = get_model_price("openrouter/openai/gpt-4o")

        assert gateway is not None, "openrouter/openai/gpt-4o should resolve via canonicalize()"
        assert bare is not None
        assert gateway.input_cost_per_token == bare.input_cost_per_token

    def test_azure_prefix_gpt4o_prices_correctly(self) -> None:
        """Arrange: 'azure/gpt-4o'.
        Act: get_model_price.
        Assert: resolves to same price as bare 'gpt-4o'.
        """
        from frugon.pricing import get_model_price

        bare = get_model_price("gpt-4o")
        gateway = get_model_price("azure/gpt-4o")

        assert gateway is not None, "azure/gpt-4o should resolve via canonicalize()"
        assert bare is not None
        assert gateway.input_cost_per_token == bare.input_cost_per_token

    def test_gateway_prefixed_model_original_name_preserved(self) -> None:
        """Arrange: gateway-prefixed model resolves successfully.
        Act: check ModelPrice.model field.
        Assert: the ORIGINAL (user-supplied) model name is stored, not the canonical form.
        """
        from frugon.pricing import get_model_price

        price = get_model_price("openai/gpt-4o")
        assert price is not None
        assert price.model == "openai/gpt-4o", (
            f"Expected original name 'openai/gpt-4o', got {price.model!r}"
        )

    def test_unknown_gateway_prefix_returns_none(self) -> None:
        """Arrange: completely unknown gateway prefix and model.
        Act: get_model_price.
        Assert: returns None — no phantom price invented.
        """
        from frugon.pricing import get_model_price

        result = get_model_price("my-custom-gw/imaginary-model-xyz-2099")
        assert result is None


class TestPricingCache:
    """The override-table + per-model caches that make a large log fast.

    These lock in the optimization: the pricing file is parsed once per
    identity, repeated lookups are served from memory, and a real
    ``frugon pricing update`` (a new mtime/size) invalidates transparently.
    """

    def test_load_pricing_override_reads_file_once_across_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: a real pricing file, cache cleared.
        Act: call load_pricing_override() many times.
        Assert: the underlying JSON read happens exactly once (cache hit after).
        """
        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, load_pricing_override

        clear_pricing_cache()
        calls = {"n": 0}
        real_loader = pricing_module.load_json_or_empty

        def _counting_loader(user_path: Path, seed_path: Path) -> dict[str, object]:
            calls["n"] += 1
            return real_loader(user_path, seed_path)

        monkeypatch.setattr(pricing_module, "load_json_or_empty", _counting_loader)

        first = load_pricing_override()
        for _ in range(50):
            again = load_pricing_override()
            assert again == first

        assert calls["n"] == 1, f"expected one disk parse, got {calls['n']}"
        clear_pricing_cache()

    def test_get_model_price_is_memoized_per_model(self) -> None:
        """Arrange: cache cleared.
        Act: resolve the same model twice.
        Assert: the second resolution is an lru_cache hit (hits counter advances).
        """
        from frugon.pricing import (
            _resolve_model_price,
            clear_pricing_cache,
            get_model_price,
        )

        clear_pricing_cache()
        get_model_price("gpt-4o")
        hits_before = _resolve_model_price.cache_info().hits
        get_model_price("gpt-4o")
        hits_after = _resolve_model_price.cache_info().hits
        assert hits_after == hits_before + 1
        clear_pricing_cache()

    def test_cached_price_matches_uncached_after_clear(self) -> None:
        """Arrange: resolve a model, clear the cache, resolve again.
        Act: compare the two ModelPrice results.
        Assert: identical — caching never changes the resolved value.
        """
        from frugon.pricing import clear_pricing_cache, get_model_price

        clear_pricing_cache()
        first = get_model_price("gpt-4o-mini")
        clear_pricing_cache()
        second = get_model_price("gpt-4o-mini")
        assert first == second

    def test_pricing_update_invalidates_cache_via_mtime(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Arrange: point pricing at a writable file, resolve a model.
        Act: rewrite the file with a different price, resolve again.
        Assert: the new price is returned — the mtime/size key invalidated the
        memo without any explicit clear (the production update path).
        """
        import json
        import os
        import time

        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        pricing_file = tmp_path / "pricing.json"
        pricing_file.write_text(
            json.dumps(
                {
                    "_last_synced": "2026-01-01",
                    "cache-test-model": {
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", pricing_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", pricing_file)
        clear_pricing_cache()

        first = get_model_price("cache-test-model")
        assert first is not None
        assert first.input_cost_per_token == Decimal("0.000001")

        # Rewrite with a different price.  Bump mtime explicitly so the change is
        # detectable even on a coarse-resolution filesystem clock.
        time.sleep(0.01)
        pricing_file.write_text(
            json.dumps(
                {
                    "_last_synced": "2026-02-02",
                    "cache-test-model": {
                        "input_cost_per_token": 0.000009,
                        "output_cost_per_token": 0.000008,
                    },
                }
            ),
            encoding="utf-8",
        )
        future = time.time() + 5
        os.utime(pricing_file, (future, future))

        second = get_model_price("cache-test-model")
        assert second is not None
        assert second.input_cost_per_token == Decimal("0.000009"), (
            "a rewritten pricing.json must invalidate the cache via mtime/size"
        )
        clear_pricing_cache()


class TestPinnedPricingIdentity:
    """Contract tests for ``pinned_pricing_identity`` (FRG-OSS-038 perf fix).

    ``get_model_price`` normally stats the pricing file's identity on every
    call via ``_override_cache_key()`` — correct, but costly when a hot loop
    calls it tens of thousands of times in a row (see the module's own
    "Pinned cache-key batching" docstring for the ~14s profiling finding that
    motivated this).  ``pinned_pricing_identity()`` lets a batch caller opt
    into computing that stat ONCE and reusing it for the whole block.

    This is a new public API on the most correctness-critical module in the
    codebase (§2a cost-math carve-out), so it gets its own explicit contract
    tests rather than relying on incidental coverage from
    ``analyze_records``'s hot-path usage:

      1. The pin stats once and reuses the key for every call inside the block.
      2. Nested (reentrant) pins are safe: the inner pin reuses the outer's
         key without re-stating; exiting the inner pin does not clear the
         outer pin; only exiting the OUTERMOST pin fully clears it.
      3. An exception raised inside a pinned block still restores state — the
         pin is unwound (back to unpinned) even on a raised exception.
      4. Outside any pin, the existing mid-process mtime-invalidation
         guarantee is untouched: a rewritten pricing file IS observed on the
         very next call — the explicit counterpoint proving the pin is
         opt-in, never a change to the unpinned default behaviour.
    """

    def test_pin_stats_once_and_reuses_the_key_for_the_whole_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: count calls to the underlying unconditional stat helper.
        Act: call _override_cache_key() many times inside one pin.
        Assert: the underlying stat helper ran exactly once; every call
        inside the block returned that SAME key.
        """
        import frugon.pricing as pricing_module
        from frugon.pricing import _override_cache_key, pinned_pricing_identity

        calls = {"n": 0}
        real_compute = pricing_module._compute_cache_key

        def _counting_compute() -> tuple[str, str, int, int]:
            calls["n"] += 1
            return real_compute()

        monkeypatch.setattr(pricing_module, "_compute_cache_key", _counting_compute)

        with pinned_pricing_identity():
            first = _override_cache_key()
            for _ in range(50):
                again = _override_cache_key()
                assert again == first

        assert calls["n"] == 1, (
            f"expected the pin to stat exactly once for the whole block, got {calls['n']}"
        )

    def test_nested_pins_reuse_outer_key_and_only_outermost_exit_clears(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: count calls to the underlying stat helper.
        Act: enter an outer pin, capture its key, enter a nested inner pin,
        capture ITS key, exit the inner pin, capture the key again, exit the
        outer pin.
        Assert: (a) the inner pin's key equals the outer's (reused, not
        re-stated — only 1 stat total across both); (b) after the inner pin
        exits, the key is STILL the outer's (not cleared); (c) after the
        outer pin exits, the pin is fully cleared (a fresh call re-stats).
        """
        import frugon.pricing as pricing_module
        from frugon.pricing import _override_cache_key, pinned_pricing_identity

        calls = {"n": 0}
        real_compute = pricing_module._compute_cache_key

        def _counting_compute() -> tuple[str, str, int, int]:
            calls["n"] += 1
            return real_compute()

        monkeypatch.setattr(pricing_module, "_compute_cache_key", _counting_compute)

        with pinned_pricing_identity():
            outer_key = _override_cache_key()
            with pinned_pricing_identity():
                inner_key = _override_cache_key()
                assert inner_key == outer_key, (
                    "a nested pin must reuse the outer pin's key, not re-stat"
                )
            # Inner pin has exited; the outer pin must still be active.
            after_inner_exit = _override_cache_key()
            assert after_inner_exit == outer_key, (
                "exiting the INNER pin must not clear the OUTER pin's key"
            )

        assert calls["n"] == 1, (
            f"expected exactly one stat across both nested pins, got {calls['n']}"
        )

        # Outermost pin has now exited — the pin is fully cleared, so the
        # next call re-stats (a fresh, unpinned lookup).
        _override_cache_key()
        assert calls["n"] == 2, (
            "after the OUTERMOST pin exits, the pin must be fully cleared so "
            "the next call re-stats"
        )

    def test_exception_inside_pinned_block_still_restores_state(self) -> None:
        """Arrange: none.
        Act: raise inside a pinned block.
        Assert: the pin is unwound (back to None / unpinned) even though the
        block exited via an exception, not a normal return.
        """
        import frugon.pricing as pricing_module
        from frugon.pricing import pinned_pricing_identity

        assert pricing_module._pinned_cache_key_var.get() is None, (
            "precondition: no pin active before this test"
        )

        pin_was_active_before_raise = False

        def _raise_inside_pin() -> None:
            nonlocal pin_was_active_before_raise
            with pinned_pricing_identity():
                pin_was_active_before_raise = (
                    pricing_module._pinned_cache_key_var.get() is not None
                )
                raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _raise_inside_pin()

        assert pin_was_active_before_raise, (
            "precondition: the pin must actually be active before the "
            "exception fires, or this test proves nothing about restoration"
        )
        assert pricing_module._pinned_cache_key_var.get() is None, (
            "a raised exception inside the pinned block must still restore "
            "the pin to unpinned — the context manager must not leak pinned "
            "state across an exceptional exit"
        )

    def test_unpinned_calls_still_observe_a_mid_process_rewrite(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Arrange: point pricing at a writable file, resolve a model — OUTSIDE
        any pin, exactly as :func:`test_pricing_update_invalidates_cache_via_mtime`
        does.
        Act: rewrite the file with a different price, resolve again — still
        with no pin active anywhere.
        Assert: the new price is returned — proving the pin is OPT-IN: with no
        pin engaged, unpinned callers keep the existing live mid-process
        mtime-invalidation guarantee unchanged, byte-for-byte the same
        behaviour as before this feature existed.
        """
        import json
        import os
        import time

        import frugon.pricing as pricing_module
        from frugon.pricing import clear_pricing_cache, get_model_price

        pricing_file = tmp_path / "pinned-counterpoint-pricing.json"
        pricing_file.write_text(
            json.dumps(
                {
                    "_last_synced": "2026-01-01",
                    "pin-counterpoint-model": {
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", pricing_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", pricing_file)
        clear_pricing_cache()

        assert pricing_module._pinned_cache_key_var.get() is None, (
            "precondition: no pin active — this test proves the UNPINNED path"
        )

        first = get_model_price("pin-counterpoint-model")
        assert first is not None
        assert first.input_cost_per_token == Decimal("0.000001")

        time.sleep(0.01)
        pricing_file.write_text(
            json.dumps(
                {
                    "_last_synced": "2026-02-02",
                    "pin-counterpoint-model": {
                        "input_cost_per_token": 0.000009,
                        "output_cost_per_token": 0.000008,
                    },
                }
            ),
            encoding="utf-8",
        )
        future = time.time() + 5
        os.utime(pricing_file, (future, future))

        second = get_model_price("pin-counterpoint-model")
        assert second is not None
        assert second.input_cost_per_token == Decimal("0.000009"), (
            "outside any pin, a rewritten pricing.json must STILL invalidate "
            "the cache via mtime/size — the pin changes nothing for the "
            "default, unpinned call path"
        )
        clear_pricing_cache()


class TestPricingFetchUserAgent:
    """The registry fetch must send an identifying User-Agent — some hosts reject
    the default ``Python-urllib`` agent (the same gap broke the quality refresh)."""

    def test_fetch_sends_identifying_user_agent(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from frugon import USER_AGENT
        from frugon.pricing import _LITELLM_REGISTRY_URL, fetch_and_update_pricing

        registry = json.dumps(
            {"gpt-4o": {"input_cost_per_token": 2.5e-06, "output_cost_per_token": 1e-05}}
        ).encode("utf-8")
        captured: list[Any] = []

        class FakeResponse:
            def read(self, *args: object) -> bytes:
                return registry

            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                pass

        def capturing_urlopen(req: Any, *args: object, **kwargs: object) -> FakeResponse:
            captured.append(req)
            return FakeResponse()

        with patch("urllib.request.urlopen", capturing_urlopen):
            result = fetch_and_update_pricing(
                registry_url=_LITELLM_REGISTRY_URL,
                output_path=tmp_path / "pricing.json",
                today_date_str="2026-06-04",
            )

        assert result["models_synced"] == 1
        assert captured, "expected a registry request"
        ua = captured[0].get_header("User-agent")
        assert ua is not None, "registry request sent without a User-Agent header"
        assert ua == USER_AGENT


class TestPricingFetchResilience:
    """The registry fetch (the cost-math path) retries on HTTP 429 / HTTP 5xx and
    transient network errors, the same as the quality fetch — raw.githubusercontent.com
    5xxs sporadically under load, and one bad response must not fail the whole update.
    A 4xx client error remains a permanent failure (not retried)."""

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

    def test_fetch_retries_on_transient_5xx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sporadic 5xx on the registry fetch is retried, not treated as fatal."""
        from unittest.mock import patch

        from frugon.pricing import _LITELLM_REGISTRY_URL, fetch_and_update_pricing

        registry = json.dumps(
            {"gpt-4o": {"input_cost_per_token": 2.5e-06, "output_cost_per_token": 1e-05}}
        ).encode("utf-8")
        calls = {"n": 0}

        def flaky_urlopen(req: Any, *args: object, **kwargs: object) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(req.full_url, 500, "Server Error", None, None)  # type: ignore[arg-type]
            return self._fake_response(registry)

        monkeypatch.setattr("time.sleep", lambda *args: None)  # skip real backoff
        with patch("urllib.request.urlopen", flaky_urlopen):
            result = fetch_and_update_pricing(
                registry_url=_LITELLM_REGISTRY_URL,
                output_path=tmp_path / "pricing.json",
                today_date_str="2026-06-04",
            )

        assert calls["n"] >= 2, "a transient 500 must be retried, not fatal"
        assert result["models_synced"] == 1

    def test_fetch_does_not_retry_on_4xx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 4xx client error is permanent — raised immediately, not retried."""
        from unittest.mock import patch

        from frugon.pricing import (
            _LITELLM_REGISTRY_URL,
            PricingUpdateError,
            fetch_and_update_pricing,
        )

        calls = {"n": 0}

        def always_404(req: Any, *args: object, **kwargs: object) -> Any:
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)  # type: ignore[arg-type]

        monkeypatch.setattr("time.sleep", lambda *args: None)
        output = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", always_404):
            with pytest.raises(PricingUpdateError):
                fetch_and_update_pricing(
                    registry_url=_LITELLM_REGISTRY_URL,
                    output_path=output,
                    today_date_str="2026-06-04",
                )

        assert calls["n"] == 1, "4xx must not be retried"
        assert not output.exists(), "output must NOT be written on a 4xx failure"

    def test_exhausted_5xx_retries_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A persistent 5xx exhausts the bounded retry budget then raises — the
        budget is exactly _FETCH_MAX_RETRIES + 1 total attempts."""
        from unittest.mock import patch

        from frugon.pricing import (
            _FETCH_MAX_RETRIES,
            _LITELLM_REGISTRY_URL,
            PricingUpdateError,
            fetch_and_update_pricing,
        )

        calls = {"n": 0}

        def always_500(req: Any, *args: object, **kwargs: object) -> Any:
            calls["n"] += 1
            raise urllib.error.HTTPError(req.full_url, 500, "Server Error", None, None)  # type: ignore[arg-type]

        monkeypatch.setattr("time.sleep", lambda *args: None)
        output = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", always_500):
            with pytest.raises(PricingUpdateError, match="pricing registry unavailable"):
                fetch_and_update_pricing(
                    registry_url=_LITELLM_REGISTRY_URL,
                    output_path=output,
                    today_date_str="2026-06-04",
                )

        assert calls["n"] == _FETCH_MAX_RETRIES + 1, (
            f"Expected {_FETCH_MAX_RETRIES + 1} attempts; got {calls['n']}"
        )
        assert not output.exists(), "output must NOT be written when retries exhausted"
