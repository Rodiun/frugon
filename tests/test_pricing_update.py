"""Tests for pricing update -- atomic write, network resilience, malformed-payload rejection.

Covers:
  - Atomic write: temp-then-rename, no leftover temp files
  - Network failure: existing pricing.json untouched on URLError
  - Malformed payload: empty dict / list / invalid JSON all rejected before write
  - _last_synced stamp: provided date written correctly
  - Models without pricing fields excluded from output
  - is_pricing_stale: threshold logic, None handling, bad-date handling
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frugon.pricing import _LITELLM_REGISTRY_URL

_VALID_REGISTRY: dict[str, object] = {
    "gpt-4o": {
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.00001,
        "max_tokens": 128000,
    },
    "gpt-4o-mini": {
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.0000006,
        "max_tokens": 128000,
    },
    "model-without-pricing": {
        "max_tokens": 4096,
    },
}


def _mock_urlopen(payload: object) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.close = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=resp)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _mock_urlopen_raw(raw: bytes) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = raw
    resp.close = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=resp)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


class TestAtomicWrite:
    """fetch_and_update_pricing writes atomically -- temp-then-rename."""

    def test_fetch_and_update_writes_pricing_json(self, tmp_path: Path) -> None:
        """Valid registry creates pricing.json with correct structure."""
        out = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_REGISTRY)):
            from frugon.pricing import fetch_and_update_pricing

            result = fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-06-01")
        assert out.exists()
        data: dict[str, object] = json.loads(out.read_text(encoding="utf-8"))
        assert data["_last_synced"] == "2026-06-01"
        assert result["models_synced"] == 2

    def test_no_temp_file_left_on_success(self, tmp_path: Path) -> None:
        """No .tmp file remains after a successful update."""
        out = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_REGISTRY)):
            from frugon.pricing import fetch_and_update_pricing

            fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-06-01")
        assert list(tmp_path.glob("*.tmp")) == []

    def test_network_failure_does_not_corrupt_existing_file(self, tmp_path: Path) -> None:
        """Existing pricing.json is untouched when the network fetch fails."""
        out = tmp_path / "pricing.json"
        original: dict[str, object] = {
            "_last_synced": "2026-01-01",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        }
        out.write_text(json.dumps(original), encoding="utf-8")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            from frugon.pricing import PricingUpdateError, fetch_and_update_pricing

            with pytest.raises(PricingUpdateError, match="Network error"):
                fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-06-01")

        data: dict[str, object] = json.loads(out.read_text(encoding="utf-8"))
        assert data["_last_synced"] == "2026-01-01"


class TestMalformedPayload:
    """Invalid registry payloads are rejected before any write."""

    def test_empty_object_rejected(self, tmp_path: Path) -> None:
        """Empty dict raises PricingUpdateError; file not created."""
        out = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen({})):
            from frugon.pricing import PricingUpdateError, fetch_and_update_pricing

            with pytest.raises(PricingUpdateError, match="no priced models"):
                fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-06-01")
        assert not out.exists()

    def test_list_payload_rejected(self, tmp_path: Path) -> None:
        """Array response raises PricingUpdateError; file not created."""
        out = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_raw(b"[1, 2, 3]")):
            from frugon.pricing import PricingUpdateError, fetch_and_update_pricing

            with pytest.raises(PricingUpdateError, match="unexpected"):
                fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-06-01")
        assert not out.exists()

    def test_invalid_json_rejected(self, tmp_path: Path) -> None:
        """Malformed JSON raises PricingUpdateError; file not created."""
        out = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen_raw(b"not json!!!")):
            from frugon.pricing import PricingUpdateError, fetch_and_update_pricing

            with pytest.raises(PricingUpdateError, match="JSON"):
                fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-06-01")
        assert not out.exists()


class TestLastSyncedStamp:
    """_last_synced is stamped with the provided date; non-priced models excluded."""

    def test_last_synced_matches_provided_date(self, tmp_path: Path) -> None:
        out = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_REGISTRY)):
            from frugon.pricing import fetch_and_update_pricing

            fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2099-12-31")
        data: dict[str, object] = json.loads(out.read_text(encoding="utf-8"))
        assert data["_last_synced"] == "2099-12-31"

    def test_models_without_pricing_excluded(self, tmp_path: Path) -> None:
        out = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_REGISTRY)):
            from frugon.pricing import fetch_and_update_pricing

            fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-06-01")
        data: dict[str, object] = json.loads(out.read_text(encoding="utf-8"))
        assert "model-without-pricing" not in data
        assert "gpt-4o" in data


class TestStalenessCheck:
    """is_pricing_stale threshold logic."""

    def test_fresh_pricing_not_stale(self) -> None:
        from frugon.pricing import is_pricing_stale

        assert is_pricing_stale("2026-06-01", max_days=30, today="2026-06-10") is False

    def test_stale_pricing_over_threshold(self) -> None:
        from frugon.pricing import is_pricing_stale

        assert is_pricing_stale("2026-04-01", max_days=30, today="2026-06-01") is True

    def test_exactly_at_threshold_is_stale(self) -> None:
        from frugon.pricing import is_pricing_stale

        assert is_pricing_stale("2026-05-02", max_days=30, today="2026-06-01") is True

    def test_one_day_before_threshold_not_stale(self) -> None:
        from frugon.pricing import is_pricing_stale

        assert is_pricing_stale("2026-05-03", max_days=30, today="2026-06-01") is False

    def test_none_last_synced_not_stale(self) -> None:
        from frugon.pricing import is_pricing_stale

        assert is_pricing_stale(None, max_days=30, today="2026-06-01") is False

    def test_invalid_date_format_not_stale(self) -> None:
        from frugon.pricing import is_pricing_stale

        assert is_pricing_stale("not-a-date", max_days=30, today="2026-06-01") is False


class TestRefreshSeedPrices:
    """refresh_seed_prices — curated seed updated in-place without adding/removing keys."""

    # Registry payloads used across tests in this class.  Deliberately richer
    # than the seed so we can verify that extra registry keys are never added.
    _SEED: dict[str, object] = {
        "_last_synced": "2026-01-01",
        "_source": "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
        "_note": "curated seed",
        "gpt-4o": {
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001,
        },
        "gpt-4o-mini": {
            "input_cost_per_token": 0.00000015,
            "output_cost_per_token": 0.0000006,
        },
        "claude-3-5-sonnet-20241022": {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
        },
    }

    # Registry has updated prices for gpt-4o only; gpt-4o-mini and the Anthropic
    # model are absent.  Many extra keys that must NEVER be added to the seed.
    _REGISTRY_PARTIAL_UPDATE: dict[str, object] = {
        "gpt-4o": {
            "input_cost_per_token": 0.000005,   # changed
            "output_cost_per_token": 0.00002,   # changed
            "max_tokens": 128000,
        },
        "registry-only-model": {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
        },
        "another-registry-only-model": {
            "input_cost_per_token": 0.0000005,
            "output_cost_per_token": 0.000001,
        },
        # Collision: exact key + prefixed key with different price.
        "claude-x": {
            "input_cost_per_token": 0.00001,
            "output_cost_per_token": 0.00003,
        },
        "bedrock/us-east-1/claude-x": {
            "input_cost_per_token": 0.000099,
            "output_cost_per_token": 0.000199,
        },
    }

    def _seed_file(self, tmp_path: Path) -> Path:
        """Write _SEED to a temp file and return its path."""
        p = tmp_path / "pricing.json"
        p.write_text(json.dumps(self._SEED), encoding="utf-8")
        return p

    def test_refresh_seed_prices_updates_matching_key_returns_updated_count(
        self, tmp_path: Path
    ) -> None:
        """Exact-match key with both cost fields updated; return value reflects count."""
        seed_path = self._seed_file(tmp_path)
        registry = {
            "gpt-4o": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.00002,
            }
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            from frugon.pricing import refresh_seed_prices

            result = refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2026-06-29")

        assert result["updated"] >= 1
        assert result["checked"] >= 1
        data: dict[str, object] = json.loads(seed_path.read_text(encoding="utf-8"))
        entry = data["gpt-4o"]
        assert isinstance(entry, dict)
        assert entry["input_cost_per_token"] == 0.000005
        assert entry["output_cost_per_token"] == 0.00002
        assert data["_last_synced"] == "2026-06-29"

    def test_refresh_seed_prices_key_absent_from_registry_price_retained(
        self, tmp_path: Path
    ) -> None:
        """Seed key absent from registry keeps its curated price; not counted as updated."""
        seed_path = self._seed_file(tmp_path)
        # Registry has gpt-4o but NOT gpt-4o-mini or claude-3-5-sonnet.
        registry = {
            "gpt-4o": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.00002,
            }
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            from frugon.pricing import refresh_seed_prices

            result = refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2026-06-29")

        # gpt-4o-mini and claude-3-5-sonnet-20241022 were NOT in registry → keep.
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        mini = data["gpt-4o-mini"]
        assert isinstance(mini, dict)
        assert mini["input_cost_per_token"] == 0.00000015
        assert mini["output_cost_per_token"] == 0.0000006
        sonnet = data["claude-3-5-sonnet-20241022"]
        assert isinstance(sonnet, dict)
        assert sonnet["input_cost_per_token"] == 0.000003
        # Only gpt-4o was updated.
        assert result["updated"] == 1
        assert result["checked"] == 3  # 3 model keys in seed

    def test_refresh_seed_prices_registry_entry_missing_one_cost_field_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Registry entry with only input_cost (no output) leaves seed entry unchanged."""
        seed_path = self._seed_file(tmp_path)
        registry = {
            "gpt-4o": {
                "input_cost_per_token": 0.000005,
                # output_cost_per_token deliberately absent
                "max_tokens": 128000,
            }
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            from frugon.pricing import refresh_seed_prices

            result = refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2026-06-29")

        # Nothing changed → file not written, so _last_synced stays "2026-01-01".
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        assert data["_last_synced"] == "2026-01-01"
        gpt4o = data["gpt-4o"]
        assert isinstance(gpt4o, dict)
        assert gpt4o["input_cost_per_token"] == 0.0000025  # original
        assert result["updated"] == 0

    def test_refresh_seed_prices_extra_registry_keys_not_added_to_seed(
        self, tmp_path: Path
    ) -> None:
        """Registry models not already in the seed are never added."""
        seed_path = self._seed_file(tmp_path)
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(self._REGISTRY_PARTIAL_UPDATE),
        ):
            from frugon.pricing import refresh_seed_prices

            refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2026-06-29")

        data = json.loads(seed_path.read_text(encoding="utf-8"))
        assert "registry-only-model" not in data
        assert "another-registry-only-model" not in data
        assert "claude-x" not in data
        assert "bedrock/us-east-1/claude-x" not in data
        # Seed key set is exactly what we started with (minus metadata keys).
        model_keys = [k for k in data if not k.startswith("_")]
        assert set(model_keys) == {"gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"}

    def test_refresh_seed_prices_absent_key_not_removed_even_if_not_in_registry(
        self, tmp_path: Path
    ) -> None:
        """A seed key absent from the registry is retained, never deleted."""
        seed_path = self._seed_file(tmp_path)
        # Registry contains only gpt-4o; gpt-4o-mini and the Anthropic model absent.
        registry = {
            "gpt-4o": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.00002,
            }
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            from frugon.pricing import refresh_seed_prices

            refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2026-06-29")

        data = json.loads(seed_path.read_text(encoding="utf-8"))
        assert "gpt-4o-mini" in data
        assert "claude-3-5-sonnet-20241022" in data

    def test_refresh_seed_prices_no_change_file_not_written_last_synced_not_bumped(
        self, tmp_path: Path
    ) -> None:
        """When no price changed, file is not written and _last_synced is not bumped."""
        seed_path = self._seed_file(tmp_path)
        mtime_before = seed_path.stat().st_mtime_ns
        original_bytes = seed_path.read_bytes()

        # Registry has identical prices to what's in the seed already.
        registry = {
            "gpt-4o": {
                "input_cost_per_token": 0.0000025,   # same as seed
                "output_cost_per_token": 0.00001,    # same as seed
            },
            "gpt-4o-mini": {
                "input_cost_per_token": 0.00000015,  # same
                "output_cost_per_token": 0.0000006,  # same
            },
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            from frugon.pricing import refresh_seed_prices

            result = refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2099-12-31")

        assert result["updated"] == 0
        assert result["checked"] == 3
        # File bytes must be bit-for-bit identical (not written).
        assert seed_path.read_bytes() == original_bytes
        # mtime must not have advanced.
        assert seed_path.stat().st_mtime_ns == mtime_before
        # _last_synced still the original value, not the new date.
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        assert data["_last_synced"] == "2026-01-01"
        assert data["_last_synced"] != "2099-12-31"

    def test_refresh_seed_prices_metadata_keys_preserved(self, tmp_path: Path) -> None:
        """All metadata keys (_source, _note, etc.) are preserved after a price update."""
        seed_path = self._seed_file(tmp_path)
        registry = {
            "gpt-4o": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.00002,
            }
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            from frugon.pricing import refresh_seed_prices

            refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2026-06-29")

        data = json.loads(seed_path.read_text(encoding="utf-8"))
        assert data["_source"] == self._SEED["_source"]
        assert data["_note"] == self._SEED["_note"]
        # _last_synced was bumped because a price changed.
        assert data["_last_synced"] == "2026-06-29"

    def test_refresh_seed_prices_collision_exact_key_wins_not_prefixed(
        self, tmp_path: Path
    ) -> None:
        """Seed key 'claude-x' gets the EXACT registry entry price, not the prefixed one."""
        seed: dict[str, object] = {
            "_last_synced": "2026-01-01",
            "claude-x": {
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00003,
            },
        }
        seed_path = tmp_path / "pricing.json"
        seed_path.write_text(json.dumps(seed), encoding="utf-8")

        # Registry has both the exact key and a prefixed variant with a
        # DIFFERENT price.  The function must use the exact key's price.
        registry = {
            "claude-x": {
                "input_cost_per_token": 0.00002,   # updated price
                "output_cost_per_token": 0.00006,  # updated price
            },
            "bedrock/us-east-1/claude-x": {
                "input_cost_per_token": 0.000099,
                "output_cost_per_token": 0.000199,
            },
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            from frugon.pricing import refresh_seed_prices

            result = refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2026-06-29")

        assert result["updated"] == 1
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        entry = data["claude-x"]
        assert isinstance(entry, dict)
        # Must be the EXACT key's price, not the prefixed variant's.
        assert entry["input_cost_per_token"] == 0.00002
        assert entry["output_cost_per_token"] == 0.00006


class TestHttpsPin:
    """fetch_and_update_pricing: HTTPS-only + host allowlist enforcement."""

    def test_http_url_raises_value_error(self, tmp_path: Path) -> None:
        """HTTP URL is rejected before any network call."""
        from frugon.pricing import fetch_and_update_pricing

        with pytest.raises(ValueError, match="HTTPS"):
            fetch_and_update_pricing(
                "http://raw.githubusercontent.com/bad.json",
                tmp_path / "pricing.json",
                "2026-01-01",
            )

    def test_unknown_host_raises_value_error(self, tmp_path: Path) -> None:
        """HTTPS URL with disallowed host is rejected before any network call."""
        from frugon.pricing import fetch_and_update_pricing

        with pytest.raises(ValueError, match="allowed"):
            fetch_and_update_pricing(
                "https://evil.example.com/prices.json",
                tmp_path / "pricing.json",
                "2026-01-01",
            )

    def test_allowed_host_passes_validation(self, tmp_path: Path) -> None:
        """raw.githubusercontent.com passes validation and the fetch proceeds."""
        from frugon.pricing import fetch_and_update_pricing

        out = tmp_path / "pricing.json"
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_REGISTRY)):
            result = fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-01-01")
        assert result["models_synced"] == 2

    def test_response_read_called_with_16mb_cap(self, tmp_path: Path) -> None:
        """resp.read() is called with the 16 MB limit, not without a bound."""
        from frugon.pricing import _MAX_RESPONSE_BYTES, fetch_and_update_pricing

        out = tmp_path / "pricing.json"
        resp = MagicMock()
        resp.read.return_value = json.dumps(_VALID_REGISTRY).encode("utf-8")
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=resp)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=ctx):
            fetch_and_update_pricing(_LITELLM_REGISTRY_URL, out, "2026-01-01")

        resp.read.assert_called_once_with(_MAX_RESPONSE_BYTES)


class TestRefreshSeedPricesFixedPoint:
    """refresh_seed_prices byte-stability, fixed-point, and error-branch tests."""

    # A minimal seed already at the json.dumps fixed point (sci-notation + trailing newline).
    # json.dumps(loaded, indent=2) + "\n" on this text must reproduce it exactly.
    _FIXED_POINT_SEED: dict[str, object] = {
        "_last_synced": "2026-01-01",
        "_source": "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
        "_note": "curated",
        "gpt-4o": {
            "input_cost_per_token": 2.5e-06,
            "output_cost_per_token": 1e-05,
        },
        "gpt-4o-mini": {
            "input_cost_per_token": 1.5e-07,
            "output_cost_per_token": 6e-07,
        },
        "claude-3-5-sonnet-20241022": {
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
        },
    }

    def _write_fixed_point_seed(self, tmp_path: Path) -> Path:
        """Write a seed already at the fixed point; return its path."""
        import json

        p = tmp_path / "pricing.json"
        p.write_text(json.dumps(self._FIXED_POINT_SEED, indent=2) + "\n", encoding="utf-8")
        return p

    def test_single_price_change_is_not_a_whole_file_reformat(
        self, tmp_path: Path
    ) -> None:
        """Byte-stability regression: changing exactly one key's price changes only
        the expected lines (that key's cost lines + _last_synced) — not the whole file.

        This is the primary guard against the sci-notation reformat bug: if atomic_write_json
        omits trailing_newline=True, or the seed was not at the fixed point, the diff
        would span ~50 lines instead of ~3.
        """
        seed_path = self._write_fixed_point_seed(tmp_path)
        old_lines = seed_path.read_text(encoding="utf-8").splitlines()

        # Registry raises gpt-4o's price; other keys unchanged.
        registry = {
            "gpt-4o": {
                "input_cost_per_token": 5e-06,   # changed from 2.5e-06
                "output_cost_per_token": 2e-05,  # changed from 1e-05
            },
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            from frugon.pricing import refresh_seed_prices

            result = refresh_seed_prices(_LITELLM_REGISTRY_URL, seed_path, "2026-06-29")

        assert result["updated"] == 1
        new_lines = seed_path.read_text(encoding="utf-8").splitlines()

        # Identify exactly which lines changed.
        differing = {
            i for i, (old, new) in enumerate(zip(old_lines, new_lines, strict=False)) if old != new
        }
        # Also flag any lines added or removed at the end.
        if len(old_lines) != len(new_lines):
            for i in range(min(len(old_lines), len(new_lines)), max(len(old_lines), len(new_lines))):
                differing.add(i)

        # We expect ONLY 3 changed lines:
        #   - _last_synced line
        #   - gpt-4o input_cost_per_token line
        #   - gpt-4o output_cost_per_token line
        # If this assertion fails, the file underwent a whole-file reformat.
        assert len(differing) == 3, (
            f"Expected exactly 3 changed lines (gpt-4o input/output + _last_synced), "
            f"got {len(differing)} changed lines: {sorted(differing)}. "
            f"This indicates a whole-file reformat (sci-notation / trailing-newline bug)."
        )

        # Confirm the changed lines are the expected ones.
        changed_content = {new_lines[i] for i in differing}
        assert any("_last_synced" in line for line in changed_content), (
            "_last_synced line must be among the changed lines"
        )
        assert any("5e-06" in line or "2e-05" in line for line in changed_content), (
            "gpt-4o new prices must appear in the changed lines"
        )

        # Confirm the file still ends with a trailing newline (fixed-point requirement).
        assert seed_path.read_text(encoding="utf-8").endswith("\n"), (
            "Seed file must end with a trailing newline after refresh_seed_prices write"
        )

    def test_shipped_seed_is_at_fixed_point(self) -> None:
        """Guard: the committed src/frugon/data/pricing.json must be byte-identical
        to json.dumps(loaded, indent=2) + "\\n".

        If this test fails, a hand-edit has introduced decimal literals (or removed
        the trailing newline) that will cause whole-file reformat churn on the next
        real price sync.  Fix by running:
            python -c "import json; from pathlib import Path; \\
                p=Path('src/frugon/data/pricing.json'); \\
                p.write_text(json.dumps(json.loads(p.read_text()), indent=2)+'\\n', encoding='utf-8')"
        """
        import json
        from pathlib import Path

        seed = Path(__file__).parent.parent / "src" / "frugon" / "data" / "pricing.json"
        raw = seed.read_text(encoding="utf-8")
        loaded = json.loads(raw)
        normalized = json.dumps(loaded, indent=2) + "\n"
        assert raw == normalized, (
            "src/frugon/data/pricing.json is NOT at the json.dumps fixed point. "
            "Hand-edits have introduced decimal literals or removed the trailing newline. "
            "Re-normalize the seed with: "
            "python -c \"import json; from pathlib import Path; "
            "p=Path('src/frugon/data/pricing.json'); "
            "p.write_text(json.dumps(json.loads(p.read_text()), indent=2)+'\\\\n', encoding='utf-8')\""
        )

    def test_missing_or_unparseable_seed_raises_pricing_update_error(
        self, tmp_path: Path
    ) -> None:
        """refresh_seed_prices raises PricingUpdateError when the seed is missing
        or contains garbage (the except (OSError, json.JSONDecodeError) branch).

        Two sub-cases:
          a. Seed file does not exist at all → OSError branch.
          b. Seed file exists but contains garbage → json.JSONDecodeError branch.
        """
        from frugon.pricing import PricingUpdateError, refresh_seed_prices

        registry = {
            "gpt-4o": {
                "input_cost_per_token": 2.5e-06,
                "output_cost_per_token": 1e-05,
            },
        }

        # Case a: nonexistent seed → OSError.
        nonexistent = tmp_path / "does_not_exist.json"
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            with pytest.raises(PricingUpdateError, match="Cannot read seed file"):
                refresh_seed_prices(_LITELLM_REGISTRY_URL, nonexistent, "2026-06-29")

        # Case b: garbage seed → json.JSONDecodeError.
        garbage_seed = tmp_path / "garbage.json"
        garbage_seed.write_text("this is not json }{", encoding="utf-8")
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(registry),
        ):
            with pytest.raises(PricingUpdateError, match="Cannot read seed file"):
                refresh_seed_prices(_LITELLM_REGISTRY_URL, garbage_seed, "2026-06-29")
