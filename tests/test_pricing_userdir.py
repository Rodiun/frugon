"""Tests for user-data-dir pricing location.

Covers the fix that moves the writable pricing table out of the installed
wheel so reinstalls no longer silently revert user-synced prices.

Scenarios tested:
  - Reads from user-dir file when it is present.
  - Seeds from bundled file when the user-dir file does not yet exist
    (first install / empty data dir).
  - ``frugon pricing update`` (via fetch_and_update_pricing) writes to the
    user-dir path, NOT the wheel-bundled path.
  - Simulated reinstall: after a successful update, replacing the bundled
    seed with old data does not affect the user-dir file (reinstall
    durability).
  - Legacy migration: if an older version left a modified pricing.json at
    the bundled-seed location, the first access copies it to the user dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frugon.pricing import _LITELLM_REGISTRY_URL

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_REGISTRY: dict[str, object] = {
    "gpt-4o": {
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.00001,
    },
    "gpt-4o-mini": {
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.0000006,
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


def _write_pricing_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReadsFromUserDir:
    """load_pricing_override reads from the user-dir file when it exists."""

    def test_reads_user_dir_file_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: user-dir pricing.json exists with a custom price.
        Act: load_pricing_override.
        Assert: the custom price is returned; source is 'pricing.json'.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import get_model_price, load_pricing_override

        user_file = tmp_path / "user" / "pricing.json"
        bundled_file = tmp_path / "bundled" / "pricing.json"

        # User-dir file has a distinct price for gpt-4o.
        _write_pricing_json(user_file, {
            "_last_synced": "2099-01-01",
            "gpt-4o": {"input_cost_per_token": 0.9999, "output_cost_per_token": 1.9999},
        })
        # Bundled seed has a different price to confirm user-dir wins.
        _write_pricing_json(bundled_file, {
            "_last_synced": "2024-01-01",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        })

        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", bundled_file)

        table, last_synced = load_pricing_override()
        assert "gpt-4o" in table
        assert float(table["gpt-4o"]["input_cost_per_token"]) == pytest.approx(0.9999)
        assert last_synced == "2099-01-01"

        price = get_model_price("gpt-4o")
        assert price is not None
        assert price.source == "pricing.json"
        assert float(price.input_cost_per_token) == pytest.approx(0.9999)

    def test_last_synced_exposed_from_user_dir_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: user-dir file has _last_synced = '2099-06-01'.
        Act: load_pricing_override.
        Assert: last_synced matches.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import load_pricing_override

        user_file = tmp_path / "pricing.json"
        _write_pricing_json(user_file, {
            "_last_synced": "2099-06-01",
            "some-model": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
        })

        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", user_file)

        _, last_synced = load_pricing_override()
        assert last_synced == "2099-06-01"


class TestSeedsFromBundledOnFirstInstall:
    """When user-dir file is absent, the bundled seed is used as fallback."""

    def test_bundled_seed_used_when_user_dir_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: user-dir pricing.json does not exist; bundled seed has gpt-4o.
        Act: load_pricing_override.
        Assert: gpt-4o price is returned (seeded from bundled file).
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import get_model_price

        user_file = tmp_path / "user" / "pricing.json"
        bundled_file = tmp_path / "bundled" / "pricing.json"
        _write_pricing_json(bundled_file, {
            "_last_synced": "2025-01-01",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        })
        # User-dir file intentionally absent.
        assert not user_file.exists()

        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", bundled_file)

        price = get_model_price("gpt-4o")
        assert price is not None
        assert price.source == "pricing.json"

    def test_seed_copy_created_in_user_dir_on_first_use(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: user-dir file absent; bundled seed present.
        Act: load_pricing_override (triggers _ensure_user_pricing_exists).
        Assert: user-dir file now exists as a copy of the seed.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import load_pricing_override

        user_file = tmp_path / "user" / "pricing.json"
        bundled_file = tmp_path / "bundled" / "pricing.json"
        _write_pricing_json(bundled_file, {
            "_last_synced": "2025-01-01",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        })
        assert not user_file.exists()

        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", bundled_file)

        load_pricing_override()

        # The user-dir file should now exist (seeded from bundle).
        assert user_file.exists()
        data: dict[str, object] = json.loads(user_file.read_text(encoding="utf-8"))
        assert data.get("_last_synced") == "2025-01-01"


class TestUpdateWritesToUserDir:
    """fetch_and_update_pricing writes to the path given as output_path.

    The CLI passes _PRICING_JSON (the user-dir path), so updates land in
    the user data dir, never in the wheel package directory.
    """

    def test_update_writes_to_user_dir_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: user-dir path provided as output_path.
        Act: fetch_and_update_pricing.
        Assert: user-dir file updated; bundled seed unchanged.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import fetch_and_update_pricing

        user_file = tmp_path / "user" / "pricing.json"
        bundled_file = tmp_path / "bundled" / "pricing.json"
        _write_pricing_json(bundled_file, {
            "_last_synced": "2024-01-01",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        })

        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", bundled_file)

        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_REGISTRY)):
            result = fetch_and_update_pricing(
                _LITELLM_REGISTRY_URL,
                user_file,  # caller passes _PRICING_JSON (the user-dir path)
                "2099-06-01",
            )

        assert result["models_synced"] == 2
        assert user_file.exists()

        data: dict[str, object] = json.loads(user_file.read_text(encoding="utf-8"))
        assert data["_last_synced"] == "2099-06-01"

        # Bundled seed must not have been modified.
        bundled_data: dict[str, object] = json.loads(bundled_file.read_text(encoding="utf-8"))
        assert bundled_data["_last_synced"] == "2024-01-01"

    def test_update_result_readable_via_load_pricing_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After update, load_pricing_override returns the freshly synced data."""
        from frugon import pricing as pricing_module
        from frugon.pricing import fetch_and_update_pricing, load_pricing_override

        user_file = tmp_path / "pricing.json"
        bundled_file = tmp_path / "seed_pricing.json"
        _write_pricing_json(bundled_file, {
            "_last_synced": "2024-01-01",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        })

        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", bundled_file)

        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_REGISTRY)):
            fetch_and_update_pricing(_LITELLM_REGISTRY_URL, user_file, "2099-06-01")

        table, last_synced = load_pricing_override()
        assert last_synced == "2099-06-01"
        assert "gpt-4o" in table
        assert "gpt-4o-mini" in table


class TestReinstallDurability:
    """Replacing the bundled seed (simulating reinstall) must not revert synced prices."""

    def test_user_dir_persists_after_bundled_seed_reverts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: user has run 'pricing update'; the bundled seed is then replaced
        with an older snapshot (simulating reinstall with an older wheel).
        Act: load_pricing_override after the seed reverts.
        Assert: the user-dir file is unchanged; synced prices are preserved.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import fetch_and_update_pricing, load_pricing_override

        user_file = tmp_path / "user" / "pricing.json"
        bundled_file = tmp_path / "bundled" / "pricing.json"

        _write_pricing_json(bundled_file, {
            "_last_synced": "2024-01-01",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        })

        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", bundled_file)

        # Step 1: user runs 'frugon pricing update' -- writes fresh prices.
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_REGISTRY)):
            fetch_and_update_pricing(
                _LITELLM_REGISTRY_URL, user_file, "2099-06-01"
            )

        user_data_before: dict[str, object] = json.loads(
            user_file.read_text(encoding="utf-8")
        )
        assert user_data_before["_last_synced"] == "2099-06-01"

        # Step 2: simulate reinstall -- overwrite the bundled seed with older data.
        _write_pricing_json(bundled_file, {
            "_last_synced": "2022-01-01",
            "gpt-4o": {"input_cost_per_token": 0.00006, "output_cost_per_token": 0.00012},
        })

        # Step 3: read pricing again -- must still see the user-synced date.
        table, last_synced = load_pricing_override()
        assert last_synced == "2099-06-01", (
            "Reinstall (bundled seed reverted) silently reset synced prices; "
            "user-dir file must take precedence over the bundled seed."
        )
        assert float(table["gpt-4o"]["input_cost_per_token"]) == pytest.approx(0.0000025), (
            "Price from synced user-dir file must not be overwritten by bundled seed."
        )


class TestFirstRunSeedBehavior:
    """Characterisation tests for the seed-copy-on-first-run path."""

    def test_seed_content_preserved_in_user_dir_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: user-dir file absent; bundled seed contains custom prices.
        Act: load_pricing_override (triggers _ensure_user_pricing_exists).
        Assert: user-dir file is created as an exact copy of the bundled seed,
                including the _last_synced date and all model entries.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import load_pricing_override

        user_file = tmp_path / "user" / "pricing.json"
        bundled_file = tmp_path / "bundled" / "pricing.json"
        _write_pricing_json(bundled_file, {
            "_last_synced": "2025-03-15",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
            "gpt-4o-mini": {"input_cost_per_token": 0.00000015, "output_cost_per_token": 0.0000006},
        })
        assert not user_file.exists()

        monkeypatch.setattr(pricing_module, "_PRICING_JSON", user_file)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", bundled_file)

        table, last_synced = load_pricing_override()

        assert user_file.exists(), "Seed copy must create user-dir file on first run."
        data: dict[str, object] = json.loads(user_file.read_text(encoding="utf-8"))
        assert data.get("_last_synced") == "2025-03-15", (
            "User-dir file must have the same _last_synced as the bundled seed."
        )
        assert "gpt-4o" in data
        assert "gpt-4o-mini" in data
        assert "gpt-4o" in table
        assert last_synced == "2025-03-15"

    def test_missing_bundled_seed_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arrange: both user-dir file and bundled seed are absent.
        Act: load_pricing_override.
        Assert: no exception; returns empty dict so tokencost fallback takes over.
        """
        from frugon import pricing as pricing_module
        from frugon.pricing import load_pricing_override

        nonexistent = tmp_path / "nonexistent.json"
        monkeypatch.setattr(pricing_module, "_PRICING_JSON", nonexistent)
        monkeypatch.setattr(pricing_module, "_BUNDLED_SEED_PATH", nonexistent)

        table, last_synced = load_pricing_override()
        assert isinstance(table, dict)
        assert last_synced is None
