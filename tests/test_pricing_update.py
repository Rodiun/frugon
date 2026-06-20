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
