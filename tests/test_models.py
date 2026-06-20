"""Tests for the `frugon models` discovery command (Item 3).

`frugon models [QUERY]` lists the model names frugon can price from the LOCAL
pricing table — the same table `--candidates` resolves against — so the names
shown are exactly what `--candidates` accepts.  Pure local read, no network.

Two layers under test:
  * ``list_priced_models`` — the name-sorted, optionally-substring-filtered data
    rows (name + per-token input/output cost + quality tier).
  * the ``frugon models`` CLI command — table render, count footer, and the
    clean no-match message (no traceback, exit 0).
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

import frugon.pricing as pricing_module
from frugon.cli import app
from frugon.pricing import list_priced_models
from frugon.quality import UNRATED_TIER, get_model_tier, tier_name

sys.path.insert(0, str(Path(__file__).parent))
from conftest import install_synthetic_quality

runner = CliRunner()


# A small, controlled pricing table so the data-layer assertions do not couple to
# the live bundled snapshot.  gpt-4o / gpt-4o-mini are rated in the quality table
# (Elite / Capable); "made-up-model" is not, so it exercises the unrated path.
_FIXTURE_TABLE = {
    "gpt-4o": {"input_cost_per_token": 2.5e-06, "output_cost_per_token": 1e-05},
    "gpt-4o-mini": {"input_cost_per_token": 1.5e-07, "output_cost_per_token": 6e-07},
    "made-up-model": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
}


@pytest.fixture
def _fixed_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force list_priced_models to read a small, deterministic pricing table."""
    monkeypatch.setattr(
        pricing_module,
        "load_pricing_override",
        lambda: (_FIXTURE_TABLE, "2026-06-04"),
    )


# ---------------------------------------------------------------------------
# Data layer — list_priced_models
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_fixed_table")
def test_list_priced_models_lists_all_sorted_by_name() -> None:
    # Act
    rows = list_priced_models()

    # Assert — every table model, name-sorted, with costs carried through.
    assert [r.model for r in rows] == ["gpt-4o", "gpt-4o-mini", "made-up-model"]
    gpt4o = rows[0]
    assert gpt4o.input_cost_per_token == Decimal("2.5e-06")
    assert gpt4o.output_cost_per_token == Decimal("1e-05")


@pytest.mark.usefixtures("_fixed_table")
def test_list_priced_models_filters_by_case_insensitive_substring() -> None:
    # Act — uppercase query still matches the lowercase names.
    rows = list_priced_models("GPT-4O-MINI")

    # Assert — only the matching model.
    assert [r.model for r in rows] == ["gpt-4o-mini"]


@pytest.mark.usefixtures("_fixed_table")
def test_list_priced_models_empty_for_no_match() -> None:
    # Act
    rows = list_priced_models("zzz-no-such-model")

    # Assert — empty list, not an error.
    assert rows == []


@pytest.mark.usefixtures("_fixed_table")
def test_list_priced_models_carries_quality_tier_when_known(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Pin the quality tiers so the row-threading assertion is EXACT and
    # drift-proof: the contract is "each row carries get_model_tier(name), and an
    # unrated model falls back to the sentinel". A leaderboard re-anchor re-bands
    # real models, so the test owns the tiers rather than reading the live seed.
    install_synthetic_quality(monkeypatch, tmp_path, {"gpt-4o": 0, "gpt-4o-mini": 2})

    # Act
    by_name = {r.model: r for r in list_priced_models()}

    # Assert — rated models carry their pinned tier; unrated falls back to the sentinel.
    assert by_name["gpt-4o"].quality_tier == 0  # pinned Elite
    assert by_name["gpt-4o-mini"].quality_tier == 2  # pinned Capable
    assert by_name["made-up-model"].quality_tier == UNRATED_TIER


# ---------------------------------------------------------------------------
# CLI — frugon models
# ---------------------------------------------------------------------------


def test_models_lists_all_against_live_table() -> None:
    # Act — no query: lists the real bundled pricing table.
    result = runner.invoke(app, ["models"], catch_exceptions=False)

    # Assert — exits clean, shows well-known names + the count footer.
    assert result.exit_code == 0, result.output
    assert "gpt-4o" in result.output
    assert "models" in result.output  # the "N models" footer


def test_models_filters_by_substring() -> None:
    # Act
    result = runner.invoke(app, ["models", "gpt-4o"], catch_exceptions=False)

    # Assert — every shown name contains the query; the footer echoes it.
    assert result.exit_code == 0, result.output
    assert "gpt-4o" in result.output
    assert 'query "gpt-4o"' in result.output
    # The filter must not surface an obviously unrelated family.
    assert "claude-3-haiku" not in result.output


def test_models_shows_quality_tier_when_known() -> None:
    # Act — a bare model name that is rated in the quality table.
    result = runner.invoke(app, ["models", "gpt-4o"], catch_exceptions=False)

    # Assert — the human-readable tier label appears for a rated model.
    # Compute the expected band from the live table so the assertion does not
    # couple to a fixed tier: re-anchoring the leaderboard moves gpt-4o between
    # bands, but whatever band it currently holds must render in the output.
    assert result.exit_code == 0, result.output
    tier = get_model_tier("gpt-4o")
    assert tier != UNRATED_TIER, "gpt-4o must be rated for this test to be meaningful"
    label = tier_name(tier)
    assert label is not None
    assert label in result.output


def test_models_empty_match_is_clean_and_exits_zero() -> None:
    # Act — a query that matches nothing.
    result = runner.invoke(app, ["models", "zzznomatch"], catch_exceptions=False)

    # Assert — friendly hint, no traceback, exit 0.
    assert result.exit_code == 0, result.output
    assert "no models match" in result.output
    assert "zzznomatch" in result.output
    assert "Traceback" not in result.output
