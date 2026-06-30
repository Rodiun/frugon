"""Shared test fixtures for frugon.

The unrated-behaviour tests need a model that is PRICED (so a routing split
forms and a saving is computed) yet UNRATED (absent from the quality table).
Pinning that role to a real model is fragile: as the bundled registries update,
a once-unrated model gets rated and a once-priced model loses its price, which
is exactly what broke these tests. This sentinel decouples the unrated-path
BEHAVIOUR from real registry data entirely.
"""
from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

import frugon.pricing as _pricing
import frugon.quality as _quality
from frugon.cli import app

# ---------------------------------------------------------------------------
# Render-independent help-text assertions
#
# Typer renders --help through Rich. On CI, the ``GITHUB_ACTIONS`` env var makes
# typer force-enable terminal mode (``rich_utils.FORCE_TERMINAL``), so the help
# is emitted WITH ANSI colour codes and wrapped to an 80-column fallback width
# (Rich probes ``os.get_terminal_size()`` and ignores ``COLUMNS`` once a terminal
# is forced). Two artifacts break naive substring checks that pass locally:
#
#   1. The OptionHighlighter styles the leading dash of a flag separately, so the
#      bytes become ``-\x1b[0m\x1b[1;36m-concurrency`` — ``"--concurrency"`` is no
#      longer a contiguous substring.
#   2. At 80 columns, multi-word help phrases wrap across lines separated by box
#      borders, so e.g. ``"A/B order"`` is split as ``A/B │\n │ order``.
#
# ``help_text`` renders the help and returns a content-canonical form: ANSI codes
# stripped, box-drawing glyphs flattened to spaces, and whitespace collapsed. The
# CONTRACT under test (the help advertises flag/text X) is preserved exactly while
# decoupling it from terminal width and colour. This is the single source of truth
# for every help-surface assertion; do not re-implement ad-hoc strippers.
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")
# Unicode box-drawing block (U+2500–U+257F): borders Rich uses for help panels.
_BOX_TRANS = dict.fromkeys(range(0x2500, 0x2580), " ")
_runner = CliRunner()


def help_text(*args: str) -> str:
    """Render ``frugon <args> --help`` as render-independent canonical text.

    ANSI escape codes are stripped, Rich box-drawing borders are flattened to
    spaces, and runs of whitespace are collapsed to a single space. The result
    is stable across terminal width, colour mode, and OS, so substring checks
    on flag names and help copy hold identically locally and on CI.
    """
    result = _runner.invoke(app, [*args, "--help"], env={"COLUMNS": "200", "TERM": "dumb"})
    assert result.exit_code == 0, result.output
    cleaned = _ANSI_RE.sub("", result.output).translate(_BOX_TRANS)
    return re.sub(r"\s+", " ", cleaned).strip()

# A model name guaranteed absent from every quality table (always "unrated")
# and not a real provider model.
FRUGON_TEST_UNRATED = "frugon-eval-unrated-x1"

# $1 / $5 per million tokens (per-token Decimals, as strings). On the bundled
# demo log (gpt-5.5 baseline) this reproduces the priced-but-unrated
# figures exactly: split saving 31.3%, full-dataset New-spend $376.913547.
_SENTINEL_INPUT = "0.000001"
_SENTINEL_OUTPUT = "0.000005"


def install_unrated_sentinel(monkeypatch, tmp_path):
    """Point the pricing override at (bundled seed + the unrated sentinel).

    Hermetic: every pricing lookup resolves the sentinel via the override
    table, so a split forms and a New-spend is computed, while the sentinel
    stays absent from the quality table. Every other model prices exactly as
    the bundled seed prices it (no dependence on the user data dir).
    """
    bundled = json.loads(_pricing._BUNDLED_SEED_PATH.read_text(encoding="utf-8"))
    bundled[FRUGON_TEST_UNRATED] = {
        "input_cost_per_token": _SENTINEL_INPUT,
        "output_cost_per_token": _SENTINEL_OUTPUT,
    }
    override = tmp_path / "pricing_with_sentinel.json"
    override.write_text(json.dumps(bundled), encoding="utf-8")
    monkeypatch.setattr(_pricing, "_PRICING_JSON", override)
    _pricing.clear_pricing_cache()


@pytest.fixture
def unrated_model(monkeypatch, tmp_path):
    """A priced-but-unrated sentinel model name (drift-proof)."""
    install_unrated_sentinel(monkeypatch, tmp_path)
    yield FRUGON_TEST_UNRATED
    _pricing.clear_pricing_cache()


def install_synthetic_quality(monkeypatch, tmp_path, tiers):
    """Pin a synthetic quality tier table so a test CONTROLS the tiers it asserts.

    The quality-axis analogue of ``install_unrated_sentinel``: routing, judge,
    and synthesis LOGIC tests own the exact tier RELATIONSHIPS they exercise, so
    a future leaderboard re-anchor (which legitimately re-bands real models as
    the field grows) can never re-break them. This is the drift-proof pattern for
    any test whose intent is a specific tier scenario (a cross-tier gap, a
    cross-provider tie-break, an escalation rung) rather than the shipped seed's
    real bands — which the seed-validation tests in test_quality.py deliberately
    keep asserting against the real table.

    *tiers* maps base_family model names — the form ``get_model_tier`` resolves to
    after ``canonicalize`` + ``base_family`` (e.g. ``"claude-3-5-sonnet"``, not the
    dated ``"claude-3-5-sonnet-20241022"``) — to integer tiers (0=Elite … 3=Efficient).
    Any model absent from *tiers* reads as UNRATED, exactly like the real table.

    Hermetic via the single ``frugon.quality._QUALITY_JSON`` lever:
    ``load_quality_table`` reads that module global fresh on every call (uncached)
    and every tier path funnels through it — ``quality.get_model_tier``,
    ``cost._get_model_tier`` (the captured ``get_model_tier`` reference), and
    ``measure.best_judge_for_available_keys`` (call-time import) — so one patch
    governs them all. No cache to clear.
    """
    table: dict[str, object] = {
        "_last_synced": "2026-01-01",
        "_attribution": "synthetic test quality table",
    }
    table.update(tiers)
    override = tmp_path / "quality_synthetic.json"
    override.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setattr(_quality, "_QUALITY_JSON", override)
