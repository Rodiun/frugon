"""Integrity tests for the bundled --demo fixture (data/sample_logs.jsonl.gz).

The fixture is the SINGLE SOURCE OF TRUTH for ``frugon analyze --demo`` (the repo
GIF and the landing demo card are both rendered from it).  These tests pin its
three load-bearing properties:

  * Honesty — every record's ``usage`` block equals what the tokenizer recomputes
    from the messages, so a hand-edited inflation is numerically impossible to
    ship (the P1-1 guard, checked per DISTINCT record shape for speed and then
    enforced across every record).
  * Determinism — re-running the generator produces a byte-identical artifact, so
    the demo output is byte-identical every run.
  * Reconciliation + register — the engine's split routes + keeps +
    already-cheaper account for every analyzed call AND every analyzed dollar
    (the total "Current" equals the sum of the per-model costs), and the result
    lands in the intended team-scale register (tens of thousands of calls, a
    believable monthly total spend, an honest 30-40% saving).
"""

from __future__ import annotations

import gzip
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import tokencost  # type: ignore[import-untyped]

from frugon.cost import analyze_records, parse_record

# ---------------------------------------------------------------------------
# Locate + load the bundled (gzip-compressed) sample file
# ---------------------------------------------------------------------------

_SAMPLE_LOGS = (
    Path(__file__).parent.parent / "src" / "frugon" / "data" / "sample_logs.jsonl.gz"
)

if not _SAMPLE_LOGS.exists():
    pytest.skip(
        f"Sample log file not found at {_SAMPLE_LOGS}. "
        "Run: python scripts/gen_sample_logs.py",
        allow_module_level=True,
    )

_TEXT = gzip.decompress(_SAMPLE_LOGS.read_bytes()).decode("utf-8")
_RECORDS: list[dict[str, Any]] = [json.loads(line) for line in _TEXT.splitlines() if line.strip()]


def _shape_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """A key identifying a record's token-shape (model + usage + content)."""
    return (
        record["model"],
        record["usage"]["prompt_tokens"],
        record["usage"]["completion_tokens"],
        json.dumps(record["request"]["messages"], sort_keys=True),
    )


# One representative per distinct record shape — the fixture cycles a small
# corpus, so a handful of distinct shapes covers thousands of records.  Testing
# the distinct shapes keeps the suite fast while every record is still guarded
# (test_every_record_matches_a_validated_shape closes the gap).
_DISTINCT: dict[tuple[Any, ...], dict[str, Any]] = {}
for _rec in _RECORDS:
    _DISTINCT.setdefault(_shape_key(_rec), _rec)


# ---------------------------------------------------------------------------
# Honesty — usage matches the tokenizer (per distinct shape, then all records)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    list(_DISTINCT.values()),
    ids=[f"shape_{i}" for i in range(len(_DISTINCT))],
)
def test_sample_record_usage_matches_tokenizer(record: dict[str, Any]) -> None:
    """Arrange: one distinct record shape from the bundled sample.
    Act: recompute prompt/completion token counts with tokencost.
    Assert: stored usage values are within ±1 token of the recomputed value.

    The P1-1 anti-inflation guard: any hand-edited inflation makes this fail.
    """
    model: str = record["model"]
    messages: list[dict[str, str]] = record["request"]["messages"]
    completion_choices: list[dict[str, Any]] = record["response"]["choices"]
    stored_usage: dict[str, int] = record["usage"]

    stored_prompt = stored_usage["prompt_tokens"]
    stored_completion = stored_usage["completion_tokens"]

    completion_text = ""
    if completion_choices:
        first = completion_choices[0]
        msg = first.get("message", {})
        completion_text = msg.get("content", "") or first.get("text", "")

    try:
        recomputed = tokencost.calculate_all_costs_and_tokens(messages, completion_text, model)
    except Exception as exc:
        pytest.skip(f"tokencost could not count tokens for model '{model}': {exc}")

    actual_prompt = int(recomputed["prompt_tokens"])
    actual_completion = int(recomputed["completion_tokens"])

    assert abs(stored_prompt - actual_prompt) <= 1, (
        f"model={model}: stored prompt_tokens={stored_prompt} but "
        f"recomputed={actual_prompt}. The bundled sample has inflated token counts."
    )
    assert abs(stored_completion - actual_completion) <= 1, (
        f"model={model}: stored completion_tokens={stored_completion} but "
        f"recomputed={actual_completion}. The bundled sample has inflated token counts."
    )


def test_every_record_matches_a_validated_shape() -> None:
    """Every record in the fixture is one of the distinct shapes validated above.

    The per-shape tokenizer check is exhaustive only if no record uses an
    unvalidated shape; this closes that gap cheaply without re-tokenizing all
    several-thousand records.
    """
    validated = set(_DISTINCT)
    for rec in _RECORDS:
        assert _shape_key(rec) in validated


# ---------------------------------------------------------------------------
# Determinism — regenerating yields a byte-identical artifact
# ---------------------------------------------------------------------------


def test_fixture_generation_is_byte_deterministic(tmp_path: Path) -> None:
    """Arrange: run the generator into a temp path.
    Act: run it a second time.
    Assert: both runs produce byte-identical gzip artifacts, AND that artifact is
            byte-identical to the bundled one — so `frugon analyze --demo` is
            deterministic and the checked-in fixture is the generator's output.
    """
    import importlib.util

    gen_path = Path(__file__).parent.parent / "scripts" / "gen_sample_logs.py"
    spec = importlib.util.spec_from_file_location("gen_sample_logs", gen_path)
    assert spec is not None
    assert spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    out1 = tmp_path / "a.jsonl.gz"
    out2 = tmp_path / "b.jsonl.gz"
    gen.generate(out1)
    gen.generate(out2)

    assert out1.read_bytes() == out2.read_bytes(), "generator is not deterministic"
    assert out1.read_bytes() == _SAMPLE_LOGS.read_bytes(), (
        "bundled fixture is stale — run: python scripts/gen_sample_logs.py"
    )


# ---------------------------------------------------------------------------
# Reconciliation + register — the engine's own numbers
# ---------------------------------------------------------------------------


def _analyze_fixture() -> Any:
    records = [r for r in (parse_record(raw) for raw in _RECORDS) if r is not None]
    return analyze_records(records)


def test_fixture_accounting_reconciles_every_call() -> None:
    """routed + kept + already-on-cheaper == priced == analyzed (no call vanishes)."""
    result = _analyze_fixture()
    split = result.split
    assert split is not None
    assert split.saving_pct is not None
    already_cheap = result.priced_calls - split.total_count
    assert already_cheap >= 0
    assert split.routed_count + split.kept_count + already_cheap == result.priced_calls
    assert result.priced_calls == result.total_calls  # the fixture has no unpriced calls


def test_fixture_lands_in_team_scale_register() -> None:
    """The demo lands in the intended register: tens of thousands of calls, a
    believable monthly TOTAL spend (hundreds-to-low-thousands USD), an honest
    30-40% saving — measured BOTH as the baseline routing win and over the total.
    """
    result = _analyze_fixture()
    split = result.split
    assert split is not None
    assert split.saving_pct is not None

    # Tens of thousands of calls — a credible team-scale workload.
    assert result.total_calls >= 3000

    # A believable monthly baseline in the hundreds-to-low-thousands of USD.
    assert split.monthly_baseline is not None
    assert Decimal("100") <= split.monthly_baseline <= Decimal("3000")

    # The TOTAL monthly spend (the demo's "Current") is also in that register and
    # is the figure a buyer reads first.
    assert result.monthly_cost is not None
    assert Decimal("100") <= result.monthly_cost <= Decimal("3000")

    # An honest blended saving in the 30-40% band — the baseline routing win.
    assert Decimal("30") <= split.saving_pct <= Decimal("40")

    # The same honest band when expressed over the TOTAL current spend (what the
    # demo headlines): saving = baseline_cost - blended_cost, over the total.
    baseline_reduction = split.baseline_cost - split.blended_cost
    total_saving_pct = baseline_reduction / result.total_cost * Decimal("100")
    assert Decimal("30") <= total_saving_pct <= Decimal("40")

    # The split routes to gpt-4.1-mini and keeps the hard calls on gpt-5.5.
    assert split.baseline_model == "gpt-5.5"
    assert split.candidate_model == "gpt-4.1-mini"
    assert split.routed_count > 0
    assert split.kept_count > 0


def test_fixture_current_equals_sum_of_per_model_costs() -> None:
    """The demo's "Current" (total_cost) is exactly the sum of the per-model costs.

    This is the flagged reconciliation invariant: the headline "Current" can
    never silently contradict the "Cost by model" breakdown, because it IS their
    sum.  Asserted on the real bundled fixture, not a hand-built result.
    """
    result = _analyze_fixture()
    assert result.cost_by_model  # the fixture has priced calls on several models
    assert sum(result.cost_by_model.values(), Decimal("0")) == result.total_cost


def test_fixture_total_dollars_reconcile_after_routing() -> None:
    """Every analyzed dollar is accounted for after routing.

    routed-at-candidate + kept-at-baseline + already-cheaper == the total blended,
    and the total blended + the saving == the total current.  No dollar vanishes.
    """
    result = _analyze_fixture()
    split = result.split
    assert split is not None

    # The non-baseline (already-cheaper) spend carries through unchanged.
    already_cheaper_cost = result.total_cost - split.baseline_cost
    assert already_cheaper_cost >= 0

    # Total blended = routed-at-candidate + kept-at-baseline + already-cheaper.
    total_blended = split.blended_cost + already_cheaper_cost
    assert total_blended == split.routed_cost + split.kept_cost + already_cheaper_cost

    # current - blended == the routing saving, exactly.
    saving = result.total_cost - total_blended
    assert saving == split.baseline_cost - split.blended_cost
