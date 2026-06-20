"""The headline routing target and the "Candidates considered" block agree.

Regression guard for the self-contradiction a user witnessed on
``--candidates gpt-4o,frugon-eval-unrated-x1`` over the bundled demo log: the HEADLINE
cost panel routed easy calls to one model (the cheapest *full-swap* candidate)
while the block marked a DIFFERENT model ``recommended`` (the cheapest *split*
candidate) — and the loser showed two different numbers (a full-dataset split
figure in the headline vs a dominant-only ``compute_split`` figure in the block).

The fix unifies everything on ONE basis: every candidate is ranked and shown by
its FULL-DATASET split New-spend — route the dominant model's easy calls to that
candidate, keep its hard calls on the baseline, leave already-cheaper calls
untouched, over the whole analyzed dataset.  The cheapest such candidate is the
headline routing target AND the block's ``recommended`` row, and each candidate
shows exactly ONE number that reconciles between the two surfaces.

These tests pin that agreement on the user's exact scenario and on a synthetic
case constructed so the full-swap-cheapest and split-cheapest candidates differ
— proving the headline and the block can never again name different models.
"""

from __future__ import annotations

import sys as _sys
from decimal import Decimal
from pathlib import Path

import pytest

import frugon
from frugon.cost import (
    AnalysisResult,
    CandidateProjection,
    analyze_records,
    iter_records,
)
from frugon.report import _split_current_and_blended, render_html, render_terminal

_sys.path.insert(0, str(Path(__file__).parent))
from conftest import install_unrated_sentinel

assert frugon.__file__ is not None
_SAMPLE = Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"


@pytest.fixture(autouse=True)
def _sentinel_pricing(monkeypatch, tmp_path):
    install_unrated_sentinel(monkeypatch, tmp_path)
    yield
    import frugon.pricing as _p

    _p.clear_pricing_cache()


def _demo_result(candidates: list[str]) -> AnalysisResult:
    """Analyze the bundled demo log (chatgpt-4o-latest dominant + 10k already-on-gpt-4o-mini)."""
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(
        list(records),
        candidates=candidates,
        skipped_malformed=skipped,
        split_routing=True,
    )


def _block(result: AnalysisResult) -> dict[str, CandidateProjection]:
    return {p.model: p for p in result.candidate_projections}


# ---------------------------------------------------------------------------
# The user's exact scenario: --candidates gpt-4o,frugon-eval-unrated-x1 on the demo.
# ---------------------------------------------------------------------------


def test_pd_scenario_headline_equals_block_recommended() -> None:
    """gpt-4o,frugon-eval-unrated-x1 on the demo: headline target == block recommended.

    Quality-awareness (Change 1): only RATED candidates are eligible for the
    headline.  frugon-eval-unrated-x1 has the cheaper full-dataset split but is UNRATED,
    so it is held out of the recommended route and tagged ``considered``; gpt-4o
    (rated, Elite) is the cheapest ELIGIBLE candidate that beats baseline, so it
    is BOTH the headline routing target and the block's ``recommended`` row.  The
    headline==block invariant is preserved — it is just anchored on the rated pick.
    """
    result = _demo_result(["gpt-4o", "frugon-eval-unrated-x1"])

    assert result.split is not None
    # The headline panel names split.candidate_model as the routing target.
    headline_target = result.split.candidate_model
    assert headline_target == "gpt-4o"
    assert result.candidate_model == "gpt-4o"

    block = _block(result)
    recommended = [m for m, p in block.items() if p.status == "recommended"]
    assert recommended == ["gpt-4o"], (
        "the block's recommended row must be the headline routing target"
    )
    # frugon-eval-unrated-x1 beats baseline on split but is unrated → considered, and
    # held out of the recommendation (Change 1b).
    assert block["frugon-eval-unrated-x1"].status == "considered"
    assert "frugon-eval-unrated-x1" in result.excluded_unrated_models


def test_pd_scenario_recommended_row_equals_headline_newspend_to_the_cent() -> None:
    """The recommended row's monthly == the headline New-spend, to the cent.

    Anchored on the rated pick (gpt-4o) now that Change 1 routes the headline to
    the cheapest rated candidate rather than the cheapest candidate overall.
    """
    result = _demo_result(["gpt-4o", "frugon-eval-unrated-x1"])
    assert result.split is not None

    # The headline New-spend the cost panel renders (its single source).
    _current, headline_newspend, _projected = _split_current_and_blended(
        result, result.split
    )

    rec = _block(result)["gpt-4o"]
    assert rec.monthly_cost is not None
    # Exact (full-precision) equality: the recommended row and the headline share
    # the SAME underlying Decimal, not merely the same rounded display.
    assert rec.monthly_cost == headline_newspend, (
        f"recommended row {rec.monthly_cost} != headline New-spend "
        f"{headline_newspend}"
    )
    # Spot-check the known-good DISPLAY figure (chatgpt-4o-latest demo, 30-day span);
    # renderers quantize to 4 dp, which is what the user sees ($331.0269).
    assert headline_newspend.quantize(Decimal("0.0001")) == Decimal("331.0269")


def test_pd_scenario_claude_haiku_shows_exactly_one_number() -> None:
    """frugon-eval-unrated-x1 (``considered``) shows ONE full-dataset New-spend.

    The unrated candidate is held out of the recommended route (Change 1) but is
    still surfaced in the block with its REAL full-dataset split New-spend
    ($286.5187) — the one number routing easy→frugon-eval-unrated-x1 would cost — so the
    user can see the potential.  Its block figure equals the figure the Change-1b
    "could save ~26.5%" caveat quotes.
    """
    result = _demo_result(["gpt-4o", "frugon-eval-unrated-x1"])
    haiku = _block(result)["frugon-eval-unrated-x1"]
    assert haiku.monthly_cost is not None
    haiku_display = haiku.monthly_cost.quantize(Decimal("0.0001"))
    # The full-dataset New-spend of routing easy→frugon-eval-unrated-x1.
    assert haiku_display == Decimal("286.5187")
    # The old dominant-only compute_split figure must NOT be what we show.
    assert haiku_display != Decimal("286.5000")


def test_pd_scenario_renders_consistently_on_terminal_and_html(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Both the terminal panel and the HTML report quote the reconciled figures."""
    result = _demo_result(["gpt-4o", "frugon-eval-unrated-x1"])

    render_terminal(result)
    out = " ".join(capsys.readouterr().out.split())
    # Headline routes to the rated pick gpt-4o at the reconciled New-spend.
    assert "gpt-4o" in out
    assert "331.0269" in out
    # frugon-eval-unrated-x1 appears once (considered), at its full-dataset figure.
    assert "frugon-eval-unrated-x1" in out
    assert "286.5187" in out

    html_path = tmp_path / "r.html"
    render_html(result, html_path)
    html = html_path.read_text(encoding="utf-8")
    assert "286.5187" in html
    assert "331.0269" in html


# ---------------------------------------------------------------------------
# Divergent synthetic case: full-swap-cheapest != split-cheapest.
# ---------------------------------------------------------------------------


def _divergent_records() -> list[dict[str, object]]:
    """A log engineered so full-swap-cheapest and split-cheapest candidates DIFFER.

    The dominant baseline (gpt-4-turbo) has two cohorts:

      * 40 EASY calls — input-heavy (large prompt, tiny completion), below the
        difficulty threshold, so they are the ONLY calls the split routes.
      * 20 HARD calls — output-heavy (large completion), above the threshold, so
        the split keeps them on the baseline but a FULL swap reprices them.

    Paired with two candidates (priced via :func:`_patch_divergent_prices`):

      * ``cand-cheapin``  — cheap INPUT, pricey output → wins the SPLIT, because
        the routed easy calls are input-heavy.
      * ``cand-cheapout`` — pricey input, cheap OUTPUT → wins the FULL SWAP,
        because the output-heavy hard calls (which only a full swap touches)
        dominate total cost.

    So the full-swap basis prefers ``cand-cheapout`` while the split basis prefers
    ``cand-cheapin`` — the exact divergence the old code resolved inconsistently
    (headline on one basis, block on the other).
    """
    records: list[dict[str, object]] = []
    for _ in range(40):  # easy + input-heavy
        records.append(
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "classify"}]},
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                },
                "usage": {"prompt_tokens": 50, "completion_tokens": 1},
            }
        )
    for _ in range(20):  # hard + heavily output-weighted
        records.append(
            {
                "model": "gpt-4-turbo",
                "request": {"messages": [{"role": "user", "content": "x " * 250}]},
                "response": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "y " * 6000}}
                    ]
                },
                "usage": {"prompt_tokens": 500, "completion_tokens": 12000},
            }
        )
    return records


def _patch_divergent_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub pricing so cand-cheapin wins the split and cand-cheapout the full swap.

    Both candidates are strictly cheaper than the gpt-4-turbo baseline on EVERY
    axis (so each genuinely beats the baseline), but their input/output shapes
    are mirror images, which is what splits the two ranking bases apart.
    """
    from decimal import Decimal

    from frugon import cost as cost_mod
    from frugon import routing as routing_mod
    from frugon.pricing import ModelPrice, get_model_price

    base = get_model_price("gpt-4-turbo")
    assert base is not None

    stubs = {
        # cheap input (1/100th baseline), pricey-ish output (still < baseline).
        "cand-cheapin": ModelPrice(
            model="cand-cheapin",
            input_cost_per_token=Decimal("0.0000001"),
            output_cost_per_token=Decimal("0.000009"),
            source="pricing.json",
            pricing_json_last_synced="2026-06-01",
        ),
        # pricey-ish input (still < baseline), cheap output (1/100th baseline).
        "cand-cheapout": ModelPrice(
            model="cand-cheapout",
            input_cost_per_token=Decimal("0.000009"),
            output_cost_per_token=Decimal("0.0000001"),
            source="pricing.json",
            pricing_json_last_synced="2026-06-01",
        ),
    }

    def _patched(model: str) -> ModelPrice | None:
        if model in stubs:
            return stubs[model]
        return get_model_price(model)

    # Both modules resolve prices via their own ``get_model_price`` reference.
    monkeypatch.setattr(cost_mod, "get_model_price", _patched)
    monkeypatch.setattr(routing_mod, "get_model_price", _patched)
    # cost.py's explicit-candidates loop imports get_model_price lazily as _gmp /
    # _gmp_cand from frugon.pricing, so patch the source too.
    import frugon.pricing as pricing_mod

    monkeypatch.setattr(pricing_mod, "get_model_price", _patched)


def test_divergent_case_headline_and_block_always_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """full-swap-cheapest != split-cheapest, yet headline == block recommended.

    With the stubbed prices ``cand-cheapout`` is the cheapest FULL SWAP while
    ``cand-cheapin`` is the cheapest SPLIT.  The old code would have named the
    full-swap winner in the headline and the split winner in the block; the fix
    makes BOTH the split winner, and every figure reconciles.
    """
    import json
    from decimal import Decimal

    from frugon.cost import analyze_logs, compute_call_cost, iter_records

    _patch_divergent_prices(monkeypatch)

    path = tmp_path / "divergent.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in _divergent_records()) + "\n",
        encoding="utf-8",
    )

    # Prove the divergence FIRST: the two ranking bases pick different winners.
    records, _ = iter_records(path)
    priced = [compute_call_cost(r) for r in records]
    full_swap = {}
    for cand in ("cand-cheapin", "cand-cheapout"):
        from frugon.pricing import get_model_price

        price = get_model_price(cand)
        assert price is not None
        full_swap[cand] = sum(
            (
                price.input_cost_per_token * Decimal(cc.record.prompt_tokens)
                + price.output_cost_per_token * Decimal(cc.record.completion_tokens)
                for cc in priced
            ),
            Decimal("0"),
        )
    full_swap_winner = min(full_swap, key=lambda m: full_swap[m])
    assert full_swap_winner == "cand-cheapout", (
        "fixture precondition: cand-cheapout must win the full swap"
    )

    result = analyze_logs(path, candidates=["cand-cheapin", "cand-cheapout"])
    assert result.split is not None

    # The headline routing target is the SPLIT winner, NOT the full-swap winner.
    headline_target = result.split.candidate_model
    assert headline_target == "cand-cheapin"
    assert headline_target != full_swap_winner  # the bases genuinely diverged
    assert result.candidate_model == headline_target

    # The block's recommended row is that same model — headline and block agree.
    block = _block(result)
    recommended = [m for m, p in block.items() if p.status == "recommended"]
    assert recommended == [headline_target], (
        "headline routing target and block recommended must always be the same "
        f"model; headline={headline_target!r}, block recommended={recommended!r}"
    )

    # The recommended row reconciles with the headline New-spend (observed basis —
    # the synthetic log carries no timestamps, so no monthly projection).
    _current, headline_newspend, projected = _split_current_and_blended(
        result, result.split
    )
    assert not projected
    rec = block[headline_target]
    assert rec.observed_cost is not None
    assert rec.observed_cost == headline_newspend

    # Each candidate's block number is its OWN full-dataset split New-spend — a
    # losing candidate is never shown a number it doesn't actually cost, and never
    # exceeds the baseline total.
    for proj in block.values():
        assert proj.observed_cost is not None
        assert proj.observed_cost <= result.total_cost
