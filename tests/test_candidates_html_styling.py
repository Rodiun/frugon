"""Tests for the "Candidates considered" HTML table styling (Fix 2).

The bug this guards: the candidates table rendered with no column separation
(header "Monthly costVs. baseline", cells "$533.9260 / mo31.3% lower" running
together) because, on the v2 surface, it carried only a ``.tbl-candidates``
class that had NO matching CSS — so it fell back to browser-default table
styling (zero padding, no right-alignment, unstyled badges).

The fix brings it to PARITY with the cost-by-model (``.tbl``) and routing-plan
(``.tbl-plan``) tables on BOTH surfaces:

  * the table now carries the shared ``.tbl`` class (so it inherits the same
    cell padding + row separators as cost-by-model);
  * numeric columns are right-aligned + nowrap + tabular-nums;
  * the model name keeps the surface's cyan;
  * status badges get the routing-plan pill treatment, recoloured per status
    (recommended = cyan, more-expensive = amber, the rest muted);
  * the caption uses the report's dim caption style.

All tests render a real report from a synthetic multi-candidate log, fully
offline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from frugon.cost import AnalysisResult, analyze_logs
from frugon.report import render_html, render_html_v2

sys.path.insert(0, str(Path(__file__).parent))
from conftest import install_unrated_sentinel


@pytest.fixture(autouse=True)
def _sentinel_pricing(monkeypatch, tmp_path):
    install_unrated_sentinel(monkeypatch, tmp_path)
    yield
    import frugon.pricing as _p

    _p.clear_pricing_cache()


def _multi_candidate_result(tmp_path: Path) -> AnalysisResult:
    records = [
        {
            "model": "gpt-4-turbo",
            "request": {
                "messages": [{"role": "user", "content": "classify this ticket"}]
            },
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "billing"}}]
            },
            "usage": {"prompt_tokens": 200, "completion_tokens": 5},
        }
        for _ in range(100)
    ]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    # gpt-4o-mini -> recommended (cheapest beat), frugon-eval-unrated-x1 -> considered,
    # imaginary-9999 -> unpriced.  Three rows exercise every badge variant except
    # more_expensive (covered by the dedicated test below).
    result = analyze_logs(
        log, candidates=["gpt-4o-mini", "frugon-eval-unrated-x1", "imaginary-9999"]
    )
    assert len(result.candidate_projections) >= 3
    return result


def _more_expensive_result(tmp_path: Path) -> AnalysisResult:
    """A run where every candidate loses to the baseline -> more_expensive."""
    records = [
        {
            "model": "gpt-4o-mini",
            "request": {"messages": [{"role": "user", "content": "x"}]},
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "y"}}]
            },
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        }
        for _ in range(50)
    ]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = analyze_logs(log, candidates=["gpt-4o", "gpt-4-turbo"])
    statuses = {p.status for p in result.candidate_projections}
    assert statuses == {"more_expensive"}
    return result


def _candidates_table(html: str) -> str:
    m = re.search(
        r'<div class="candidates-considered">.*?</table>', html, re.DOTALL
    )
    assert m, "candidates table not found in rendered HTML"
    return m.group(0)


# ---------------------------------------------------------------------------
# Table markup — shared inner table (v1 + v2)
# ---------------------------------------------------------------------------


def test_html_v1_table_carries_shared_tbl_class(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    render_html(_multi_candidate_result(tmp_path), out)
    table = _candidates_table(out.read_text(encoding="utf-8"))
    # Shares the cost-by-model table's class so it inherits the same padding +
    # row separators (parity), not a standalone bare table.
    assert 'class="tbl tbl-candidates"' in table


def test_html_v2_table_carries_shared_tbl_class(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    render_html_v2(_multi_candidate_result(tmp_path), out)
    table = _candidates_table(out.read_text(encoding="utf-8"))
    assert 'class="tbl tbl-candidates"' in table


def test_numeric_columns_marked_for_right_alignment(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    render_html_v2(_multi_candidate_result(tmp_path), out)
    table = _candidates_table(out.read_text(encoding="utf-8"))
    # Both numeric headers AND cells carry .num so the CSS can right-align them
    # (the jumble was numeric columns rendering left-aligned, butted together).
    assert '<th class="num">Monthly cost</th>' in table
    assert '<th class="num">Vs. baseline</th>' in table
    assert table.count('<td class="num">') == 6  # 3 rows x 2 numeric cols


def test_status_badges_use_semantic_classes(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    render_html_v2(_multi_candidate_result(tmp_path), out)
    table = _candidates_table(out.read_text(encoding="utf-8"))
    assert '<span class="badge badge-recommended">recommended</span>' in table
    assert '<span class="badge badge-considered">considered</span>' in table
    assert '<span class="badge badge-unpriced">unpriced</span>' in table
    # No inline opacity hacks left behind.
    assert 'style="opacity:.7"' not in table


def test_more_expensive_badge_class(tmp_path: Path) -> None:
    # A no-recommendation (wholesale) run still surfaces every candidate it
    # considered, each tagged more_expensive — rendered via the v1 candidates
    # card (the surface that carries the wholesale candidates block).
    out = tmp_path / "r.html"
    render_html(_more_expensive_result(tmp_path), out)
    table = _candidates_table(out.read_text(encoding="utf-8"))
    assert "badge-more-expensive" in table


# ---------------------------------------------------------------------------
# CSS presence — both style blocks carry the candidates rules
# ---------------------------------------------------------------------------


def test_v1_style_block_defines_candidates_table_css(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    render_html(_multi_candidate_result(tmp_path), out)
    css = out.read_text(encoding="utf-8")
    # Right-alignment rule, per-status badge colours, and the dim caption rule
    # are all present in the v1 <style> block (the table is no longer unstyled).
    assert ".tbl-candidates th.num,.tbl-candidates td.num{" in css
    assert "text-align:right" in css
    assert ".tbl-candidates .badge-recommended{" in css
    assert ".tbl-candidates .badge-more-expensive{" in css
    assert ".candidates-caption{" in css


def test_v2_style_block_defines_candidates_table_css(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    render_html_v2(_multi_candidate_result(tmp_path), out)
    css = out.read_text(encoding="utf-8")
    assert ".tbl-candidates th.num,.tbl-candidates td.num{" in css
    assert ".tbl-candidates .badge-recommended{" in css
    assert ".tbl-candidates .badge-more-expensive{" in css
    assert ".candidates-caption{" in css
    # The shared .tbl base rule (the cost-by-model aesthetic the table inherits)
    # exists on this surface — that is what supplies the cell padding + hairlines.
    assert ".tbl td{" in css


# ---------------------------------------------------------------------------
# Self-containment is preserved (no external assets introduced by the CSS).
# ---------------------------------------------------------------------------


def test_styling_introduces_no_external_assets(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    render_html_v2(_multi_candidate_result(tmp_path), out)
    html = out.read_text(encoding="utf-8")
    assert "https://frugon.rodiun.io" in html  # the one allowed deliberate link
    # No asset-loading external refs (the SVG `xmlns="http://www.w3.org/..."`
    # namespace is a non-asset identifier and is intentionally allowed).
    assert not re.search(r'<link\b[^>]*href\s*=\s*["\']https?://', html)
    assert not re.search(r'<script\b[^>]*src\s*=\s*["\']https?://', html)
    assert not re.search(r'<img\b[^>]*src\s*=\s*["\']https?://', html)
