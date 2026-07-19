"""Tests for the three-state no-candidate distinction on md/html report surfaces.

The terminal surface is pinned in ``tests/test_report_wholesale.py``
(``TestWholesaleNoPriceableCandidates`` / ``TestWholesaleNoCandidate``); this
file pins report parity across Markdown v1/v2 and HTML v1/v2 for the same
three states:

  (a) evaluated, none cheaper       -- existing wording, unchanged.
  (b) no priceable candidate at all -- new honest state: the cost race never
      ran because nothing in the pool had a known list price (e.g. a local
      model passed via ``--candidates``).
  (c) a mixed pool                  -- existing "unpriced" tag behaviour,
      unaffected by the new state (b) wording.

All fixtures are directly-constructed ``AnalysisResult`` objects (no network,
no LLM) so these tests pin the report-layer behaviour independent of the
cost-layer flag computation, which is covered in
``tests/test_cost.py::TestNoPriceableCandidates``.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from frugon.cost import AnalysisResult, CandidateProjection
from frugon.measure import MeasureResult
from frugon.report import (
    NO_PRICEABLE_CANDIDATES_NOTE,
    NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
)


def _result_no_candidate(**kwargs: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "total_calls": 100,
        "priced_calls": 100,
        "unpriced_calls": 0,
        "total_cost": Decimal("12.00"),
        "cost_by_model": {"gpt-4o-mini": Decimal("12.00")},
        "calls_by_model": {"gpt-4o-mini": 100},
        "projected_cost": Decimal("0"),
        "candidate_model": None,
        "split": None,
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _result_no_priceable_candidates(**kwargs: Any) -> AnalysisResult:
    defaults: dict[str, Any] = {
        "no_priceable_candidates": True,
        "unpriced_candidate_names": ["ollama/llama3.2:1b"],
    }
    defaults.update(kwargs)
    return _result_no_candidate(**defaults)


def _tier0_measure_result() -> MeasureResult:
    """Minimal Tier-0 (--measure, no --judge) result -- enough for
    _quality_section_md/_quality_section_html to render a non-empty section,
    which is all the quality-comparison-clause condition checks for."""
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o-mini",
        candidates=["ollama/llama3.2:1b"],
        comparisons=[],
    )


class TestMarkdownV1NoPriceableCandidates:
    def test_state_b_names_candidate_and_notes_no_fabrication(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.md"
        render_markdown(_result_no_priceable_candidates(), out)
        md = out.read_text(encoding="utf-8")
        assert "ollama/llama3.2:1b" in md
        assert NO_PRICEABLE_CANDIDATES_NOTE in md
        assert "Recommended swap" not in md

    def test_state_b_quality_clause_absent_without_measure(self, tmp_path: Path) -> None:
        """Offline (no --measure/--judge, single candidate): no quality
        comparison renders anywhere in this report, so the clause claiming one
        covers the candidate must not appear (the live bug: --candidates
        "ollama/llama3.2:1b" with no --measure)."""
        out = tmp_path / "report.md"
        render_markdown(_result_no_priceable_candidates(), out)
        md = out.read_text(encoding="utf-8")
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE not in md

    def test_state_b_quality_clause_present_with_measure(self, tmp_path: Path) -> None:
        """--measure ran: a Quality measurement section DOES render below, so
        the clause is honest here."""
        out = tmp_path / "report.md"
        render_markdown(
            _result_no_priceable_candidates(), out, measure_result=_tier0_measure_result()
        )
        md = out.read_text(encoding="utf-8")
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE in md

    def test_state_a_is_unaffected(self, tmp_path: Path) -> None:
        """Regression: the existing evaluated-none-cheaper path stays silent
        about candidates (no swap bullets, no state (b) wording either)."""
        out = tmp_path / "report.md"
        render_markdown(_result_no_candidate(), out)
        md = out.read_text(encoding="utf-8")
        assert "No list price for" not in md
        assert NO_PRICEABLE_CANDIDATES_NOTE not in md
        assert "Recommended swap" not in md


class TestMarkdownV2NoPriceableCandidates:
    def test_state_b_names_candidate_and_notes_no_fabrication(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.md"
        render_markdown_v2(_result_no_priceable_candidates(), out)
        md = out.read_text(encoding="utf-8")
        assert "No list price for ollama/llama3.2:1b" in md
        assert NO_PRICEABLE_CANDIDATES_NOTE in md
        assert "No cheaper swap clears the quality bar" not in md

    def test_state_b_quality_clause_absent_without_measure(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        render_markdown_v2(_result_no_priceable_candidates(), out)
        md = out.read_text(encoding="utf-8")
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE not in md

    def test_state_b_quality_clause_present_with_measure(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        render_markdown_v2(
            _result_no_priceable_candidates(), out, measure_result=_tier0_measure_result()
        )
        md = out.read_text(encoding="utf-8")
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE in md

    def test_state_a_keeps_existing_wording(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        render_markdown_v2(_result_no_candidate(), out)
        md = out.read_text(encoding="utf-8")
        assert "No cheaper swap clears the quality bar." in md
        assert "No list price for" not in md


class TestHtmlV1NoPriceableCandidates:
    def test_state_b_names_candidate_and_notes_no_fabrication(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html(_result_no_priceable_candidates(), out)
        html = out.read_text(encoding="utf-8")
        assert "No list price for ollama/llama3.2:1b" in html
        # html.escape turns the apostrophe in "won't" into &#x27;, so match the
        # unambiguous, apostrophe-free portion of the shared note constant.
        assert "Local models cost $0 in API spend" in html
        assert "fabricate a price" in html
        assert "No cheaper candidate found within quality constraints" not in html

    def test_state_b_quality_clause_absent_without_measure(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        render_html(_result_no_priceable_candidates(), out)
        html = out.read_text(encoding="utf-8")
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE not in html

    def test_state_b_quality_clause_present_with_measure(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        render_html(
            _result_no_priceable_candidates(), out, measure_result=_tier0_measure_result()
        )
        html = out.read_text(encoding="utf-8")
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE in html

    def test_state_a_keeps_existing_wording(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        render_html(_result_no_candidate(), out)
        html = out.read_text(encoding="utf-8")
        assert "No cheaper candidate found within quality constraints." in html
        assert "No list price for" not in html


class TestHtmlV2NoPriceableCandidates:
    def test_state_b_names_candidate_and_notes_no_fabrication(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.html"
        render_html_v2(_result_no_priceable_candidates(), out)
        html = out.read_text(encoding="utf-8")
        assert "No list price for ollama/llama3.2:1b" in html
        assert "Local models cost $0 in API spend" in html
        assert "fabricate a price" in html
        assert "No cheaper swap clears the quality bar" not in html

    def test_state_b_quality_clause_absent_without_measure(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        render_html_v2(_result_no_priceable_candidates(), out)
        html = out.read_text(encoding="utf-8")
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE not in html

    def test_state_b_quality_clause_present_with_measure(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        render_html_v2(
            _result_no_priceable_candidates(), out, measure_result=_tier0_measure_result()
        )
        html = out.read_text(encoding="utf-8")
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE in html

    def test_state_a_keeps_existing_wording(self, tmp_path: Path) -> None:
        out = tmp_path / "report.html"
        render_html_v2(_result_no_candidate(), out)
        html = out.read_text(encoding="utf-8")
        assert "No cheaper swap clears the quality bar." in html
        assert "No list price for" not in html


class TestUnpricedCandidatesLabelHelper:
    """Multiple unpriced names join with ', ' -- pinned once here so the
    per-surface tests above don't each need a multi-name fixture too."""

    def test_multiple_names_join_with_comma(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        result = _result_no_priceable_candidates(
            unpriced_candidate_names=["ollama/llama3.2:1b", "ollama/mistral"]
        )
        render_markdown(result, out)
        md = out.read_text(encoding="utf-8")
        assert "ollama/llama3.2:1b, ollama/mistral" in md


class TestQualityClauseOfflineMultiCandidate:
    """The other way a quality comparison can render without --measure: the
    "Candidates considered" table's Quality tier column, which appears
    whenever more than one candidate was passed -- no measure run needed."""

    def _multi_candidate_result(self) -> AnalysisResult:
        return _result_no_priceable_candidates(
            unpriced_candidate_names=["ollama/llama3.2:1b", "ollama/mistral"],
            candidate_projections=[
                CandidateProjection(
                    model="ollama/llama3.2:1b", status="unpriced", tier_label="Efficient"
                ),
                CandidateProjection(
                    model="ollama/mistral", status="unpriced", tier_label="Capable"
                ),
            ],
        )

    def test_quality_clause_present_with_offline_candidates_table(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.md"
        render_markdown(self._multi_candidate_result(), out)
        md = out.read_text(encoding="utf-8")
        assert "Quality tier" in md
        assert NO_PRICEABLE_CANDIDATES_QUALITY_CLAUSE in md
