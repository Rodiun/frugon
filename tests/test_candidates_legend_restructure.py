"""The default-pool "2+3 hybrid" caption restructure (PD-directed 2026-07-03).

Before this change the default-pool "Candidates considered" block carried a
plain "Candidates considered" title and a two-paragraph prose caption below
the table (the same shape the explicit ``--candidates`` path still uses,
pinned by ``test_candidate_caption.py``).  The restructure:

  * moves the pool/shown COUNTING into the header line itself —
    "Candidates considered · N in pool · top M shown" — so a reader sees the
    scope before a single row, instead of learning it from a trailing
    sentence;
  * replaces the two prose paragraphs with a 3-line bullet legend, one fact
    per line (what each row represents, the selection rule, the actionable
    follow-ups).

The explicit ``--candidates`` path is UNTOUCHED: plain header, the existing
``_candidate_caption`` prose, no cap line — this suite proves that isolation
explicitly, alongside the new default-pool behaviour.
"""

from __future__ import annotations

from pathlib import Path

import frugon
from frugon.cost import analyze_records, iter_records
from frugon.report import (
    _candidate_legend_lines,
    _candidates_considered_html,
    _candidates_considered_md_lines,
    _candidates_header_title,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
    render_terminal,
)

assert frugon.__file__ is not None
_SAMPLE = Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"


def _demo_result():
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(list(records), skipped_malformed=skipped, split_routing=True)


def _explicit_result():
    records, skipped = iter_records(_SAMPLE)
    return analyze_records(
        list(records),
        candidates=["gpt-4o", "gpt-4.1-mini", "claude-haiku-4-5"],
        skipped_malformed=skipped,
        split_routing=True,
    )


# ---------------------------------------------------------------------------
# _candidates_header_title
# ---------------------------------------------------------------------------


class TestCandidatesHeaderTitle:
    def test_default_pool_header_names_pool_and_shown_counts(self) -> None:
        result = _demo_result()
        title = _candidates_header_title(result)
        assert title == (
            f"Candidates considered · {result.candidate_pool_size} in pool · "
            f"top {len(result.candidate_projections)} shown"
        )

    def test_explicit_candidates_header_is_unchanged_plain_title(self) -> None:
        result = _explicit_result()
        assert _candidates_header_title(result) == "Candidates considered"

    def test_shown_count_reflects_fewer_than_five_rows_honestly(
        self, monkeypatch
    ) -> None:
        """If fewer than 5 candidates beat the baseline, the header says so —
        never a hardcoded 'top 5' regardless of the actual row count."""
        import frugon.cost as cost_mod

        # Shrink the pool so the recommended candidate is the only eligible one.
        monkeypatch.setattr(cost_mod, "_ROUTING_CANDIDATES", ["deepseek-v4-flash", "gpt-5.5"])
        records, skipped = iter_records(_SAMPLE)
        result = analyze_records(list(records), skipped_malformed=skipped, split_routing=True)
        title = _candidates_header_title(result)
        shown = len(result.candidate_projections)
        assert shown < 5
        assert f"top {shown} shown" in title
        assert "top 5 shown" not in title


# ---------------------------------------------------------------------------
# _candidate_legend_lines
# ---------------------------------------------------------------------------


class TestCandidateLegendLines:
    def test_legend_has_exactly_three_lines(self) -> None:
        result = _demo_result()
        assert result.split is not None
        lines = _candidate_legend_lines(result, result.split, has_judge_section=False)
        assert len(lines) == 3

    def test_first_line_names_the_actual_baseline_model(self) -> None:
        result = _demo_result()
        assert result.split is not None
        lines = _candidate_legend_lines(result, result.split, has_judge_section=False)
        assert result.split.baseline_model in lines[0]
        assert "quality-preserving split" in lines[0]

    def test_second_line_states_the_tie_break_rule(self) -> None:
        result = _demo_result()
        assert result.split is not None
        lines = _candidate_legend_lines(result, result.split, has_judge_section=False)
        assert "higher quality tier" in lines[1]
        assert "Biggest saving wins" in lines[1]

    def test_third_line_no_judge_offers_the_measure_command(self) -> None:
        result = _demo_result()
        assert result.split is not None
        lines = _candidate_legend_lines(result, result.split, has_judge_section=False)
        assert "--candidates" in lines[2]
        assert "--measure --judge" in lines[2]
        assert "below" not in lines[2]

    def test_third_line_with_judge_references_section_below(self) -> None:
        result = _demo_result()
        assert result.split is not None
        lines = _candidate_legend_lines(result, result.split, has_judge_section=True)
        assert "--candidates" in lines[2]
        assert "scored independently" in lines[2]
        assert "below" in lines[2]


# ---------------------------------------------------------------------------
# Rendering — default-pool surface shows the new header + legend
# ---------------------------------------------------------------------------


class TestDefaultPoolRendersNewStructure:
    def test_terminal_shows_header_and_legend(self, capsys) -> None:
        result = _demo_result()
        render_terminal(result)
        out = " ".join(capsys.readouterr().out.split())
        assert f"{result.candidate_pool_size} in pool" in out
        assert f"top {len(result.candidate_projections)} shown" in out
        assert "quality-preserving split" in out
        assert "Biggest saving wins" in out
        assert "Compare specific models with --candidates" in out
        # The old prose sentences are gone from the default-pool surface.
        assert "the biggest saving is the headline recommendation, and when" not in out
        assert "showing the recommended split and the" not in out

    def test_markdown_v1_shows_header_and_bullet_list(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r.md"
        render_markdown(result, out_path)
        md = out_path.read_text(encoding="utf-8")
        assert f"## Candidates considered · {result.candidate_pool_size} in pool" in md
        assert "- Each row is the same quality-preserving split" in md
        assert "- Biggest saving wins" in md
        assert "- Compare specific models with --candidates" in md

    def test_markdown_v2_shows_header_and_bullet_list(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r2.md"
        render_markdown_v2(result, out_path)
        md = out_path.read_text(encoding="utf-8")
        assert f"{result.candidate_pool_size} in pool" in md
        assert "- Biggest saving wins" in md

    def test_html_v1_shows_header_and_ul_legend(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r.html"
        render_html(result, out_path)
        html = out_path.read_text(encoding="utf-8")
        assert f"{result.candidate_pool_size} in pool" in html
        assert '<ul class="caption candidates-caption candidates-legend">' in html
        assert "<li>Biggest saving wins" in html

    def test_html_v2_shows_header_and_ul_legend(self, tmp_path: Path) -> None:
        result = _demo_result()
        out_path = tmp_path / "r2.html"
        render_html_v2(result, out_path)
        html = out_path.read_text(encoding="utf-8")
        assert f"{result.candidate_pool_size} in pool" in html
        assert "candidates-legend" in html

    def test_md_lines_helper_matches_terminal_structure(self) -> None:
        result = _demo_result()
        lines = _candidates_considered_md_lines(result)
        joined = "\n".join(lines)
        assert f"{result.candidate_pool_size} in pool" in joined
        assert "- Biggest saving wins" in joined

    def test_html_helper_matches_terminal_structure(self) -> None:
        result = _demo_result()
        html = _candidates_considered_html(result, lambda s: s)
        assert "candidates-legend" in html
        assert "Biggest saving wins" in html


# ---------------------------------------------------------------------------
# Explicit --candidates path — untouched, proven on every surface
# ---------------------------------------------------------------------------


class TestExplicitCandidatesPathIsolation:
    def test_terminal_keeps_old_prose_no_new_header(self, capsys) -> None:
        result = _explicit_result()
        render_terminal(result)
        out = " ".join(capsys.readouterr().out.split())
        assert "in pool" not in out
        # The header line right after "Candidates considered" must be plain —
        # no "· N in pool · top M shown" suffix.  Checking the narrow window
        # right after the title (rather than a bare "top"/"shown" substring
        # anywhere in the output) avoids false positives from the unrelated,
        # unchanged caption prose ("...precision shown...").
        header_tail = out.split("Candidates considered", 1)[1][:10]
        assert "top" not in header_tail
        assert "the biggest saving is the headline recommendation" in out

    def test_markdown_keeps_plain_heading(self, tmp_path: Path) -> None:
        result = _explicit_result()
        out_path = tmp_path / "r.md"
        render_markdown(result, out_path)
        md = out_path.read_text(encoding="utf-8")
        assert "## Candidates considered\n" in md
        assert "in pool" not in md

    def test_html_keeps_p_tag_caption_not_ul(self, tmp_path: Path) -> None:
        """The explicit-candidates path never renders the <ul class="...
        candidates-legend"> markup — the shared stylesheet defines the
        ``.candidates-legend`` CSS RULE on every HTML report regardless of
        path (so the class exists whenever the default-pool surface needs
        it), but no element should actually CARRY that class here."""
        result = _explicit_result()
        out_path = tmp_path / "r.html"
        render_html(result, out_path)
        html = out_path.read_text(encoding="utf-8")
        assert 'class="caption candidates-caption candidates-legend"' not in html
        assert '<p class="caption candidates-caption">' in html
