"""Regression guards for four confirmed dangling-reference defects.

Each of the four ``test_C{n}_*`` classes pins one of the fixes from the
adversarial audit so the wording cannot silently regress on the surfaces it
was wrong on (and stays correct on the surfaces it was already right on).

  * C-1 — stray ``†`` footnote dagger with no body referent in the split
    Markdown callout (``_render_markdown_split``). The v2 wholesale callout
    DOES anchor a ``†`` to body markers in the after-swap table; the split
    path does not, so the dagger is dropped only there.
  * C-2 — Tier-0 nudge wording is now position-independent.  The string was
    "Raw samples shown above ..." which contradicted the layout on the 3
    surfaces where the nudge renders above (not below) the raw samples.
    "Raw samples are shown in the quality measurement section ..." holds on
    every surface.
  * C-3 — the ``_unrated_split_*_measured`` helpers no longer claim
    "measured below"; they point at the named quality section instead.
    Same shape as C-2 (position-independent), applied to a symmetric
    inversion on the v2 surfaces where the unrated family renders BELOW the
    quality_section.
  * C-4 — the v2-MD wholesale hero subtitle now routes through
    ``_quality_caveat_text`` like every other site, so a ``--wholesale
    --measure`` run no longer contradicts itself by telling the user to
    "run --measure to sample real outputs" in the hero while the report
    below already carries the sampled outputs.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from frugon.cost import AnalysisResult, LogRecord
from frugon.measure import Comparison, MeasureResult, SampledOutput, Tier1Tally
from frugon.report import (
    _render_markdown_split,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
)
from frugon.routing import SplitRouting

# ---------------------------------------------------------------------------
# Shared fixtures — built directly so the suite runs fully offline.
# ---------------------------------------------------------------------------


def _split() -> SplitRouting:
    return SplitRouting(
        baseline_model="gpt-4-turbo",
        candidate_model="gpt-4o-mini",
        routed_count=24,
        kept_count=3,
        routed_cost=Decimal("12.3400"),
        kept_cost=Decimal("489.2802"),
        baseline_cost=Decimal("750.0000"),
        blended_cost=Decimal("501.6202"),
        easy_threshold=Decimal("0.35"),
        monthly_baseline=Decimal("750.0000"),
        monthly_blended=Decimal("501.6202"),
    )


def _result(*, with_split: bool = True) -> AnalysisResult:
    return AnalysisResult(
        total_calls=37,
        priced_calls=37,
        unpriced_calls=0,
        total_cost=Decimal("496.7250"),
        cost_by_model={
            "gpt-4-turbo": Decimal("489.2802"),
            "gpt-4o-mini": Decimal("7.4448"),
        },
        calls_by_model={"gpt-4-turbo": 27, "gpt-4o-mini": 10},
        projected_cost=Decimal("330.0000"),
        candidate_model="gpt-4o-mini",
        observed_span_days=7.0,
        pricing_json_last_synced="2026-06-01",
        quality_json_last_synced="2026-05-15",
        split=_split() if with_split else None,
        monthly_cost=Decimal("750.0000"),
        monthly_projected=Decimal("501.6202"),
    )


def _record(prompt: str) -> LogRecord:
    return LogRecord(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _tier0() -> MeasureResult:
    """``--measure`` without ``--judge`` — two raw samples, no verdict."""
    comparisons = [
        Comparison(
            record=_record("Name three primary colours."),
            current_output=SampledOutput("gpt-4-turbo", "Red, green, blue."),
            candidate_outputs=[SampledOutput("gpt-4o-mini", "Red, blue, yellow.")],
        ),
        Comparison(
            record=_record("Capital of France?"),
            current_output=SampledOutput("gpt-4-turbo", "Paris."),
            candidate_outputs=[SampledOutput("gpt-4o-mini", "Paris, France.")],
        ),
    ]
    return MeasureResult(
        samples_requested=2,
        samples_taken=2,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=None,
    )


def _tier1() -> MeasureResult:
    """``--measure --judge`` — five-sample Tier-1 with a clear verdict."""
    verdicts = ["win", "win", "win", "tie", "win"]
    comparisons = [
        Comparison(
            record=_record(f"prompt {i}"),
            current_output=SampledOutput("gpt-4-turbo", "base"),
            candidate_outputs=[SampledOutput("gpt-4o-mini", f"cand {i}")],
            verdicts=[verdicts[i]],
        )
        for i in range(5)
    ]
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4-turbo",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=4, losses=0, ties=1)],
    )


# ---------------------------------------------------------------------------
# C-1 — stray ``†`` dagger in the split-Markdown callout.
# ---------------------------------------------------------------------------


class TestC1NoOrphanDaggerInSplitMd:
    """The split-Markdown callout never carries a ``†`` glyph (no body anchor).

    Covers ``_render_markdown_split`` which is the shared writer for both v1
    and v2 split-Markdown reports, with measure_result both present and
    absent.  The v2 wholesale callout at the end of ``render_markdown_v2``
    KEEPS its dagger — that one DOES anchor to the ``†`` markers in the
    after-swap table cells — and a separate guard pins it.
    """

    @pytest.mark.parametrize(
        ("mr", "expect_callout"),
        [
            (None, True),  # no measure: SPLIT_CAVEAT caveat applies
            (_tier0(), True),  # Tier-0: caveat softened, callout still present
            (_tier1(), False),  # Tier-1: caveat suppressed entirely
        ],
        ids=["no_measure", "tier0", "tier1"],
    )
    def test_split_md_callout_has_no_dagger(
        self,
        mr: MeasureResult | None,
        expect_callout: bool,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "split.md"
        _render_markdown_split(_result(), _result().split, out, measure_result=mr)  # type: ignore[arg-type]
        md = out.read_text(encoding="utf-8")

        assert (
            "†" not in md
        ), "split-MD path emits no `†` body marker for the callout to anchor to"
        assert "**† Before you switch:**" not in md
        if expect_callout:
            assert "**Before you switch:**" in md

    def test_v2_wholesale_md_keeps_dagger(self, tmp_path: Path) -> None:
        """Parity guard: the v2 wholesale callout DOES anchor to body daggers.

        ``render_markdown_v2`` emits ``†`` markers in the after-swap table
        cells; the closing callout's dagger ties to those markers.  Removing
        that dagger would orphan the table marker — the inverse of C-1.
        """
        out = tmp_path / "v2.md"
        render_markdown_v2(_result(with_split=False), out)
        md = out.read_text(encoding="utf-8")
        assert "> **† Before you switch:**" in md
        # And the body marker the dagger ties to.
        assert "†" in md.split("> **† Before you switch:**")[0]


# ---------------------------------------------------------------------------
# C-2 — Tier-0 nudge is now position-independent.
# ---------------------------------------------------------------------------


class TestC2Tier0NudgePositionIndependent:
    """The Tier-0 ``--judge`` nudge no longer claims "shown above".

    The string was inverted on MD-v1 wholesale, HTML v1 split, and HTML v1
    wholesale — three surfaces where the nudge renders ABOVE the raw
    samples it points at.  Pointing at the named section neutralises every
    direction.
    """

    RENDERERS = {
        "md_v1_wholesale": render_markdown,
        "md_v2_wholesale": render_markdown_v2,
        "html_v1": render_html,
        "html_v2": render_html_v2,
    }

    @pytest.mark.parametrize("variant", list(RENDERERS))
    def test_tier0_nudge_no_shown_above(self, variant: str, tmp_path: Path) -> None:
        out = tmp_path / "r"
        self.RENDERERS[variant](_result(with_split=False), out, measure_result=_tier0())
        text = out.read_text(encoding="utf-8")

        assert "shown above" not in text
        # The reframed wording — present in every variant when Tier-0 ran.
        # The HTML renderers wrap ``--judge`` in a ``<code>`` element.
        if variant.startswith("html"):
            assert "run <code>--judge</code> for a scored verdict" in text
            assert "in the quality measurement section" in text
        else:
            assert (
                "Raw samples are shown in the quality measurement section — "
                "run --judge for a scored verdict."
            ) in text

    def test_tier0_nudge_in_md_split(self, tmp_path: Path) -> None:
        """Split path was already correct (nudge renders below the samples),
        and the reframed wording stays correct there."""
        out = tmp_path / "split.md"
        _render_markdown_split(
            _result(), _result().split, out, measure_result=_tier0()  # type: ignore[arg-type]
        )
        md = out.read_text(encoding="utf-8")
        assert "shown above" not in md
        assert "in the quality measurement section" in md


# ---------------------------------------------------------------------------
# C-4 — v2-MD wholesale hero subtitle is routed through the measure helper.
# ---------------------------------------------------------------------------


class TestC4V2MdHeroSubtitleMeasureAware:
    """The v2-MD wholesale hero subtitle no longer hard-codes the static
    ``QUALITY_CAVEAT``.  It threads ``_quality_caveat_text`` so a
    ``--wholesale --measure --report out.md`` (default v2 style) run never
    tells the user to "run --measure" after --measure has already run, and a
    Tier-1 run drops the contradictory "Quality is not verified" claim."""

    def test_no_measure_hero_byte_identical(self, tmp_path: Path) -> None:
        """Without ``--measure`` the hero subtitle is unchanged — the static
        caveat is the canonical wording."""
        out = tmp_path / "r.md"
        render_markdown_v2(_result(with_split=False), out)
        md = out.read_text(encoding="utf-8")
        # The hero block is "## Bottom line" through the next blank-line tail.
        bottom = md.split("## Bottom line", 1)[1].split("## What we found", 1)[0]
        assert "run --measure to sample real outputs" in bottom
        assert "Quality is not verified" in bottom

    def test_tier0_hero_softens_caveat(self, tmp_path: Path) -> None:
        """With ``--measure`` (no judge) the hero carries the softened
        ``--judge`` nudge — never the contradictory "run --measure" claim."""
        out = tmp_path / "r.md"
        render_markdown_v2(
            _result(with_split=False), out, measure_result=_tier0()
        )
        md = out.read_text(encoding="utf-8")
        bottom = md.split("## Bottom line", 1)[1].split("## What we found", 1)[0]

        assert "run --measure to sample real outputs" not in bottom
        assert "Quality is not verified" not in bottom
        # The reframed (C-2) Tier-0 nudge — position-independent.
        assert "in the quality measurement section" in bottom
        assert "run --judge for a scored verdict" in bottom

    def test_tier1_hero_bare_projection(self, tmp_path: Path) -> None:
        """With ``--judge`` the hero is the bare projection label — no
        contradictory "Quality is not verified" claim in the subtitle (the
        scored verdict rides the quality section below)."""
        out = tmp_path / "r.md"
        render_markdown_v2(
            _result(with_split=False), out, measure_result=_tier1()
        )
        md = out.read_text(encoding="utf-8")
        bottom = md.split("## Bottom line", 1)[1].split("## What we found", 1)[0]

        assert "Quality is not verified" not in bottom
        assert "run --measure to sample real outputs" not in bottom
        assert "run --judge for a scored verdict" not in bottom


# ---------------------------------------------------------------------------
# C-5 — v2 HTML hero dagger must not dangle on the Tier-1 path.
# ---------------------------------------------------------------------------


class TestC5V2HtmlHeroDaggerHasReferent:
    """The v2 HTML after-swap dagger (and its "see note below" pointer) only
    render when the footer fineprint footnote they reference is present.

    The footnote is gated on ``_quality_caveat_text(...) is not None`` — which
    is ``None`` on the Tier-1 (``--judge``) path, where the report carries a
    scored verdict instead of a "run --measure" caveat.  On that path the
    dagger marker and the "see note below" pointer must BOTH be suppressed so
    the marker can never dangle without a referent.  Covers both v2 hero
    builders: the per-call split body and the wholesale hero.
    """

    @pytest.mark.parametrize("with_split", [True, False], ids=["split", "wholesale"])
    def test_tier1_hero_has_no_dangling_dagger(
        self, with_split: bool, tmp_path: Path
    ) -> None:
        """Tier-1 v2 HTML: no ``class="dagger"`` marker and no "see note below"
        pointer in the hero (the footnote they reference is suppressed)."""
        out = tmp_path / "r.html"
        render_html_v2(
            _result(with_split=with_split), out, measure_result=_tier1()
        )
        html = out.read_text(encoding="utf-8")

        # The footnote (the .fineprint dagger anchor) is suppressed on Tier-1.
        assert 'class="fineprint"' not in html
        # Therefore no marker may point at it, and the pointer is downgraded.
        assert 'class="dagger"' not in html
        assert "see note below" not in html
        # The honest estimate label survives without the dangling pointer.
        assert "List-price estimate." in html

    @pytest.mark.parametrize("with_split", [True, False], ids=["split", "wholesale"])
    def test_tier0_hero_keeps_dagger_with_referent(
        self, with_split: bool, tmp_path: Path
    ) -> None:
        """Tier-0 v2 HTML: the dagger marker and "see note below" pointer DO
        render, because the .fineprint footnote they anchor to is present — the
        marker has a referent."""
        out = tmp_path / "r.html"
        render_html_v2(
            _result(with_split=with_split), out, measure_result=_tier0()
        )
        html = out.read_text(encoding="utf-8")

        # The footnote renders on Tier-0, so the marker has a valid anchor.
        assert 'class="fineprint"' in html
        assert 'class="dagger"' in html
        assert "see note below" in html
