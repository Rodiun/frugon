"""Tests for the report-side Quality-measurement section (--measure / --judge).

A written report is the FULL view (there is no --verbose for a file), so when a
--measure / --judge run produced a MeasureResult the four report renderers
(render_markdown / render_markdown_v2 / render_html / render_html_v2) append a
"Quality measurement" section mirroring render_quality_terminal's content in the
report's own design language:

  * Tier-1 (--judge): the win/loss/tie tally table, the verdict synthesis line
    (sharing _classify_verdict with the terminal, so report and terminal NEVER
    disagree), and the per-prompt detail with [WIN]/[TIE]/[LOSS] labels.
  * Tier-0 (--measure, no --judge): the per-prompt side-by-side outputs + the
    "run --judge for a scored verdict" framing.

Two invariants are load-bearing and tested here:

  1. measure_result=None → the report is BYTE-IDENTICAL to a report rendered
     before the parameter existed (existing callers/tests unaffected).
  2. The verdict TEXT in the report reconciles with the terminal for the same
     tally (confirmed / borderline / not-confirmed / not-verified) — the
     classifier is shared, so the two surfaces can never drift.

All tests run fully offline — the MeasureResult is constructed directly, no
provider call is made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from frugon.cost import LogRecord
from frugon.measure import (
    Comparison,
    MeasureResult,
    SampledOutput,
    Tier1Tally,
)
from frugon.report import (
    _quality_section_html,
    _quality_section_md,
    _render_tier1_synthesis,
    render_html,
    render_html_v2,
    render_markdown,
    render_markdown_v2,
)

# ---------------------------------------------------------------------------
# Fixtures — construct MeasureResults directly (no provider call)
# ---------------------------------------------------------------------------


def _record(prompt: str, system: str | None = None) -> LogRecord:
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return LogRecord(
        model="gpt-4o",
        messages=messages,
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _tier1_mixed_result() -> MeasureResult:
    """Tier-1 with a borderline tally (0W / 2L / 3T) and per-prompt verdicts.

    The verdicts carry at least one LOSS so the per-prompt detail exercises the
    [LOSS] label path (the shareable proof the section exists for).
    """
    verdicts = ["tie", "tie", "loss", "tie", "loss"]
    prompts = [
        ("Summarise the French Revolution in one line.", "You are terse."),
        ("What is 17 * 23?", None),
        ("Explain recursion to a five-year-old.", "Answer as a pirate."),
        ("Translate 'good morning' to Japanese.", None),
        ("Write a haiku about databases.", None),
    ]
    comparisons = [
        Comparison(
            record=_record(prompt, system),
            current_output=SampledOutput(model="gpt-4o", content=f"baseline {i}"),
            candidate_outputs=[
                SampledOutput(model="gpt-4o-mini", content=f"candidate {i}")
            ],
            verdicts=[verdicts[i]],
        )
        for i, (prompt, system) in enumerate(prompts)
    ]
    return MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=[Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=2, ties=3)],
    )


def _tier0_result() -> MeasureResult:
    """Tier-0 (no tallies): raw side-by-side outputs, no per-prompt verdicts."""
    comparisons = [
        Comparison(
            record=_record("Name three primary colours.", "You are concise."),
            current_output=SampledOutput(model="gpt-4o", content="Red, green, blue."),
            candidate_outputs=[
                SampledOutput(model="gpt-4o-mini", content="Red, blue, yellow.")
            ],
        ),
        Comparison(
            record=_record("Capital of France?"),
            current_output=SampledOutput(model="gpt-4o", content="Paris."),
            candidate_outputs=[
                SampledOutput(model="gpt-4o-mini", content="Paris, France.")
            ],
        ),
    ]
    return MeasureResult(
        samples_requested=2,
        samples_taken=2,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=comparisons,
        tier1_tallies=None,
    )


# ---------------------------------------------------------------------------
# Tier-1 — Markdown section
# ---------------------------------------------------------------------------


def test_quality_section_md_tier1_emits_tally_verdict_and_loss_label() -> None:
    # Arrange
    result = _tier1_mixed_result()

    # Act
    md = "\n".join(_quality_section_md(result))

    # Assert — header, tally table, borderline synthesis, per-prompt LOSS label.
    assert "## Quality measurement" in md
    assert "| Candidate | Win | Loss | Tie | Error | Summary |" in md
    assert "| `gpt-4o-mini` | 0 | 2 | 3 | 0 |" in md
    # Fix B — the status word is bolded in Markdown (no colour available).
    assert "Estimate **borderline**" in md
    assert "review the losses before routing these calls" in md
    assert "[LOSS]" in md
    assert "[TIE]" in md
    assert "### Per-prompt detail" in md
    # System message surfaced for the prompts that carried one.
    assert "_System:_ You are terse." in md
    assert "_System:_ Answer as a pirate." in md
    # Privacy line present.
    assert "Nothing was sent to Rodiun or any Frugon endpoint." in md


def test_quality_section_md_none_is_empty() -> None:
    # measure_result=None → no section at all (byte-identity guarantee).
    assert _quality_section_md(None) == []


# ---------------------------------------------------------------------------
# Tier-0 — Markdown section
# ---------------------------------------------------------------------------


def test_quality_section_md_tier0_emits_side_by_side_and_judge_framing() -> None:
    # Arrange
    result = _tier0_result()

    # Act
    md = "\n".join(_quality_section_md(result))

    # Assert — Tier-0 framing + no tally table + no verdict labels.
    assert "Raw samples" in md
    assert "run `--judge` for a scored verdict" in md
    assert "| Candidate | Win |" not in md  # no tally table in Tier-0
    assert "[WIN]" not in md
    assert "[LOSS]" not in md
    assert "[TIE]" not in md
    # Both models' outputs appear side by side, each fenced as a literal code
    # block under its label line (untrusted model output is never spliced as live
    # Markdown — see _md_fenced_output_lines).  The label stays a bullet; the
    # output text is intact inside the fence.
    assert "- `gpt-4o` (current):" in md
    assert "Red, green, blue." in md
    assert "- `gpt-4o-mini`:" in md
    assert "Red, blue, yellow." in md
    assert "Nothing was sent to Rodiun or any Frugon endpoint." in md


# ---------------------------------------------------------------------------
# Tier-1 + Tier-0 — HTML section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["v1", "v2"])
def test_quality_section_html_tier1_verdict_classes_match_tally_table(style: str) -> None:
    # Arrange — a tally with a loss, plus a per-prompt LOSS verdict.
    result = _tier1_mixed_result()

    # Act
    html = _quality_section_html(result, style=style)

    # Assert — section present with the tally and verdict; colour discipline:
    # verdict labels carry table-matching classes (verdict-win green /
    # verdict-loss red / verdict-tie yellow per the CSS); the section markup
    # uses class names only, so no raw colour token appears here.
    assert "Quality measurement" in html
    assert 'class="quality-tally"' in html
    # Fix B — borderline status word carries the TIE tally class (amber) so it
    # stands out and agrees with the tally table.
    assert 'Estimate <span class="verdict-tie">borderline</span>' in html
    assert 'class="verdict-loss">[LOSS]' in html
    assert 'class="verdict-tie">[TIE]' in html
    assert "var(--green" not in html
    assert "var(--red" not in html
    assert "Nothing was sent to Rodiun or any Frugon endpoint." in html
    # v1 wraps in a card; v2 wraps in a below-the-fold section.
    if style == "v2":
        assert html.startswith('<section class="below">')
    else:
        assert html.startswith('<div class="card">')


def test_quality_section_html_tier0_has_no_tally_and_judge_framing() -> None:
    # Arrange
    result = _tier0_result()

    # Act
    html = _quality_section_html(result, style="v1")

    # Assert — Tier-0: side-by-side outputs, the --judge framing, no tally.
    assert "Raw samples" in html
    assert "<code>--judge</code>" in html
    assert 'class="quality-tally"' not in html
    assert "[LOSS]" not in html
    assert "Red, blue, yellow." in html


def test_quality_section_html_none_is_empty() -> None:
    assert _quality_section_html(None, style="v1") == ""
    assert _quality_section_html(None, style="v2") == ""


def test_quality_section_html_escapes_untrusted_output() -> None:
    # A model output containing HTML must be escaped, never injected raw.
    comp = Comparison(
        record=_record("hi"),
        current_output=SampledOutput(model="gpt-4o", content="<script>alert(1)</script>"),
        candidate_outputs=[SampledOutput(model="gpt-4o-mini", content="plain")],
    )
    result = MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[comp],
        tier1_tallies=None,
    )
    html = _quality_section_html(result, style="v1")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Verdict reconciliation — the report and the terminal MUST agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tally", "expected_state"),
    [
        (Tier1Tally(candidate="gpt-4o-mini", wins=5, losses=0, ties=0), "confirmed"),
        (Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=2, ties=3), "borderline"),
        (Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=4, ties=1), "not_confirmed"),
        (Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=0, ties=0, errors=5), "not_verified"),
    ],
)
def test_report_verdict_reconciles_with_terminal_via_shared_classifier(
    tally: Tier1Tally, expected_state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report verdict and the terminal verdict are the SAME _classify_verdict.

    The reconciliation guarantee is structural, not coincidental: BOTH the
    terminal synthesis (``_render_tier1_synthesis``) and the report sections call
    the one ``_classify_verdict``.  We prove it by spying on the classifier — the
    terminal renderer must call it, the call must return the expected state, and
    the EXACT plain text it returns must appear verbatim in the report Markdown.
    Spying on the shared function is immune to terminal-capture fragility (the
    table render path writes through a separate console), so this test reconciles
    the two surfaces without depending on rendered-output capture.
    """
    import sys

    # Resolve the EXACT module object the renderers live in — robust even if an
    # earlier test reloaded frugon.report via sys.modules (which would make a
    # plain ``import frugon.report`` bind a different module instance, so a spy
    # on its attributes would never be seen by the already-bound renderers).
    report = sys.modules[_render_tier1_synthesis.__module__]

    result = MeasureResult(
        samples_requested=5,
        samples_taken=5,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[],
        tier1_tallies=[tally],
    )

    # Spy on the shared classifier, delegating to the real implementation.
    real_classify = report._classify_verdict
    calls: list[tuple[str, str]] = []

    def _spy(t: Tier1Tally, current: str, **kwargs: object) -> tuple[str, str]:
        out = real_classify(t, current, **kwargs)  # type: ignore[arg-type]
        calls.append(out)
        return out

    monkeypatch.setattr(report, "_classify_verdict", _spy)

    # Drive the TERMINAL synthesis and the report MD through the SAME resolved
    # module — both must route through the spied classifier.
    report._render_tier1_synthesis(result)
    assert calls, "terminal synthesis did not call the shared _classify_verdict"
    term_state, term_text = calls[-1]

    # The report Markdown carries the SAME classifier output (verbatim wording).
    md = "\n".join(report._quality_section_md(result))

    # Assert — same state and expected state.  The wording is verbatim except the
    # status word, which Fix B emphasises (bold in Markdown) in place; so the
    # sentence with its status word bolded must appear (same single-source rule
    # the renderer applies), proving both reconciliation AND the emphasis.
    assert term_state == expected_state
    emphasised = report._emphasise_verdict_status(
        term_text, term_state, report._verdict_status_md(term_state)
    )
    assert emphasised in md, f"report MD missing the emphasised verdict text: {emphasised!r}"
    # And the status word is bold in the MD.
    assert f"**{report._VERDICT_STATUS_PHRASE[term_state]}**" in md


# ---------------------------------------------------------------------------
# Byte-identity — measure_result=None must not change the report at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "renderer",
    [render_markdown, render_markdown_v2, render_html, render_html_v2],
)
def test_renderer_byte_identical_with_and_without_default_measure_kwarg(
    renderer: object, tmp_path: Path
) -> None:
    """Passing measure_result=None explicitly == omitting it == today's output."""
    from frugon.cost import analyze_records, iter_records

    sample = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "frugon"
        / "data"
        / "sample_logs.jsonl.gz"
    )
    records, skipped = iter_records(sample)
    result = analyze_records(records, skipped_malformed=skipped, split_routing=True)

    out_default = tmp_path / "default.out"
    out_explicit = tmp_path / "explicit.out"
    renderer(result, out_default)  # type: ignore[operator]
    renderer(result, out_explicit, measure_result=None)  # type: ignore[operator]

    assert out_default.read_bytes() == out_explicit.read_bytes()


@pytest.mark.parametrize(
    "renderer",
    [render_markdown, render_markdown_v2, render_html, render_html_v2],
)
def test_renderer_with_measure_appends_quality_section(
    renderer: object, tmp_path: Path
) -> None:
    """With a MeasureResult, the written report carries the quality section."""
    from frugon.cost import analyze_records, iter_records

    sample = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "frugon"
        / "data"
        / "sample_logs.jsonl.gz"
    )
    records, skipped = iter_records(sample)
    result = analyze_records(records, skipped_malformed=skipped, split_routing=True)

    out = tmp_path / "with_measure.out"
    renderer(result, out, measure_result=_tier1_mixed_result())  # type: ignore[operator]
    text = out.read_text(encoding="utf-8")

    assert "Quality measurement" in text
    # Fix B emphasises just the status word in place (bold in MD, a tally-class
    # span in HTML), so the sentence reads "Estimate <emph>borderline</emph>:".
    assert "Estimate " in text
    assert "borderline" in text
    assert "**borderline**" in text or '"verdict-tie">borderline' in text
    # The loss label appears in both the MD ([LOSS]) and HTML (verdict-loss).
    assert "[LOSS]" in text or "verdict-loss" in text


# ---------------------------------------------------------------------------
# Untrusted-output fencing (Markdown structural-injection / raw-HTML XSS)
# ---------------------------------------------------------------------------


_ADVERSARIAL_OUTPUT = (
    "#### Scalability\n"
    "1. **First point**\n"
    "## A fake report heading\n"
    "a fenced block:\n"
    "```python\nprint('hi')\n```\n"
    "<script>alert(1)</script>\n"
    "plain multi-line\ntext continues"
)


def _adversarial_result(*, tier1: bool) -> MeasureResult:
    """A measure result whose model outputs try to impersonate report structure."""
    comp = Comparison(
        record=_record("Describe the system.", "# You are a ## helper"),
        current_output=SampledOutput(model="gpt-4o", content=_ADVERSARIAL_OUTPUT),
        candidate_outputs=[
            SampledOutput(model="gpt-4o-mini", content=_ADVERSARIAL_OUTPUT)
        ],
        verdicts=["loss"] if tier1 else [],
    )
    return MeasureResult(
        samples_requested=1,
        samples_taken=1,
        current_model="gpt-4o",
        candidates=["gpt-4o-mini"],
        comparisons=[comp],
        tier1_tallies=(
            [Tier1Tally(candidate="gpt-4o-mini", wins=0, losses=1, ties=0)]
            if tier1
            else None
        ),
    )


def _md_lines_outside_fences(md: str) -> list[str]:
    """Return the lines of *md* that sit OUTSIDE any backtick code fence."""
    outside: list[str] = []
    in_fence = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            outside.append(line)
    return outside


@pytest.mark.parametrize("tier1", [True, False])
def test_quality_section_md_fences_model_output_no_structural_injection(
    tier1: bool,
) -> None:
    """Model output is fenced — its #-headings and raw <script> never escape.

    Arrange: a measure result whose outputs contain markdown headings, an ordered
    list, an inner ``` fence, multi-line text, and a raw <script>.
    Act: render the Markdown quality section (Tier-1 AND Tier-0 paths).
    Assert: no heading line and no raw <script> appears OUTSIDE a fence (only the
    report's own ## / ### structure may), and the literal output text survives
    INSIDE the fences intact.
    """
    md = "\n".join(_quality_section_md(_adversarial_result(tier1=tier1)))

    outside = _md_lines_outside_fences(md)
    # The ONLY '#'-led lines outside a fence are the report's own headings.
    own_headings = {"## Quality measurement", "### Per-prompt detail"}
    leaked_headings = [
        ln for ln in outside if ln.lstrip().startswith("#") and ln.strip() not in own_headings
    ]
    assert leaked_headings == [], leaked_headings
    # No raw <script> escapes the fence (raw-HTML XSS guard).
    assert all("<script>" not in ln for ln in outside)
    # The output text is intact, literal, inside the fences.
    assert "#### Scalability" in md
    assert "<script>alert(1)</script>" in md
    # An output containing ``` forces a longer (````) fence so it cannot close early.
    assert "````" in md
    # The System preview is inline-escaped, not raw (no live '# heading').
    assert "\\# You are a \\#\\# helper" in md


# ---------------------------------------------------------------------------
# Self-judge caution (F3) — present on every report surface + terminal
# ---------------------------------------------------------------------------


def _self_judged_result() -> MeasureResult:
    r = _tier1_mixed_result()
    r.judge_model = "gpt-4o-mini"
    r.self_judged_models = ["gpt-4o-mini"]
    return r


def _independent_judge_result() -> MeasureResult:
    r = _tier1_mixed_result()
    r.judge_model = "claude-3-5-sonnet"
    r.self_judged_models = []
    return r


def test_self_judge_caution_identical_wording_across_surfaces() -> None:
    """The self-judge caution reads IDENTICALLY on the terminal helper, MD, HTML.

    A single helper (_self_judge_caution_text) is the one source, so the three
    surfaces can never drift.  When self_judged_models is empty, the caution is
    absent everywhere.
    """
    from frugon.report import _self_judge_caution_text

    sj = _self_judged_result()
    wording = _self_judge_caution_text(sj)
    assert wording is not None
    assert "self-biased" in wording

    md = "\n".join(_quality_section_md(sj))
    html_v1 = _quality_section_html(sj, style="v1")
    html_v2 = _quality_section_html(sj, style="v2")
    assert f"> ⚠ {wording}" in md
    assert wording in html_v1
    assert wording in html_v2

    # Independent judge → no caution anywhere.
    indep = _independent_judge_result()
    assert _self_judge_caution_text(indep) is None
    assert "self-biased" not in "\n".join(_quality_section_md(indep))
    assert "self-biased" not in _quality_section_html(indep, style="v1")
    assert "self-biased" not in _quality_section_html(indep, style="v2")
