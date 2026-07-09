"""frugon report renderer.

Produces terminal, HTML, and Markdown reports from an AnalysisResult.
All formatting is pure local arithmetic — no network, no LLM.
"""

from __future__ import annotations

import html as _html_escape
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich import get_console as _get_console
from rich import print as rprint
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from frugon._store import atomic_write_text as _atomic_write_text
from frugon.cost import (
    AnalysisResult,
    EscalationSuggestion,
    _display_pct,
    compute_saving_pct,
    next_rung_up,
    window_contradicts_span,
)
from frugon.pricing import is_pricing_stale as _is_pricing_stale
from frugon.quality import get_attribution as _get_attribution
from frugon.quality import get_model_tier as _get_model_tier
from frugon.quality import is_quality_stale as _is_quality_stale
from frugon.quality import is_unrated as _is_unrated
from frugon.quality import tier_name as _tier_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from frugon.measure import Comparison, MeasureResult, Tier1Tally
    from frugon.pricing import PricedModelRow
    from frugon.routing import SplitRouting

# Canonical "saving / success" green.  Used for the saving figure in the terminal
# and HTML reports so report, terminal, and the landing page all agree that
# green == saving.
SAVING_GREEN = "#10B981"

# Terminal colour law (CLI redesign):
#   green  = the money win ONLY (the saving / hero) — green always means payoff
#   cyan   = brand / model names / commands
#   amber  = caution ("quality not verified", "tier unknown")
# These are the only three accent colours; everything else is plain or dim.
BRAND_CYAN = "cyan"
CAUTION_AMBER = "yellow"

# Loss fraction at or above which a judge result that technically *held*
# (wins + ties > losses) is downgraded from a flat "confirmed" to the
# amber "borderline" band: a 3-2 win should not read identically to 5-0.
# At 0.20, 1/10 losses still reads confirmed while 2/5 reads borderline.
# Tunable — raise it to be more forgiving, lower it to be stricter.
_VERDICT_BORDERLINE_LOSS_FRACTION = 0.20

# Confidence-aware "small sample" nudge.
#
# A judged tally is "low-confidence" — thin enough that one more scored prompt
# could plausibly flip the verdict band — when EITHER:
#   * fewer than this many prompts were actually scored, OR
#   * the held-vs-lost margin is within one prompt of flipping
#     (|held − losses| ≤ 1 — e.g. a 3-2 split).
# When low-confidence, the synthesis appends a dim line nudging the user to
# re-sample with more prompts.  A decisive result (e.g. 9/10 hold, 8/10 lose)
# is NOT nudged — its margin is wide and its sample already adequate.  A fully
# unverified tally (every comparison errored) is never nudged: the fix there is
# to retry/fix access, not to widen the sample.
_LOW_CONFIDENCE_MIN_SCORED = 10
_LOW_CONFIDENCE_MARGIN = 1
# The sample size the nudge suggests re-running with — a firmer read than the
# n=10 default without being punitively slow.
_NUDGE_RESAMPLE_SAMPLES = 25


def _is_low_confidence(tally: Tier1Tally) -> bool:
    """Return True when *tally* is statistically thin for its sample size.

    Low-confidence ⇔ too few scored prompts OR a held-vs-lost margin within one
    prompt of flipping the verdict band.  Errored-only tallies (nothing scored)
    return False — the nudge is about widening a thin-but-real sample, not about
    a run that produced no verdict at all (that path already tells the user to
    retry / check access).
    """
    scored = tally.wins + tally.losses + tally.ties
    if scored == 0:
        return False
    held = tally.wins + tally.ties
    if scored < _LOW_CONFIDENCE_MIN_SCORED:
        return True
    return abs(held - tally.losses) <= _LOW_CONFIDENCE_MARGIN


def _nudge_text(tally: Tier1Tally) -> str | None:
    """Return the plain-text small-sample nudge for *tally*, or None when decisive.

    Shared verbatim by the terminal, Markdown and HTML synthesis so the nudge
    wording can never drift across surfaces.  ``N`` is the number of prompts
    actually scored (errors excluded), matching the figure in the verdict line
    just above it.
    """
    if not _is_low_confidence(tally):
        return None
    scored = tally.wins + tally.losses + tally.ties
    return (
        f"Small sample ({scored:,}) — re-run with "
        f"--samples {_NUDGE_RESAMPLE_SAMPLES} for a firmer read."
    )


def _escalation_for_tally(
    tally: Tier1Tally, current_model: str
) -> EscalationSuggestion | None:
    """Return the next-rung-up suggestion for a NOT-confirmed *tally*, or None.

    Thin adapter over :func:`frugon.cost.next_rung_up`: the candidate that failed
    the judge is ``tally.candidate``; the baseline the user pays today is
    *current_model*.  Returns ``None`` when no model is both a quality tier above
    the failed candidate and still cheaper than the baseline (the honest
    dead-end — the surfaces then keep the "keep these on the baseline" guidance).
    """
    return next_rung_up(tally.candidate, current_model)


def _escalation_lead(tally: Tier1Tally) -> str:
    """The 'try the next rung up' lead-in, reusing the tally's loss count.

    Reconciles with the verdict line directly above it (same losses/scored
    figures) so the escalation reads as a continuation, not a restatement.
    """
    scored = tally.wins + tally.losses + tally.ties
    return (
        f"Estimate NOT confirmed: {tally.candidate} was worse in "
        f"{tally.losses:,}/{scored:,} — try the next rung up:"
    )


def _escalation_detail(suggestion: EscalationSuggestion, current_model: str) -> str:
    """The escalation body naming the model, its tier, the saving and the command.

    Shared verbatim across terminal / Markdown / HTML so the suggested model,
    its tier label, the ~NN% cheaper-than-baseline figure and the ready command
    are byte-identical on every surface.
    """
    return (
        f"{suggestion.model} ({suggestion.tier_label} tier, still "
        f"~{suggestion.pct_cheaper_than_baseline}% cheaper than {current_model}): "
        f"{suggestion.command}"
    )

# The two load-bearing caveats kept in the default terminal view (CLI redesign
# point 3).  Everything else — the wholesale upper bound, the easy/hard heuristic
# explanation, and the automated-routing upsell — moves to --verbose.
# The quality caveat is rendered as TWO deliberate lines: the assertion, then
# the call to action on its own line beneath it.  The break is forced (not left
# to soft-wrap) so the "run --measure …" instruction always reads as a distinct,
# scannable second line rather than a tail that bleeds past the left margin when
# the terminal is narrow.  Each half then reflows independently with a hanging
# indent (see ``_render_split_footer``).
QUALITY_NOT_VERIFIED_ASSERTION = (
    "Quality is not verified — 'within tolerance' is an offline estimate;"
)
# The wholesale headline has no "within tolerance" band (it moves every call), so
# its caveat names the full-swap quality change directly rather than the band.
QUALITY_NOT_VERIFIED_ASSERTION_WHOLESALE = (
    "Quality is not verified — a full swap can change output quality;"
)
QUALITY_NOT_VERIFIED_ACTION = (
    "run --measure to confirm it on your real outputs before you switch."
)
# The single-string form is retained for callers/tests that want the full
# caveat as one sentence (e.g. the wholesale path and the Markdown/HTML
# renderers, where wrapping is the viewer's concern, not ours).
QUALITY_NOT_VERIFIED = f"{QUALITY_NOT_VERIFIED_ASSERTION} {QUALITY_NOT_VERIFIED_ACTION}"
# Shown in place of the quality-risk caveat when the recommended candidate is the
# SAME or HIGHER quality tier than the baseline (tier_drop <= 0).  No quality
# downgrade is involved, so the "verify before you switch" risk framing would be
# dishonest in the wrong direction — replace it with the stronger, accurate claim.
QUALITY_EQUAL_OR_BETTER = (
    "Candidate is rated same or better quality than the baseline — "
    "this is a quality-neutral or quality-improving move."
)
PRIVACY_CAVEAT = "Your data never leaves your machine."

# The footer privacy line for the split headline — the two clauses a buyer needs
# at a glance.  Named (and kept) distinct from cli.PRIVACY_LINE (the fuller
# three-clause line the wholesale path prints) so each surface reads at its
# own length — the two constants are DELIBERATELY different strings for
# different surfaces, not a duplicate (FRG-OSS-005b): a bare shared name
# `PRIVACY_LINE` in both modules was the real smell (import-the-wrong-one
# risk), fixed by giving this one a name that states what it actually is.
SPLIT_FOOTER_PRIVACY_LINE = (
    "Your data never leaves your machine. Your keys go to your own providers."
)

# The single upsell destination shown in the footer (cyan link).  The product
# pitch ("route every call automatically and hold the savings") lives in the
# footer text; this is just the URL.
FUNNEL_URL = "https://frugon.rodiun.io"

# Disclosure shown whenever the per-call split routing is the headline: it is a
# transparent offline heuristic, not a trained router or a quality guarantee.
SPLIT_CAVEAT = (
    "Easy/hard split is a local heuristic over prompt and completion length "
    "(RouteLLM-style), computed offline with no LLM calls. "
    "Quality is not verified — run --measure to sample real outputs before you switch."
)
# Variant for the case where tier_drop <= 0 (same or better quality): the method
# explanation is identical but the "Quality is not verified" risk tail is replaced
# with the accurate positive statement.  Used in the Markdown/HTML taglines.
SPLIT_CAVEAT_EQUAL_OR_BETTER = (
    "Easy/hard split is a local heuristic over prompt and completion length "
    "(RouteLLM-style), computed offline with no LLM calls. "
    "Candidate is rated same or better quality — no quality downgrade."
)

# Shown for a model that has no entry in the LMArena quality tier table
# (``tier_name`` returns None for ``UNRATED_TIER``).  The gap is surfaced — never
# omitted — so the reader sees "what's known, and what isn't" side by side.
_TIER_UNRATED_LABEL = "unrated"


def _tier_label(model: str) -> str:
    """Return a model's published quality-tier label, or ``"unrated"``.

    Wraps :func:`frugon.quality.get_model_tier` + :func:`frugon.quality.tier_name`
    so EVERY surface (terminal, HTML v1/v2, Markdown) renders the SAME label for
    the SAME model — the one reconciliation point for the quality-tier disclosure,
    exactly as :func:`_split_report_figures` is the one point for the money figures.
    ``None`` (an unrated model) renders as ``_TIER_UNRATED_LABEL`` rather than
    being dropped, so the disclosure marks the gap instead of hiding it.
    """
    return _tier_name(_get_model_tier(model)) or _TIER_UNRATED_LABEL


def _is_equal_or_better_quality(result: AnalysisResult) -> bool:
    """True when the recommended candidate is rated same or HIGHER quality than baseline.

    ``result.tier_drop`` is the quality tier delta (candidate_tier - baseline_tier):
    a negative value means the candidate is a HIGHER tier (better quality), zero
    means same tier.  Both are equal-or-better moves — no quality risk to disclose.

    Returns False when tier_drop is None (either model is unrated, or no candidate
    was selected): the unrated path has its own separate disclosure and must never
    be silenced by this helper.
    """
    return result.tier_drop is not None and result.tier_drop <= 0


def _shown_quality_phrase(result: AnalysisResult | None) -> str:
    """Return the quality phrase actually rendered in the panel/headline for *result*.

    The --measure / --judge synthesis lines back-reference the phrase the user
    already saw in the offline routing panel.  When the candidate is rated same
    or better quality (``tier_drop <= 0``) the panel said "same or better
    quality"; for a genuine step-down (``tier_drop >= 1``) or an unrated result
    (``tier_drop is None``) the panel said "within tolerance".  Threading this
    helper through every synthesis back-reference keeps them truthful regardless
    of which branch fired — the §6 honesty invariant on the --judge path.

    CAUTION: this phrase is only meaningful for the model the offline panel
    actually recommended (``result.split.candidate_model`` / ``result.candidate_model``).
    A caller measuring a DIFFERENT model (e.g. ``--demo --measure``'s pinned
    try-out candidate) must not attribute this phrase to that model — see
    :func:`_measured_model_is_the_recommendation` and
    :func:`_recommendation_divergence_note`.
    """
    if result is not None and _is_equal_or_better_quality(result):
        return "same or better quality"
    return "within tolerance"


def _recommended_model(result: AnalysisResult | None) -> str | None:
    """Return the model the offline panel actually recommended for *result*.

    Single source of truth for "the headline recommendation" — the per-call
    split's candidate when a split exists, else the wholesale candidate.
    Mirrors the resolution cli.py uses when defaulting --measure's candidate
    (measure_candidates' default_candidate), so this always names the SAME
    model the routing panel's hero line showed.
    """
    if result is None:
        return None
    if result.split is not None and result.split.candidate_model:
        return result.split.candidate_model
    return result.candidate_model


def _measured_model_is_the_recommendation(
    measured_model: str, result: AnalysisResult | None
) -> bool:
    """True iff *measured_model* is the SAME model the offline panel recommended.

    False when the caller measured a model other than the headline
    recommendation — the case ``--demo --measure`` exercises via its pinned
    single try-out candidate (``_DEMO_MEASURE_CANDIDATE``), which is chosen
    only so the demo's try-out path needs a single provider key, and is NOT
    necessarily the model the offline panel recommends.  Also False when
    *result* carries no recommendation to compare against.
    """
    recommended = _recommended_model(result)
    return recommended is not None and measured_model == recommended


def _recommendation_divergence_note(measured_model: str, result: AnalysisResult | None) -> str | None:
    """Return a disclosure sentence when *measured_model* != the headline recommendation.

    Returns ``None`` when the measured model IS the recommendation (the common
    case — no disclosure needed) or when there is no recommendation to compare
    against.  When they diverge, callers must render this note ALONGSIDE the
    verdict rather than let the verdict imply the headline recommendation was
    verified — the honesty gap this guards is a sample-model's offline-quality
    phrase being silently attributed to a different model the panel actually
    recommends (§6 honesty invariant).
    """
    recommended = _recommended_model(result)
    if recommended is None or measured_model == recommended:
        return None
    return (
        f"{measured_model} is the demo's try-out sample model, not the headline "
        f"recommendation ({recommended}); this verifies the --measure flow, not "
        "the recommended switch — the demo pins its sample model so the "
        "try-out needs only an OpenAI key."
    )


# ---------------------------------------------------------------------------
# Unrated-recommendation message family (audit findings #1 + #4)
# ---------------------------------------------------------------------------
#
# Two related honesty disclosures, written as ONE message family so they read
# consistently and can never drift across surfaces (terminal + Markdown v1/v2 +
# HTML v1/v2 all build from these single string sources):
#
#   #1 — the RECOMMENDED candidate is unrated.  The default-pool routing path
#        never recommends an unrated model (it is filtered out), but an EXPLICIT
#        ``--candidates X`` can recommend one (the user asked for it).  We do not
#        block — we attach a clear, non-blocking quality caveat next to the
#        recommendation pointing at ``--measure`` to confirm it.
#
#   #4 — an unrated candidate was held out of the per-call split.  When the only
#        viable candidate is unrated the split is unavailable and the report
#        falls back to the wholesale full-swap projection; the user is told why
#        rather than silently seeing a full swap.
#
# Both name the model and point at ``--measure`` — the same remedy, the same
# voice.  Each surface styles these strings in its own idiom (terminal: cyan
# model + amber caution; Markdown: code-spanned model in a callout; HTML: the
# amber ``.caution`` class) but the wording is shared verbatim.
#
# Measurement-awareness (per model): the family is built against the set of
# models the CURRENT run actually JUDGED (those with a Tier-1 tally — the same
# condition that means a quality verdict renders below).  For a model judged in
# THIS run the "run --measure to confirm" remedy is redundant and contradictory
# — the verdict line below already verified it — so:
#
#   * the #1 recommendation caveat is SUPPRESSED entirely, and
#   * the #4 held-out / wholesale-fallback notes keep the still-true fact but
#     DROP the "run --measure" imperative and point DOWN at the measured verdict
#     ("its quality is measured below").
#
# For a model NOT judged in this run (a cost-only report, or a model the user
# did not pass to the judge) the original wording is preserved verbatim — the
# "run --measure --judge --candidates X to verify" advice is correct there.  Call sites
# with no measurement context pass an empty ``judged_models`` set, which yields
# exactly the original (not-judged) wording, so the cost-only path is unchanged.
#
# Severity (per message): each family line carries a ``_SEV_WARNING`` /
# ``_SEV_INFO`` tag so every surface can style it by what it MEANS, not by the
# fact that it is an unrated-family note:
#
#   * ``_SEV_WARNING`` — the quality is genuinely UNVERIFIED this run, so the
#     "run --measure …" remedy is live.  Rendered as the amber ⚠ caution (the
#     real caution it always was): the recommendation caveat and the not-judged
#     held-out / wholesale-fallback notes.
#   * ``_SEV_INFO`` — the ``_measured`` variants, fired ONLY when the model WAS
#     judged this run.  These merely explain why the candidate isn't in the
#     routing plan and point DOWN at the verdict the quality section already
#     confirmed — informational, not a caution.  Alarm-styling a row about a
#     model the section below verified at 10/10 is wrong, so these render in the
#     report's neutral/dim note idiom (no ⚠, no amber) on every surface.

_SEV_WARNING = "warning"
_SEV_INFO = "info"

# Terminal (Rich) style per severity, single-sourced so the footer mapping can
# never drift: a real caution reads amber; an informational note reads dim (the
# report's standard muted note idiom), with NO ⚠ glyph on either — the footer's
# tier-note lines never carried a glyph; the severity governs colour alone here.
_SEV_TERMINAL_STYLE: dict[str, str] = {
    _SEV_WARNING: CAUTION_AMBER,
    _SEV_INFO: "dim",
}


def _unrated_recommendation_caveat(model: str) -> str:
    """Finding #1 — non-blocking quality caveat for an UNRATED recommended model.

    Shared verbatim across every surface.  Names the model and the exact command
    that verifies it, so a user who explicitly asked for an unrated candidate is
    told plainly that its quality is unverified — without being blocked (they
    asked for it on purpose).  The command is ``--measure --judge``: only the
    judge produces the scored verdict that VERIFIES the model's quality;
    ``--measure`` alone just samples raw outputs without a verdict.
    """
    return (
        f"{model} is unrated — its quality is unverified. "
        f"Run --measure --judge --candidates {model} to verify it before you switch."
    )


def _candidate_block_saving_pct(
    result: AnalysisResult, model: str
) -> Decimal | None:
    """Return *model*'s block split saving% (monthly, else observed), or None.

    Reads the SAME figure the "Candidates considered" block quotes for this model
    (``candidate_projections``), so the Change-1b caveat's "could save ~X%" and
    the Change-2 promotion's percentages never drift from the number printed in
    the block one line away.  Prefers the monthly saving% (the headline basis);
    falls back to the observed saving% when there is no monthly projection.
    """
    for proj in result.candidate_projections:
        if proj.model == model:
            if proj.saving_pct is not None:
                return proj.saving_pct
            return proj.observed_saving_pct
    return None


def _excluded_unrated_caveat(model: str, saving_pct: Decimal | None) -> str:
    """Change 1b — an unrated candidate beats baseline but is held out of the route.

    Shared verbatim across every surface.  States the potential saving (so the
    user sees what verifying could unlock), names the model, and gives the exact
    command that makes it eligible.  The command is ``--measure --judge``: only
    the judge yields the scored verdict that VERIFIES the model and unlocks it as
    the recommendation; ``--measure`` alone samples raw outputs without a verdict.
    When the saving% is unknown (no projection basis) the phrasing degrades
    gracefully to omit the figure rather than print a bare placeholder.
    """
    if saving_pct is not None:
        lead = f"{model} could save ~{float(saving_pct):.1f}%, but it's unrated"
    else:
        lead = f"{model} beats your baseline, but it's unrated"
    return (
        f"{lead} — it's excluded from the recommended route. "
        f"Run --measure --judge --candidates {model} to verify it and unlock it "
        "as the recommendation."
    )


def _recommended_unrated_model(result: AnalysisResult) -> str | None:
    """Return the UNRATED model being RECOMMENDED on this surface, or None.

    The recommendation is unrated when EITHER the headline candidate is unrated
    (``candidate_is_unrated`` — covers the wholesale full-swap and the
    single-candidate paths) OR the multi-candidate "Candidates considered" block
    tags an unrated model as the cheapest "recommended" split.  Returns the model
    name so the caller can build the shared #1 caveat; None when the recommended
    candidate is rated (e.g. the default demo's gpt-4o-mini → no caveat).
    """
    if result.candidate_is_unrated and result.candidate_model is not None:
        return result.candidate_model
    for proj in result.candidate_projections:
        if proj.status == "recommended" and _is_unrated(proj.model):
            return proj.model
    return None


def judged_models_from_measure(
    measure_result: MeasureResult | None,
) -> frozenset[str]:
    """Return the set of model names JUDGED in *measure_result* (have a Tier-1 tally).

    A model with a tally is one for which a quality verdict ("Estimate
    confirmed/borderline/not-confirmed: <model> …") renders in the measurement
    section — exactly the condition under which a "run --measure to confirm"
    caveat for that model becomes redundant and contradictory.  ``None`` (no
    --measure) or a Tier-0 sample (--measure without --judge, no tallies) yields
    an empty set, so the unrated family keeps its original "run --measure" wording.
    """
    if measure_result is None or measure_result.tier1_tallies is None:
        return frozenset()
    return frozenset(tally.candidate for tally in measure_result.tier1_tallies)


def _unrated_family_messages(
    result: AnalysisResult,
    judged_models: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    """Return the unrated-recommendation caveat (finding #1) for *result*, if any.

    Each item is a ``(message, severity)`` tuple where *severity* is
    ``_SEV_WARNING`` — the ONE source every surface (terminal footer + Markdown
    v1/v2 + HTML v1/v2) builds from, so the wording reads byte-identically
    everywhere (parity discipline) and the icon/colour is decided ONCE here.

    Under the unified split basis every priced candidate (rated or not) competes
    on its full-dataset split New-spend, and the cheapest becomes the headline
    routing target.  An unrated candidate is therefore never "held out of the
    split" or forced into a wholesale fallback — those finding-#4 notes are
    obsolete and removed.  The only remaining honesty disclosure is the #1
    caveat: when the chosen ROUTING TARGET is unrated, name it and the verify
    command so a user who routed onto an unknown-quality model is told plainly.

    *judged_models* is the set of models the CURRENT run judged (those whose
    quality verdict renders in the measurement section below — see
    :func:`judged_models_from_measure`).  When the recommended unrated model was
    judged this run the #1 caveat is SUPPRESSED: the measurement section's own
    verdict line ("Estimate confirmed: <model> …") IS the verification, so a "run
    --measure to confirm" caveat would be contradictory.

    Empty list when the recommendation is rated — e.g. the default demo
    (gpt-4o-mini is rated) and every rated single-candidate run.
    """
    messages: list[tuple[str, str]] = []

    recommended_unrated = _recommended_unrated_model(result)
    if recommended_unrated is not None and recommended_unrated not in judged_models:
        messages.append((_unrated_recommendation_caveat(recommended_unrated), _SEV_WARNING))

    # Change 1b — unrated candidates that beat baseline on split but were EXCLUDED
    # from the recommended route for being unrated (a rated candidate was
    # recommended instead).  Each gets a clear "could save ~X%, but it's unrated —
    # excluded until you verify it; run --measure to check" caveat.  When the model
    # was judged THIS run the caveat is suppressed entirely: the measurement
    # section's own verdict line is the verification, so "run --measure to check"
    # would be contradictory (and the promotion callout, Change 2, takes over).
    for excluded in result.excluded_unrated_models:
        if excluded in judged_models:
            continue
        saving_pct = _candidate_block_saving_pct(result, excluded)
        messages.append((_excluded_unrated_caveat(excluded, saving_pct), _SEV_WARNING))

    return messages


def _data_quality_terminal(result: AnalysisResult) -> None:
    """Print fail-loud data-quality notes (skipped + approximated) when present."""
    if result.skipped_malformed:
        rprint(
            f"[yellow]{result.skipped_malformed} malformed record(s) skipped "
            "(not counted in the figures above).[/yellow]"
        )
    if result.approximated_calls:
        rprint(
            f"[yellow]{result.approximated_calls} call(s) used approximate token "
            "counts (tokenizer unavailable for the model) — those costs are estimates.[/yellow]"
        )


def _data_quality_md(result: AnalysisResult) -> list[str]:
    """Return Markdown lines for the data-quality notes (empty when none apply)."""
    lines: list[str] = []
    if result.skipped_malformed:
        lines.append(
            f"> ⚠ {result.skipped_malformed} malformed record(s) skipped "
            "(not counted in the figures above)."
        )
    if result.approximated_calls:
        lines.append(
            f"> ⚠ {result.approximated_calls} call(s) used approximate token counts "
            "(tokenizer unavailable for the model) — those costs are estimates."
        )
    if lines:
        lines.append("")
    return lines


def _data_quality_html(result: AnalysisResult) -> str:
    """Return an HTML note for the data-quality disclosures (empty when none apply)."""
    bits: list[str] = []
    if result.skipped_malformed:
        bits.append(
            f"{result.skipped_malformed} malformed record(s) skipped "
            "(not counted in the figures above)."
        )
    if result.approximated_calls:
        bits.append(
            f"{result.approximated_calls} call(s) used approximate token counts "
            "(tokenizer unavailable) — those costs are estimates."
        )
    if not bits:
        return ""
    # Amber (a disclosure, not an error) — matches the terminal/MD ⚠ yellow.
    return (
        '<p class="note" style="color:#F59E0B">&#9888; '
        + " ".join(_html_escape.escape(b) for b in bits)
        + "</p>"
    )


def _has_split(result: AnalysisResult) -> bool:
    """True when a per-call split recommendation should be the report headline.

    Requires a split with at least one routed (easy) call and a positive blended
    saving; otherwise the report falls back to the wholesale recommendation.
    """
    split = result.split
    return (
        split is not None
        and split.routed_count > 0
        and split.saving_pct is not None
        and split.saving_pct > Decimal("0")
    )

# Canonical quality caveat — shown whenever a routing recommendation is displayed.
# Anchors the saving to list-price arithmetic and prompts the user to verify
# quality with --measure before switching models (honest-savings policy).
QUALITY_CAVEAT = (
    "Estimated from list prices against your logged token counts. "
    "Quality is not verified — run --measure to sample real outputs before you switch."
)


# Funnel line shown at the end of every report when a saving is present.
# Points to the automated paid layer without overselling.
FUNNEL_LINE = (
    "This is a one-time snapshot. Frugon can route every call automatically and hold the savings for you."
    " → https://frugon.rodiun.io"
)


# ---------------------------------------------------------------------------
# USD formatting — adaptive precision
# ---------------------------------------------------------------------------


def _fmt_usd(amount: Decimal) -> str:
    """Format a USD amount with adaptive decimal precision and ROUND_HALF_UP.

    Three tiers, ascending precision, so a real non-zero cost never prints as
    $0.00 at standard precision when it is genuinely sub-cent:

    * amounts < $0.0001  → 6 dp  (e.g. $0.000030)
    * amounts < $0.01    → 4 dp  (e.g. $0.0050)
    * all other amounts  → 2 dp  (e.g. $389.88)
    * zero               → 2 dp  ($0.00)
    """
    _TWO = Decimal("0.01")
    _FOUR = Decimal("0.0001")
    _SIX = Decimal("0.000001")
    if Decimal("0") < amount < _FOUR:
        return "$" + str(amount.quantize(_SIX, rounding=ROUND_HALF_UP))
    if Decimal("0") < amount < _TWO:
        return "$" + str(amount.quantize(_FOUR, rounding=ROUND_HALF_UP))
    return "$" + str(amount.quantize(_TWO, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Terminal renderer
# ---------------------------------------------------------------------------


def render_terminal(
    result: AnalysisResult,
    suppress_caveat: bool = False,
    verbose: bool = False,
    *,
    has_judge_section: bool = False,
    judged_models: frozenset[str] = frozenset(),
) -> None:
    """Print the cost analysis report to the terminal.

    Two headlines share ONE design language — a rounded cyan panel carrying the
    decision and its payoff, muted reconciliation lines, the same quiet footer:

      * the per-call **split** routing recommendation (the default), and
      * the **wholesale** single-model swap (``--wholesale`` / when no split is
        available).

    When ``suppress_caveat=True``, the quality caveat is omitted — used by
    ``analyze --measure`` so the measured-quality section replaces the "quality
    unverified" caveat.

    When ``verbose=True``, each view appends the supporting detail that the
    default (pared-down) view moves out of the way: the per-model cost table plus
    a Notes block (the split's easy/hard heuristic and wholesale upper-bound, or —
    on the wholesale view — a pointer back to the conservative split), and the
    automated-routing upsell.

    *judged_models* — the models this run will judge (the explicit ``--candidates``
    under ``--measure --judge``, whose quality verdict renders in the section
    below).  Threaded into the unrated-message family so a model verified by that
    verdict gets no contradictory "run --measure to confirm" caveat; empty for a
    cost-only run, which preserves the original wording.
    """

    # --- No priced calls guard (P3-4) ---
    if result.priced_calls == 0:
        rprint(
            Panel(
                "[yellow]No priced calls found.[/yellow]\n\n"
                f"Analyzed [bold]{result.total_calls:,}[/bold] records — "
                "none could be priced (unknown models or missing usage data).\n\n"
                "Run [bold]frugon pricing update[/bold] to refresh the pricing table, "
                "or check that your log records include a [dim]model[/dim] field.",
                title="[bold]frugon — no priced calls[/bold]",
                border_style="yellow",
            )
        )
        return

    # --- Per-call split routing is the headline when available ---
    if _has_split(result):
        assert result.split is not None  # narrowed by _has_split
        _render_split_terminal(
            result,
            result.split,
            suppress_caveat,
            verbose,
            has_judge_section=has_judge_section,
            judged_models=judged_models,
        )
        return

    # --- Wholesale single-model swap headline (same design language as split) ---
    _render_wholesale_terminal(
        result,
        suppress_caveat,
        verbose,
        has_judge_section=has_judge_section,
        judged_models=judged_models,
    )


def _render_cost_by_model_table(result: AnalysisResult) -> None:
    """Print the per-model cost breakdown as a borderless, column-aligned table.

    Borderless (``box=None``) by design: the framed decision panel is the only
    box on screen, so this breakdown sits quietly beneath it and lets monospace
    column alignment — not box-drawing — carry the structure (CLI redesign
    points 1, 3, 5).
    """
    if not result.cost_by_model:
        return
    table = Table(
        title="Cost by model",
        title_justify="left",
        title_style="dim",
        box=None,
        show_header=False,
        pad_edge=False,
        padding=(0, 2, 0, 0),
    )
    table.add_column("Model", style=BRAND_CYAN, no_wrap=True)
    table.add_column("Calls", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("% of total", justify="right", style="dim")

    for model, cost in sorted(result.cost_by_model.items(), key=lambda kv: kv[1], reverse=True):
        pct = (cost / result.total_cost * 100) if result.total_cost else Decimal("0")
        calls = result.calls_by_model.get(model, 0)
        table.add_row(model, f"{calls:,}", _fmt_usd(cost), f"{float(pct):.1f}%")

    rprint(Padding(table, (1, 0, 0, 2)))




# ---------------------------------------------------------------------------
# Multi-candidate "Candidates considered" block (terminal + MD + HTML v1/v2)
# ---------------------------------------------------------------------------
#
# The cost-projection headline picks ONE winner (the cheapest priced candidate
# that beat the baseline).  When the user passed multiple --candidates, the
# others were judged too — silently dropping them from the cost surface is
# dishonest, even though the measure/judge section already scored every
# candidate independently below.  These helpers render a small block surfacing
# every candidate (recommended / considered / more_expensive / unpriced) with
# its own projected monthly cost and saving% — and a one-line caption explaining
# why one was picked as the headline.  Fires only when
# len(result.candidate_projections) > 1, so the single-candidate path (and
# the no-candidate --demo path) stay byte-identical to before.

# Per-status label + style.  Kept in one map so the four labels read the same
# way on every surface (terminal label is rich-styled; MD/HTML use the plain
# text and surface-specific styling).
_CANDIDATE_STATUS_LABEL = {
    "recommended":   "recommended",
    "considered":    "considered",
    "more_expensive": "more expensive",
    "unpriced":      "unpriced",
}
def _candidate_caption(has_judge_section: bool) -> str:
    """Caption under the \"Candidates considered\" block.

    The first two sentences are invariant; the trailing clause is CONDITIONAL
    on whether a per-candidate judge tally is actually rendered below this
    block in the SAME output.  A cost-only report (no ``--measure --judge``)
    has no such section, so the caption must not point \"below\" at a section
    that does not exist — it offers the actionable command instead.

    The first sentence names the axis the pick is ACTUALLY made on — the
    biggest saving% — never "cheapest" (a dollar-column claim the recommended
    row can visibly contradict: a display-tied candidate can print a lower
    dollar figure than the recommended row and still lose, because the
    recommendation is picked on saving% with a quality tie-break, not on the
    dollar column alone — see :func:`frugon.cost._select_cheapest_eligible`).
    The second sentence states the quality tie-break rule (PD-ratified
    2026-07-02): when two rows print the SAME saving% at the precision shown,
    the higher quality tier wins — the Quality tier column is what proves the
    pick honest.

    Args:
        has_judge_section: True iff a Tier-1 judge tally (the per-candidate
            quality measurement) follows this block in the same surface.
    """
    base = (
        "Each candidate is shown under the same quality-preserving split (easy"
        " calls to the candidate, hard calls kept on baseline); the biggest"
        " saving is the headline recommendation, and when savings tie at the"
        " precision shown the higher quality tier wins."
    )
    if has_judge_section:
        return base + (
            " Each candidate is scored independently in the quality"
            " measurement below."
        )
    return base + " Run --measure --judge to score each candidate's quality."


def _candidate_cap_caption(result: AnalysisResult) -> str | None:
    """Cap caption for the DEFAULT-pool "Candidates considered" block, or None.

    The default pool (no explicit ``--candidates``) has up to 23 priced, rated
    models — showing all of them would bury the recommendation, so the block
    caps to the recommended candidate plus the next-4-cheapest that beat the
    baseline (5 rows max; see ``analyze_records``).  This one honest line, shown
    below the rows, tells the user the FULL pool was considered and how to see
    more — it never replaces the existing pool-source notice (OpenRouter /
    LiteLLM freshness line), which the CLI prints separately.

    Returns None when the cap does not apply — an explicit ``--candidates`` run,
    or a default-pool run where the block did not fire — so callers can skip the
    line entirely rather than print a caption for a block that isn't capped.

    Retained (unchanged) for the explicit ``--candidates`` path's caption
    behaviour and as the historical prose this function documents; the
    DEFAULT-pool surface now renders the header-line + bullet-legend built by
    :func:`_candidates_header_title` and :func:`_candidate_legend_lines`
    instead of this sentence (PD-directed 2026-07-03 "2+3 hybrid" restructure)
    — see those functions' docstrings.
    """
    if not result.used_default_pool:
        return None
    if len(result.candidate_projections) <= 1:
        return None
    pool_size = result.candidate_pool_size
    shown = len(result.candidate_projections)
    return (
        f"{pool_size} candidates considered — showing the recommended split and "
        f"the {shown - 1} next-cheapest. Pass --candidates to compare specific "
        "models."
    )


# ---------------------------------------------------------------------------
# Default-pool "2+3 hybrid" caption restructure (PD-directed 2026-07-03)
# ---------------------------------------------------------------------------
#
# The two-paragraph caption above (``_candidate_caption`` + the cap line from
# ``_candidate_cap_caption``) stays EXACTLY as-is for the explicit
# ``--candidates`` path — that surface never shows a capped pool, so the
# "cheapest split is the headline recommendation" prose is the whole story and
# ``test_candidate_caption.py`` pins it verbatim.
#
# The DEFAULT pool (no explicit ``--candidates``) instead gets a "2+3 hybrid":
# the pool/shown COUNTING moves into the table's header line (so the reader
# sees "23 in pool, top 5 shown" before a single row), and the two paragraphs
# of prose collapse into a three-bullet legend, one fact per line.  Both
# helpers below are used ONLY on the default-pool branch — callers gate on
# ``result.used_default_pool`` before reaching for them.


def _candidates_header_title(result: AnalysisResult) -> str:
    """Table title for the "Candidates considered" block.

    Explicit ``--candidates`` path: unchanged — plain "Candidates considered",
    no pool/cap claim (there is no cap on that path; every passed candidate is
    shown).

    Default-pool path: absorbs the counting that used to live in the trailing
    cap-caption sentence — "Candidates considered · N in pool · top M shown".
    Both counts are DERIVED, never hardcoded: the pool size comes from
    ``result.candidate_pool_size`` (the live ``_ROUTING_CANDIDATES`` length)
    and the shown count from the actual rendered row count
    (``len(result.candidate_projections)``), so a roster change or a
    fewer-than-5-row cap (not enough candidates beat the baseline) is
    reflected honestly — "top 3 shown" when only 3 rows render, never a
    hardcoded "top 5".
    """
    if not result.used_default_pool:
        return "Candidates considered"
    pool_size = result.candidate_pool_size
    shown = len(result.candidate_projections)
    return f"Candidates considered · {pool_size} in pool · top {shown} shown"


def _candidate_legend_lines(
    result: AnalysisResult, split: SplitRouting, *, has_judge_section: bool
) -> list[str]:
    """The default-pool block's 3-line bullet legend (replaces the old prose).

    One fact per line, each prefixed with the same ``·`` bullet the header
    line itself uses (visual consistency), dim-styled on every surface:

      1. What every row represents — the shared quality-preserving split,
         naming the actual baseline model so the claim is concrete, not
         "baseline" as a generic noun.
      2. The selection rule — biggest saving wins; the display-precision
         quality tie-break (PD-ratified 2026-07-02, unchanged from the old
         caption's second sentence).
      3. The two actionable follow-ups — compare specific models, or measure
         quality — conditional on whether a judge tally already follows this
         block in the same output (mirrors the old caption's "below"-honesty
         rule: never point at a section that is not actually there).

    Only used on the DEFAULT-pool path; the explicit ``--candidates`` surface
    keeps :func:`_candidate_caption`'s two-sentence prose unchanged.
    """
    line1 = (
        f"Each row is the same quality-preserving split — easy calls "
        f"→ candidate, hard calls kept on {split.baseline_model}"
    )
    line2 = (
        "Biggest saving wins; ties at the precision shown go to the "
        "higher quality tier"
    )
    if has_judge_section:
        line3 = (
            "Compare specific models with --candidates · each is "
            "scored independently in the quality measurement below"
        )
    else:
        line3 = (
            "Compare specific models with --candidates · score "
            "quality with --measure --judge"
        )
    return [line1, line2, line3]


def _fmt_candidate_saving(pct_val: Decimal) -> str:
    """Format a saving% for the candidates block.

    Positive pct_val means cheaper than baseline  -> "X.X% lower".
    Negative pct_val means more expensive          -> "X.X% higher".

    Rounds via :func:`frugon.cost._display_pct` — the SAME Decimal
    ROUND_HALF_UP quantizer the selector's display-precision tie-break reads
    (:func:`frugon.cost._select_cheapest_eligible`) — rather than Python's
    binary-float ``.1f`` (round-half-to-EVEN), which disagrees with
    ROUND_HALF_UP at exact .x5 boundaries (e.g. 37.25 -> "37.2" under
    round-half-even vs "37.3" under ROUND_HALF_UP).  Routing both the selector
    and every renderer through one shared quantizer is what makes the
    selector's tie-set PROVABLY the same set a reader sees printed — the whole
    point of the caption-truth invariant.
    """
    quantized = _display_pct(pct_val)
    if quantized >= 0:
        return f"{quantized}% lower"
    return f"{-quantized}% higher"


def _render_candidates_considered_terminal(
    result: AnalysisResult, *, has_judge_section: bool = False
) -> None:
    """Print the "Candidates considered" block beneath the cost panel.

    Borderless dim table (mirrors :func:) so it sits
    quietly under the framed decision/cost panel and above the
    Accounting/Quality-tier/Prices rows.  Fires whenever ``result`` carries
    more than one candidate projection — either an explicit ``--candidates``
    run with >1 model, or the default pool's capped transparency block (see
    ``analyze_records``) — and is a no-op on the single-candidate path, so
    that surface stays byte-identical.

    The header line and the caption below it differ by path (PD-directed
    2026-07-03 "2+3 hybrid" restructure): the default pool absorbs its
    pool/shown counts into the header title and swaps the old two-paragraph
    prose for a three-bullet legend (see :func:`_candidates_header_title` /
    :func:`_candidate_legend_lines`); the explicit ``--candidates`` path is
    untouched — plain title, :func:`_candidate_caption` prose, no cap line.
    """
    projs = result.candidate_projections
    if len(projs) <= 1:
        return

    table = Table(
        title=_candidates_header_title(result),
        title_justify="left",
        title_style="dim",
        box=None,
        show_header=False,
        pad_edge=False,
        padding=(0, 2, 0, 0),
    )
    table.add_column("Model", style=BRAND_CYAN, no_wrap=True)
    table.add_column("Monthly", justify="right")
    table.add_column("Vs. baseline", justify="right", style="dim")
    table.add_column("Quality tier", justify="left", style="dim")
    table.add_column("Status", justify="left")

    for proj in projs:
        # Money column: monthly when projection available, observed otherwise,
        # em-dash when unpriced (nothing to show).
        if proj.status == "unpriced":
            money = "—"
        elif proj.monthly_cost is not None:
            money = _fmt_usd(proj.monthly_cost) + " / mo"
        elif proj.observed_cost is not None:
            money = _fmt_usd(proj.observed_cost)
        else:  # pragma: no cover — defensive
            money = "—"

        # Saving% column.  Positive % when cheaper than baseline (matches the
        # hero's "you save N%"); negative when more expensive; em-dash when
        # unpriced or no figure.
        pct_val: Decimal | None
        if proj.saving_pct is not None:
            pct_val = proj.saving_pct
        elif proj.observed_saving_pct is not None:
            pct_val = proj.observed_saving_pct
        else:
            pct_val = None
        if pct_val is None:
            saving = "—"
        else:
            saving = _fmt_candidate_saving(pct_val)

        # Status tag style: recommended is cyan (matches the hero candidate
        # name); considered/more_expensive/unpriced stay dim.  Green is reserved
        # for the money win — never bled onto a status tag.
        label = _CANDIDATE_STATUS_LABEL[proj.status]
        if proj.status == "recommended":
            status_cell = Text(label, style=BRAND_CYAN)
        elif proj.status == "more_expensive":
            status_cell = Text(label, style="dim " + CAUTION_AMBER)
        else:
            status_cell = Text(label, style="dim")

        table.add_row(proj.model, money, saving, proj.tier_label, status_cell)

    rprint(Padding(table, (1, 0, 0, 2)))
    if result.used_default_pool and result.split is not None:
        # Default-pool "2+3 hybrid": the header already carries the pool/shown
        # counts, so the caption below is a plain 3-bullet legend — one fact
        # per line, no counting prose to repeat.  Printed via _print_hanging
        # (the split-footer precedent) rather than a raw Padding(Text(...)) so
        # a wrapped continuation on a narrow terminal hangs under the bullet
        # text at column 4 instead of falling back to the left margin — the
        # SAME hang-indent discipline every other footer line in this module
        # already gets.  MD/HTML render real list markup, which already hangs
        # natively, so this fix is terminal-only.
        for line in _candidate_legend_lines(
            result, result.split, has_judge_section=has_judge_section
        ):
            _print_hanging(
                Text(line, style="dim"),
                hang=4,
                prefix=Text("  · ", style="dim"),
            )
    else:
        # Explicit --candidates path — unchanged two-paragraph prose.
        rprint(
            Padding(
                Text(_candidate_caption(has_judge_section), style="dim"),
                (0, 0, 0, 2),
            )
        )
        cap_caption = _candidate_cap_caption(result)
        if cap_caption is not None:
            rprint(
                Padding(
                    Text(cap_caption, style="dim"),
                    (0, 0, 0, 2),
                )
            )


def _split_accounting(result: AnalysisResult, split: SplitRouting) -> tuple[int, list[str]]:
    """Reconcile every analyzed call: routed + kept + already-cheap == analyzed.

    The split only ever targets the dominant *baseline* model's calls (the
    routing target).  Calls already running on other, cheaper models are not the
    routing target — they are counted here as "already cheap" so no analyzed call
    silently vanishes (CLI redesign point 6 — transparent accounting).

    Returns ``(already_cheap_count, other_model_names)`` where the names are the
    non-baseline priced models, cheapest-cost-first, for the fine-print line.
    """
    routing_target = split.total_count  # routed + kept (the baseline calls)
    already_cheap = result.priced_calls - routing_target
    if already_cheap < 0:  # pragma: no cover — defensive; arithmetic guarantees >= 0
        already_cheap = 0
    other_models = sorted(
        (m for m in result.calls_by_model if m != split.baseline_model),
        key=lambda m: result.cost_by_model.get(m, Decimal("0")),
    )
    return already_cheap, other_models


def _render_split_terminal(
    result: AnalysisResult,
    split: SplitRouting,
    suppress_caveat: bool,
    verbose: bool = False,
    *,
    has_judge_section: bool = False,
    judged_models: frozenset[str] = frozenset(),
) -> None:
    """Print the per-call split-routing report — bordered panel + quiet detail.

    This is the headline ``frugon analyze --demo`` output.  The layout follows
    the CLI redesign brief: one framed panel carries the decision and its payoff;
    everything else sits quietly beneath it.

      1. Panel (rounded, cyan border) — three groups separated by blank lines:
           (a) summary  — analyzed calls + baseline, current monthly spend
           (b) decision — plain-English Route / Keep, with "within tolerance"
           (c) outcome  — blended monthly, then the SAVING hero (green, the
                          visual climax — green means the money win and nothing
                          else)
      2. Accounting + Prices — muted single lines that reconcile every call.
      3. Footer — the quality+tolerance caveat + privacy + one upsell line.

    Alignment and a single frame carry the structure.  Supporting detail (the
    per-model cost table, the wholesale upper bound, the easy/hard heuristic,
    verbose tier notes) moves to ``--verbose``.
    """
    already_cheap, other_models = _split_accounting(result, split)

    # ----- The framed decision panel (the one box on screen) -----------------
    _render_split_panel(result, split)

    # ----- Candidates considered (multi-candidate transparency) --------------
    # No-op when the user passed 0 or 1 --candidates (the headline already names
    # the chosen one).  Fires only on the multi-candidate surface — surfaces each
    # candidate the user passed with its projected monthly + status tag, so the
    # cost projection never silently drops a candidate the measure section is
    # still scoring below.
    _render_candidates_considered_terminal(
        result, has_judge_section=has_judge_section
    )

    # ----- Muted reconciliation lines ----------------------------------------
    # Thread *verbose* so the Upper-bound row's trailing hint points at the Notes
    # block (rendered below under --verbose) instead of re-suggesting --verbose.
    _render_split_accounting(
        result, split, already_cheap, other_models, verbose=verbose
    )

    # ----- Verbose-only supporting detail ------------------------------------
    # The per-model cost breakdown is detail, not headline: the default view is
    # the decision panel + accounting + footer.  The cost-by-model table moves
    # under --verbose with the rest of the supporting detail (CLI redesign
    # clarity pass).
    if verbose:
        _render_cost_by_model_table(result)
        # Blank line: the "Cost by model" table and the supporting notes below
        # (Upper bound / Method / Automate) are DISTINCT concerns — separate them
        # so the notes don't read as extra rows of the cost table.
        rprint("")
        _render_split_verbose(result, split)

    # ----- Quiet footer: caveats + privacy + one upsell ----------------------
    if not suppress_caveat:
        _render_split_footer(result, split, judged_models=judged_models)

    _data_quality_terminal(result)


# Width the label column is padded to inside the panel so every value starts in
# the same column — the monospace alignment that carries the sophistication.
# Wide enough for the longest label ("Current spend") so every value still
# starts in one column.
_PANEL_LABEL_WIDTH = 14


def _split_current_and_blended(
    result: AnalysisResult, split: SplitRouting
) -> tuple[Decimal, Decimal, bool]:
    """Return ``(current, blended, projected)`` on the FULL-dataset basis.

    The single source for the panel's Current/New-spend/SAVING figures AND any
    note that quotes the headline percentage — both derive from this helper so
    they can never disagree.  ``current`` is the TOTAL spend across every
    analyzed call/model; ``blended`` is that total after routing (only the
    baseline's easy calls move); ``projected`` is True when monthly projections
    are available (the figures are then monthly).
    """
    projected = (
        split.monthly_baseline is not None
        and split.monthly_blended is not None
        and result.monthly_cost is not None
    )
    if projected:
        assert result.monthly_cost is not None  # narrowed above
        assert split.monthly_baseline is not None
        assert split.monthly_blended is not None
        # Current = TOTAL monthly across ALL models (== sum of cost-by-model rows).
        current = result.monthly_cost
        # The routing reduction applies only to the baseline model's easy calls.
        baseline_reduction = split.monthly_baseline - split.monthly_blended
    else:
        current = result.total_cost
        baseline_reduction = split.baseline_cost - split.blended_cost
    # Blended TOTAL = the full current spend minus the baseline routing reduction;
    # every non-routed call (hard baseline + already-cheaper) carries through.
    return current, current - baseline_reduction, projected


@dataclass(frozen=True)
class _SplitReportFigures:
    """The full-dataset split figures every renderer (terminal + reports) shares.

    All money figures reconcile to the TOTAL analyzed dataset, never to the
    baseline model in isolation:

      * ``current``  — TOTAL spend across every analyzed call/model (== the sum
        of the Cost-by-model rows == ``monthly_cost`` when projected, else
        ``total_cost``).
      * ``blended``  — that TOTAL *after* routing: only the baseline's easy calls
        move; every hard-baseline and already-on-a-cheaper-model call is
        unchanged.
      * ``saved``    — ``current - blended`` (the routing win on the easy calls).
      * ``total_pct``— ``saved / current`` as a percent of the TOTAL current
        spend (NOT ``split.saving_pct``, which is baseline-only).
      * ``projected``— True when monthly projections are available (figures are
        then monthly); the unit suffix follows.
      * ``already_cheap`` / ``already_cheap_cost`` / ``other_models`` — the
        already-on-a-cheaper-model bucket so the routing plan reconciles to ALL
        analyzed calls (routed + kept + already-cheap == ``priced_calls``), not
        just the baseline routing target.

    Reports MUST derive every split headline/routing figure from this single
    source so they can never diverge from the terminal panel (the cross-surface
    correctness invariant).
    """

    current: Decimal
    blended: Decimal
    saved: Decimal
    total_pct: Decimal
    projected: bool
    already_cheap: int
    already_cheap_cost: Decimal
    other_models: list[str]


def _split_report_figures(
    result: AnalysisResult, split: SplitRouting
) -> _SplitReportFigures:
    """Compute the shared full-dataset split figures (see :class:`_SplitReportFigures`).

    This is the ONE arithmetic the terminal panel and all four report renderers
    consume, so a report figure can never contradict the terminal.  Current and
    blended come from :func:`_split_current_and_blended` (identical to the panel);
    the already-cheap bucket comes from :func:`_split_accounting` (identical to
    the panel's "already optimal" line and the Accounting reconciliation row).
    """
    current, blended, projected = _split_current_and_blended(result, split)
    # RECONCILIATION: round the two components to the displayed precision before
    # deriving the saving so that SAVING == Current − New is verifiable from the
    # printed figures.  The displayed precision for amounts ≥ $0.01 is 2 dp.
    _DP2 = Decimal("0.01")
    current = current.quantize(_DP2, rounding=ROUND_HALF_UP)
    blended = blended.quantize(_DP2, rounding=ROUND_HALF_UP)
    saved = current - blended
    # SAVING percent is honest over the TOTAL current spend (saved / current),
    # NOT the baseline-only split.saving_pct.  Guard the zero-total edge.
    total_pct = (saved / current * Decimal("100")) if current else Decimal("0")
    already_cheap, other_models = _split_accounting(result, split)
    # The already-cheap bucket's cost is the sum of the non-baseline (cheaper)
    # per-model costs — the exact rows that carry through routing unchanged.
    already_cheap_cost = sum(
        (result.cost_by_model.get(m, Decimal("0")) for m in other_models),
        Decimal("0"),
    )
    return _SplitReportFigures(
        current=current,
        blended=blended,
        saved=saved,
        total_pct=total_pct,
        projected=projected,
        already_cheap=already_cheap,
        already_cheap_cost=already_cheap_cost,
        other_models=other_models,
    )


def _reconciled_delta_pct(
    current: Decimal, projected: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Quantize *current* and *projected* to their _fmt_usd display precision, then
    derive the saving percent from the quantized values.

    Contract: the returned *pct* equals ``round(printed_save / printed_current * 100, 1)``
    for the dollar figures actually printed by ``_fmt_usd``, so the percent on screen
    is always verifiable from the adjacent Current and After numbers.

    Precision tiers mirror :func:`_fmt_usd`:
      * amount < $0.0001  → 6 dp
      * amount < $0.01    → 4 dp
      * otherwise         → 2 dp (including zero)

    Returns ``(cur_q, proj_q, pct)`` where *cur_q* and *proj_q* are the quantized
    amounts and *pct* is ``Decimal`` with 1 dp precision.  If *cur_q* rounds to zero
    (so no non-zero current cost is printed), *pct* is ``Decimal('0.0')``.
    """
    _TWO = Decimal("0.01")
    _FOUR = Decimal("0.0001")
    _SIX = Decimal("0.000001")

    def _precision(amount: Decimal) -> Decimal:
        if Decimal("0") < amount < _FOUR:
            return _SIX
        if Decimal("0") < amount < _TWO:
            return _FOUR
        return _TWO

    cur_q = current.quantize(_precision(current), rounding=ROUND_HALF_UP)
    proj_q = projected.quantize(_precision(projected), rounding=ROUND_HALF_UP)

    if cur_q == Decimal("0"):
        return cur_q, proj_q, Decimal("0.0")

    saved = cur_q - proj_q
    # Compute to full Decimal precision then round to 1 dp, matching the display
    # format ``f"{float(pct):.1f}%"`` used at every call site.
    pct = (saved / cur_q * Decimal("100")).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )
    return cur_q, proj_q, pct


def _call_share_pcts(counts: list[int]) -> list[float]:
    """Return each count's share of the total as one-decimal percents summing to 100.0.

    Used by the routing-plan tables to show what fraction of all analyzed calls
    each bucket (routed / kept / already-optimal) carries.  Naive per-bucket
    ``round(count/total*100, 1)`` does NOT generally sum to 100.0 (e.g. the demo's
    64.3 + 17.8 + 17.8 = 99.9), which would read as a reconciliation error next to
    a 100%-anchored total row.  Largest-remainder rounding fixes this: round every
    share DOWN to one decimal, then distribute the leftover tenths to the buckets
    with the largest fractional remainders, so the displayed figures sum EXACTLY
    to 100.0 while each stays within 0.1 of its true share.

    Returns a list of percentages (one per input count) in input order.  An empty
    input or a zero total yields all-zero shares (the caller then shows no bar).
    """
    total = sum(counts)
    if total <= 0:
        return [0.0] * len(counts)
    # Work in tenths-of-a-percent integers so the sum target is exactly 1000.
    exact = [c / total * 1000 for c in counts]
    floored = [int(x) for x in exact]
    remainder = 1000 - sum(floored)
    # Hand the leftover tenths to the largest fractional parts, tie-broken by the
    # larger bucket first (stable, deterministic) so the result is reproducible.
    order = sorted(
        range(len(counts)),
        key=lambda i: (exact[i] - floored[i], counts[i]),
        reverse=True,
    )
    for i in order[: max(0, remainder)]:
        floored[i] += 1
    return [tenths / 10 for tenths in floored]


def _render_split_panel(result: AnalysisResult, split: SplitRouting) -> None:
    """Render the single bordered panel: summary, decision, and the saving hero.

    Every figure in the panel reconciles to the FULL analyzed dataset, never to a
    single model in isolation:

      * ``Current``  is the TOTAL spend across every analyzed call/model — it is
        exactly the sum of the "Cost by model" rows beneath the panel.
      * ``Blended``  is that TOTAL *after* routing: the easy baseline calls move
        to the cheaper candidate; every other call (the hard baseline calls and
        the calls already on a cheaper model) is unchanged.
      * ``SAVING``   is ``Current - Blended`` (the routing win on the baseline's
        easy calls), shown as money and as a percent of the TOTAL current spend.

    The routing only ever touches the dominant baseline model's calls, so the
    routing reduction equals ``baseline_cost - blended_cost`` for that model; the
    already-on-a-cheaper-model calls are already optimal and carry through both
    sides unchanged.  This keeps "Current" honest as a total while the per-call
    Route/Keep lines describe exactly which calls move.

    Money figures are monthly when a projection is available (the honest, headline
    cadence a buyer cares about) and fall back to the observed totals otherwise.
    Green is reserved for the saving hero (and, softly, the blended "after"): on
    this screen green always means the money win.
    """
    # Consume the ONE shared figure source — the same arithmetic the reports use — so the
    # terminal panel can never diverge from them. current/blended/saved/total_pct are all
    # rounded-then-derived there, so SAVING == (Current - New) is guaranteed on screen.
    fig = _split_report_figures(result, split)
    current, blended, projected = fig.current, fig.blended, fig.projected
    suffix = " / mo" if projected else ""
    cadence = "monthly" if projected else "observed"

    # Per-bucket share of ALL analyzed calls (routed + kept + already-optimal),
    # at 1-dp via the shared :func:`_call_share_pcts` (largest-remainder, sums to
    # exactly 100.0).  This is the SAME denominator the Markdown and HTML reports
    # use — the panel and the reports tell one story (the earlier panel divided by
    # routed+kept only, which read as 78%/22% while the reports read 64.4/17.8/17.8
    # for the identical data).  The already-optimal bucket therefore also carries
    # its share here, not just a bare count.
    already_cheap, other_models = fig.already_cheap, fig.other_models
    _share_counts = [split.routed_count, split.kept_count]
    if already_cheap > 0:
        _share_counts.append(already_cheap)
    _shares = _call_share_pcts(_share_counts)
    routed_share = _shares[0]
    kept_share = _shares[1]
    already_share = _shares[2] if already_cheap > 0 and len(_shares) > 2 else 0.0
    # Right-justify the share percentages to a common width so the "→ candidate"
    # arrows line up across the Route and Keep rows.
    routed_pct = f"{routed_share:.1f}%"
    kept_pct = f"{kept_share:.1f}%"
    pct_width = max(len(routed_pct), len(kept_pct))
    routed_pct = routed_pct.rjust(pct_width)
    kept_pct = kept_pct.rjust(pct_width)

    def label(text: str) -> Text:
        return Text(text.ljust(_PANEL_LABEL_WIDTH), style="dim")

    body = Text()

    # --- (a) summary: what was looked at, and what it costs today ---
    body.append(label("Analyzed"))
    body.append(f"{result.total_calls:,} calls", style="bold")
    body.append("  ·  ", style="dim")
    body.append("baseline ", style="dim")
    body.append(split.baseline_model, style=BRAND_CYAN)
    body.append(" (your current model)", style="dim")
    if result.unpriced_calls:
        body.append(f"   ({result.unpriced_calls:,} unpriced)", style=CAUTION_AMBER)
    body.append("\n")
    body.append(label("Current spend"))
    body.append(f"{_fmt_usd(current)}{suffix}", style="bold")
    body.append("\n\n")

    # --- (b) decision: plain-English Route / Keep ---
    # Right-justify both counts to a common width so the "easy/hard calls" phrases
    # line up under each other — monospace alignment carries the structure.
    count_width = max(len(f"{split.routed_count:,}"), len(f"{split.kept_count:,}"))
    routed_str = f"{split.routed_count:,}".rjust(count_width)
    kept_str = f"{split.kept_count:,}".rjust(count_width)

    # Plain two-space indent (NOT a leading "→") so the arrow glyph only ever
    # means "to" — the model arrow in "easy calls  →  candidate".  Two meanings of
    # "→" on one line read as a contradiction (CLI redesign clarity pass).
    body.append("  ")
    body.append("Route ".ljust(7), style="bold")
    body.append(f"{routed_str} easy calls", style="bold")
    body.append(f" ({routed_pct})  →  ", style="dim")
    body.append(split.candidate_model, style=BRAND_CYAN)
    # Quality badge on the Route line — three distinct cases:
    #
    # 1. Candidate is UNRATED (no tier to claim) — omit any quality badge; the
    #    unrated caveat in the footer discloses the gap (Change 1).
    # 2. tier_drop <= 0 (same or better quality) — show "same or better quality"
    #    so the reader immediately sees this is a stronger claim than tolerance, not
    #    a risk.  The footer echoes this with the positive note (no risk caveat).
    # 3. tier_drop >= 1 (genuine step-down) — show "within tolerance" muted, never
    #    green, matching the existing band-estimate disclosure.
    if not _is_unrated(split.candidate_model):
        if _is_equal_or_better_quality(result):
            body.append("   same or better quality", style="dim")
        else:
            body.append("   within tolerance", style="dim")
    body.append("\n")
    body.append("  ")
    body.append("Keep ".ljust(7), style="bold")
    body.append(f"{kept_str} hard calls", style="bold")
    body.append(f" ({kept_pct})  →  ", style="dim")
    body.append(split.baseline_model, style=BRAND_CYAN)
    body.append("\n")

    # Already-on-a-cheaper-model calls are already optimal — no action.  Naming
    # them here (muted, never green) closes the loop so the reader never wonders
    # where the rest of the analyzed calls went: they reconcile in the panel, not
    # only in the fine print.
    if already_cheap > 0:
        already_pct = f"{already_share:.1f}%".rjust(pct_width)
        body.append("  ")
        body.append("Keep ".ljust(7), style="dim")
        body.append(f"{already_cheap:,} already on ", style="dim")
        body.append(
            ", ".join(other_models) if other_models else "a cheaper model",
            style=BRAND_CYAN,
        )
        body.append(f" ({already_pct})", style="dim")
        body.append("   already optimal — no action", style="dim")
        body.append("\n")
    body.append("\n")

    # --- (c) outcome: blended, then the saving hero ---
    body.append(label("New spend"))
    body.append(f"{_fmt_usd(blended)}{suffix}", style=SAVING_GREEN)
    body.append("\n\n")

    # saved + total_pct come from the shared source: saved == (Current - New) on the
    # rounded figures, and the percent stays honest over the TOTAL current spend — so the
    # printed SAVING reconciles with the printed Current and New.
    saved = fig.saved
    total_pct = fig.total_pct
    pct = f"{float(total_pct):.1f}%"
    body.append("SAVING".ljust(_PANEL_LABEL_WIDTH), style=f"bold {SAVING_GREEN}")
    body.append(f"{_fmt_usd(saved)}{suffix}", style=f"bold {SAVING_GREEN}")
    body.append("    ·    ", style="dim")
    body.append(f"{pct} lower", style=f"bold {SAVING_GREEN}")
    if not projected:
        body.append(f"   ({cadence})", style="dim")

    panel = Panel(
        body,
        title="[bold cyan]frugon · cost analysis[/bold cyan]",
        title_align="left",
        border_style=BRAND_CYAN,
        box=box.ROUNDED,
        padding=(1, 3),
    )
    rprint(panel)


# ---------------------------------------------------------------------------
# Responsive, margin-safe line rendering
# ---------------------------------------------------------------------------
#
# The terminal footer/accounting/caveat lines carry a deliberate left margin
# (the two- or four-column indent that visually offsets them from the report
# panel).  A long line that simply soft-wraps reflows its continuation back to
# column 0, bleeding past that margin and looking unpolished on a narrow
# terminal.  ``_print_hanging`` solves this the way responsive HTML does: it
# measures the live console width, wraps the text to fit, and re-indents every
# continuation row to the hang column so wrapped text aligns under the body —
# never into the margin — at any terminal width.
#
# Determinism: Rich's console is width-aware and falls back to a fixed 80 columns
# when stdout is not a TTY (piped, redirected, or under pytest's capture), so the
# wrap geometry — and therefore ``--demo`` / piped / test output — is byte-stable.


# The reconciliation rows ("Accounting", "Prices") use a fixed two-column left
# margin plus a 13-column label field, so the body always starts at column 15.
# Keeping these as constants means every row hangs under the same body column.
_LABEL_MARGIN = 2
_LABEL_FIELD_WIDTH = 13
_LABEL_HANG = _LABEL_MARGIN + _LABEL_FIELD_WIDTH  # body/continuation column

# The footer caveat/upsell lines hang at column 2: the marker ("⚠"/"→") sits in
# the left margin (column 0, matching the ✓ progress checkpoints) and the body
# plus every wrapped continuation align under it at column 2.
_FOOTER_HANG = 2


def _label_prefix(label: str) -> Text:
    """Build the dim, fixed-width first-row prefix for a reconciliation row.

    Returns ``"  " + label`` left-padded to the body column (``_LABEL_HANG``)
    so the body starts at a consistent column and columns align across rows.
    """
    return Text(
        " " * _LABEL_MARGIN + label.ljust(_LABEL_FIELD_WIDTH),
        style="dim",
    )


def _render_console() -> Console:
    """Return the console used to MEASURE + wrap the responsive terminal lines.

    This MUST be the very console the wrapped rows are emitted to, so the width
    used to wrap a line exactly matches the width it is printed at — otherwise a
    line wrapped for one width and printed at another would be re-wrapped by the
    receiving console, bleeding a trailing fragment back to column 0 (the margin
    bug this whole mechanism exists to prevent).

    For the CLI that console is Rich's global one: width-aware on a TTY (so the
    report reflows to the user's terminal — responsive, like HTML) and a fixed 80
    columns when stdout is not a TTY (Rich's own non-TTY default), which keeps
    piped / redirected / ``--demo`` / test output deterministic and byte-stable.
    Tests that drive the renderer through a custom console patch this function to
    return theirs, so measurement and emission still agree on one width.
    """
    return _get_console()


def _print_hanging(
    body: Text,
    *,
    hang: int,
    prefix: Text | None = None,
) -> None:
    """Print *body* width-reflowed with a hanging indent at column *hang*.

    The first visual row begins with *prefix* (a marker like ``⚠`` or a label
    field) drawn in the gutter; every continuation row is indented to *hang* so
    wrapped text lines up under the body and never wraps back into the left
    margin.  *prefix* is expected to be ``hang`` cells wide (its content plus
    enough trailing space to reach the hang column); the body then flows
    immediately after it on the first row and under it on the rest.

    The line is measured/wrapped at the active render width and each composed row
    is emitted via the module ``rprint`` hook, so a test that patches ``rprint``
    to its own console captures the wrapped rows exactly as the terminal shows
    them.  Styling is preserved — wrapping operates on the styled :class:`Text`,
    so each span keeps its colour across the break.
    """
    console = _render_console()
    avail = max(1, console.width - hang)
    wrapped = body.wrap(console, avail)

    gutter = Text(" " * hang)
    first_prefix = prefix if prefix is not None else gutter
    for index, segment in enumerate(wrapped):
        line = (first_prefix if index == 0 else gutter).copy()
        line.append_text(segment)
        # The composed row already fits the measured width, so emission is a
        # plain print — the row will not re-wrap.  ``rprint`` is the module hook
        # tests patch to capture output through their own console.
        rprint(line)


def _days_old(last_synced: str) -> int | None:
    """Return whole days between *last_synced* (ISO date) and today, or None.

    Used only to fill the staleness annotation's "<N> days old" figure; returns
    None when the stored date cannot be parsed so a malformed value never renders
    a nonsense count (the predicate has already gated whether the row is stale).
    """
    from datetime import date as _date

    try:
        return (_date.today() - _date.fromisoformat(last_synced)).days
    except ValueError:  # pragma: no cover — predicate already screens parse errors
        return None


def _print_synced_row(
    label: str,
    last_synced: str | None,
    *,
    stale: bool,
    refresh_command: str,
) -> None:
    """Render one freshness row ("Prices"/"Quality") in the Accounting block.

    The row mirrors the design language of the surrounding reconciliation rows: a
    dim, fixed-width label prefix (:func:`_label_prefix`) and a hanging-indented
    body (:func:`_print_hanging`).  The body is always the dim ``synced <date>``;
    when *stale* it carries an amber caution annotation naming the age and the
    cyan refresh command — colour discipline intact (amber = caution only, cyan =
    command, the date itself stays dim).  Renders nothing when no date exists.
    """
    if not last_synced:
        return
    body = Text(f"synced {last_synced}", style="dim")
    if stale:
        days = _days_old(last_synced)
        age = f"{days} days old" if days is not None else "out of date"
        body.append(f" — ⚠ {age}; refresh with ", style=CAUTION_AMBER)
        body.append(refresh_command, style=BRAND_CYAN)
    _print_hanging(body, hang=_LABEL_HANG, prefix=_label_prefix(label))


def _print_window_caution_row(result: AnalysisResult) -> None:
    """Render the amber Window caution when ``--window`` contradicts the real span.

    ``--window N`` overrides the monthly-projection basis (``total_cost × 30/N``),
    so passing a window that materially disagrees with the log's actual observed
    span silently scales the monthly figure (e.g. ``--window 7`` on a ~30-day log
    overstates it ~4.3×).  This row surfaces that contradiction in the Accounting
    block so the numbers — which are arithmetically correct but answering the wrong
    question — are never read as a clean monthly projection.

    Gated by :func:`window_contradicts_span`, so it renders ONLY when ``--window``
    was given, the span is known from timestamps, and the two materially disagree.
    When the window matches the span, or there are no timestamps, or ``--window``
    was absent, nothing is printed.

    Design language matches the Prices/Quality stale annotations exactly: a dim,
    fixed-width label prefix (:func:`_label_prefix`), a dim base body, and an amber
    caution.  The ``--window N`` token and the ``Drop --window`` hint carry a subtle
    cyan accent (consistent with command styling); the caution itself stays amber.
    Green is never used.  The body reflows with the shared hanging indent.
    """
    window = result.window_days
    span = result.observed_span_days
    if not window_contradicts_span(window, span):
        return
    # Narrowed by the predicate above (both are present and positive).
    assert window is not None
    assert span is not None

    span_days = round(span)
    body = Text("", style="dim")
    body.append("⚠ ", style=CAUTION_AMBER)
    body.append("--window ", style=BRAND_CYAN)
    body.append(f"{window}", style=BRAND_CYAN)
    body.append(
        f" overrides your log's actual ~{span_days}-day span — the monthly",
        style=CAUTION_AMBER,
    )
    body.append(
        " figure is projected as if the data covered ", style=CAUTION_AMBER
    )
    body.append(f"{window}", style=CAUTION_AMBER)
    body.append(" days. Drop ", style=CAUTION_AMBER)
    body.append("--window", style=BRAND_CYAN)
    body.append(" to project from the real span.", style=CAUTION_AMBER)
    _print_hanging(body, hang=_LABEL_HANG, prefix=_label_prefix("Window"))


def _render_freshness_rows(result: AnalysisResult) -> None:
    """Render the Prices + Quality freshness rows beneath the Accounting line.

    Shared by both the split and wholesale paths so the two surfaces never drift:
    each disclosure (pricing.json and quality.json last-synced) is its own labelled
    row, annotated amber when the table is stale (>30 days for prices, >60 for
    quality — tier tables drift slower).  A row is omitted entirely when its date
    is absent, so the Quality row simply does not appear on a table that predates
    the freshness stamp.

    A Window caution row follows the freshness rows whenever ``--window`` materially
    contradicts the log's observed span (:func:`_print_window_caution_row`), so the
    same warning appears consistently in BOTH the split and wholesale paths.
    """
    _print_synced_row(
        "Prices",
        result.pricing_json_last_synced,
        stale=_is_pricing_stale(result.pricing_json_last_synced, max_days=30),
        refresh_command="frugon pricing update",
    )
    _print_synced_row(
        "Quality",
        result.quality_json_last_synced,
        stale=_is_quality_stale(result.quality_json_last_synced, max_days=60),
        refresh_command="frugon quality update",
    )
    _print_window_caution_row(result)


def _render_split_accounting(
    result: AnalysisResult,
    split: SplitRouting,
    already_cheap: int,
    other_models: list[str],
    *,
    verbose: bool = False,
) -> None:
    """Muted reconciliation lines: every analyzed call accounted for, + prices.

    Each reconciliation row reflows to the console width with a hanging indent
    (see :func:`_label_line`) so a long accounting string never wraps back into
    the deliberate left margin.

    *verbose* is threaded to the Upper-bound row so its trailing hint can point at
    the Notes block (verbose, where the full breakdown is shown) instead of
    re-suggesting ``--verbose`` when it is already in force.
    """
    accounting_body = Text(
        f"{split.routed_count:,} routed + {split.kept_count:,} kept "
        f"({split.baseline_model}) + {already_cheap:,} already on cheaper "
        + (f"{', '.join(other_models)}" if other_models else "models")
        + f"  =  {result.priced_calls:,} analyzed",
        style="dim",
    )
    if result.unpriced_calls:
        accounting_body.append(
            f"  (+{result.unpriced_calls:,} unpriced, not in the figures)",
            style="dim",
        )

    # One blank line above the block (matching the previous top padding), then
    # each reconciliation row as its own width-aware, margin-safe line.  The
    # label sits at the two-column margin and the body hangs under its own start
    # column, so a long accounting string reflows under the body — never back
    # under the label or into the margin.
    rprint("")
    _print_hanging(accounting_body, hang=_LABEL_HANG, prefix=_label_prefix("Accounting"))
    # Order: reconciliation -> Upper bound (decision) -> Quality tier -> Prices ->
    # Quality -> window caution.  The Upper-bound row and the Quality-tier row are
    # the decision context (what the swap costs and which benchmark class it moves
    # between), so they sit together ABOVE the freshness metadata; the freshness
    # rows (last-synced dates) then follow, ending with the window caution (see
    # _render_freshness_rows).
    _render_split_upper_bound_row(result, split, detail_shown=verbose)
    # The Quality-tier comparison rides directly under the Upper-bound swap context
    # — the two together describe the aggressive full-swap and the benchmark class
    # it crosses — and ABOVE the freshness (last-synced) metadata.
    _render_split_quality_tier_row(result, split)
    _render_freshness_rows(result)


def _render_split_quality_tier_row(result: AnalysisResult, split: SplitRouting) -> None:
    """One dim 'Quality tier' row comparing the baseline and candidate tiers.

    Shows the published LMArena quality CLASS the routing moves between, e.g.

        Quality tier   gpt-4-turbo: unrated  →  gpt-4o-mini: Capable   (LMArena)

    This is the third quality signal in the report, distinct from its neighbours:
    "within tolerance" is the offline HEURISTIC (panel), ``--measure`` is the
    MEASURED verdict (on the user's own outputs), and this tier is the published
    BENCHMARK class — a coarse, external sanity check on the routing's quality
    step.  Model names are cyan; the tier labels and the "(LMArena)" source tag
    stay dim/neutral (green is reserved for the money win).  An unrated model
    reads ``unrated`` rather than being omitted, so the gap is marked, not hidden.

    Uses the shared :func:`_tier_label` so the tier shown here reconciles with the
    HTML and Markdown surfaces.  Mirrors the hang-indent design language of the
    Accounting / Upper bound / freshness rows (:func:`_label_prefix` +
    :func:`_print_hanging`).
    """
    body = Text()
    body.append(split.baseline_model, style=BRAND_CYAN)
    body.append(f": {_tier_label(split.baseline_model)}", style="dim")
    body.append("  →  ", style="dim")
    body.append(split.candidate_model, style=BRAND_CYAN)
    body.append(f": {_tier_label(split.candidate_model)}", style="dim")
    body.append("   (LMArena)", style="dim")
    _print_hanging(body, hang=_LABEL_HANG, prefix=_label_prefix("Quality tier"))


def _render_split_upper_bound_row(
    result: AnalysisResult, split: SplitRouting, *, detail_shown: bool = False
) -> None:
    """One dim Upper-bound row in the DEFAULT split Accounting block.

    Surfaces the range the conservative split headline lives inside: a full swap to
    the wholesale candidate saves materially more (~70% on the demo vs the 34%
    split headline), and the verbose note carries the full aggressive-vs-conservative
    explanation.  Kept to ONE dim line — only the model name is cyan, there is no
    second green number (green stays reserved for the saving hero).

    The trailing hint is context-aware (*detail_shown*).  When the full
    aggressive-vs-conservative breakdown is NOT on screen — the default,
    non-verbose terminal view — it reads ``run with --verbose for detail``.  When
    that breakdown IS already present — under ``--verbose``, where
    :func:`_render_split_verbose` renders the full Upper-bound note in the Notes
    block below — it instead reads ``see notes below for detail`` so the hint
    points at detail that exists rather than re-suggesting a flag already in force.

    Rendered under the SAME guard as the verbose Upper-bound note and gated on the SAME
    reconciled percent (:func:`_split_report_figures` ``.total_pct`` — the single source
    the panel hero and the verbose note also read), so this default-view hint, the verbose
    note, and the wholesale panel can never disagree.
    """
    # The full-swap upper bound is surfaced even when the wholesale winner is the
    # SAME model as the split's easy-call target (audit finding #2): the split
    # moves only the easy baseline calls while the full swap moves every call, so
    # the figures differ and the user must see the aggressive basis.  The
    # ``wholesale_saving <= split_total_pct`` check below keeps it honest and
    # non-redundant (no upper-bound row when the split already moves the whole
    # dataset and the two savings coincide).
    if not result.candidate_model:
        return
    # Non-displayed use: compute_saving_pct is intentional here.  This value is used
    # only as context text ("a full swap saves ~X%") — it is NOT printed adjacent to
    # a Current / After dollar pair that the user could verify by subtraction.  The
    # reconciling rounding is therefore not required; raw precision is the correct
    # behaviour for a standalone informational upper-bound estimate.
    wholesale_saving = compute_saving_pct(result.total_cost, result.projected_cost)
    # Gate on the SAME reconciled percent the verbose Upper-bound note and the panel hero
    # use (`_split_report_figures(...).total_pct`) — so this default-view hint and the
    # verbose note appear/disappear in lockstep; no raw re-derivation can drift the gate.
    split_total_pct = _split_report_figures(result, split).total_pct
    if wholesale_saving is None or wholesale_saving <= split_total_pct:
        return
    hint = "see notes below for detail" if detail_shown else "run with --verbose for detail"
    upper = Text("a full swap to ", style="dim")
    upper.append(result.candidate_model, style=BRAND_CYAN)
    upper.append(
        f" saves ~{float(wholesale_saving):.1f}% — {hint}",
        style="dim",
    )
    _print_hanging(upper, hang=_LABEL_HANG, prefix=_label_prefix("Upper bound"))


def _render_footer_core(
    tier_notes: list[Text],
    assertion: str | None = QUALITY_NOT_VERIFIED_ASSERTION,
    quality_note: str | None = None,
) -> None:
    """Render the shared quiet footer — the two caveats, the upsell, one trailing blank.

    Both the split and the wholesale headline close with the same footer, so the
    copy strings (the quality-not-verified caveat, the privacy line, the upsell)
    live in one place and are never duplicated across the two paths.

    *assertion* is the first (amber) caveat line.  It defaults to the split's
    "'within tolerance' is an offline estimate" wording; the wholesale path passes
    its own assertion (a full swap can change quality) since it has no tolerance
    band.  The call-to-action second line is shared.  Pass ``None`` to suppress
    the quality-risk block entirely — used when the candidate is rated same or
    better quality than the baseline (no downgrade to caution against).

    *quality_note* is an optional dim note rendered instead of the quality-risk
    block when *assertion* is None.  Typically the :data:`QUALITY_EQUAL_OR_BETTER`
    string confirming the candidate is a quality-neutral or quality-improving move.

    *tier_notes* are zero or more amber lines that sit on their OWN lines directly
    beneath the quality caveat — a distinct, scannable fact (e.g. "<model> has no
    known quality tier."), never a phrase trailing/wrapping the caveat sentence.
    The split path passes the unrated-baseline note; the wholesale path passes the
    unrated baseline AND/OR candidate notes.
    """
    # One blank line above the footer block; each caveat/upsell line is emitted
    # independently so it reflows to the console width with its own hanging indent.
    rprint("")

    if assertion is not None:
        # Caveat 1 — quality is not verified (amber, the one real caution).  The
        # assertion and the "run --measure …" call to action are TWO deliberate
        # lines: the break is forced, not left to soft-wrap, so the instruction
        # always reads as a distinct second line.  Each half hangs under the caveat
        # text (column 4) on a narrow terminal rather than bleeding into the margin.
        _print_hanging(
            Text(assertion, style=CAUTION_AMBER),
            hang=_FOOTER_HANG,
            prefix=Text("⚠ ", style=CAUTION_AMBER),
        )
        _print_hanging(
            Text(QUALITY_NOT_VERIFIED_ACTION, style=CAUTION_AMBER),
            hang=_FOOTER_HANG,
        )
    elif quality_note is not None:
        # Positive quality note — candidate is same or better quality.  Dim (not
        # amber) because there is no caution to convey; the "✓" prefix signals a
        # positive fact rather than a warning.
        _print_hanging(
            Text(quality_note, style="dim"),
            hang=_FOOTER_HANG,
            prefix=Text("✓ ", style="dim"),
        )

    # Each tier note is its OWN line — aligned under the caveat text, never
    # trailing the sentence above (a tidy, scannable footer).
    for note in tier_notes:
        _print_hanging(note, hang=_FOOTER_HANG)

    # Blank line: separate the quality-caveat group (assertion / action / tier
    # note) from the privacy statement — two distinct messages, breathing room.
    rprint("")

    # Caveat 2 — privacy (muted, confident).
    _print_hanging(Text(SPLIT_FOOTER_PRIVACY_LINE, style="dim"), hang=_FOOTER_HANG)

    # One upsell line → the product (cyan link).  The pitch and the URL share a
    # logical line that hangs under the text after the "→" gutter on wrap.
    upsell = Text("Route every call automatically and hold the savings:  ", style="dim")
    upsell.append(FUNNEL_URL, style=BRAND_CYAN)
    _print_hanging(upsell, hang=_FOOTER_HANG, prefix=Text("→ ", style="dim"))

    # A single trailing blank so the plain `analyze` output ends with exactly one
    # empty line before the shell prompt — matching the --measure path's footer.
    rprint("")


def _render_split_footer(
    result: AnalysisResult,
    split: SplitRouting,
    *,
    judged_models: frozenset[str] = frozenset(),
) -> None:
    """The quiet footer for the split headline — delegates to the shared core.

    When the baseline model has no known quality tier, that note sits on its own
    line directly beneath the quality caveat.
    """
    tier_notes: list[Text] = []
    if result.baseline_is_unrated:
        tier_notes.append(
            Text(f"{split.baseline_model} has no known quality tier.", style=CAUTION_AMBER)
        )
    # Findings #1 + #4 — the shared unrated-message family (recommendation caveat
    # and held-out notes); the same strings render in the MD and HTML reports so
    # the surfaces never drift.  *judged_models* makes the family
    # measurement-aware (see :func:`_unrated_family_messages`), and each line's
    # severity decides its colour: a genuine "quality unverified" caution reads
    # amber, a measured-below informational note reads dim.
    for message, severity in _unrated_family_messages(result, judged_models):
        tier_notes.append(Text(message, style=_SEV_TERMINAL_STYLE[severity]))
    # Select the footer quality block:
    #
    # 1. tier_drop <= 0 (same or better quality) — suppress the risk caveat entirely
    #    and show the positive "same or better quality" note instead.  No downgrade
    #    to disclose; the "verify before you switch" framing would be dishonest.
    #
    # 2. Candidate is UNRATED — the "within tolerance" band reference would be a
    #    phantom (Change 1); use the band-free wholesale assertion that names the
    #    quality change directly instead.  The unrated caveat in tier_notes carries
    #    the specific gap disclosure.
    #
    # 3. Default (tier_drop >= 1) — genuine quality step-down; show the standard
    #    "within tolerance is an offline estimate" risk caveat unchanged.
    if _is_equal_or_better_quality(result):
        assertion: str | None = None
        quality_note: str | None = QUALITY_EQUAL_OR_BETTER
    elif _is_unrated(split.candidate_model):
        assertion = QUALITY_NOT_VERIFIED_ASSERTION_WHOLESALE
        quality_note = None
    else:
        assertion = QUALITY_NOT_VERIFIED_ASSERTION
        quality_note = None
    _render_footer_core(tier_notes, assertion=assertion, quality_note=quality_note)


def _render_log_span_row(result: AnalysisResult) -> None:
    """Render the observed log time span as a labelled verbose row.

    Dim, fixed-label-column row in the same design language as the other Notes
    rows (:func:`_label_prefix` + :func:`_print_hanging`).  Emitted only when
    both span dates are present (>= 2 parseable timestamps in the log); silent
    otherwise, so a timestamp-free log never shows an empty or partial span.
    The figure echoes the span the monthly projection is computed from, so the
    disclosed window and the projection always agree.
    """
    start = result.observed_span_start
    end = result.observed_span_end
    if start is None or end is None:
        return
    body = Text(f"{start} → {end}", style="dim")
    if result.observed_span_days is not None:
        body.append(f" ({result.observed_span_days:.1f} days)", style="dim")
    _print_hanging(body, hang=_LABEL_HANG, prefix=_label_prefix("Log span"))


def _render_split_verbose(result: AnalysisResult, split: SplitRouting) -> None:
    """Verbose-only supporting detail moved out of the default split view.

    Carries the wholesale upper-bound, the easy/hard heuristic explanation, and
    the automated-routing upsell — the material the pared-down default view sends
    here so the headline reads confident, not defensive (CLI redesign point 3).
    """
    # Group header — mirrors the "Cost by model" table title so the notes read
    # as their own labelled section, not orphaned rows trailing the table.
    rprint(Text("  Notes", style="dim"))

    # Each note is a labelled row in the same fixed label column as the
    # Accounting/Prices block, wrapping with a hanging indent under the body —
    # continuations never bleed back to the left margin.

    # Order (Item 7): Upper bound -> Log span -> Method -> Automate.  The
    # Upper-bound decision note leads; the Log-span disclosure follows it.
    # Wholesale upper-bound — the larger, less-conservative full-swap figure.
    # The quoted split percentage IS the panel hero's percent: both read
    # `_split_report_figures(...).total_pct` (total-dataset basis, derived from the
    # rounded Current/New), so "the X% split above" always echoes the headline exactly —
    # no second, independently-rounded derivation can drift from it.
    # Surfaced even when the wholesale winner == the split's easy-call target
    # (audit finding #2): full swap moves every call, the split moves only the
    # easy baseline calls, so the figures differ; the strictly-greater check below
    # keeps it non-redundant when the two coincide (100%-routed wholesale case).
    if result.candidate_model:
        # Non-displayed use: compute_saving_pct is intentional here.  This value is used
        # only as context text ("a full swap saves ~X%") — it is NOT printed adjacent to
        # a Current / After dollar pair that the user could verify by subtraction.
        wholesale_saving = compute_saving_pct(result.total_cost, result.projected_cost)
        split_total_pct = _split_report_figures(result, split).total_pct
        if wholesale_saving is not None and wholesale_saving > split_total_pct:
            upper = Text("moving every call to ", style="dim")
            upper.append(result.candidate_model, style=BRAND_CYAN)
            upper.append(
                f" saves ~{float(wholesale_saving):.1f}% — the aggressive end; "
                f"the {float(split_total_pct):.1f}% split above is the conservative, "
                "quality-respecting recommendation. Quality-check the full swap: ",
                style="dim",
            )
            upper.append(
                f"--wholesale --measure --candidates {result.candidate_model}",
                style=BRAND_CYAN,
            )
            _print_hanging(upper, hang=_LABEL_HANG, prefix=_label_prefix("Upper bound"))

    # Observed log span — the window the projection is computed from (after the
    # Upper-bound decision note per Item 7).
    _render_log_span_row(result)

    # Easy/hard heuristic explanation.
    _print_hanging(
        Text(
            "easy/hard is a local heuristic over prompt and completion length "
            "(RouteLLM-style), computed offline with no LLM calls",
            style="dim",
        ),
        hang=_LABEL_HANG,
        prefix=_label_prefix("Method"),
    )

    # Automated-routing upsell.
    automate = Text(
        "frugon can route every call automatically and hold the savings for you → ",
        style="dim",
    )
    automate.append("https://frugon.rodiun.io", style=BRAND_CYAN)
    _print_hanging(automate, hang=_LABEL_HANG, prefix=_label_prefix("Automate"))

    rprint("")


# ---------------------------------------------------------------------------
# Wholesale (single-model swap) headline — same design language as the split
# ---------------------------------------------------------------------------
#
# The wholesale path moves EVERY call to one candidate model.  It shares the
# split's visual language exactly — one rounded cyan panel carrying the decision
# and its payoff, muted reconciliation lines, the shared quiet footer — so the
# tool never shows two design languages.  The only structural differences reflect
# the different recommendation: a single "Swap … (full swap)" line instead of the
# Route/Keep split, and an accounting line that separates the calls that move from
# the calls already on the candidate (already optimal, unchanged by the swap).


def _wholesale_current_and_new(result: AnalysisResult) -> tuple[Decimal, Decimal, bool]:
    """Return ``(current, new, projected)`` on the FULL-dataset basis.

    The single source for the wholesale panel's Current/New/SAVING figures so they
    can never disagree.  ``current`` is the TOTAL spend across every analyzed
    call/model; ``new`` is that total after a full swap (every priced call costed
    at the candidate's list price — the calls already on the candidate carry
    through unchanged, the rest move); ``projected`` is True when monthly
    projections are available (the figures are then monthly).

    Both figures reconcile to the full dataset exactly like the split panel:
    ``current`` is the sum of the "Cost by model" rows, and ``new`` is the
    analysis engine's full-dataset candidate projection (``projected_cost``).
    """
    projected = result.monthly_cost is not None and result.monthly_projected is not None
    if projected:
        assert result.monthly_cost is not None  # narrowed above
        assert result.monthly_projected is not None
        return result.monthly_cost, result.monthly_projected, True
    return result.total_cost, result.projected_cost, False


def _wholesale_accounting(result: AnalysisResult) -> tuple[int, int, list[str]]:
    """Reconcile every analyzed call for a full swap: swapped + already-on == priced.

    A wholesale swap moves every priced call to the single candidate.  Calls
    already running on that candidate model do not move — they are already on the
    target — so they are counted separately as "already on <candidate>" and the
    rest are the swapped calls.  Returns ``(swapped, already_on_candidate,
    swapped_model_names)`` where the names are the priced models other than the
    candidate, costliest-first (the swap is dominated by the most expensive one).
    """
    already_on_candidate = (
        result.calls_by_model.get(result.candidate_model, 0)
        if result.candidate_model
        else 0
    )
    swapped = result.priced_calls - already_on_candidate
    if swapped < 0:  # pragma: no cover — defensive; arithmetic guarantees >= 0
        swapped = 0
    swapped_models = sorted(
        (m for m in result.calls_by_model if m != result.candidate_model),
        key=lambda m: result.cost_by_model.get(m, Decimal("0")),
        reverse=True,
    )
    return swapped, already_on_candidate, swapped_models


def _render_wholesale_terminal(
    result: AnalysisResult,
    suppress_caveat: bool,
    verbose: bool = False,
    *,
    has_judge_section: bool = False,
    judged_models: frozenset[str] = frozenset(),
) -> None:
    """Print the wholesale single-model-swap report in the split design language.

    Mirrors :func:`_render_split_terminal` exactly: one bordered panel carries the
    decision and its payoff, muted reconciliation lines reconcile every call, the
    shared quiet footer carries the caveats + privacy + one upsell.  Supporting
    detail (the per-model cost table, the conservative-split pointer, the method
    note) moves under ``--verbose``.
    """
    has_recommendation = result.candidate_model is not None

    _render_wholesale_panel(result)

    # ----- Candidates considered (multi-candidate transparency) --------------
    # No-op when the user passed 0 or 1 --candidates.  See split-path comment.
    _render_candidates_considered_terminal(
        result, has_judge_section=has_judge_section
    )

    _render_wholesale_accounting(result)

    if verbose:
        _render_cost_by_model_table(result)
        rprint("")
        _render_wholesale_verbose(result)

    # The footer carries the quality caveat + the automated-routing upsell, both
    # of which only make sense when there IS a recommendation to switch to.  With
    # no cheaper candidate there is nothing to verify and nothing to upsell, so
    # the footer is suppressed — the panel honestly says "no cheaper candidate".
    if has_recommendation and not suppress_caveat:
        _render_wholesale_footer(result, judged_models=judged_models)

    _data_quality_terminal(result)


def _render_wholesale_panel(result: AnalysisResult) -> None:
    """Render the single bordered panel: summary, the swap decision, the saving hero.

    Every figure reconciles to the FULL analyzed dataset, never to one model in
    isolation: ``Current spend`` is the sum of the "Cost by model" rows; ``New
    spend`` is that total after the full swap; ``SAVING`` is ``Current − New``,
    shown as money and as a percent of the total current spend.
    """
    current, new, projected = _wholesale_current_and_new(result)
    suffix = " / mo" if projected else ""
    cadence = "monthly" if projected else "observed"

    dominant = _dominant_model(result)

    def label(text: str) -> Text:
        return Text(text.ljust(_PANEL_LABEL_WIDTH), style="dim")

    body = Text()

    # --- (a) summary: what was looked at, and what it costs today ---
    body.append(label("Analyzed"))
    body.append(f"{result.total_calls:,} calls", style="bold")
    if dominant:
        body.append("  ·  ", style="dim")
        body.append("baseline ", style="dim")
        body.append(dominant, style=BRAND_CYAN)
        body.append(" (your current model)", style="dim")
    if result.unpriced_calls:
        body.append(f"   ({result.unpriced_calls:,} unpriced)", style=CAUTION_AMBER)
    body.append("\n")
    body.append(label("Current spend"))
    body.append(f"{_fmt_usd(current)}{suffix}", style="bold")

    # No cheaper candidate found — there is no swap to recommend.  Say so honestly
    # (muted, never green) and stop: no phantom "New spend $0 / SAVING 100%" line,
    # which a None candidate's zero projected_cost would otherwise produce.
    if not result.candidate_model:
        body.append("\n\n")
        body.append("  ")
        body.append("no cheaper candidate found", style="dim")
        panel = Panel(
            body,
            title="[bold cyan]frugon · cost analysis[/bold cyan]",
            title_align="left",
            border_style=BRAND_CYAN,
            box=box.ROUNDED,
            padding=(1, 3),
        )
        rprint(panel)
        return
    body.append("\n\n")

    # --- (b) decision: the full swap, in plain English ---
    # Two-space indent (NOT a leading "→") so the "→" glyph only ever means "to" —
    # the model arrow in "every call  →  candidate" (same rule as the split panel).
    body.append("  ")
    body.append("Swap ".ljust(7), style="bold")
    body.append("every call", style="bold")
    body.append("  →  ", style="dim")
    body.append(result.candidate_model, style=BRAND_CYAN)
    body.append("   (full swap)", style="dim")
    body.append("\n")

    # Already-on-the-candidate calls do not move — naming them here (muted, never
    # green) closes the loop so the reader never wonders where the rest went.
    _swapped, already_on_candidate, _swapped_models = _wholesale_accounting(result)
    if already_on_candidate > 0:
        body.append("  ")
        body.append("Keep ".ljust(7), style="dim")
        body.append(f"{already_on_candidate:,} already on ", style="dim")
        body.append(result.candidate_model, style=BRAND_CYAN)
        body.append("   already on target — no change", style="dim")
        body.append("\n")
    body.append("\n")

    # --- (c) outcome: new spend, then the saving hero ---
    body.append(label("New spend"))
    body.append(f"{_fmt_usd(new)}{suffix}", style=SAVING_GREEN)
    body.append("\n\n")

    # RECONCILIATION: round the displayed components first so SAVING == Current − New
    # is verifiable from the printed figures.
    _DP2 = Decimal("0.01")
    current = current.quantize(_DP2, rounding=ROUND_HALF_UP)
    new = new.quantize(_DP2, rounding=ROUND_HALF_UP)
    saved = current - new
    # SAVING percent is honest over the TOTAL current spend (saved / current) so
    # every figure in the panel reconciles to the full dataset.  Guard the
    # zero-total edge so a degenerate fixture cannot divide by zero.
    total_pct = (saved / current * Decimal("100")) if current else Decimal("0")
    pct = f"{float(total_pct):.1f}%"
    body.append("SAVING".ljust(_PANEL_LABEL_WIDTH), style=f"bold {SAVING_GREEN}")
    body.append(f"{_fmt_usd(saved)}{suffix}", style=f"bold {SAVING_GREEN}")
    body.append("    ·    ", style="dim")
    body.append(f"{pct} lower", style=f"bold {SAVING_GREEN}")
    if not projected:
        body.append(f"   ({cadence})", style="dim")

    panel = Panel(
        body,
        title="[bold cyan]frugon · cost analysis[/bold cyan]",
        title_align="left",
        border_style=BRAND_CYAN,
        box=box.ROUNDED,
        padding=(1, 3),
    )
    rprint(panel)


def _render_wholesale_accounting(result: AnalysisResult) -> None:
    """Muted reconciliation lines: every analyzed call accounted for, + prices.

    Reconciles ``swapped + already-on-candidate == analyzed`` so no analyzed call
    silently vanishes — the same transparent-accounting discipline as the split
    path, reflowed to the console width with a hanging indent.
    """
    swapped, already_on_candidate, swapped_models = _wholesale_accounting(result)
    candidate = result.candidate_model

    accounting_body = Text(style="dim")
    if candidate and already_on_candidate > 0:
        accounting_body.append(
            f"{swapped:,} swapped "
            + (f"({', '.join(swapped_models)})" if swapped_models else "")
            + f" + {already_on_candidate:,} already on {candidate}"
            f"  =  {result.priced_calls:,} analyzed",
            style="dim",
        )
    else:
        # No calls already on the candidate (or no candidate): every priced call
        # is a swap, so the reconciliation is the single analyzed total.
        accounting_body.append(
            f"{result.priced_calls:,} analyzed"
            + (f", all swapped to {candidate}" if candidate else ""),
            style="dim",
        )
    if result.unpriced_calls:
        accounting_body.append(
            f"  (+{result.unpriced_calls:,} unpriced, not in the figures)",
            style="dim",
        )

    rprint("")
    _print_hanging(accounting_body, hang=_LABEL_HANG, prefix=_label_prefix("Accounting"))
    _render_freshness_rows(result)


def _render_wholesale_footer(
    result: AnalysisResult,
    *,
    judged_models: frozenset[str] = frozenset(),
) -> None:
    """The quiet footer for the wholesale headline — delegates to the shared core.

    Folds the unrated-baseline and unrated-candidate cautions into the footer
    caveat area (each on its own line beneath the quality caveat), the same way
    the split footer folds its unrated-baseline note — so the "quality tier
    unknown …" cautions no longer float mid-output.
    """
    dominant = _dominant_model(result)
    tier_notes: list[Text] = []
    if result.baseline_is_unrated and dominant:
        tier_notes.append(Text(f"{dominant} has no known quality tier.", style=CAUTION_AMBER))

    # Findings #1 + #4 — the shared unrated-message family.  When it names the
    # recommended candidate (the #1 caveat carries the model + --measure command),
    # it SUPERSEDES the bare "has no known quality tier." note for that candidate;
    # we fall back to the bare note only when the family is silent about it (e.g.
    # the candidate is unrated but not the recommendation — never the wholesale
    # headline, where the candidate IS the recommendation).
    family = _unrated_family_messages(result, judged_models)
    for message, severity in family:
        tier_notes.append(Text(message, style=_SEV_TERMINAL_STYLE[severity]))
    candidate_named = _recommended_unrated_model(result) == result.candidate_model
    if (
        not candidate_named
        and result.candidate_is_unrated
        and result.candidate_model
    ):
        tier_notes.append(
            Text(f"{result.candidate_model} has no known quality tier.", style=CAUTION_AMBER)
        )
    _render_footer_core(tier_notes, assertion=QUALITY_NOT_VERIFIED_ASSERTION_WHOLESALE)


def _render_wholesale_verbose(result: AnalysisResult) -> None:
    """Verbose-only supporting detail for the wholesale headline.

    Wholesale IS the upper bound (every call swapped), so there is no "Upper
    bound" note here — instead a pointer BACK to the conservative split, which
    routes only the easy calls and keeps quality changes small (the default view).
    The method note (the offline list-price arithmetic) applies and is kept.
    """
    rprint(Text("  Notes", style="dim"))

    # Observed log span — the window the projection is computed from.
    _render_log_span_row(result)

    # Point back to the conservative default — wholesale is the aggressive end, so
    # the honest counterweight is the split that keeps quality changes small.
    if result.candidate_model:
        split_note = Text("routing only the easy calls to ", style="dim")
        split_note.append(result.candidate_model, style=BRAND_CYAN)
        split_note.append(
            " keeps quality changes small — the default view (drop ", style="dim"
        )
        split_note.append("--wholesale", style=BRAND_CYAN)
        split_note.append(")", style="dim")
        _print_hanging(split_note, hang=_LABEL_HANG, prefix=_label_prefix("Split"))

    # Method note — the same offline list-price arithmetic disclosure as elsewhere.
    _print_hanging(
        Text(
            "estimated from list prices against your logged token counts, "
            "computed offline with no LLM calls",
            style="dim",
        ),
        hang=_LABEL_HANG,
        prefix=_label_prefix("Method"),
    )

    # Default-pool disclosure — only when frugon auto-selected from the built-in
    # pool (no explicit --candidates).  Kept honest and dim.
    if result.used_default_pool and result.candidate_model:
        _print_hanging(
            Text(
                f"considered {result.candidate_pool_size} built-in candidates; "
                "pass --candidates to compare against any priced model",
                style="dim",
            ),
            hang=_LABEL_HANG,
            prefix=_label_prefix("Pool"),
        )

    # Automated-routing upsell — same as the split verbose block.
    automate = Text(
        "frugon can route every call automatically and hold the savings for you → ",
        style="dim",
    )
    automate.append("https://frugon.rodiun.io", style=BRAND_CYAN)
    _print_hanging(automate, hang=_LABEL_HANG, prefix=_label_prefix("Automate"))

    rprint("")


# ---------------------------------------------------------------------------
# Format-neutral verdict synthesis — the single source of truth for the
# confirmed / borderline / not-confirmed / not-verified verbiage.
#
# The terminal (``_render_tier1_synthesis``), the Markdown report and the HTML
# report ALL derive their synthesis line from :func:`_classify_verdict` so the
# verdict a user reads in a shared report can NEVER disagree with the verdict
# they saw in their terminal for the same tally.  The classifier returns plain
# text and a ``state`` token; each surface paints it in its own design language
# (Rich markup, Markdown prose, HTML spans) without re-deciding the verdict.
# ---------------------------------------------------------------------------

# The four mutually-exclusive verdict states a judged tally can resolve to.
#   confirmed      — held cleanly (wins+ties dominate, losses a small minority)
#   borderline     — held on balance but lost on a non-trivial share
#   not_confirmed  — losses dominate; the candidate was materially worse
#   not_verified   — every comparison errored; no scored verdict at all
#   unmeasured     — at least half of the samples errored; too few scored
#                    verdicts to call a quality outcome.  The honest read is
#                    that the candidate could not be sampled, not that it lost.
_VERDICT_CONFIRMED = "confirmed"
_VERDICT_BORDERLINE = "borderline"
_VERDICT_NOT_CONFIRMED = "not_confirmed"
_VERDICT_NOT_VERIFIED = "not_verified"
_VERDICT_UNMEASURED = "unmeasured"
# Every comparison errored AND the cause was the CURRENT/baseline model failing
# to sample on every sampled prompt (rate limit / API error) — so there was
# nothing for the candidate to be compared against.  Distinct from
# ``not_verified`` (which blames "every <candidate> comparison errored", wrongly
# implying the candidate failed): here the honest read is that the baseline
# could not be sampled, so the candidate was never given a fair comparison.
_VERDICT_BASELINE_FAILED = "baseline_failed"

# When at least this fraction of the samples errored, the synthesis treats the
# candidate as UNMEASURED rather than scoring the surviving handful.  >=50% is
# the bug-mode threshold: claude-haiku returned 9/10 errors against a key it
# could not access, and the one scored verdict (whatever it was) is too thin to
# call a quality outcome.  Below this threshold we still classify on the scored
# verdicts (the user has enough signal to act on).
_VERDICT_UNMEASURED_ERROR_FRACTION = 0.5

# Which verdict states read as a caution (amber) vs neutral-positive (cyan).
# Green stays reserved for the money saving, so even a clean "confirmed" reads
# cyan, never green.
_VERDICT_CAUTION_STATES = frozenset(
    {
        _VERDICT_BORDERLINE,
        _VERDICT_NOT_CONFIRMED,
        _VERDICT_NOT_VERIFIED,
        _VERDICT_UNMEASURED,
        _VERDICT_BASELINE_FAILED,
    }
)

# Semantic status-word colour (Fix B).  The whole synthesis line renders in its
# state accent (cyan for confirmed, amber for the cautions); that makes the one
# word that carries the verdict — "confirmed" / "borderline" / "NOT confirmed" /
# "not verified" — fail to stand out.  So colour JUST that word using the SAME
# WIN/LOSS/TIE palette the judge tally table uses, single-sourced here so the
# terminal, HTML and Markdown can never drift:
#
#   confirmed                  → tally WIN  (green)
#   borderline                 → tally TIE  (amber)
#   NOT confirmed              → tally LOSS (red)
#   not verified / unmeasured  → neutral / dim (no strong colour — nothing was
#                                actually measured, so no win/loss/tie applies)
#
# Each state maps to a ``"win" | "loss" | "tie" | None`` token (None == neutral)
# plus the exact status PHRASE the renderers wrap.  The phrase is identical
# across all surfaces (the plain text from :func:`_classify_verdict` and the
# terminal's bold span both use these literals), so one mapping styles them all.
_VERDICT_STATUS_PHRASE: dict[str, str] = {
    _VERDICT_CONFIRMED: "confirmed",
    _VERDICT_BORDERLINE: "borderline",
    _VERDICT_NOT_CONFIRMED: "NOT confirmed",
    _VERDICT_NOT_VERIFIED: "not verified",
    _VERDICT_UNMEASURED: "unmeasured",
    # The synthesis sentence leads "Could not verify — your current model …",
    # so the emphasised status phrase is "Could not verify".
    _VERDICT_BASELINE_FAILED: "Could not verify",
}

# state → WIN/LOSS/TIE tally token (or None for neutral/dim).  Reused by every
# surface to pick the matching colour from its own palette.
_VERDICT_STATUS_TALLY: dict[str, str | None] = {
    _VERDICT_CONFIRMED: "win",
    _VERDICT_BORDERLINE: "tie",
    _VERDICT_NOT_CONFIRMED: "loss",
    _VERDICT_NOT_VERIFIED: None,
    _VERDICT_UNMEASURED: None,
    _VERDICT_BASELINE_FAILED: None,
}

# Terminal (Rich) colour per WIN/LOSS/TIE token — the SAME palette as the tally
# table's per-prompt labels (:data:`_VERDICT_LABEL_STYLE`): WIN green, LOSS red,
# TIE yellow.  A neutral token (None) keeps the word in the line's own colour
# (dim/cyan/amber) — nothing was measured, so no semantic colour applies.
_VERDICT_STATUS_TERMINAL_COLOUR: dict[str, str] = {
    "win": "green",
    "loss": "red",
    "tie": "yellow",
}

# HTML class per WIN/LOSS/TIE token — reuses the tally cell classes
# (:data:`_VERDICT_HTML_CLASS`-family ``.verdict-*``) already injected into both
# report stylesheets (WIN green, LOSS red, TIE amber), so the verdict word and
# the tally table can never show different greens/reds.
_VERDICT_STATUS_HTML_CLASS: dict[str, str] = {
    "win": "verdict-win",
    "loss": "verdict-loss",
    "tie": "verdict-tie",
}


def _verdict_status_phrase(state: str) -> str:
    """Return the status PHRASE for *state* (e.g. ``"NOT confirmed"``)."""
    return _VERDICT_STATUS_PHRASE[state]


def _verdict_status_terminal_markup(state: str) -> str:
    """Rich markup for the bold, semantically-coloured status word of *state*.

    The whole synthesis sentence is already painted its state accent; this wraps
    just the status word in the matching WIN/LOSS/TIE colour (green/red/amber)
    so the verdict word stands out.  A neutral state keeps the line's own colour
    (only bold), since nothing was measured to win, lose or tie.
    """
    phrase = _VERDICT_STATUS_PHRASE[state]
    token = _VERDICT_STATUS_TALLY[state]
    if token is None:
        return f"[bold]{phrase}[/bold]"
    colour = _VERDICT_STATUS_TERMINAL_COLOUR[token]
    return f"[bold {colour}]{phrase}[/bold {colour}]"


def _verdict_status_html(state: str) -> str:
    """HTML for the semantically-coloured status word of *state*.

    Wraps the status word in the matching ``.verdict-*`` tally class (WIN green,
    LOSS red, TIE amber) so it stands out from the rest of the sentence and
    agrees with the tally table.  A neutral state renders the bare (already-safe
    literal) phrase — no strong colour, nothing was measured."""
    phrase = _VERDICT_STATUS_PHRASE[state]
    token = _VERDICT_STATUS_TALLY[state]
    if token is None:
        return phrase
    cls = _VERDICT_STATUS_HTML_CLASS[token]
    return f'<span class="{cls}">{phrase}</span>'


def _verdict_status_md(state: str) -> str:
    """Markdown for the status word of *state* — bold (no colour available).

    Markdown has no colour, so the status word is emphasised with ``**bold**``
    so it still stands out; we never fabricate colour."""
    return f"**{_VERDICT_STATUS_PHRASE[state]}**"


def _emphasise_verdict_status(text: str, state: str, replacement: str) -> str:
    """Wrap the status word inside *text* with *replacement* (surface-styled).

    *text* is a verdict sentence whose status word (e.g. ``"NOT confirmed"``)
    must be emphasised in place — the rest of the sentence is left untouched.
    Only the FIRST occurrence is replaced (the leading "Estimate <phrase>:"), so
    a later incidental mention is never re-styled.  Shared by the Markdown and
    HTML synthesis so the colour/bold mapping is single-sourced and cannot drift.
    For HTML, *text* is already escaped; the phrases contain no HTML-special
    characters, so the literal phrase still matches after escaping."""
    phrase = _VERDICT_STATUS_PHRASE[state]
    return text.replace(phrase, replacement, 1)


def _baseline_all_errored(measure_result: MeasureResult) -> bool:
    """True when the baseline/current model failed to sample on EVERY prompt.

    The signal that flips the ``scored == 0`` synthesis from "the candidate
    errored" to "your current model failed to sample".  Conservatively requires
    at least one sampled comparison AND every one of them carrying a
    baseline-side error (``current_output.error is not None``) — so a run with no
    samples, or one where even a single baseline call succeeded, never claims the
    baseline failed wholesale.
    """
    comps = measure_result.comparisons
    if not comps:
        return False
    return all(c.current_output.error is not None for c in comps)


def _classify_verdict(
    tally: Tier1Tally,
    current_model: str,
    *,
    baseline_all_errored: bool = False,
    result: AnalysisResult | None = None,
) -> tuple[str, str]:
    """Classify one judged tally into a ``(state, plain_text)`` pair.

    *state* is one of the ``_VERDICT_*`` tokens above; *plain_text* is the
    human sentence — identical wording to the terminal's synthesis line, but
    with NO colour markup — that every surface reuses verbatim.  Centralising
    both the thresholds (``_VERDICT_BORDERLINE_LOSS_FRACTION``) and the wording
    here is what guarantees the terminal and the reports never diverge.

    *baseline_all_errored* is True when the CURRENT/baseline model itself failed
    to sample on every sampled prompt (rate limit / API error).  In that state a
    "every <candidate> comparison errored" message wrongly implies the candidate
    failed, so the classifier emits the baseline-failed sentence instead — the
    honest read is that there was nothing to compare against.

    *result* is the :class:`AnalysisResult` that generated the offline routing
    panel for this run.  When supplied it allows the confirmed-verdict text to
    back-reference the SAME quality phrase the panel showed — "same or better
    quality" when ``tier_drop <= 0``, or "within tolerance" for a genuine
    step-down — so the synthesis is always truthful (§6 honesty invariant).
    """
    scored = tally.wins + tally.losses + tally.ties
    held = tally.wins + tally.ties
    if scored == 0:
        if baseline_all_errored:
            return (
                _VERDICT_BASELINE_FAILED,
                f"Could not verify — your current model ({current_model}) failed "
                "to sample (rate limit or API error), so there was nothing to "
                f"compare against. Retry, or check {current_model} access.",
            )
        return (
            _VERDICT_NOT_VERIFIED,
            f"Estimate not verified: every {tally.candidate} comparison errored "
            f"({tally.errors}/{tally.total}) — no scored verdict. "
            f"Retry, or check {tally.candidate} access.",
        )
    # Predominantly-errored: the candidate could not be sampled cleanly (the
    # canonical bug-mode is a project that lacks access to the model — every
    # call comes back '[unavailable — …]').  Calling it NOT-confirmed in
    # that state would imply we measured it and it lost; the honest read is
    # we could not measure it at all.  Surface the access-check action instead
    # of a quality verdict.
    if tally.total > 0 and tally.errors / tally.total >= _VERDICT_UNMEASURED_ERROR_FRACTION:
        return (
            _VERDICT_UNMEASURED,
            f"Estimate unmeasured: {tally.candidate} could not be sampled in "
            f"{tally.errors}/{tally.total} comparison(s) — check API access / "
            f"model name for {tally.candidate}.",
        )
    loss_fraction = tally.losses / scored
    if held > tally.losses and loss_fraction < _VERDICT_BORDERLINE_LOSS_FRACTION:
        divergence = _recommendation_divergence_note(tally.candidate, result)
        if divergence is not None:
            # The measured model is NOT the headline recommendation (e.g. the
            # demo's pinned try-out candidate) — do not back-reference the
            # RECOMMENDATION's offline quality phrase against a different
            # model. Disclose the divergence instead (§6 honesty invariant).
            return (
                _VERDICT_CONFIRMED,
                f"Estimate confirmed: {tally.candidate} held quality in "
                f"{held:,}/{scored:,} sampled prompt(s). {divergence}",
            )
        _quality_phrase = _shown_quality_phrase(result)
        return (
            _VERDICT_CONFIRMED,
            f"Estimate confirmed: {tally.candidate} held quality in "
            f"{held:,}/{scored:,} sampled prompt(s) "
            f"(offline '{_quality_phrase}' → verified on your data).",
        )
    if held > tally.losses:
        return (
            _VERDICT_BORDERLINE,
            f"Estimate borderline: {tally.candidate} held in "
            f"{held:,}/{scored:,} but was worse in {tally.losses:,}/{scored:,} — "
            f"review the losses before routing these calls.",
        )
    return (
        _VERDICT_NOT_CONFIRMED,
        f"Estimate NOT confirmed: {tally.candidate} was worse in "
        f"{tally.losses:,}/{scored:,} sampled prompt(s) — "
        f"consider keeping these calls on {current_model}.",
    )


def _render_tier1_synthesis(
    measure_result: MeasureResult,
    result: AnalysisResult | None = None,
) -> None:
    """Tie each judged candidate's tally back to the offline estimate.

    For every candidate the offline split proposed routing easy calls to, the
    judge tally either confirms the estimate (the candidate held quality on the
    sampled prompts) or refutes it (the candidate was materially worse).  This
    line is the missing link between the measured outputs and the headline
    offline saving — without it the user sees a table and an estimate with no
    stated relationship.

    *result* carries the :class:`AnalysisResult` that generated the routing
    panel for this run.  It is threaded to :func:`_classify_verdict` so the
    confirmed-verdict back-reference uses the SAME quality phrase the panel
    showed ("same or better quality" when ``tier_drop <= 0``, "within
    tolerance" for a genuine step-down), keeping the §6 honesty invariant on
    the --judge surface.

    Colour law: green is reserved for the money saving only, so a *confirmed*
    estimate reads in cyan (neutral-positive), a *not-confirmed* one in amber
    (caution).  The candidate named is exactly the one that was measured, and
    the count reconciles with the scored-sample total (errors excluded).

    The verdict (state + wording) is decided once in :func:`_classify_verdict`
    and shared with the Markdown/HTML reports, so a terminal verdict and a
    report verdict for the same tally are always the same sentence.
    """
    tallies = measure_result.tier1_tallies
    if not tallies:
        return
    current = measure_result.current_model
    # Did the baseline itself fail to sample on every prompt?  Decided ONCE for
    # the whole run and shared by every tally's verdict so the "current model
    # failed" reading is consistent across candidates and surfaces.
    baseline_failed = _baseline_all_errored(measure_result)
    rprint("")  # one blank line separating the table from its reading
    for tally in tallies:
        state, _text = _classify_verdict(
            tally, current, baseline_all_errored=baseline_failed, result=result
        )
        # Each branch keeps its bespoke Rich markup (bold spans on the key
        # figures) for the terminal's richer styling; the verdict it renders is
        # the one _classify_verdict chose, so it can never disagree with the
        # report.  The plain _text is the reports' verbatim copy of the same
        # sentence.
        scored = tally.wins + tally.losses + tally.ties
        held = tally.wins + tally.ties
        # Fix B — colour JUST the status word semantically (WIN green / TIE amber
        # / LOSS red / neutral-dim), reusing the tally palette; the rest of the
        # sentence keeps its state accent.  The status markup nests inside the
        # outer colour tag and overrides only its own span.
        status = _verdict_status_terminal_markup(state)
        if state == _VERDICT_BASELINE_FAILED:
            # The status phrase ("Could not verify") already opens the sentence,
            # so it is not re-prefixed with "Estimate".
            rprint(
                f"[{CAUTION_AMBER}]{status} — your current model "
                f"([bold]{current}[/bold]) failed to sample (rate limit or API "
                "error), so there was nothing to compare against. "
                f"Retry, or check {current} access.[/{CAUTION_AMBER}]"
            )
        elif state == _VERDICT_NOT_VERIFIED:
            rprint(
                f"[{CAUTION_AMBER}]Estimate {status}: "
                f"every {tally.candidate} comparison errored "
                f"({tally.errors:,}/{tally.total:,}) — no scored verdict. "
                f"Retry, or check {tally.candidate} access.[/{CAUTION_AMBER}]"
            )
        elif state == _VERDICT_UNMEASURED:
            rprint(
                f"[{CAUTION_AMBER}]Estimate {status}: "
                f"{tally.candidate} could not be sampled in "
                f"[bold]{tally.errors:,}/{tally.total:,}[/bold] comparison(s) — "
                f"check API access / model name for "
                f"{tally.candidate}.[/{CAUTION_AMBER}]"
            )
        elif state == _VERDICT_CONFIRMED:
            divergence = _recommendation_divergence_note(tally.candidate, result)
            if divergence is not None:
                # The measured model is NOT the headline recommendation — never
                # back-reference the RECOMMENDATION's offline quality phrase
                # against a different model (§6 honesty invariant).
                rprint(
                    f"[{BRAND_CYAN}]Estimate {status}:[/{BRAND_CYAN}] "
                    f"[cyan]{tally.candidate}[/cyan] held quality in "
                    f"[bold]{held:,}/{scored:,}[/bold] sampled prompt(s). "
                    f"[dim]{divergence}[/dim]"
                )
            else:
                _quality_phrase = _shown_quality_phrase(result)
                rprint(
                    f"[{BRAND_CYAN}]Estimate {status}:[/{BRAND_CYAN}] "
                    f"[cyan]{tally.candidate}[/cyan] held quality in "
                    f"[bold]{held:,}/{scored:,}[/bold] sampled prompt(s) "
                    f"(offline [dim]'{_quality_phrase}'[/dim] → verified on your data)."
                )
        elif state == _VERDICT_BORDERLINE:
            rprint(
                f"[{CAUTION_AMBER}]Estimate {status}: "
                f"{tally.candidate} held in "
                f"[bold]{held:,}/{scored:,}[/bold] but was worse in "
                f"[bold]{tally.losses:,}/{scored:,}[/bold] — "
                f"review the losses before routing these calls.[/{CAUTION_AMBER}]"
            )
        else:  # _VERDICT_NOT_CONFIRMED
            suggestion = _escalation_for_tally(tally, current)
            if suggestion is not None:
                # Actionable next step: point at the next rung up the quality
                # ladder that still saves money.  Amber prose carries the
                # caution; the model name + ready command are cyan; the saving is
                # a comparison figure (not a realised saving), so it stays amber,
                # never money-green.
                rprint(
                    f"[{CAUTION_AMBER}]Estimate {status}: "
                    f"{tally.candidate} was worse in "
                    f"[bold]{tally.losses:,}/{scored:,}[/bold] — "
                    f"try the next rung up:[/{CAUTION_AMBER}]"
                )
                rprint(
                    f"[cyan]{suggestion.model}[/cyan] "
                    f"[{CAUTION_AMBER}]({suggestion.tier_label} tier, still "
                    f"~{suggestion.pct_cheaper_than_baseline}% cheaper than "
                    f"{current}):[/{CAUTION_AMBER}] "
                    f"[cyan]{suggestion.command}[/cyan]"
                )
            else:
                # Honest dead-end — no cheaper higher-tier model exists.
                rprint(
                    f"[{CAUTION_AMBER}]Estimate {status}: "
                    f"{tally.candidate} was worse in "
                    f"[bold]{tally.losses:,}/{scored:,}[/bold] sampled prompt(s) — "
                    f"consider keeping these calls on {current}.[/{CAUTION_AMBER}]"
                )
        # Confidence-aware nudge — a dim line beneath any scored verdict whose
        # sample is thin enough that one more prompt could flip the band.  A
        # decisive result (wide margin, adequate n) and an unverified tally get
        # no nudge.  It is a quiet CAVEAT, not part of the action above it, so a
        # blank line separates it from the verdict/escalation.
        nudge = _nudge_text(tally)
        if nudge is not None:
            rprint("")
            rprint(f"[dim]{nudge}[/dim]")


def _render_tier0_framing(
    measure_result: MeasureResult,
    result: AnalysisResult | None = None,
) -> None:
    """One-line framing for a Tier-0 (--measure, no --judge) sample.

    Tier-0 produces raw side-by-side outputs with no score, so the relationship
    to the headline estimate is necessarily weaker than Tier-1's: the user is
    the judge.  State that plainly, and that the offline quality estimate
    remains UNVERIFIED until --judge scores it — so the reader knows what they
    are looking at and is not misled into treating the diffs as a verdict.

    *result* is the :class:`AnalysisResult` that drove the routing panel; it
    selects the quality phrase actually shown to the user ("same or better
    quality" vs "within tolerance") so this back-reference is always truthful
    (§6 honesty invariant on the --measure-only path).  When *measure_result*
    sampled a model OTHER than the panel's headline recommendation (e.g. the
    demo's pinned try-out candidate), the offline phrase belongs to a
    different model — this framing discloses that instead of back-referencing
    it (same §6 invariant the Tier-1 verdict enforces).
    """
    candidates = measure_result.candidates
    named = candidates[0] if len(candidates) == 1 else "the candidate"
    if len(candidates) == 1:
        divergence = _recommendation_divergence_note(candidates[0], result)
    else:
        divergence = None
    if divergence is not None:
        rprint(
            f"\n[dim]These are raw side-by-side outputs for you to compare "
            f"({named} vs {measure_result.current_model}). {divergence} "
            f"Run [/dim][cyan]--judge[/cyan][dim] for a scored verdict.[/dim]"
        )
    else:
        _quality_phrase = _shown_quality_phrase(result)
        rprint(
            f"\n[dim]These are raw side-by-side outputs for you to compare "
            f"({named} vs {measure_result.current_model}). "
            f"'{_quality_phrase}' above is an offline estimate — "
            f"run [/dim][cyan]--judge[/cyan][dim] for a scored verdict.[/dim]"
        )


# ---------------------------------------------------------------------------
# Post-measurement promotion (Change 2) — retrospective recommendation
# ---------------------------------------------------------------------------
#
# A candidate that was EXCLUDED from the offline recommendation purely for being
# unrated (Change 1) can become the better route once --measure --judge verifies
# it.  When that candidate CONFIRMS its quality on the user's own data AND saves
# MORE than the headline rated recommendation, we surface a positive promotion
# callout — green ✓, not a warning — single-sourced across terminal + MD + HTML.
#
# Fire ONLY when the candidate: was excluded for being unrated, confirmed
# (held quality), and saves STRICTLY more than the headline recommendation.  A
# candidate that confirmed but saves ≤ the headline, failed quality, or was
# already the recommendation does NOT promote.  When multiple qualify the
# cheapest (biggest saving) is promoted — deterministic.  The headline at the top
# of the report stays the offline rated pick; this is the measurement-section
# upgrade prompt, never a silent rewrite of the headline.


@dataclass(frozen=True)
class _Promotion:
    """A confirmed unrated candidate that now beats the headline recommendation.

    *candidate* held quality on the user's data and saves *cand_pct* — strictly
    more than the headline *headline_model*'s *headline_pct*.  The percentages are
    the SAME block split saving figures the "Candidates considered" block quotes
    (read from ``candidate_projections``), so the promotion can never quote a
    figure that disagrees with the block.
    """

    candidate: str
    cand_pct: Decimal
    headline_model: str
    headline_pct: Decimal
    held: int
    scored: int


def _detect_promotion(
    result: AnalysisResult,
    measure_result: MeasureResult | None,
) -> _Promotion | None:
    """Return the promotable confirmed-unrated candidate, or None.

    Single source of truth for the Change-2 promotion across every surface.
    Returns the cheapest (biggest-saving) candidate that (a) was excluded from
    the recommendation for being unrated, (b) CONFIRMED its quality this run
    (verdict state == confirmed/held), and (c) saves strictly more than the
    headline recommendation's saving%.  None when no candidate qualifies, when
    there is no headline recommendation, or when --measure produced no Tier-1
    tallies.
    """
    if measure_result is None or measure_result.tier1_tallies is None:
        return None
    headline_model = result.candidate_model
    if headline_model is None:
        return None
    # The headline must itself be a saving recommendation with a known saving%.
    headline_pct = _candidate_block_saving_pct(result, headline_model)
    if headline_pct is None:
        return None

    excluded = set(result.excluded_unrated_models)
    if not excluded:
        return None

    baseline_failed = _baseline_all_errored(measure_result)
    current = measure_result.current_model

    candidates: list[_Promotion] = []
    for tally in measure_result.tier1_tallies:
        cand = tally.candidate
        if cand not in excluded:
            continue
        state, _text = _classify_verdict(
            tally, current, baseline_all_errored=baseline_failed
        )
        if state != _VERDICT_CONFIRMED:
            continue
        cand_pct = _candidate_block_saving_pct(result, cand)
        if cand_pct is None or cand_pct <= headline_pct:
            continue
        scored = tally.wins + tally.losses + tally.ties
        held = tally.wins + tally.ties
        candidates.append(
            _Promotion(
                candidate=cand,
                cand_pct=cand_pct,
                headline_model=headline_model,
                headline_pct=headline_pct,
                held=held,
                scored=scored,
            )
        )
    if not candidates:
        return None
    # Promote the biggest saving; deterministic tie-break by model name.
    candidates.sort(key=lambda p: (-p.cand_pct, p.candidate))
    return candidates[0]


def _promotion_message(promo: _Promotion) -> str:
    """Plain-text promotion callout — shared verbatim across every surface.

    The percentages are the block split saving figures, so this sentence and the
    "Candidates considered" block always quote the same numbers.
    """
    return (
        f"{promo.candidate} held quality on your data "
        f"({promo.held:,}/{promo.scored:,}) — and at {float(promo.cand_pct):.1f}% "
        f"it saves more than the recommended {promo.headline_model} "
        f"({float(promo.headline_pct):.1f}%). Now that it's verified, it's the "
        f"better route: re-run with --candidates {promo.candidate} to switch."
    )


# Tier-0 preview lengths.  The prompt gets enough room to convey what was
# asked; each model's output gets more, since the outputs are what the human
# is judging side-by-side.  Both truncate with a visible ellipsis.
_PROMPT_PREVIEW_CHARS = 160
_OUTPUT_PREVIEW_CHARS = 240


@dataclass(frozen=True)
class PreviewLimits:
    """Per-render preview truncation limits for the quality-sample previews.

    Carries the OUTPUT preview length (``output_chars``) and the PROMPT preview
    length (``prompt_chars``) for ONE surface (terminal *or* report), plus a
    ``no_truncate`` flag that, when True, shows every preview in full (no
    ``…`` cut, both lengths ignored).

    The dataclass is *immutable* (``frozen=True``) and threaded explicitly
    through the renderers — no module-global is mutated — so tests can build
    independent limits and call the renderers concurrently without one run's
    settings bleeding into another.  The module-level constants
    (``_PROMPT_PREVIEW_CHARS`` etc.) remain the per-surface DEFAULTS, surfaced
    via :func:`terminal_default` / :func:`report_default`; an unset CLI flag
    yields exactly those, so default output is byte-identical to before.
    """

    prompt_chars: int
    output_chars: int
    no_truncate: bool = False

    @staticmethod
    def terminal_default() -> PreviewLimits:
        """The terminal's historical limits (prompt 160, output 240)."""
        return PreviewLimits(
            prompt_chars=_PROMPT_PREVIEW_CHARS,
            output_chars=_OUTPUT_PREVIEW_CHARS,
        )

    @staticmethod
    def report_default() -> PreviewLimits:
        """The report's historical limits (prompt 400, output 800)."""
        return PreviewLimits(
            prompt_chars=_REPORT_PROMPT_PREVIEW_CHARS,
            output_chars=_REPORT_OUTPUT_PREVIEW_CHARS,
        )

    def truncate_prompt(self, text: str) -> str:
        """Truncate a PROMPT/system preview per these limits."""
        return _truncate(text, self.prompt_chars, no_truncate=self.no_truncate)

    def truncate_output(self, text: str) -> str:
        """Truncate a model OUTPUT preview per these limits."""
        return _truncate(text, self.output_chars, no_truncate=self.no_truncate)


def _scaled_preview_limits(
    base_prompt: int, base_output: int, *, override_output: int | None, no_truncate: bool
) -> PreviewLimits:
    """Resolve a surface's PreviewLimits from the CLI display flags.

    *base_prompt* / *base_output* are the surface's historical defaults (terminal
    160/240, report 400/800).  When *override_output* (``--preview-chars N``) is
    given it replaces the OUTPUT length on this surface and the PROMPT length is
    scaled to preserve the surface's existing prompt:output ratio
    (``prompt = round(N × base_prompt / base_output)``, floored at 1 so a tiny N
    never collapses the prompt preview to nothing).  *no_truncate* maps straight
    through to :class:`PreviewLimits` (the lengths are then irrelevant, but kept
    for a stable shape).  With neither flag set the surface defaults are returned
    unchanged, so default output is byte-identical to before this existed.
    """
    if no_truncate:
        return PreviewLimits(
            prompt_chars=base_prompt, output_chars=base_output, no_truncate=True
        )
    if override_output is None:
        return PreviewLimits(prompt_chars=base_prompt, output_chars=base_output)
    scaled_prompt = max(1, round(override_output * base_prompt / base_output))
    return PreviewLimits(prompt_chars=scaled_prompt, output_chars=override_output)


def resolve_preview_limits(
    *, preview_chars: int | None = None, no_truncate: bool = False
) -> tuple[PreviewLimits, PreviewLimits]:
    """Resolve ``(terminal_limits, report_limits)`` from the CLI display flags.

    Single source of truth for turning ``--preview-chars`` / ``--no-truncate``
    into the two per-surface :class:`PreviewLimits`.  ``--preview-chars N``
    overrides the OUTPUT length on BOTH surfaces (prompt scaled per surface
    ratio); ``--no-truncate`` shows everything in full on both.  With neither
    flag set, the historical terminal (160/240) and report (400/800) defaults
    are returned, so default rendering is unchanged.  The two flags are mutually
    exclusive at the CLI boundary; if both somehow arrive here, ``no_truncate``
    wins (full text is the safer, information-preserving choice).
    """
    terminal_limits = _scaled_preview_limits(
        _PROMPT_PREVIEW_CHARS,
        _OUTPUT_PREVIEW_CHARS,
        override_output=preview_chars,
        no_truncate=no_truncate,
    )
    report_limits = _scaled_preview_limits(
        _REPORT_PROMPT_PREVIEW_CHARS,
        _REPORT_OUTPUT_PREVIEW_CHARS,
        override_output=preview_chars,
        no_truncate=no_truncate,
    )
    return terminal_limits, report_limits


# ---------------------------------------------------------------------------
# Multi-format report writer — one analysis pass, every requested format
# ---------------------------------------------------------------------------
#
# The extension on the ``--report`` value (or ``FRUGON_REPORT_PATH``) decides the
# format, and a name with NO recognised extension is treated as a PREFIX that
# emits the FULL set.  Every file is rendered from the SAME in-memory
# ``AnalysisResult`` (+ optional ``MeasureResult``) the analysis already produced
# — there is no recompute and no extra provider call, whatever the format count.
# This is the ONE place that maps a target path to render calls, so the CLI never
# duplicates the render-selection logic and the formats can never drift.

# Markdown has a single canonical layout (v1 and v2 render byte-identically — the
# ``--report-style`` flag governs HTML design only), so the canonical Markdown
# renderer is fixed here rather than selected by style.
_CANONICAL_MARKDOWN = "v2"


def report_paths_for(target: Path) -> list[Path]:
    """Return the exact file path(s) a ``--report`` *target* resolves to.

    The pure path arithmetic behind :func:`write_reports`, exposed so callers can
    learn WHICH files a target would write WITHOUT rendering (e.g. for a dry-run
    or a test).  A ``.md`` / ``.html`` target maps to itself; a target with no
    recognised extension is a PREFIX that maps to ``<prefix>.md``,
    ``<prefix>.v1.html`` and ``<prefix>.v2.html`` (Markdown is style-agnostic, so
    one ``.md`` — never ``.v1.md`` / ``.v2.md``).
    """
    suffix = target.suffix.lower()
    if suffix in (".md", ".html"):
        return [target]
    # No recognised extension → treat the whole value as a prefix and emit every
    # format.  ``with_name`` appends to the final path component so a prefix that
    # itself contains dots (e.g. ``report.2026``) still gains the format suffix
    # rather than replacing the user's text.
    return [
        target.with_name(f"{target.name}.md"),
        target.with_name(f"{target.name}.v1.html"),
        target.with_name(f"{target.name}.v2.html"),
    ]


def write_reports(
    result: AnalysisResult,
    target: Path,
    *,
    report_style: str = "v2",
    measure_result: MeasureResult | None = None,
    limits: PreviewLimits | None = None,
) -> list[Path]:
    """Render *result* to *target* in every format its extension implies.

    Extension heuristic (one analysis pass feeds all of them):

      * ``<name>.md``   → Markdown only (single canonical layout).
      * ``<name>.html`` → HTML only, styled per *report_style* (``v1``/``v2``).
      * ``<name>`` with NO recognised extension → the FULL set from this one pass:
        ``<name>.md``, ``<name>.v1.html`` and ``<name>.v2.html``.  *report_style*
        is ignored in the prefix case — every HTML style is emitted.

    All files render from the SAME *result* / *measure_result* already in memory;
    nothing is recomputed and no provider call is made per format.  Returns the
    list of paths actually written, in write order, so the caller can print one
    confirmation line per file.

    *report_style* must already be a validated ``"v1"``/``"v2"`` value (the CLI
    normalises and validates it before calling); an unexpected value falls back
    to the canonical ``v2`` HTML design rather than raising mid-write.
    """
    style = report_style if report_style in ("v1", "v2") else "v2"
    suffix = target.suffix.lower()

    written: list[Path] = []
    if suffix == ".md":
        # Markdown is style-agnostic (v1 and v2 render identically), but the
        # single-target path still routes through the requested style's renderer
        # so the dispatch is consistent with the HTML branch and with callers
        # that patch a specific renderer.  The bytes are the same either way.
        _write_markdown(result, target, style, measure_result, limits)
        written.append(target)
    elif suffix == ".html":
        _write_html(result, target, style, measure_result, limits)
        written.append(target)
    else:
        # Prefix case — emit every format from this one pass.  --report-style is
        # ignored here (both HTML styles are written), and Markdown is emitted via
        # its single canonical renderer (one .md, never .v1.md / .v2.md).
        md_path, v1_path, v2_path = report_paths_for(target)
        _write_markdown(result, md_path, _CANONICAL_MARKDOWN, measure_result, limits)
        _write_html(result, v1_path, "v1", measure_result, limits)
        _write_html(result, v2_path, "v2", measure_result, limits)
        written.extend([md_path, v1_path, v2_path])
    return written


def _write_markdown(
    result: AnalysisResult,
    path: Path,
    style: str,
    measure_result: MeasureResult | None,
    limits: PreviewLimits | None,
) -> None:
    """Render Markdown via the canonical renderer (no duplicated render logic)."""
    renderer = render_markdown_v2 if style == "v2" else render_markdown
    renderer(result, path, measure_result=measure_result, limits=limits)


def _write_html(
    result: AnalysisResult,
    path: Path,
    style: str,
    measure_result: MeasureResult | None,
    limits: PreviewLimits | None,
) -> None:
    """Render HTML in the requested style via the existing renderers."""
    renderer = render_html_v2 if style == "v2" else render_html
    renderer(result, path, measure_result=measure_result, limits=limits)


def _truncate(text: str, limit: int, *, no_truncate: bool = False) -> str:
    """Cut *text* at *limit* characters, appending an ellipsis when it was cut.

    When *no_truncate* is True the text is returned verbatim (no cut, no
    ellipsis) — the ``--no-truncate`` display path — regardless of *limit*.
    """
    if no_truncate or len(text) <= limit:
        return text
    return text[:limit] + "…"


# Per-prompt verdict label styling for the verbose Tier-1 detail block.  Colour
# law: these labels MATCH the judge tally table (WIN green, LOSS red, TIE
# yellow) so the per-prompt detail and the table never disagree.  Green-as-win
# is legitimate inside the quality detail; the "green = money" discipline
# governs the COST panel only.  An unparseable/errored judge collapses to a dim
# "[error]".  These are the cue that makes a lost prompt visually findable.
_VERDICT_LABEL_STYLE: dict[str, str] = {
    "win": "green",
    "tie": "yellow",
    "loss": "red",
}

# Refined per-prompt label token for the "the candidate worked but couldn't be
# scored because the BASELINE failed to sample" case.  The judge tally counts
# this as a neutral ``error`` (the comparison was impossible — see measure.py
# §baseline_errored), but rendering a bare red ``[error]`` next to a candidate
# whose OWN output succeeded wrongly reads as the candidate failing.  This token
# labels it neutrally as ``[no comparison]`` so a working candidate is never
# shown as failing.  A candidate that genuinely errored — or a judge that errored
# on two good outputs — still resolves to plain ``error``.
_VERDICT_NO_COMPARISON = "no_comparison"


def _refine_prompt_verdict(comp: Comparison, pos: int) -> str:
    """Refine one prompt's ``error`` verdict into ``no_comparison`` where apt.

    Returns the verdict token the per-prompt label should render.  For a non-
    error verdict it is returned unchanged.  For an ``error`` verdict it becomes
    :data:`_VERDICT_NO_COMPARISON` ONLY when the candidate's own output succeeded
    (no ``error`` on its :class:`SampledOutput`) while the baseline's output
    failed — i.e. there was simply nothing to compare against, not a candidate
    failure.  Every other ``error`` (the candidate itself errored, or the judge
    errored on two good outputs) stays ``error``.  Shared by the terminal,
    Markdown and HTML per-prompt renderers so the three surfaces label the same
    prompt identically.
    """
    if pos >= len(comp.verdicts):
        return "error"
    verdict = comp.verdicts[pos]
    if verdict != "error":
        return verdict
    baseline_errored = comp.current_output.error is not None
    cand_errored = (
        pos < len(comp.candidate_outputs)
        and comp.candidate_outputs[pos].error is not None
    )
    if baseline_errored and not cand_errored:
        return _VERDICT_NO_COMPARISON
    return "error"


def _verdict_label(verdict: str) -> Text:
    """Render a candidate's per-prompt verdict as a styled ``[WIN]``/``[LOSS]``…

    Returns a one-space-padded styled fragment to splice into the candidate's
    output prefix (``model [LOSS]: …``).  Unknown/empty verdicts and the judge's
    own "error" collapse to a dim ``[error]`` so the row is never blank or
    mislabelled; a :data:`_VERDICT_NO_COMPARISON` token renders a neutral
    ``[no comparison]`` (a working candidate the baseline failure left unscored).
    """
    if verdict == _VERDICT_NO_COMPARISON:
        fragment = Text(" ")
        fragment.append("[no comparison]", style="dim")
        return fragment
    style = _VERDICT_LABEL_STYLE.get(verdict)
    fragment = Text(" ")
    if style is None:
        fragment.append("[error]", style="dim")
    else:
        fragment.append(f"[{verdict.upper()}]", style=style)
    return fragment


def _render_comparison_prompt(
    comp: Comparison, index: int, limits: PreviewLimits
) -> None:
    """Print one sampled prompt's System (if any) + Prompt lines (Tier-0 shape).

    Shared by the Tier-0 side-by-side and the verbose Tier-1 per-prompt detail
    so both use the SAME truncation, hang-indent and dim styling.  *limits*
    carries the (possibly flag-overridden) preview lengths for the terminal
    surface; with no display flag set it is :meth:`PreviewLimits.terminal_default`
    so output is unchanged.
    """
    last_msg = comp.record.messages[-1]["content"]
    preview = limits.truncate_prompt(last_msg)
    rprint("")
    # When the sampled record carries a system message, surface it (dim) above
    # the prompt — it explains outputs the prompt alone would not.
    system_msg = next(
        (
            m["content"]
            for m in comp.record.messages
            if m.get("role") == "system" and m.get("content")
        ),
        None,
    )
    if system_msg:
        sys_prefix = Text("  ")
        sys_prefix.append(f"System {index}: ", style="dim")
        _print_hanging(
            Text(limits.truncate_prompt(system_msg), style="dim"),
            hang=sys_prefix.cell_len,
            prefix=sys_prefix,
        )
    prompt_prefix = Text("  ")
    prompt_prefix.append(f"Prompt {index}: ", style="dim")
    _print_hanging(Text(preview), hang=prompt_prefix.cell_len, prefix=prompt_prefix)


def _render_comparison_outputs(
    comp: Comparison, *, label_verdicts: bool, limits: PreviewLimits
) -> None:
    """Print the baseline + each candidate output for one sampled prompt.

    The baseline (current model) prints in cyan, exactly as Tier-0.  Each
    candidate prints under its model name; when *label_verdicts* is True and a
    per-prompt verdict exists for that candidate, the verdict label is spliced
    into the prefix (``gpt-4o-mini [LOSS]: …``) in its caution-colour band so a
    lost prompt is visually findable.  Same hang-indent as Tier-0; *limits*
    carries the (possibly flag-overridden) terminal preview lengths.
    """
    cur_text = limits.truncate_output(comp.current_output.content) or "[no output]"
    cur_prefix = Text("    ")
    cur_prefix.append(f"{comp.current_output.model}:", style=BRAND_CYAN)
    cur_prefix.append(" ")
    _print_hanging(Text(cur_text), hang=cur_prefix.cell_len, prefix=cur_prefix)
    for pos, cand_out in enumerate(comp.candidate_outputs):
        cand_text = (
            limits.truncate_output(cand_out.content)
            or cand_out.error
            or "[no output]"
        )
        cand_prefix = Text("    ")
        # Tier-0 (no verdicts) keeps the historical green model label; the
        # verbose Tier-1 detail recolours the label to the verdict band so the
        # whole row carries the win/tie/loss signal, not a money-green name.
        if label_verdicts and pos < len(comp.verdicts):
            verdict = _refine_prompt_verdict(comp, pos)
            # No-comparison is neutral (the candidate worked; only the baseline
            # failed), so the model name stays dim rather than red.
            label_style = _VERDICT_LABEL_STYLE.get(verdict, "dim")
            cand_prefix.append(f"{cand_out.model}", style=label_style)
            cand_prefix.append_text(_verdict_label(verdict))
            cand_prefix.append(": ")
        else:
            cand_prefix.append(f"{cand_out.model}:", style="green")
            cand_prefix.append(" ")
        _print_hanging(Text(cand_text), hang=cand_prefix.cell_len, prefix=cand_prefix)


def _judge_provenance_text(measure_result: MeasureResult) -> str | None:
    """Return the dim judge-provenance CAPTION for a Tier-1 sample, or ``None``.

    Provenance answers "how this was measured": which judge scored the run, where
    it came from, and that the A/B order was randomised.  It is the CAPTION that
    sits directly under the "Quality sample — judge results …" title (above the
    tally), shared verbatim by the terminal, Markdown and HTML so the three
    surfaces can never drift.

    The prompt count is deliberately OMITTED — the title already carries it
    (``… (N prompts, current: …)``), so repeating it here would be redundant.

    Returns ``None`` when no judge ran (Tier-0 / ``judge_model is None``).

    The honest one-word qualifier on the judge name mirrors the long-standing
    methodology logic:

      * ``(your highest-tier model)`` — the judge was auto-selected as the
        strongest model in the user's OWN log (``judge_from_log``).  This is the
        case most likely to coincide with a compared model, so it NEVER claims
        independence; it states plainly where the judge came from.
      * ``(independent)`` — ONLY when the judge is genuinely neither a compared
        model NOR the log-best auto-pick (an explicit external ``--judge-model``).
      * no qualifier — an explicit judge that IS one of the compared models (the
        self-judge caution already flags the bias separately).
    """
    judge = measure_result.judge_model
    if judge is None:  # Tier-0 — no judge ran
        return None
    if measure_result.judge_from_log:
        qualifier = " (your highest-tier model)"
    elif not measure_result.self_judged_models:
        qualifier = " (independent)"
    else:
        qualifier = ""
    return f"Judge: {judge}{qualifier} · A/B order randomised"


def _render_judge_provenance_caption(measure_result: MeasureResult) -> None:
    """Print the dim judge-provenance caption (under the title), when a judge ran.

    No-op for a Tier-0 sample.  Rendered dim — it is provenance ("how this was
    measured"), not the verdict; it belongs as a quiet caption above the tally,
    never as a line competing with the takeaway below it.
    """
    caption = _judge_provenance_text(measure_result)
    if caption is not None:
        rprint(f"[dim]{caption}[/dim]")


def _self_judge_caution_text(measure_result: MeasureResult) -> str | None:
    """Return the self-judge caution sentence, or ``None`` when it does not apply.

    The SINGLE source of the self-judge caution wording, shared verbatim by the
    terminal, Markdown and HTML surfaces so they can never drift.  The caution
    fires WHENEVER the resolved judge is itself one of the models it scored
    (``self_judged_models`` non-empty) — a trust signal, never gated on
    ``--verbose`` — and reads identically everywhere; only the per-surface
    styling (terminal amber-dim / MD blockquote / HTML amber paragraph) differs.

    Returns ``None`` for a Tier-0 sample (no judge) or when the judge is
    independent of every compared model.
    """
    if measure_result.judge_model is None:  # Tier-0 — no judge ran
        return None
    if not measure_result.self_judged_models:
        return None
    named = ", ".join(measure_result.self_judged_models)
    return (
        f"Caution: the judge ({measure_result.judge_model}) is grading a model "
        f"it IS ({named}) — the verdict may be self-biased. Pass --judge-model "
        "to use an independent judge."
    )


def _render_judge_methodology(measure_result: MeasureResult, *, verbose: bool) -> None:
    """Surface the self-judge CAUTION beneath a Tier-1 (--judge) sample.

    The self-judge caution is shown WHENEVER the resolved judge is itself one of
    the models it scored (judge == a candidate or == the baseline).  This is a
    trust signal, not a verbosity detail, so it is NOT gated on --verbose: the
    user must always know when a verdict is partly self-assessment.

    The judge-provenance line ("how it was measured") is no longer rendered here:
    it has moved UP to be the dim caption directly under the tally title
    (:func:`_render_judge_provenance_caption`), shown on every run (not
    verbose-gated), so the cluster reads provenance → tally → verdict → caveat.
    *verbose* is retained for signature stability with the caller and possible
    future verbose-only methodology detail.
    """
    judge = measure_result.judge_model
    if judge is None:  # Tier-0 — no judge ran
        return

    caution = _self_judge_caution_text(measure_result)
    if caution is not None:
        rprint(f"[{CAUTION_AMBER}][dim]{caution}[/dim][/{CAUTION_AMBER}]")


def render_quality_terminal(
    measure_result: MeasureResult,
    verbose: bool = False,
    limits: PreviewLimits | None = None,
    *,
    result: AnalysisResult | None = None,
) -> None:
    """Print the quality measurement section — replaces the unverified caveat.

    Tier-0: side-by-side output diffs (human is the judge).
    Tier-1: win/loss/tie tallies from the LLM judge.

    After the raw outputs / tallies, a synthesis line ties the measured result
    back to the offline 'within tolerance' estimate (confirmed / not confirmed
    for Tier-1, an unverified-estimate framing for Tier-0) so the two are never
    left disconnected.

    When *verbose* is True, the Tier-1 view additionally prints each sampled
    prompt's side-by-side outputs with every candidate output labelled by its
    per-prompt verdict (``gpt-4o-mini [LOSS]: …``), so a user can investigate
    WHICH prompts the candidate lost on — the detail the synthesis line's
    "review the losses" instruction points at.  Non-verbose Tier-1 output is
    byte-identical to the compact table + synthesis it has always been.
    """
    n = measure_result.samples_taken
    current = measure_result.current_model
    # Default to the terminal's historical limits when the caller passes none, so
    # a plain run (no --preview-chars / --no-truncate) is byte-identical to before.
    if limits is None:
        limits = PreviewLimits.terminal_default()

    # NOTE: no leading blank here — the CLI prints the separating blank BEFORE
    # the live sampling counter starts, so the gap below the Prices line exists
    # while the counter animates, not only once this section renders.

    if measure_result.tier1_tallies is not None:
        # Tier-1: show judge tallies.  The cluster reads top-to-bottom as
        # provenance (how it was measured) → tally → verdict/action → caveat:
        #
        #   1. the title,
        #   2. the dim judge-provenance CAPTION directly beneath it,
        #   3. the win/loss/tie tally table,
        #   4. the verdict + action (escalation ladder or confirmed/not line),
        #   5. a blank line, then the dim small-sample nudge (when it fires).
        #
        # The title is rendered as its own bold line (not the rich Table title)
        # so the provenance caption can sit BETWEEN the title and the tally — a
        # rich Table renders its title and grid as one unit, with no room for a
        # caption in between.
        rprint(
            f"[bold]Quality sample — judge results "
            f"({n:,} prompts, current: {current})[/bold]"
        )
        _render_unique_prompts_note_terminal(measure_result)
        _render_judge_provenance_caption(measure_result)
        judge_table = Table(
            show_header=True,
            header_style="bold",
        )
        judge_table.add_column("Candidate", style="cyan")
        judge_table.add_column("Win", justify="right", style="green")
        judge_table.add_column("Loss", justify="right", style="red")
        judge_table.add_column("Tie", justify="right", style="yellow")
        judge_table.add_column("Error", justify="right", style="dim")
        judge_table.add_column("Summary", justify="right", style="dim")

        for tally in measure_result.tier1_tallies:
            non_error = tally.wins + tally.losses + tally.ties
            summary = (
                f"{tally.wins + tally.ties}/{non_error} equivalent or better"
                if non_error > 0
                else "—"
            )
            judge_table.add_row(
                tally.candidate,
                str(tally.wins),
                str(tally.losses),
                str(tally.ties),
                str(tally.errors),
                summary,
            )
        rprint(judge_table)
        # Synthesis: tie the tally back to the offline quality estimate.
        _render_tier1_synthesis(measure_result, result=result)
        # Change 2 — promotion: a candidate excluded from the offline route for
        # being unrated that now CONFIRMED on the user's data AND saves more than
        # the headline rated pick is the better route.  A positive (green ✓)
        # upgrade prompt, not a warning.  Only fires with the AnalysisResult in
        # hand (the cost path that owns the recommendation); never on a measure-
        # only render.
        if result is not None:
            promo = _detect_promotion(result, measure_result)
            if promo is not None:
                rprint(f"[green]✓ {_promotion_message(promo)}[/green]")
        # Methodology: a self-judge caution (always, when applicable) and — under
        # --verbose — a dim Note naming the judge + A/B-order randomisation, so the
        # reader knows HOW the verdict was reached and whether it is arm's-length.
        _render_judge_methodology(measure_result, verbose=verbose)
        # Verbose only: the per-prompt detail the synthesis line points at.
        # Placed AFTER the synthesis ("review the losses" → here is what you
        # review), under a dim header, each candidate output verdict-labelled.
        if verbose:
            rprint("\n[dim]Per-prompt detail (verbose):[/dim]")
            for i, comp in enumerate(measure_result.comparisons, start=1):
                _render_comparison_prompt(comp, i, limits)
                _render_comparison_outputs(comp, label_verdicts=True, limits=limits)
    else:
        # Tier-0: side-by-side diffs (human judge)
        rprint(
            f"[bold]Quality sample[/bold] — {n:,} prompt(s), "
            f"current: [cyan]{current}[/cyan]"
        )
        _render_unique_prompts_note_terminal(measure_result)
        for i, comp in enumerate(measure_result.comparisons, start=1):
            _render_comparison_prompt(comp, i, limits)
            _render_comparison_outputs(comp, label_verdicts=False, limits=limits)
        # Framing: Tier-0 outputs are unscored; name what the user is looking at.
        _render_tier0_framing(measure_result, result=result)

    # Measurement-cost accounting — what this run cost the user (dim, quiet),
    # after the quality section and before the privacy closer.  No-op when the
    # run captured no usage to price.
    #
    # A blank line ABOVE it separates this RUN-LEVEL figure (the whole sample +
    # judge spend) from the last per-prompt detail directly above — so it reads as
    # a summary of the run, never as the final prompt's own cost.  It then groups
    # with the privacy footer below as one trailing block.
    if _measurement_cost_text(measure_result) is not None:
        rprint("")
    _render_measurement_cost_terminal(measure_result)

    rprint(
        f"\n[dim]{_QUALITY_PRIVACY_LINE}[/dim]\n"
    )


# ---------------------------------------------------------------------------
# Quality-measurement report section (Markdown + HTML)
# ---------------------------------------------------------------------------
#
# A written --report is the FULL view (there is no --verbose for a file), so the
# quality-measurement section mirrors render_quality_terminal's CONTENT in the
# report's own design language — never verbose-gated.  Both Tier-1 (--judge) and
# Tier-0 (--measure) shapes are rendered:
#
#   Tier-1 : the win/loss/tie tally table, the shared verdict synthesis line
#            (_classify_verdict — so report and terminal never disagree), and the
#            per-prompt detail (System/Prompt + each model's output, each carrying
#            its [WIN]/[TIE]/[LOSS] verdict label — the shareable proof).
#   Tier-0 : the per-prompt side-by-side outputs + the "raw samples — run --judge
#            for a scored verdict" framing.
#
# Colour discipline (HTML): WIN cyan, LOSS amber, TIE muted; green stays
# money-only.  Markdown uses text labels ([WIN]/[LOSS]/[TIE]) and ⚠ for the
# not-confirmed / borderline caution.  Outputs are truncated (longer limits than
# the terminal, since a report has room) to keep the file reasonable.

# The privacy invariant line, shared verbatim by the terminal and both report
# formats so the three surfaces never drift.
_QUALITY_PRIVACY_LINE = (
    "Quality measured on your own data using your own API keys. "
    "Nothing was sent to Rodiun or any Frugon endpoint."
)


def _fmt_measurement_money(amount: Decimal) -> str:
    """Format the measurement cost as 4-dp money, falling back to 6 dp sub-cent.

    The spec calls for 4-dp money (``$0.0043``).  A genuinely tiny run (a handful
    of short prompts) can cost less than $0.0001, where 4 dp would round to
    ``$0.0000`` and read as "free" — dishonest.  So, exactly like
    :func:`_fmt_usd`, a strictly-positive sub-$0.0001 amount falls back to 6 dp;
    everything else uses 4 dp.  Zero stays ``$0.0000`` (all calls were unpriced).
    """
    if Decimal("0") < amount < Decimal("0.0001"):
        return f"${float(amount):.6f}"
    return f"${float(amount):.4f}"


def _measurement_cost_text(measure_result: MeasureResult) -> str | None:
    """Build the plain (unstyled) measurement-cost line, or ``None`` to omit it.

    Returns ``None`` when no per-call usage was captured (a directly-constructed
    MeasureResult, or a run that made no calls) so a caller renders nothing.
    Otherwise returns the SINGLE canonical sentence shared verbatim by the
    terminal, Markdown and HTML surfaces (each adds only its own dim styling), so
    the three can never drift:

        Measurement cost  ~$0.0569 · 15 calls (your spend at list prices —
        check your provider dashboard for the exact bill)

    The cost is COMPUTED by :func:`frugon.measure.measurement_cost` from the run's
    own captured token counts and the LiteLLM list prices in frugon's pricing
    table.  It is not the provider's authoritative invoice (which may apply tier
    discounts, batch-API pricing, etc.) and the user's provider dashboard may
    lag minutes-to-hours — hence "your spend at list prices" (an estimate of
    what the user paid) rather than "your bill" (which implied authority frugon
    cannot have).  The privacy footer just below the cost line carries the
    "never sent to Rodiun/Frugon" trust statement; the older "your bill, never
    Frugon's" phrasing was the misleading half.

    When some calls' models were not in the pricing table, a ``· N calls
    unpriced`` clause is inserted BEFORE the parenthetical so the user sees the
    coverage caveat as part of the call-count clause, not trailing after the
    explanation:

        Measurement cost  ~$0.0569 · 15 calls · 3 calls unpriced (your spend at
        list prices — check your provider dashboard for the exact bill)
    """
    from frugon.measure import measurement_cost

    cost = measurement_cost(measure_result)
    if cost is None:
        return None
    money = _fmt_measurement_money(cost.total_cost)
    calls = f"{cost.call_count:,} call{'s' if cost.call_count != 1 else ''}"
    head = f"Measurement cost  ~{money} · {calls}"
    if cost.unpriced_calls:
        head += f" · {cost.unpriced_calls:,} calls unpriced"
    return (
        f"{head} (your spend at list prices — "
        "check your provider dashboard for the exact bill)"
    )


# Canonical, plain-text "honest sample size" note when the input log has FEWER
# distinct prompts than --samples requested.  Built and shared by every surface
# (terminal / Markdown / HTML) via :func:`_unique_prompts_note_text` so the
# wording cannot drift.  None when no note is owed (the log carried at least as
# many unique prompts as the user asked for, or the field was never populated).
def _unique_prompts_note_text(measure_result: MeasureResult) -> str | None:
    """Return the dim "honest unique-prompts" note for *measure_result*, or None.

    A run with a duplicate-heavy log can only compare on the distinct prompts
    it actually carries; sampling 10 of 3 unique prompts gives 3 comparisons,
    not 10.  When the renderer would otherwise read as a deceptive
    "Sampled 10 prompts", this surface tells the truth instead.

    Returns ``None`` when:
      * ``unique_prompts_available`` is 0 — the field was never populated
        (directly-constructed MeasureResult, or a run that bypassed
        :func:`frugon.measure.sample_records`);
      * the log carried at least as many unique prompts as the user asked for
        (``unique_prompts_available >= samples_requested``) — no note owed.
    """
    uniques = measure_result.unique_prompts_available
    requested = measure_result.samples_requested
    if uniques <= 0 or uniques >= requested:
        return None
    taken = measure_result.samples_taken
    return (
        f"Sampled {taken:,} prompt{'s' if taken != 1 else ''} "
        f"({uniques:,} unique available — your log has fewer distinct "
        "prompts than --samples requested)."
    )


def _render_unique_prompts_note_terminal(measure_result: MeasureResult) -> None:
    """Print the dim "honest unique-prompts" note on the terminal, or nothing.

    Quiet aside, rendered dim and matching the small-sample-nudge style — it
    qualifies the title above (which says "Sampled N prompts") with the truth
    when N exceeds what the log could supply.  No-op when no note is owed.
    """
    text = _unique_prompts_note_text(measure_result)
    if text is not None:
        rprint(f"[dim]{text}[/dim]")


def _render_measurement_cost_terminal(measure_result: MeasureResult) -> None:
    """Print the dim measurement-cost line on the terminal, or nothing.

    Rendered dim — it is a quiet accounting aside (what the run cost), not the
    verdict.  No-op when there is no captured usage to price.
    """
    text = _measurement_cost_text(measure_result)
    if text is not None:
        rprint(f"[dim]{text}[/dim]")

# Report-side preview limits.  A report has more room than a terminal line, so
# the prompt/output previews are longer than the terminal's
# _PROMPT_PREVIEW_CHARS / _OUTPUT_PREVIEW_CHARS — but still capped so a verbose
# sample cannot bloat the file unboundedly.
_REPORT_PROMPT_PREVIEW_CHARS = 400
_REPORT_OUTPUT_PREVIEW_CHARS = 800


def _comparison_system_message(comp: Comparison) -> str | None:
    """Return the first non-empty system message in a comparison's record, or None.

    Mirrors the terminal's :func:`_render_comparison_prompt` system-message
    lookup so the report surfaces the same context line.
    """
    return next(
        (
            m["content"]
            for m in comp.record.messages
            if m.get("role") == "system" and m.get("content")
        ),
        None,
    )


def _verdict_md_label(verdict: str) -> str:
    """Map a per-prompt verdict to its Markdown text label (``[WIN]`` …).

    Unknown/empty verdicts and the judge's own "error" collapse to ``[error]``
    so a row is never blank or mislabelled — mirroring the terminal's
    :func:`_verdict_label`.
    """
    if verdict in ("win", "loss", "tie"):
        return f"[{verdict.upper()}]"
    if verdict == _VERDICT_NO_COMPARISON:
        return "[no comparison]"
    return "[error]"


# HTML verdict-label classes — these are the per-prompt [WIN]/[LOSS]/[TIE]
# labels, recoloured to MATCH the tally table (WIN green, LOSS red, TIE yellow)
# so the per-prompt detail and the table never disagree.  Green-as-win is
# legitimate INSIDE the quality section; the "green = money" discipline governs
# the cost panel only.  Maps to the .verdict-* CSS injected into both report
# stylesheets.
_VERDICT_HTML_CLASS: dict[str, str] = {
    "win": "verdict-win",
    "loss": "verdict-loss",
    "tie": "verdict-tie",
}


def _verdict_html_label(verdict: str) -> str:
    """Render a per-prompt verdict as a styled HTML ``<span>`` label.

    WIN reads green, LOSS red, TIE yellow (matching the tally table); an
    unknown/errored verdict collapses to a muted ``[error]``.  The verdict token
    is already a safe literal, so no escaping is required.
    """
    if verdict in ("win", "loss", "tie"):
        cls = _VERDICT_HTML_CLASS[verdict]
        return f'<span class="{cls}">[{verdict.upper()}]</span>'
    if verdict == _VERDICT_NO_COMPARISON:
        return '<span class="verdict-error">[no comparison]</span>'
    return '<span class="verdict-error">[error]</span>'


# CSS for the quality-measurement section, appended to BOTH report stylesheets.
# Colour law INSIDE the quality section: WIN green, LOSS red, TIE yellow, error
# muted — matching the tally table so the per-prompt labels and the table agree.
# (Green-as-win is legitimate here; the cost panel keeps green=money discipline.)
# The green/red tokens reuse the report palette (--green / --red, with literal
# fallbacks) and yellow reuses the existing caution amber (#F59E0B).
_QUALITY_HTML_CSS = """
/* Quality-measurement section (--measure / --judge). WIN green, LOSS red,
   TIE yellow — matching the tally table; the cost panel keeps green=money. */
.quality-tally{width:100%;border-collapse:collapse;margin:.5rem 0 1rem}
.quality-tally th,.quality-tally td{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:.82rem;padding:.45rem .7rem;text-align:right;border-bottom:1px solid var(--border,#2C2C2C);
}
.quality-tally th{text-transform:uppercase;letter-spacing:.08em;font-size:.66rem;text-align:right;color:var(--ink-dim,#6B6B72)}
/* Judge-provenance caption — "how it was measured", directly under the title
   and above the tally.  Dim, quiet; sits as a caption, never competing with the
   verdict/action below the tally. */
.quality-provenance{font-size:.78rem;line-height:1.5;margin:-.25rem 0 .5rem;color:var(--ink-dim,#6B6B72)}
.quality-tally th:first-child,.quality-tally td:first-child{text-align:left}
.quality-tally td.cand{color:var(--cyan,#00D1FF)}
.quality-tally td.t-win{color:var(--green,#10B981)}
.quality-tally td.t-loss{color:var(--red,#F87171)}
.quality-tally td.t-tie{color:#F59E0B}
.quality-tally td.summary{text-align:right;color:var(--ink-dim,#6B6B72)}
.quality-verdict{font-size:.86rem;line-height:1.6;margin:.4rem 0 1rem}
.quality-verdict.neutral{color:var(--cyan,#00D1FF)}
.quality-verdict.caution{color:#F59E0B}
/* Escalation ladder — the next rung up after a NOT-confirmed verdict. Amber
   prose carries the caution; the model name + command are cyan; the saving is a
   comparison figure (not realised money), so it stays amber, never green. */
.quality-escalation{font-size:.86rem;line-height:1.6;margin:-.6rem 0 1rem;color:#F59E0B}
.quality-escalation .model-name{color:var(--cyan,#00D1FF)}
.quality-escalation .esc-cmd{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  color:var(--cyan,#00D1FF);font-size:.82rem;
}
/* Small-sample nudge — a quiet caveat, set off from the verdict/action above it
   with a small top gap (it is an aside, not the takeaway). */
.quality-nudge{font-size:.8rem;line-height:1.5;margin:.5rem 0 1rem;color:var(--ink-dim,#6B6B72)}
/* Self-judge caution — amber trust signal, adjacent to the provenance caption,
   shown when the judge is itself one of the models it scored. */
.quality-self-judge{font-size:.8rem;line-height:1.5;margin:-.25rem 0 .75rem;color:#F59E0B}
.quality-prompt{margin:1rem 0 .25rem;padding-top:.75rem;border-top:1px solid var(--surface-2,#151515)}
.quality-prompt .qp-label{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-dim,#6B6B72);
}
.quality-prompt .qp-sys{display:block;color:var(--ink-dim,#6B6B72);font-size:.8rem;margin:.15rem 0}
.quality-prompt .qp-text{display:block;color:var(--ink-mute,#A1A1AA);font-size:.85rem;margin:.15rem 0}
.quality-output{margin:.3rem 0 .3rem 1rem;font-size:.85rem;line-height:1.55}
.quality-output .qo-model{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  color:var(--ink-mute,#A1A1AA);
}
.quality-output.baseline .qo-model{color:var(--cyan,#00D1FF)}
.quality-output .qo-current{color:var(--ink-dim,#6B6B72)}
.quality-output .qo-text{color:var(--ink-mute,#A1A1AA);white-space:pre-wrap;word-break:break-word}
.verdict-win{color:var(--green,#10B981);font-weight:600}
.verdict-loss{color:var(--red,#F87171);font-weight:600}
.verdict-tie{color:#F59E0B;font-weight:600}
.verdict-error{color:var(--ink-dim,#6B6B72)}
/* Promotion callout (Change 2) — a measured-confirmed unrated candidate that now
   saves more than the headline rated pick.  Positive (green), NOT amber: it is
   good news, the better route unlocked by verification. */
.quality-promotion{font-size:.85rem;line-height:1.5;margin:.5rem 0 1rem;color:var(--green,#10B981);font-weight:600}
.quality-privacy{font-size:.78rem;color:var(--ink-dim,#6B6B72);margin-top:.75rem;font-style:italic}
/* Measurement-cost line — what the run cost the user. Dim accounting aside,
   sits below the quality detail and above the privacy closer. */
/* The run-level measurement-cost line — the WHOLE measure+judge spend, not a
   per-prompt figure.  A clear top margin detaches it from the last per-prompt
   block above so it reads as a run summary, and it sits flush with the privacy
   line below it as one trailing block. */
.quality-measure-cost{font-size:.78rem;color:var(--ink-dim,#6B6B72);margin:1.5rem 0 .25rem}
"""


def _quality_css(measure_result: MeasureResult | None) -> str:
    """Return the quality-section CSS to splice into a stylesheet, or ``""``.

    Returns ``_QUALITY_HTML_CSS`` ONLY when a measure run actually produced a
    section to style; otherwise the empty string, so a report without a measure
    run keeps a byte-identical ``<style>`` block (no dead rules) — preserving the
    byte-identity guarantee end to end, not just in the body.
    """
    return _QUALITY_HTML_CSS if measure_result is not None else ""


def _quality_section_md(
    measure_result: MeasureResult | None,
    limits: PreviewLimits | None = None,
    *,
    result: AnalysisResult | None = None,
) -> list[str]:
    """Build the Markdown 'Quality measurement' section, or ``[]`` when absent.

    Returns a list of Markdown lines (no trailing blank) when *measure_result*
    is given, and the EMPTY LIST when it is None — so a report without a measure
    run is byte-identical to today.

    *limits* carries the (possibly flag-overridden) REPORT preview lengths;
    ``None`` defaults to :meth:`PreviewLimits.report_default` so a report with no
    display flag set is byte-identical to before.

    Tier-1 (tallies present): the win/loss/tie table, the shared verdict
    synthesis line(s) (verbatim from :func:`_classify_verdict`, with a ``⚠`` for
    the caution states) and the per-prompt detail with ``[WIN]``/``[TIE]``/
    ``[LOSS]`` labels.  Tier-0: the per-prompt side-by-side + the "run --judge"
    framing.  Both end with the shared privacy line.
    """
    if measure_result is None:
        return []
    if limits is None:
        limits = PreviewLimits.report_default()

    n = measure_result.samples_taken
    current = measure_result.current_model
    lines: list[str] = ["## Quality measurement", ""]

    if measure_result.tier1_tallies is not None:
        lines += [
            f"Judge results — **{n:,}** sampled prompt(s), current model `{current}`.",
            "",
        ]
        # Honest unique-prompts note — qualifies the count above when the log
        # had fewer distinct prompts than --samples requested.  Dim italic
        # (markdown _emphasis_), shared verbatim across surfaces via
        # :func:`_unique_prompts_note_text` so the wording cannot drift.
        unique_note = _unique_prompts_note_text(measure_result)
        if unique_note is not None:
            lines += [f"_{unique_note}_", ""]
        # Provenance CAPTION (how it was measured) directly under the title,
        # above the tally — dim, rendered as a Markdown italic line.  Shared
        # verbatim with the terminal/HTML via _judge_provenance_text.
        provenance = _judge_provenance_text(measure_result)
        if provenance is not None:
            lines += [f"_{provenance}_", ""]
        # Self-judge caution — the provenance says WHO judged; this qualifies it
        # when the judge is itself one of the compared models.  Shared verbatim
        # with the terminal/HTML via _self_judge_caution_text; rendered as an
        # amber-marked blockquote so it reads as a caution, not body text.
        self_judge = _self_judge_caution_text(measure_result)
        if self_judge is not None:
            lines += [f"> ⚠ {self_judge}", ""]
        lines += [
            "| Candidate | Win | Loss | Tie | Error | Summary |",
            "|-----------|----:|-----:|----:|------:|--------:|",
        ]
        for tally in measure_result.tier1_tallies:
            non_error = tally.wins + tally.losses + tally.ties
            summary = (
                f"{tally.wins + tally.ties}/{non_error} equivalent or better"
                if non_error > 0
                else "—"
            )
            lines.append(
                f"| `{tally.candidate}` | {tally.wins:,} | {tally.losses:,} "
                f"| {tally.ties:,} | {tally.errors:,} | {summary} |"
            )
        lines.append("")
        # Verdict synthesis — the SAME sentence the terminal renders, so the
        # report and the terminal never disagree.  Caution states carry a ⚠.
        # On a NOT-confirmed verdict, if a cheaper higher-tier model exists the
        # synthesis escalates to the next rung up instead of the dead-end
        # "keep these on the baseline" line — verbatim the terminal's wording.
        md_baseline_failed = _baseline_all_errored(measure_result)
        for tally in measure_result.tier1_tallies:
            state, text = _classify_verdict(
                tally,
                current,
                baseline_all_errored=md_baseline_failed,
                result=result,
            )
            mark = "⚠ " if state in _VERDICT_CAUTION_STATES else ""
            # Fix B — Markdown has no colour, so bold JUST the status word so it
            # still stands out from the rest of the sentence (single-sourced).
            md_status = _verdict_status_md(state)
            suggestion = (
                _escalation_for_tally(tally, current)
                if state == _VERDICT_NOT_CONFIRMED
                else None
            )
            if suggestion is not None:
                lead = _emphasise_verdict_status(
                    _escalation_lead(tally), state, md_status
                )
                lines.append(f"> {mark}{lead}")
                lines.append(f"> `{_escalation_detail(suggestion, current)}`")
            else:
                lines.append(f"> {mark}{_emphasise_verdict_status(text, state, md_status)}")
            nudge = _nudge_text(tally)
            if nudge is not None:
                # A blank quote line separates the dim caveat from the verdict
                # /action above it — the nudge is a quiet aside, not the takeaway.
                lines.append(">")
                lines.append(f"> _{nudge}_")
        lines.append("")
        # Change 2 — promotion callout (green ✓, positive): a confirmed unrated
        # candidate that now saves more than the headline rated pick.  Single-
        # sourced via _detect_promotion / _promotion_message; only with the
        # AnalysisResult in hand.  Rendered as its own ✓ blockquote (NOT a ⚠
        # caution) so it reads as the positive upgrade it is.
        if result is not None:
            promo = _detect_promotion(result, measure_result)
            if promo is not None:
                lines.append(f"> ✓ {_promotion_message(promo)}")
                lines.append("")
        # Per-prompt detail — the shareable proof: each sampled prompt with each
        # model's output and its [WIN]/[TIE]/[LOSS] verdict label.
        lines += ["### Per-prompt detail", ""]
        for i, comp in enumerate(measure_result.comparisons, start=1):
            lines += _quality_prompt_md(comp, i, label_verdicts=True, limits=limits)
    else:
        lines += [
            f"Raw samples — **{n:,}** sampled prompt(s), current model `{current}`. "
            "These are side-by-side outputs for you to compare; "
            "run `--judge` for a scored verdict.",
            "",
        ]
        # Honest unique-prompts note — same wording + placement discipline as
        # the Tier-1 path; shared verbatim via _unique_prompts_note_text.
        unique_note = _unique_prompts_note_text(measure_result)
        if unique_note is not None:
            lines += [f"_{unique_note}_", ""]
        for i, comp in enumerate(measure_result.comparisons, start=1):
            lines += _quality_prompt_md(comp, i, label_verdicts=False, limits=limits)

    # Measurement-cost accounting (dim italic) — what this RUN cost the user (the
    # whole sample + judge spend), after the quality detail and before the privacy
    # closer.  The leading "" forces a true paragraph break so the run-level figure
    # detaches from the last per-prompt block above (never read as that prompt's
    # own cost); it then groups with the privacy line below as one trailing block.
    # Omitted when the run captured no usage to price.
    cost_text = _measurement_cost_text(measure_result)
    if cost_text is not None:
        lines += ["", f"_{cost_text}_"]

    lines += ["", f"_{_QUALITY_PRIVACY_LINE}_"]
    return lines


def _md_inline_escape(text: str) -> str:
    """Neutralise Markdown control characters in a SINGLE-LINE inline preview.

    Used for the System/Prompt preview lines, which are already truncated to one
    short line: backslash-escaping the inline control characters keeps the text
    literal (a model-authored ``# heading`` or ``*bold*`` renders as plain text,
    never as report structure) while staying a readable inline bullet.  Newlines
    are folded to a single space so a multi-line preview can never break the
    bullet — the full, unmodified output is shown fenced below, this is only the
    glanceable prompt context.
    """
    folded = " ".join(text.split())
    out: list[str] = []
    for ch in folded:
        # The inline-meaningful set: emphasis/code (`*_`` ` ``), link/heading
        # (`[]<>#`), and the table/escape glyphs (`|\`).  A bare `.`/`-`/`!`/
        # parenthesis mid-line carries no inline structure, so it is left
        # readable — only structure-bearing glyphs are neutralised.
        if ch in "\\`*_[]<>#|":
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _md_fence_for(text: str) -> str:
    """Return a backtick fence long enough to safely wrap *text* verbatim.

    A code fence must be longer than the longest backtick run INSIDE the content,
    otherwise the content's own ``` would close the fence early and leak the
    remainder as live Markdown.  Scan for the longest backtick run and return a
    fence of at least three backticks, one longer than that run.
    """
    longest = 0
    run = 0
    for ch in text:
        if ch == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return "`" * max(3, longest + 1)


def _md_fenced_output_lines(label: str, body: str, *, indent: str = "") -> list[str]:
    """Render a model output as a label line + a fenced literal code block.

    The *label* (already containing its own inline code spans / verdict tag, e.g.
    ``` `gpt-4o-mini` [LOSS]: ```) stays a bullet line; the *body* — the raw model
    output — is emitted inside a backtick fence on the following lines so NOTHING
    in it is interpreted as Markdown structure (headings, lists, raw ``<script>``
    all render as literal text).  The fence length adapts to any backtick run in
    *body* (see :func:`_md_fence_for`).  *indent* is prepended to the fence lines
    for clean rendering under a nested list item.
    """
    fence = _md_fence_for(body)
    lines = [f"- {label}", "", f"{indent}{fence}"]
    # The body is literal: preserve its own line breaks, indenting each line so a
    # nested fence renders correctly.  An empty body still yields a visible empty
    # fenced block (the "[no output]" sentinel is handled by the caller).
    for raw_line in body.split("\n"):
        lines.append(f"{indent}{raw_line}" if indent else raw_line)
    lines.append(f"{indent}{fence}")
    lines.append("")
    return lines


def _quality_prompt_md(
    comp: Comparison, index: int, *, label_verdicts: bool, limits: PreviewLimits
) -> list[str]:
    """Markdown for one sampled prompt: System/Prompt + each model's output.

    Model-authored text is UNTRUSTED: each model output is rendered inside a
    backtick code fence (see :func:`_md_fenced_output_lines`) so its own headings,
    lists, fences, or raw ``<script>`` render as literal text and can never
    impersonate the report's own structure or inject HTML.  The short System /
    Prompt previews are inline-escaped (:func:`_md_inline_escape`) for the same
    reason — they stay glanceable bullet lines while staying literal.  *limits*
    carries the (possibly flag-overridden) report preview lengths.
    """
    lines: list[str] = []
    system_msg = _comparison_system_message(comp)
    last_msg = comp.record.messages[-1]["content"]
    lines.append(f"**Prompt {index}**")
    lines.append("")
    if system_msg:
        lines.append(
            f"- _System:_ {_md_inline_escape(limits.truncate_prompt(system_msg))}"
        )
    lines.append(
        f"- _Prompt:_ {_md_inline_escape(limits.truncate_prompt(last_msg))}"
    )
    lines.append("")
    cur_body = limits.truncate_output(comp.current_output.content)
    if cur_body:
        lines += _md_fenced_output_lines(
            f"`{comp.current_output.model}` (current):", cur_body
        )
    else:
        lines.append(f"- `{comp.current_output.model}` (current): [no output]")
        lines.append("")
    for pos, cand_out in enumerate(comp.candidate_outputs):
        cand_body = limits.truncate_output(cand_out.content)
        label = ""
        if label_verdicts and pos < len(comp.verdicts):
            label = f" {_verdict_md_label(_refine_prompt_verdict(comp, pos))}"
        if cand_body:
            lines += _md_fenced_output_lines(f"`{cand_out.model}`{label}:", cand_body)
        else:
            # No content — show the captured error (inline-escaped) or a sentinel;
            # neither is model-authored long-form, so an inline bullet is fine.
            fallback = _md_inline_escape(cand_out.error) if cand_out.error else "[no output]"
            lines.append(f"- `{cand_out.model}`{label}: {fallback}")
            lines.append("")
    return lines


def _quality_section_html(
    measure_result: MeasureResult | None,
    *,
    style: str = "v1",
    limits: PreviewLimits | None = None,
    result: AnalysisResult | None = None,
) -> str:
    """Build the HTML 'Quality measurement' section, or ``""`` when absent.

    Returns a wrapped section (a ``<div class="card">`` for the v1 surface, a
    ``<section class="below">`` for v2) when *measure_result* is given, and the
    EMPTY STRING when it is None — so a report without a measure run is
    byte-identical to today.  Colour discipline: WIN cyan, LOSS amber, TIE
    muted; green is never used here.  The section's CSS (``_QUALITY_HTML_CSS``)
    is appended to BOTH stylesheets, so the same markup styles correctly in
    either surface.  *limits* carries the (possibly flag-overridden) report
    preview lengths; ``None`` defaults to :meth:`PreviewLimits.report_default`.
    """
    if measure_result is None:
        return ""
    if limits is None:
        limits = PreviewLimits.report_default()

    esc = _html_escape.escape
    n = measure_result.samples_taken
    current = esc(measure_result.current_model)
    open_tag, close_tag = (
        ('<section class="below">', "</section>")
        if style == "v2"
        else ('<div class="card">', "</div>")
    )
    parts: list[str] = [
        open_tag,
        '<div class="eyebrow">Quality measurement</div>',
    ]

    if measure_result.tier1_tallies is not None:
        parts.append(
            f'<p class="caveat">Judge results — <strong>{n:,}</strong> sampled '
            f'prompt(s), current model <span class="model-name">{current}</span>.</p>'
        )
        # Honest unique-prompts note — qualifies the title above when the log
        # had fewer distinct prompts than --samples requested.  Shared verbatim
        # with terminal/Markdown via :func:`_unique_prompts_note_text`; reuses
        # the .quality-provenance dim style so it sits as a quiet caption
        # alongside (and above) the existing judge-provenance line.
        unique_note = _unique_prompts_note_text(measure_result)
        if unique_note is not None:
            parts.append(
                f'<p class="quality-provenance">{esc(unique_note)}</p>'
            )
        # Provenance CAPTION (how it was measured) directly under the title,
        # above the tally — dim, shared verbatim with the terminal/Markdown.
        provenance = _judge_provenance_text(measure_result)
        if provenance is not None:
            parts.append(f'<p class="quality-provenance">{esc(provenance)}</p>')
        # Self-judge caution — adjacent to the provenance caption (provenance says
        # WHO judged; this qualifies it when the judge IS one of the compared
        # models).  Amber, shared verbatim with terminal/Markdown via
        # _self_judge_caution_text so the surfaces cannot drift.
        self_judge = _self_judge_caution_text(measure_result)
        if self_judge is not None:
            parts.append(
                f'<p class="quality-self-judge">&#9888; {esc(self_judge)}</p>'
            )
        rows = ""
        for tally in measure_result.tier1_tallies:
            non_error = tally.wins + tally.losses + tally.ties
            summary = (
                f"{tally.wins + tally.ties}/{non_error} equivalent or better"
                if non_error > 0
                else "—"
            )
            rows += (
                "<tr>"
                f'<td class="cand">{esc(tally.candidate)}</td>'
                f'<td class="t-win">{tally.wins:,}</td>'
                f'<td class="t-loss">{tally.losses:,}</td>'
                f'<td class="t-tie">{tally.ties:,}</td>'
                f"<td>{tally.errors:,}</td>"
                f'<td class="summary">{esc(summary)}</td>'
                "</tr>"
            )
        parts.append(
            '<table class="quality-tally"><thead><tr>'
            "<th>Candidate</th><th>Win</th><th>Loss</th><th>Tie</th>"
            "<th>Error</th><th>Summary</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )
        # Verdict synthesis — verbatim from the shared classifier.  A
        # NOT-confirmed verdict escalates to the next rung up (cyan model name +
        # command) when a cheaper higher-tier model exists, matching the terminal
        # and Markdown; otherwise it keeps the dead-end guidance.  The
        # small-sample nudge, when applicable, follows in a dim line.
        current_model = measure_result.current_model
        html_baseline_failed = _baseline_all_errored(measure_result)
        for tally in measure_result.tier1_tallies:
            state, text = _classify_verdict(
                tally,
                current_model,
                baseline_all_errored=html_baseline_failed,
                result=result,
            )
            cls = "caution" if state in _VERDICT_CAUTION_STATES else "neutral"
            mark = "&#9888; " if state in _VERDICT_CAUTION_STATES else ""
            # Fix B — colour JUST the status word with the matching tally class
            # (WIN green / TIE amber / LOSS red; neutral states stay uncoloured)
            # so the verdict word stands out and agrees with the tally table.
            # The text is escaped first; the phrase has no HTML-special chars, so
            # it still matches for the in-place wrap.
            html_status = _verdict_status_html(state)
            suggestion = (
                _escalation_for_tally(tally, current_model)
                if state == _VERDICT_NOT_CONFIRMED
                else None
            )
            if suggestion is not None:
                lead = _emphasise_verdict_status(
                    esc(_escalation_lead(tally)), state, html_status
                )
                parts.append(f'<p class="quality-verdict {cls}">{mark}{lead}</p>')
                parts.append(
                    '<p class="quality-escalation">'
                    f'<span class="model-name">{esc(suggestion.model)}</span> '
                    f"({esc(suggestion.tier_label)} tier, still "
                    f"~{suggestion.pct_cheaper_than_baseline}% cheaper than "
                    f"{esc(current_model)}): "
                    f'<code class="esc-cmd">{esc(suggestion.command)}</code></p>'
                )
            else:
                body = _emphasise_verdict_status(esc(text), state, html_status)
                parts.append(
                    f'<p class="quality-verdict {cls}">{mark}{body}</p>'
                )
            nudge = _nudge_text(tally)
            if nudge is not None:
                parts.append(f'<p class="quality-nudge">{esc(nudge)}</p>')
        # Change 2 — promotion callout (positive ✓): a confirmed unrated candidate
        # that now saves more than the headline rated pick.  Single-sourced via
        # _detect_promotion / _promotion_message; only with the AnalysisResult in
        # hand.  Rendered in the positive ``.quality-promotion`` class (green ✓),
        # NOT the amber ``.caution`` — it is good news, not a warning.
        if result is not None:
            promo = _detect_promotion(result, measure_result)
            if promo is not None:
                parts.append(
                    '<p class="quality-promotion">&#10003; '
                    f"{esc(_promotion_message(promo))}</p>"
                )
        # Per-prompt detail — the shareable proof.
        for i, comp in enumerate(measure_result.comparisons, start=1):
            parts.append(
                _quality_prompt_html(comp, i, label_verdicts=True, limits=limits)
            )
    else:
        parts.append(
            f'<p class="caveat">Raw samples — <strong>{n:,}</strong> sampled '
            f'prompt(s), current model <span class="model-name">{current}</span>. '
            "These are side-by-side outputs for you to compare; "
            "run <code>--judge</code> for a scored verdict.</p>"
        )
        # Honest unique-prompts note — same placement discipline as the Tier-1
        # path; shared verbatim via _unique_prompts_note_text.
        unique_note = _unique_prompts_note_text(measure_result)
        if unique_note is not None:
            parts.append(
                f'<p class="quality-provenance">{esc(unique_note)}</p>'
            )
        for i, comp in enumerate(measure_result.comparisons, start=1):
            parts.append(
                _quality_prompt_html(comp, i, label_verdicts=False, limits=limits)
            )

    # Measurement-cost accounting (dim) — what this run cost the user, after the
    # quality detail and before the privacy closer.  Omitted when no usage was
    # captured to price.
    cost_text = _measurement_cost_text(measure_result)
    if cost_text is not None:
        parts.append(f'<p class="quality-measure-cost">{esc(cost_text)}</p>')

    parts.append(
        f'<p class="quality-privacy">{esc(_QUALITY_PRIVACY_LINE)}</p>'
    )
    parts.append(close_tag)
    return "".join(parts)


def _quality_prompt_html(
    comp: Comparison, index: int, *, label_verdicts: bool, limits: PreviewLimits
) -> str:
    """HTML for one sampled prompt: System/Prompt + each model's output.

    *limits* carries the (possibly flag-overridden) report preview lengths.
    """
    esc = _html_escape.escape
    system_msg = _comparison_system_message(comp)
    last_msg = comp.record.messages[-1]["content"]
    parts: list[str] = ['<div class="quality-prompt">']
    parts.append(f'<span class="qp-label">Prompt {index}</span>')
    if system_msg:
        parts.append(
            '<span class="qp-sys"><strong>System:</strong> '
            f"{esc(limits.truncate_prompt(system_msg))}</span>"
        )
    parts.append(
        '<span class="qp-text"><strong>Prompt:</strong> '
        f"{esc(limits.truncate_prompt(last_msg))}</span>"
    )
    parts.append("</div>")
    cur_text = (
        esc(limits.truncate_output(comp.current_output.content))
        or "[no output]"
    )
    parts.append(
        '<div class="quality-output baseline">'
        f'<span class="qo-model">{esc(comp.current_output.model)}</span> '
        '<span class="qo-current">(current)</span>: '
        f'<span class="qo-text">{cur_text}</span></div>'
    )
    for pos, cand_out in enumerate(comp.candidate_outputs):
        cand_text = (
            esc(limits.truncate_output(cand_out.content))
            or esc(cand_out.error or "")
            or "[no output]"
        )
        label = ""
        if label_verdicts and pos < len(comp.verdicts):
            label = " " + _verdict_html_label(_refine_prompt_verdict(comp, pos))
        parts.append(
            '<div class="quality-output">'
            f'<span class="qo-model">{esc(cand_out.model)}</span>{label}: '
            f'<span class="qo-text">{cand_text}</span></div>'
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Shared report disclosure helpers (Markdown + HTML)
# ---------------------------------------------------------------------------
#
# The reports are the FULL view -- there is no verbose mode -- so every
# freshness, caution, log-span and upper-bound disclosure the terminal surfaces
# under --verbose (or in its Accounting block) belongs in every report too.
# These helpers compute that content ONCE, from the SAME predicates the terminal
# uses (``_days_old`` / ``_is_pricing_stale`` / ``_is_quality_stale`` /
# ``window_contradicts_span`` / ``compute_saving_pct``), so a number can never
# drift between the terminal and a report or between the four report variants.

_PRICING_REFRESH_COMMAND = "frugon pricing update"
_QUALITY_REFRESH_COMMAND = "frugon quality update"


def _staleness_age(last_synced: str) -> str:
    """Return the age phrase ("N days old" / "out of date") for a stale date.

    Mirrors the terminal's :func:`_print_synced_row`: the day count is computed
    with the SAME :func:`_days_old` helper, and a date the helper cannot parse
    falls back to "out of date" rather than a nonsense count.
    """
    days = _days_old(last_synced)
    return f"{days} days old" if days is not None else "out of date"


def _upper_bound_pct(result: AnalysisResult) -> Decimal | None:
    """Return the wholesale full-swap saving % when it beats the split headline.

    The single gate the terminal's Upper-bound row and verbose note both use, and
    the one source the Markdown/HTML reports key off too: a wholesale full-swap
    candidate exists and its full-dataset saving (via :func:`compute_saving_pct`
    on ``total_cost`` then ``projected_cost``) is strictly larger than the
    conservative split saving.

    The full-swap basis context is surfaced even when the wholesale winner is the
    SAME model as the split's easy-call target (audit finding #2).  The split
    routes only the *easy baseline* calls to that model, whereas the full swap
    moves *every* call to it, so the two figures differ — e.g. a single
    ``--candidates gpt-4o-mini`` run shows a ~35% split headline but an ~98%
    full-swap upper bound, and the user must be able to tell that the larger
    figure is the aggressive full-swap basis.  The strictly-greater check
    (``wholesale <= split.saving_pct`` → None) keeps it honest and non-redundant:
    when the split already moves the whole dataset (a 100%-routed wholesale case),
    the split and full-swap savings coincide and no separate upper bound is shown.

    Returns None when there is no candidate or the full swap does not beat the
    split -- so the reports never quote an "upper bound" equal to or below the
    headline.
    """
    split = result.split
    if split is None or split.saving_pct is None:
        return None
    if not result.candidate_model:
        return None
    # Non-displayed use: compute_saving_pct is intentional here.  The returned value
    # is used only as an informational upper-bound context note ("see upper-bound
    # section") — it is NOT printed adjacent to a Current / After dollar pair.
    wholesale = compute_saving_pct(result.total_cost, result.projected_cost)
    if wholesale is None or wholesale <= split.saving_pct:
        return None
    return wholesale


# --- Markdown wrappers (no colour -> a leading-warning callout line) ---------


def _md_quality_tier_line(split: SplitRouting) -> str:
    """Markdown ``- **Quality tier:**`` bullet for the split section's Details block.

    Discloses the published LMArena quality CLASS the routing moves between, e.g.
    ``- **Quality tier:** `gpt-4-turbo` unrated → `gpt-4o-mini` Capable (LMArena)``.
    Uses the shared :func:`_tier_label` so the tier shown here reconciles with the
    terminal and HTML surfaces; an unrated model reads ``unrated`` rather than
    being dropped, so the gap is marked.  Model names are code-spanned (the
    Markdown analogue of the terminal/HTML cyan); the source tag stays plain.
    """
    return (
        f"- **Quality tier:** `{split.baseline_model}` {_tier_label(split.baseline_model)} "
        f"→ `{split.candidate_model}` {_tier_label(split.candidate_model)} (LMArena)"
    )


def _md_freshness_lines(result: AnalysisResult) -> list[str]:
    """Markdown lines disclosing pricing.json + quality.json freshness.

    One ``**... last synced:** <date>`` bullet per table that carries a date,
    each followed by a ``>`` blockquote caution when the table is stale (pricing
    at >30 days, quality at >60 -- the terminal's thresholds).  A table without a
    stored date is omitted entirely.  Empty list when neither date exists.
    """
    lines: list[str] = []
    for label, date_str, stale, refresh in _freshness_rows(result):
        lines.append(f"- **{label} last synced:** {date_str}")
        if stale:
            lines.append(
                f"> ⚠ {_staleness_age(date_str)} — refresh with `{refresh}`."
            )
    return lines


def _freshness_rows(
    result: AnalysisResult,
) -> list[tuple[str, str, bool, str]]:
    """Return ``(label, date, stale, refresh_command)`` for each table with a date.

    The single source the Markdown freshness/staleness wrappers iterate over so
    pricing/quality labels, the 30/60-day thresholds and the refresh commands
    are declared once.  A table whose date is absent is omitted.
    """
    rows: list[tuple[str, str, bool, str]] = []
    if result.pricing_json_last_synced:
        rows.append(
            (
                "Pricing",
                result.pricing_json_last_synced,
                _is_pricing_stale(result.pricing_json_last_synced, max_days=30),
                _PRICING_REFRESH_COMMAND,
            )
        )
    if result.quality_json_last_synced:
        rows.append(
            (
                "Quality",
                result.quality_json_last_synced,
                _is_quality_stale(result.quality_json_last_synced, max_days=60),
                _QUALITY_REFRESH_COMMAND,
            )
        )
    return rows


def _md_staleness_callouts(result: AnalysisResult) -> list[str]:
    """Markdown ``>`` staleness callouts only (no synced bullets).

    For the v2 surface, where the synced dates already ride an inline meta line;
    this emits just the ``> WARNING`` caution for each table that is stale, naming
    the age and the refresh command.  Empty list when nothing is stale.
    """
    lines: list[str] = []
    for label, date_str, stale, refresh in _freshness_rows(result):
        if stale:
            lines.append(
                f"> ⚠ {label} table is {_staleness_age(date_str)} — refresh "
                f"with `{refresh}`."
            )
    return lines


def _md_window_caution_lines(result: AnalysisResult) -> list[str]:
    """Markdown ``>`` caution callout when ``--window`` contradicts the span.

    Gated by :func:`window_contradicts_span` exactly as the terminal's Window
    caution row is, with the same wording.  Empty list otherwise.
    """
    window = result.window_days
    span = result.observed_span_days
    if not window_contradicts_span(window, span):
        return []
    assert window is not None  # narrowed by the predicate
    assert span is not None
    span_days = round(span)
    return [
        f"> ⚠ `--window {window}` overrides your log's actual ~{span_days}-day "
        f"span — the monthly figure is projected as if the data covered "
        f"{window} days. Drop `--window` to project from the real span."
    ]


def _md_log_span_line(result: AnalysisResult) -> str | None:
    """Markdown line disclosing the observed log span, or None when absent.

    Matches the terminal's verbose Log-span row: ``earliest -> latest (N.d
    days)``, emitted only when both span bounds are present.
    """
    start = result.observed_span_start
    end = result.observed_span_end
    if start is None or end is None:
        return None
    span = (
        f" ({result.observed_span_days:.1f} days)"
        if result.observed_span_days is not None
        else ""
    )
    return f"- **Log span:** {start} → {end}{span}"


def _md_unrated_family_lines(
    result: AnalysisResult,
    judged_models: frozenset[str] = frozenset(),
) -> list[str]:
    """Markdown callout lines for the unrated-message family (findings #1 + #4).

    Each shared message (from :func:`_unrated_family_messages`) becomes its own
    blockquote callout so the disclosure is as scannable in the report as the
    footer line is in the terminal — same wording, surface-appropriate styling.
    Severity decides the glyph: a genuine "quality unverified" CAUTION keeps the
    ``> ⚠`` amber callout, while a measured-below INFORMATIONAL note renders as a
    plain ``> `` blockquote WITHOUT the ⚠ (Markdown has no colour; dropping the
    glyph is how the info idiom reads as neutral, not a caution).  Model names
    already read inline (no code-span) so the sentence stays a single quoted
    line.  *judged_models* (the models measured in this run) makes the family
    measurement-aware — see :func:`_unrated_family_messages`.  Empty list when no
    unrated model is recommended/held out (so rated runs and the --demo path are
    byte-identical).
    """
    return [
        f"> ⚠ {message}" if severity == _SEV_WARNING else f"> {message}"
        for message, severity in _unrated_family_messages(result, judged_models)
    ]


def _md_upper_bound_lines(result: AnalysisResult) -> list[str]:
    """Markdown upper-bound disclosure for the split headline (Item 6 parity).

    Mirrors MD v1's existing line and the terminal's Upper-bound note: names the
    aggressive full-swap saving and frames it against the conservative split
    headline.  Empty list when no distinct candidate beats the split.
    """
    upper = _upper_bound_pct(result)
    if upper is None or result.split is None or result.split.saving_pct is None:
        return []
    # The quoted split percentage is the SAME total-dataset figure the headline
    # shows (via the shared helper), NOT split.saving_pct (baseline-only) — so
    # the note can never contradict the "save ~34%" headline above it (the
    # terminal's verbose note quotes this same total-basis figure).
    split_total_pct = _split_report_figures(result, result.split).total_pct
    return [
        f"_Upper bound: moving every call to `{result.candidate_model}` saves "
        f"~{float(upper):.1f}% — the aggressive end; the "
        f"~{float(split_total_pct):.1f}% split above is the conservative, "
        "quality-respecting recommendation (a full swap is a larger quality change)._"
    ]


# --- HTML wrappers (amber ``.caution`` inline note) -------------------------


def _html_caution(message_html: str) -> str:
    """Wrap an already-escaped HTML fragment in the amber ``.caution`` note.

    The single styling hook for every report caution (staleness + window): an
    inline warning glyph and an amber span consistent with the data-quality note.
    """
    return f'<span class="caution">&#9888; {message_html}</span>'


def _html_note(message_html: str) -> str:
    """Wrap an already-escaped HTML fragment in the neutral/dim note style.

    The informational counterpart to :func:`_html_caution`: NO ⚠ glyph and the
    report's muted ink-dim colour (with a hardcoded fallback so it reads dim on
    both the v1 and v2 themes) instead of amber.  Used for unrated-family lines
    whose quality the measurement section already confirmed below — neutral, not
    a caution.
    """
    return f'<span class="note" style="color:var(--ink-dim,#6B6B72)">{message_html}</span>'


def _html_unrated_family_html(
    result: AnalysisResult,
    judged_models: frozenset[str] = frozenset(),
) -> str:
    """HTML for the unrated-message family (findings #1 + #4), or "" when silent.

    Each shared message (from :func:`_unrated_family_messages`) becomes its own
    paragraph carrying the same wording the terminal footer and the Markdown
    reports carry, escaped for HTML.  Severity decides the styling hook: a genuine
    "quality unverified" CAUTION uses the amber ``.caution`` glyph note, while a
    measured-below INFORMATIONAL line uses the neutral dim ``.note`` style (no ⚠,
    no amber) — alarm-styling a row about a model the quality section confirmed
    below would be wrong.  *judged_models* (the models measured in this run) makes
    the family measurement-aware — see :func:`_unrated_family_messages`.  Empty
    string when no unrated model is recommended or held out, so rated runs render
    byte-identically.
    """
    messages = _unrated_family_messages(result, judged_models)
    if not messages:
        return ""
    parts: list[str] = []
    for message, severity in messages:
        escaped = _html_escape.escape(message)
        inner = _html_caution(escaped) if severity == _SEV_WARNING else _html_note(escaped)
        parts.append(f'<p class="caveat">{inner}</p>')
    return "".join(parts)


def _html_staleness_note(date_str: str, stale: bool, refresh_command: str) -> str:
    """Return the amber staleness note for a freshness row, or "" when fresh.

    *date_str* is assumed pre-escaped by the caller (it is an ISO date).  The
    refresh command is wrapped in ``<code>`` so the CLI invocation reads as a
    command, matching the methodology/swap-note treatment.
    """
    if not stale:
        return ""
    return " " + _html_caution(
        f"{_html_escape.escape(_staleness_age(date_str))} — refresh with "
        f"<code>{_html_escape.escape(refresh_command)}</code>"
    )


def _html_pricing_synced_html(result: AnalysisResult) -> str:
    """Return the escaped ``pricing synced <date>`` fragment, or "" when absent.

    Carries the >30-day staleness caution so the v2 meta line flags an old
    pricing table the same way the freshness rows/stats do elsewhere.
    """
    date_str = result.pricing_json_last_synced
    if not date_str:
        return ""
    note = _html_staleness_note(
        date_str,
        _is_pricing_stale(date_str, max_days=30),
        _PRICING_REFRESH_COMMAND,
    )
    return f"pricing synced {_html_escape.escape(date_str)}{note}"


def _html_quality_synced_html(result: AnalysisResult) -> str:
    """Return the escaped ``quality synced <date>`` fragment, or "" when absent.

    The quality counterpart to the ``pricing synced`` fragment the reports
    already show; carries its own staleness note (>60-day window).
    """
    date_str = result.quality_json_last_synced
    if not date_str:
        return ""
    note = _html_staleness_note(
        date_str,
        _is_quality_stale(date_str, max_days=60),
        _QUALITY_REFRESH_COMMAND,
    )
    return f"quality synced {_html_escape.escape(date_str)}{note}"


def _html_v2_meta_lines(result: AnalysisResult) -> list[str]:
    """Build the v2 ``<p class="meta-line">`` block under the routing plan.

    Item E: the calls-priced count and the synced dates ride SEPARATE lines —
    "N calls priced" on its own, then "pricing synced … · quality synced …" on a
    second line. The split is deliberate even when the two would fit on one line:
    a large priced-call count (e.g. 250k records) would otherwise push the synced
    dates past the container edge. The synced-dates line is emitted only when at
    least one synced date exists; otherwise just the calls-priced line is shown,
    matching the previous single-line behaviour minus the (now absent) dates.
    """
    if result.unpriced_calls:
        calls = (
            f"{result.priced_calls:,} priced "
            f"<span class='warn'>+ {result.unpriced_calls:,} unpriced</span>"
        )
    else:
        calls = f"{result.priced_calls:,} calls priced"
    lines = [f'<p class="meta-line">{calls}</p>']
    synced_bits: list[str] = []
    _pricing_synced = _html_pricing_synced_html(result)
    if _pricing_synced:
        synced_bits.append(_pricing_synced)
    _quality_synced = _html_quality_synced_html(result)
    if _quality_synced:
        synced_bits.append(_quality_synced)
    if synced_bits:
        lines.append(
            '<p class="meta-line">'
            + "<span class='sep'>&middot;</span>".join(synced_bits)
            + "</p>"
        )
    return lines


def _html_window_caution(result: AnalysisResult) -> str:
    """Return the amber window-caution note for HTML, or "" when it does not apply."""
    window = result.window_days
    span = result.observed_span_days
    if not window_contradicts_span(window, span):
        return ""
    assert window is not None
    assert span is not None
    span_days = round(span)
    return _html_caution(
        f"<code>--window {window}</code> overrides your log's actual "
        f"~{span_days}-day span — the monthly figure is projected as if the "
        f"data covered {window} days. Drop <code>--window</code> to project from "
        "the real span."
    )


def _html_log_span_html(result: AnalysisResult) -> str:
    """Return the escaped ``Log span: start -> end (N.d days)`` text, or "" when absent."""
    start = result.observed_span_start
    end = result.observed_span_end
    if start is None or end is None:
        return ""
    span = (
        f" ({result.observed_span_days:.1f} days)"
        if result.observed_span_days is not None
        else ""
    )
    return (
        f"Log span: {_html_escape.escape(start)} &rarr; "
        f"{_html_escape.escape(end)}{_html_escape.escape(span)}"
    )


def _html_quality_tier_line(split: SplitRouting, *, model_class: str) -> str:
    """Return the ``Quality tier`` disclosure line HTML for the split section.

    Renders ``Quality tier <baseline>: <label> &rarr; <candidate>: <label>
    (LMArena)`` with each model name wrapped in *model_class* (``model-name`` for
    HTML v1, ``route-to`` for v2 — the cyan model-name class on each surface) and
    the tier labels / source tag left plain/dim.  Uses the shared
    :func:`_tier_label`, so the tier shown in the HTML reconciles exactly with the
    terminal and Markdown surfaces; an unrated model reads ``unrated`` rather than
    being dropped.  The caller wraps this in the surface's note element
    (``projection-note`` for v1, ``meta-line`` for v2).
    """
    esc = _html_escape.escape
    return (
        f'<span class="qt-label">Quality tier</span> '
        f'<span class="{model_class}">{esc(split.baseline_model)}</span>: '
        f"{esc(_tier_label(split.baseline_model))} &rarr; "
        f'<span class="{model_class}">{esc(split.candidate_model)}</span>: '
        f"{esc(_tier_label(split.candidate_model))} "
        f'<span class="qt-src">(LMArena)</span>'
    )


def _dominant_model(result: AnalysisResult) -> str | None:
    """Return the model with the highest cost in *result*, or None if empty."""
    if not result.cost_by_model:
        return None
    return max(result.cost_by_model, key=lambda m: result.cost_by_model[m])


def _projection_label(result: AnalysisResult) -> str:
    """Human-readable projection disclosure string."""
    if result.window_days is not None:
        return f"projected from a {result.window_days}-day window"
    if result.observed_span_days is not None:
        return f"projected from a {result.observed_span_days:.1f}-day sample"
    return "across analyzed calls"


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

_METHODOLOGY_NOTE = (
    "Estimated from list prices via the LiteLLM registry. "
    "No LLM calls. No network. No data leaves your machine."
)

# The unconditional privacy clause (cost analysis is fully local). Kept VERBATIM
# wherever no measure run was performed so the no-measure path stays byte-for-byte
# identical to before the measure-aware reframing existed.
_PRIVACY_ABSOLUTE = "No LLM calls. No network. No data leaves your machine."

# The "0 LLM calls" methodology tail shared by the Markdown and HTML footers.
_METHODOLOGY_TAIL_ZERO = "0 LLM calls made for this analysis"


# ---------------------------------------------------------------------------
# Measure-aware honesty reframing (Items C + D)
# ---------------------------------------------------------------------------
#
# Once --measure / --judge has run, the report CONTAINS quality results, so the
# pre-measure copy becomes contradictory or untrue:
#
#   * The "Quality is not verified — run --measure …" caveat (Item C) now sits
#     beside the very verdict it tells the reader to go obtain — drop it on the
#     Tier-1 path, soften it to a "run --judge" nudge on the Tier-0 path.
#   * The "No LLM calls. No network. No data leaves your machine." privacy
#     absolute and the "0 LLM calls" methodology tail (Item D) are TRUE of the
#     cost analysis but FALSE of the measurement step, which DID call the user's
#     own provider with the sampled prompts. Reframe honestly while preserving
#     the "nothing reaches Frugon" guarantee, which holds in BOTH cases.
#
# All four report variants (md v1/v2, html v1/v2) and the terminal route every
# such string through the helpers below so the surfaces can never drift.


def _measure_sample_count(measure_result: MeasureResult | None) -> int:
    """Sampled-prompt count for the measure-aware copy (0 when no run)."""
    return measure_result.samples_taken if measure_result is not None else 0


def _is_tier1(measure_result: MeasureResult | None) -> bool:
    """True when a --judge (scored) run produced tallies."""
    return measure_result is not None and measure_result.tier1_tallies is not None


def _quality_caveat_text(
    base_caveat: str, measure_result: MeasureResult | None
) -> str | None:
    """Return the quality caveat to show beneath a recommendation, measure-aware.

    * No measure run  → *base_caveat* unchanged (today's wording — preserves the
      byte-identical no-measure path).
    * Tier-1 (--judge) → ``None``: the report already carries a scored verdict,
      so "Quality is not verified — run --measure …" would contradict it.
    * Tier-0 (--measure, no judge) → a softened nudge that makes no
      "not verified" claim and points at ``--judge`` for a scored verdict.
    """
    if measure_result is None:
        return base_caveat
    if _is_tier1(measure_result):
        return None
    return (
        "Raw samples are shown in the quality measurement section — "
        "run --judge for a scored verdict."
    )


def _md_projection_caveat_line(
    projection_label: str, base_caveat: str, measure_result: MeasureResult | None
) -> str:
    """Build the italic ``_<projection>. <caveat>_`` summary tagline, measure-aware.

    On the Tier-1 path the caveat clause is dropped (the report carries a
    verdict), leaving the bare projection label; otherwise the projection and
    the (possibly softened) caveat ride together exactly as before.
    """
    caveat = _quality_caveat_text(base_caveat, measure_result)
    if caveat is None:
        return f"_{projection_label}._"
    return f"_{projection_label}. {caveat}_"


def _privacy_sentence(measure_result: MeasureResult | None) -> str:
    """Plain-text privacy line, measure-aware (Item D).

    Returns the unconditional absolute when no measurement ran; otherwise an
    honest split that owns the provider calls the measurement made while keeping
    the "never to Frugon" guarantee intact.
    """
    if measure_result is None:
        return _PRIVACY_ABSOLUTE
    n = measure_result.samples_taken
    prompt_word = "prompt" if n == 1 else "prompts"
    return (
        "Cost analysis: 0 LLM calls, fully local (no network). "
        f"Quality measurement: {n:,} {prompt_word} sent to your own provider using "
        "your own keys — never to Frugon."
    )


def _privacy_html(measure_result: MeasureResult | None) -> str:
    """Inner HTML of the ``<p class="privacy">`` footer line, measure-aware.

    No-measure → the unconditional absolute wrapped in the green ``em`` span
    (byte-identical to before). After a measure run, the green ``em`` emphasis
    stays on the cost-analysis clause (still fully local), and the honest
    quality-measurement clause follows in plain text — the green-money / local
    emphasis is never falsely extended over the provider calls the sampling made.
    """
    esc = _html_escape.escape
    if measure_result is None:
        return f"<span class='em'>{_PRIVACY_ABSOLUTE}</span>"
    n = measure_result.samples_taken
    prompt_word = "prompt" if n == 1 else "prompts"
    return (
        "<span class='em'>Cost analysis: 0 LLM calls, fully local (no network).</span> "
        + esc(
            f"Quality measurement: {n:,} {prompt_word} sent to your own provider "
            "using your own keys — never to Frugon."
        )
    )


def _methodology_note_text(measure_result: MeasureResult | None) -> str:
    """The methodology/privacy note (Item D), measure-aware.

    No-measure → ``_METHODOLOGY_NOTE`` verbatim (byte-identical path). After a
    measure run, the registry-pricing sentence is kept but the privacy absolute
    is swapped for the honest measure-aware split.
    """
    if measure_result is None:
        return _METHODOLOGY_NOTE
    return (
        "Estimated from list prices via the LiteLLM registry. "
        + _privacy_sentence(measure_result)
    )


def _methodology_tail_text(measure_result: MeasureResult | None) -> str:
    """The "… · 0 LLM calls made for this analysis" footer tail, measure-aware.

    No-measure → the historical "0 LLM calls made for this analysis" (byte-
    identical). After a measure run, the cost analysis still made zero calls but
    the measurement did, so the tail names that the count is the cost pass only.
    """
    if measure_result is None:
        return _METHODOLOGY_TAIL_ZERO
    n = measure_result.samples_taken
    prompt_word = "prompt" if n == 1 else "prompts"
    return (
        "0 LLM calls for the cost analysis · "
        f"{n:,} {prompt_word} sampled for quality (your own provider)"
    )



def _candidates_considered_md_lines(
    result: AnalysisResult, *, has_judge_section: bool = False
) -> list[str]:
    """Markdown lines for the "Candidates considered" block — empty when not needed.

    Empty list when the user passed <=1 candidate (no block rendered, MD stays
    byte-identical).  Otherwise returns a level-2 heading + a small table:
    Model | Monthly cost | Vs. baseline | Quality tier | Status — one row per
    candidate.  Sits below the cost-analysis section and above the
    "## Details" / freshness block on every Markdown surface (split + wholesale).

    Heading + caption differ by path (PD-directed 2026-07-03 "2+3 hybrid"):
    the default pool absorbs its pool/shown counts into the heading text and
    renders the caption as a 3-item bullet list instead of two prose
    paragraphs; the explicit ``--candidates`` path is unchanged (plain
    heading, the existing italic prose, no cap line).
    """
    projs = result.candidate_projections
    if len(projs) <= 1:
        return []
    lines: list[str] = [
        f"## {_candidates_header_title(result)}",
        "",
        "| Model | Monthly cost | Vs. baseline | Quality tier | Status |",
        "|-------|-------------:|-------------:|--------------|--------|",
    ]
    for proj in projs:
        if proj.status == "unpriced":
            money = "—"
        elif proj.monthly_cost is not None:
            money = f"{_fmt_usd(proj.monthly_cost)} / mo"
        elif proj.observed_cost is not None:
            money = _fmt_usd(proj.observed_cost)
        else:  # pragma: no cover — defensive
            money = "—"
        pct_val = proj.saving_pct if proj.saving_pct is not None else proj.observed_saving_pct
        saving = "—" if pct_val is None else _fmt_candidate_saving(pct_val)
        label = _CANDIDATE_STATUS_LABEL[proj.status]
        lines.append(
            f"| `{proj.model}` | {money} | {saving} | {proj.tier_label} | {label} |"
        )
    lines.append("")
    if result.used_default_pool and result.split is not None:
        for line in _candidate_legend_lines(
            result, result.split, has_judge_section=has_judge_section
        ):
            lines.append(f"- {line}")
        lines.append("")
    else:
        lines += [f"_{_candidate_caption(has_judge_section)}_", ""]
        cap_caption = _candidate_cap_caption(result)
        if cap_caption is not None:
            lines += [f"_{cap_caption}_", ""]
    return lines


def _candidates_considered_html(
    result: AnalysisResult,
    esc: Callable[[str], str],
    *,
    has_judge_section: bool = False,
) -> str:
    """HTML fragment for the "Candidates considered" block — empty when not needed.

    Shared by HTML v1 and v2: a small table inside a wrapper div (the wrappers
    differ per style, but the inner table is identical so the figures are the
    SAME across surfaces).  Status tag styling mirrors the routing-plan badge
    vocabulary so the block reads as part of the same surface.

    The caption below the table differs by path (PD-directed 2026-07-03 "2+3
    hybrid"): the default pool renders a ``<ul>`` bullet legend instead of the
    two ``<p class="caption">`` paragraphs; the explicit ``--candidates`` path
    is unchanged.  The card's own eyebrow heading (rendered by each of the
    three HTML callers, not by this function) uses
    :func:`_candidates_header_title` for the same pool/shown counting move.
    """
    projs = result.candidate_projections
    if len(projs) <= 1:
        return ""
    # Status -> badge modifier class.  The CSS in BOTH style blocks (v1 + v2)
    # gives each modifier its colour, mirroring the routing-plan badge vocabulary:
    # recommended = cyan emphasis, more_expensive = amber caution, everything else
    # = muted.  Using a class (not inline opacity) lets the candidates table
    # inherit the SAME badge treatment as the routing-plan table on each surface.
    badge_modifier = {
        "recommended": "badge-recommended",
        "considered": "badge-considered",
        "more_expensive": "badge-more-expensive",
        "unpriced": "badge-unpriced",
    }
    rows: list[str] = []
    for proj in projs:
        if proj.status == "unpriced":
            money = "&mdash;"
        elif proj.monthly_cost is not None:
            money = f"{_fmt_usd(proj.monthly_cost)} / mo"
        elif proj.observed_cost is not None:
            money = _fmt_usd(proj.observed_cost)
        else:  # pragma: no cover — defensive
            money = "&mdash;"
        pct_val = proj.saving_pct if proj.saving_pct is not None else proj.observed_saving_pct
        saving = "&mdash;" if pct_val is None else _fmt_candidate_saving(pct_val)
        label = _CANDIDATE_STATUS_LABEL[proj.status]
        modifier = badge_modifier.get(proj.status, "badge-considered")
        status_html = f'<span class="badge {modifier}">{esc(label)}</span>'
        rows.append(
            "<tr>"
            f'<td class="c-model"><span class="model-name">{esc(proj.model)}</span></td>'
            f'<td class="num">{money}</td>'
            f'<td class="num">{saving}</td>'
            f'<td class="c-tier">{esc(proj.tier_label)}</td>'
            f'<td class="c-status">{status_html}</td>'
            "</tr>"
        )
    return (
        '<div class="candidates-considered">'
        '<table class="tbl tbl-candidates"><thead><tr>'
        "<th>Model</th>"
        '<th class="num">Monthly cost</th>'
        '<th class="num">Vs. baseline</th>'
        '<th class="c-tier">Quality tier</th>'
        '<th class="c-status">Status</th>'
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + (
            '<ul class="caption candidates-caption candidates-legend">'
            + "".join(
                f"<li>{esc(line)}</li>"
                for line in _candidate_legend_lines(
                    result, result.split, has_judge_section=has_judge_section
                )
            )
            + "</ul>"
            if result.used_default_pool and result.split is not None
            else (
                '<p class="caption candidates-caption">'
                + esc(_candidate_caption(has_judge_section))
                + "</p>"
                + (
                    '<p class="caption candidates-caption">'
                    + esc(cap_caption)
                    + "</p>"
                    if (cap_caption := _candidate_cap_caption(result)) is not None
                    else ""
                )
            )
        )
        + "</div>"
    )


def _render_markdown_split(
    result: AnalysisResult,
    split: SplitRouting,
    output_path: Path,
    *,
    measure_result: MeasureResult | None = None,
    limits: PreviewLimits | None = None,
) -> None:
    """Write a Markdown split-routing report (shared by v1 and v2 markdown).

    When *measure_result* is provided, a "Quality measurement" section is
    appended after the cost detail and before the methodology footer; when None
    the output is byte-identical to before.  *limits* carries the (possibly
    flag-overridden) report preview lengths for that section.
    """
    projection_label = _projection_label(result)

    # Every split figure reconciles to the FULL analyzed dataset (identical to
    # the terminal panel) via the shared helper — NOT to the baseline model in
    # isolation.  "Current" is the TOTAL spend, the saving percent is over that
    # total, and the routing plan includes the already-on-a-cheaper-model bucket
    # so the buckets sum to every analyzed call.
    fig = _split_report_figures(result, split)
    unit = "/mo" if fig.projected else ""
    current_val = _fmt_usd(fig.current)
    blended_val = _fmt_usd(fig.blended)

    # Quality phrase for the headline and routing-plan table — three cases:
    #
    # 1. Candidate is UNRATED — omit any quality phrase; unrated caveat handles it.
    # 2. tier_drop <= 0 (same or better) — use the stronger positive phrase.
    # 3. tier_drop >= 1 (genuine step-down) — use "within tolerance" as before.
    _routed_unrated = _is_unrated(split.candidate_model)
    if _routed_unrated:
        _tol = ""
        _tol_paren = ""
    elif _is_equal_or_better_quality(result):
        _tol = "same or better quality "
        _tol_paren = " (same or better quality)"
    else:
        _tol = "within tolerance "
        _tol_paren = " (within tolerance)"

    # Per-bucket share of total analyzed calls for the routing-plan "% calls"
    # column — same bucket order as the rows below (routed, kept, already-optimal
    # when present), summing to exactly 100.0% via largest-remainder rounding.
    _md_share_counts = [split.routed_count, split.kept_count]
    if fig.already_cheap > 0:
        _md_share_counts.append(fig.already_cheap)
    _md_share = _call_share_pcts(_md_share_counts)

    lines: list[str] = [
        "# frugon — cost analysis",
        "",
        "## Bottom line",
        "",
        f"**Route {split.routed_count:,} of {result.priced_calls:,} analyzed calls "
        f"(`{split.baseline_model}` easy calls) to `{split.candidate_model}` "
        f"{_tol}— save ~{float(fig.total_pct):.1f}% "
        f"({current_val}{unit} → {blended_val}{unit}).**",
        "",
        _md_projection_caveat_line(
            projection_label,
            SPLIT_CAVEAT_EQUAL_OR_BETTER if _is_equal_or_better_quality(result) else SPLIT_CAVEAT,
            measure_result,
        ),
        "",
        "## Routing plan",
        "",
        # The "% calls" column shows each bucket's share of all analyzed calls,
        # largest-remainder rounded so the non-blended buckets sum to EXACTLY
        # 100.0% (reconciling with the 100% Blended total row) — see
        # _call_share_pcts.  The terminal stays as-is; only the written tables
        # (MD here, HTML v2) carry the share figure.
        "| Bucket | Calls | % calls | Route to | Cost |",
        "|--------|------:|--------:|----------|-----:|",
        f"| Routed · easy | {split.routed_count:,} | {_md_share[0]:.1f}% "
        f"| `{split.candidate_model}`{_tol_paren} "
        f"| {_fmt_usd(split.routed_cost)} |",
        f"| Kept · hard | {split.kept_count:,} | {_md_share[1]:.1f}% "
        f"| `{split.baseline_model}` | {_fmt_usd(split.kept_cost)} |",
    ]
    # Already-on-a-cheaper-model calls — already optimal, no action.  Including
    # them makes the routing plan reconcile to ALL analyzed calls, not just the
    # baseline routing target (the dropped-calls bug).
    if fig.already_cheap > 0:
        _already_to = (
            "`" + "`, `".join(fig.other_models) + "`"
            if fig.other_models
            else "a cheaper model"
        )
        lines.append(
            f"| Keep · already optimal | {fig.already_cheap:,} | {_md_share[2]:.1f}% "
            f"| {_already_to} (no action) "
            f"| {_fmt_usd(fig.already_cheap_cost)} |"
        )
    lines += [
        f"| **Blended** | **{result.priced_calls:,}** | **100.0%** | — "
        f"| **{_fmt_usd(fig.blended)}** |",
        "",
        f"**You save {_fmt_usd(fig.saved)}{unit} (−{float(fig.total_pct):.1f}%).**",
        "",
    ]

    # Upper bound (decision info) sits in the decision area, BEFORE the
    # freshness metadata block below (Item 7 ordering).  Shared helper keeps
    # the wording + figure identical to the terminal and the other variants.
    upper_lines = _md_upper_bound_lines(result)
    if upper_lines:
        lines += upper_lines + [""]

    # Unrated-message family (findings #1 + #4) — non-blocking quality caveat
    # when the recommended candidate is unrated and/or an unrated candidate was
    # held out of the split.  Same wording as the terminal footer + HTML reports.
    _unrated_lines = _md_unrated_family_lines(result, judged_models_from_measure(measure_result))
    if _unrated_lines:
        lines += _unrated_lines + [""]

    # Candidates considered (multi-candidate transparency) — no-op when
    # the user passed <=1 candidate (block lines is empty, byte-identical).
    _cand_lines = _candidates_considered_md_lines(
        result, has_judge_section=_is_tier1(measure_result)
    )
    if _cand_lines:
        lines += _cand_lines

    if result.cost_by_model:
        lines += [
            "## Cost by model",
            "",
            "| Model | Calls | Cost | % of total |",
            "|-------|------:|-----:|-----------:|",
        ]
        total_priced_calls = 0
        for model, cost in sorted(result.cost_by_model.items(), key=lambda kv: kv[1], reverse=True):
            pct = (cost / result.total_cost * 100) if result.total_cost else Decimal("0")
            calls = result.calls_by_model.get(model, 0)
            total_priced_calls += calls
            lines.append(f"| {model} | {calls:,} | {_fmt_usd(cost)} | {float(pct):.1f}% |")
        lines.append(
            f"| **Total** | **{total_priced_calls:,}** | "
            f"**{_fmt_usd(result.total_cost)}** | **100.0%** |"
        )
        lines.append("")

    # --- Details: quality tier + table freshness + log span + window caution ---
    # Reports are the FULL view (no verbose mode), so the disclosures the
    # terminal shows under --verbose / in its Accounting block all live here.
    # Placed AFTER the decision (incl. the Upper bound above) so decision info
    # precedes freshness metadata.  Order mirrors the terminal: the Quality-tier
    # benchmark comparison rides directly under the Upper-bound swap context and
    # ABOVE the freshness (last-synced) lines; the Log span — pure freshness
    # metadata — drops below them.
    detail_lines: list[str] = []
    detail_lines.append(_md_quality_tier_line(split))
    detail_lines += _md_freshness_lines(result)
    span_line = _md_log_span_line(result)
    if span_line:
        detail_lines.append(span_line)
    window_lines = _md_window_caution_lines(result)
    if detail_lines or window_lines:
        lines += ["## Details", ""]
        lines += detail_lines
        if window_lines:
            # Blank line separates the bullet list from the blockquote callout.
            if detail_lines:
                lines.append("")
            lines += window_lines
        lines.append("")

    lines += _data_quality_md(result)
    # Quality-measurement section — the full proof of a --measure / --judge run,
    # placed after the cost detail and before the methodology footer.  Empty
    # (byte-identical to before) when no measure run was performed.
    quality_lines = _quality_section_md(measure_result, limits=limits, result=result)
    if quality_lines:
        lines += ["", *quality_lines]
    _md_bottom_caveat = (
        SPLIT_CAVEAT_EQUAL_OR_BETTER
        if _is_equal_or_better_quality(result)
        else SPLIT_CAVEAT
    )
    _caveat = _quality_caveat_text(_md_bottom_caveat, measure_result)
    attribution = _get_attribution()
    if _caveat is not None:
        lines.append(f"> **Before you switch:** {_caveat}")
        if attribution:
            lines += [">", f"> _Source: {attribution}_"]
    elif attribution:
        lines.append(f"> _Source: {attribution}_")
    lines += [
        "",
        FUNNEL_LINE,
        "",
        "---",
        "",
        f"_{_methodology_note_text(measure_result)}_",
        "",
        "_methodology · tokencost · LiteLLM registry · "
        f"LMArena quality tiers, RouteLLM-style routing · {_methodology_tail_text(measure_result)}_",
        "",
    ]
    _atomic_write_text(output_path, "\n".join(lines))


def _render_md_wholesale_swap_plan(result: AnalysisResult) -> list[str]:
    """Markdown swap-plan table for a wholesale run.

    Mirrors '## Routing plan' so split and wholesale reports read as visual
    siblings.  Returns empty list when no candidate is set.
    """
    if not result.candidate_model or not result.cost_by_model:
        return []
    candidate = result.candidate_model
    non_cand = sorted(
        [(m, c) for m, c in result.cost_by_model.items() if m != candidate],
        key=lambda mc: mc[1],
        reverse=True,
    )
    cand_cost = result.cost_by_model.get(candidate, Decimal("0"))
    ordered: list[tuple[str, Decimal]] = non_cand + [(candidate, cand_cost)]

    counts = [result.calls_by_model.get(m, 0) for m, _ in ordered]
    shares = _call_share_pcts(counts)

    lines: list[str] = [
        "## Swap plan",
        "",
        "| Model | Calls | % calls | Current cost | Action |",
        "|-------|------:|--------:|-------------:|--------|",
    ]
    for (model, cost), share in zip(ordered, shares, strict=True):
        calls = result.calls_by_model.get(model, 0)
        action = (
            f"already on `{candidate}`"
            if model == candidate
            else f"→ swap to `{candidate}`"
        )
        lines.append(
            f"| `{model}` | {calls:,} | {share:.1f}% "
            f"| {_fmt_usd(cost)} | {action} |"
        )
    total_calls = sum(result.calls_by_model.get(m, 0) for m, _ in ordered)
    total_cost = sum((c for _, c in ordered), Decimal("0"))
    lines.append(
        f"| **Total** | **{total_calls:,}** | **100.0%** "
        f"| **{_fmt_usd(total_cost)}** | |"
    )
    lines.append("")
    return lines


def render_markdown(
    result: AnalysisResult,
    output_path: Path,
    *,
    measure_result: MeasureResult | None = None,
    limits: PreviewLimits | None = None,
) -> None:
    """Write a Markdown cost report to *output_path*.

    Fully local — no external requests. The quality caveat is included only
    when a routing recommendation is present (candidate_model is not None).

    When *measure_result* is provided (a ``--measure`` / ``--judge`` run ran),
    a "Quality measurement" section is appended mirroring the terminal's quality
    output (tally + verdict + per-prompt proof, or Tier-0 side-by-side).  When
    it is ``None`` the output is byte-identical to before this parameter
    existed — existing callers and tests are unaffected.
    """
    if _has_split(result):
        assert result.split is not None  # narrowed by _has_split
        _render_markdown_split(
            result,
            result.split,
            output_path,
            measure_result=measure_result,
            limits=limits,
        )
        return

    lines: list[str] = ["# frugon — cost analysis", ""]

    if result.priced_calls == 0:
        lines += [
            "**No priced calls found.**",
            "",
            f"Analyzed {result.total_calls:,} records — "
            "none could be priced (unknown models or missing usage data).",
            "",
            "Run `frugon pricing update` to refresh the pricing table, "
            "or check that your log records include a `model` field.",
        ]
    else:
        # RECONCILIATION: derive saving% from the quantized (displayed) dollar
        # amounts so the printed percent verifiably equals
        # round(printed_save / printed_current * 100, 1).
        # compute_saving_pct is kept for the gate check only (> Decimal("0")).
        _raw_saving_pct = compute_saving_pct(result.total_cost, result.projected_cost)
        _, _, saving_pct = _reconciled_delta_pct(result.total_cost, result.projected_cost)
        projection_label = _projection_label(result)

        lines += [
            "## Summary",
            "",
            f"- **Current cost:** {_fmt_usd(result.total_cost)} (across analyzed calls)",
            f"- **Calls analyzed:** {result.priced_calls:,} priced"
            + (f", {result.unpriced_calls:,} unpriced" if result.unpriced_calls else ""),
        ]

        if result.monthly_cost is not None:
            lines.append(
                f"- **Monthly cost:** {_fmt_usd(result.monthly_cost)} (monthly projection)"
            )

        # Freshness metadata (Items 1+2): pricing AND quality last-synced, each
        # with its staleness caution when the table is old.  Shared helper keeps
        # the dates + thresholds identical to the terminal and the split path.
        lines += _md_freshness_lines(result)
        # Observed log span (Item 4) — the window the projection is computed from.
        span_line = _md_log_span_line(result)
        if span_line:
            lines.append(span_line)

        if result.candidate_model and _raw_saving_pct is not None and _raw_saving_pct > Decimal("0"):
            baseline = _dominant_model(result)
            swap_label = (
                f"{baseline} → {result.candidate_model}"
                if baseline
                else result.candidate_model
            )
            lines += [
                f"- **Recommended swap:** {swap_label}",
                f"- **Projected cost:** {_fmt_usd(result.projected_cost)} (across analyzed calls)",
            ]
            if result.monthly_projected is not None:
                lines.append(
                    f"- **Monthly projected:** {_fmt_usd(result.monthly_projected)} (monthly projection)"
                )
            lines += [
                f"- **Estimated saving:** {float(saving_pct):.1f}%"
                f" ({projection_label})",
            ]
            _caveat = _quality_caveat_text(QUALITY_CAVEAT, measure_result)
            if _caveat is not None:
                lines += ["", f"> **Quality caveat:** {_caveat}"]
            # Unrated-message family (findings #1 + #4) — the actionable caveat
            # when the recommended candidate is unrated and/or an unrated
            # candidate forced this wholesale full-swap fallback.
            _unrated_lines = _md_unrated_family_lines(result, judged_models_from_measure(measure_result))
            if _unrated_lines:
                lines += [""] + _unrated_lines

        # Candidates considered (multi-candidate transparency) — no-op when
        # the user passed <=1 candidate (block lines is empty, byte-identical).
        _cand_lines = _candidates_considered_md_lines(
            result, has_judge_section=_is_tier1(measure_result)
        )
        if _cand_lines:
            lines += [""] + _cand_lines

        # Swap plan — wholesale only; split runs use the routing-plan section.
        _swap_plan_lines = _render_md_wholesale_swap_plan(result)
        if _swap_plan_lines:
            lines += [""] + _swap_plan_lines

        if result.cost_by_model:
            lines += [
                "",
                "## Cost by model",
                "",
                "| Model | Calls | Cost | % of total |",
                "|-------|------:|-----:|-----------:|",
            ]
            for model, cost in sorted(
                result.cost_by_model.items(), key=lambda kv: kv[1], reverse=True
            ):
                pct = (cost / result.total_cost * 100) if result.total_cost else Decimal("0")
                calls = result.calls_by_model.get(model, 0)
                lines.append(
                    f"| {model} | {calls:,} | {_fmt_usd(cost)} | {float(pct):.1f}% |"
                )

        # CC-BY attribution footer — shown when quality tiers drove the recommendation
        if result.candidate_model:
            attribution = _get_attribution()
            if attribution:
                lines += ["", f"_Source: {attribution}_"]

            lines += [
                "",
                FUNNEL_LINE,
                "",
            ]

        # Window-vs-span caution (Item 3) — a `>` callout near the figures it
        # qualifies, before the methodology footer.
        window_lines = _md_window_caution_lines(result)
        if window_lines:
            lines += [""] + window_lines

        lines += _data_quality_md(result)
        # Quality-measurement section before the methodology footer (empty /
        # byte-identical when no measure run ran).
        quality_lines = _quality_section_md(measure_result, limits=limits, result=result)
        if quality_lines:
            lines += ["", *quality_lines]
        lines += ["", "---", f"_{_methodology_note_text(measure_result)}_", ""]

    _atomic_write_text(output_path, "\n".join(lines))


def render_markdown_v2(
    result: AnalysisResult,
    output_path: Path,
    *,
    measure_result: MeasureResult | None = None,
    limits: PreviewLimits | None = None,
) -> None:
    """Write a v2 (refined) Markdown cost report to *output_path*.

    Leads with a bold bottom-line headline (the saving), then a compact
    "What we found" before/after, a cost-by-model table with a total row,
    the recommended swap, caveats as a callout, and a methodology/privacy
    footer.

    Fully local — no external requests. The quality caveat is included only
    when a routing recommendation is present (candidate_model is not None).
    All honesty caveats and projection labels carry over from v1 unchanged.
    """
    if _has_split(result):
        assert result.split is not None  # narrowed by _has_split
        _render_markdown_split(
            result,
            result.split,
            output_path,
            measure_result=measure_result,
            limits=limits,
        )
        return

    lines: list[str] = ["# frugon — cost analysis", ""]

    # --- No priced calls guard ---
    if result.priced_calls == 0:
        lines += [
            "> **No priced calls found.**",
            ">",
            f"> Analyzed {result.total_calls:,} records — none could be priced "
            "(unknown models or missing usage data).",
            "",
            "Run `frugon pricing update` to refresh the pricing table, "
            "or check that your log records include a `model` field.",
            "",
            "---",
            "",
            f"_{_METHODOLOGY_NOTE}_",
            "",
        ]
        _atomic_write_text(output_path, "\n".join(lines))
        return

    # RECONCILIATION: gate on the raw percent (so a positive but sub-display-
    # precision saving does not vanish), then derive the DISPLAYED percent and
    # delta from quantized amounts — the same pair _fmt_usd will print.
    # Prefer the monthly axis (headline basis); fall back to the sample axis.
    _raw_saving_pct = compute_saving_pct(result.total_cost, result.projected_cost)
    if result.monthly_cost is not None and result.monthly_projected is not None:
        _, _, saving_pct = _reconciled_delta_pct(
            result.monthly_cost, result.monthly_projected
        )
        _delta_unit = "/mo"
    else:
        _, _, saving_pct = _reconciled_delta_pct(
            result.total_cost, result.projected_cost
        )
        _delta_unit = ""
    projection_label = _projection_label(result)
    has_saving = (
        result.candidate_model is not None
        and _raw_saving_pct is not None
        and _raw_saving_pct > Decimal("0")
    )

    # --- Bottom line (hero) ---
    lines += ["## Bottom line", ""]
    if has_saving:
        # Prefer monthly figures for the before/after when a projection exists.
        if result.monthly_cost is not None and result.monthly_projected is not None:
            before = f"{_fmt_usd(result.monthly_cost)}/mo"
            after = f"{_fmt_usd(result.monthly_projected)}/mo"
        else:
            before = _fmt_usd(result.total_cost)
            after = _fmt_usd(result.projected_cost)
        # C-4: route through the measure-aware caveat helper instead of pasting
        # the static QUALITY_CAVEAT. Tier-1 → bare projection label (no stale
        # "run --measure" pointing at the verdict below); Tier-0 → softened nudge
        # (the C-2 wording); no --measure → byte-identical to before.
        _hero_caveat = _quality_caveat_text(QUALITY_CAVEAT, measure_result)
        # Bare projection_label has no trailing period; the caveat ends in '.'.
        # Match the original `_X. CAVEAT_` pattern — single period either way.
        _hero_tail = (
            f"{projection_label}. {_hero_caveat}"
            if _hero_caveat is not None
            else f"{projection_label}."
        )
        lines += [
            f"**Cut these calls ~{float(saving_pct):.1f}% — {before} → {after}.**",
            "",
            f"_{_hero_tail}_",
            "",
        ]
    else:
        lines += [
            "**No cheaper swap clears the quality bar.** "
            "Your current routing is already efficient for the models considered.",
            "",
        ]

    # --- What we found: Current -> After-swap comparison ---
    # Same 2x2 framing as the HTML: (Current / After swap) x (This sample /
    # Monthly). A markdown table makes the structure explicit where a flat
    # bullet list hid it. The after-swap figures carry a dagger that links to
    # the quality fine-print in the caveat callout below.
    lines += ["## What we found", ""]
    sample_col = (
        f"This sample ({result.window_days}-day)"
        if result.window_days is not None
        else "This sample"
    )
    if result.monthly_cost is not None:
        lines += [
            f"| | {sample_col} | Monthly projection |",
            "|---|------:|------:|",
            f"| **Current** | {_fmt_usd(result.total_cost)} "
            f"| {_fmt_usd(result.monthly_cost)} |",
        ]
        if has_saving:
            monthly_after = (
                f"{_fmt_usd(result.monthly_projected)} †"
                if result.monthly_projected is not None
                else "—"
            )
            lines.append(
                f"| **After recommended swap** | {_fmt_usd(result.projected_cost)} † "
                f"| {monthly_after} |"
            )
    else:
        lines += [
            f"| | {sample_col} |",
            "|---|------:|",
            f"| **Current** | {_fmt_usd(result.total_cost)} |",
        ]
        if has_saving:
            lines.append(
                f"| **After recommended swap** | {_fmt_usd(result.projected_cost)} † |"
            )
    lines.append("")

    # Saving delta — concrete dollars, grounding the headline percentage.
    # RECONCILIATION: _reconciled_delta_pct already quantized both amounts and
    # derived the percent from the quantized pair; reuse those to compute
    # delta_amt so SAVING == printed Current − printed After exactly.
    if has_saving:
        if result.monthly_cost is not None and result.monthly_projected is not None:
            _cur_q, _proj_q, _ = _reconciled_delta_pct(
                result.monthly_cost, result.monthly_projected
            )
        else:
            _cur_q, _proj_q, _ = _reconciled_delta_pct(
                result.total_cost, result.projected_cost
            )
        delta_amt = _cur_q - _proj_q
        lines += [
            f"**You save {_fmt_usd(delta_amt)}{_delta_unit} "
            f"(−{float(saving_pct):.1f}%).**",
            "",
        ]

    # Metadata line — demoted below the comparison (not part of the cost matrix).
    meta_bits = [
        f"{result.priced_calls:,} calls priced"
        + (f" + {result.unpriced_calls:,} unpriced" if result.unpriced_calls else "")
    ]
    if result.pricing_json_last_synced:
        meta_bits.append(f"pricing synced {result.pricing_json_last_synced}")
    # Quality synced rides the same meta line beside pricing synced (Item 1).
    if result.quality_json_last_synced:
        meta_bits.append(f"quality synced {result.quality_json_last_synced}")
    lines += ["_" + " · ".join(meta_bits) + "_", ""]

    # --- Details: log span + staleness/window cautions --------------------
    # The report is the full view, so the verbose-only disclosures live here,
    # after the decision matrix and its meta line.  The synced dates already
    # ride the meta line above; this block carries the log span, the staleness
    # cautions (Item 2) and the window caution (Item 3).
    detail_lines: list[str] = []
    span_line = _md_log_span_line(result)
    if span_line:
        detail_lines.append(span_line)
    callouts = _md_staleness_callouts(result) + _md_window_caution_lines(result)
    if detail_lines or callouts:
        lines += ["## Details", ""]
        lines += detail_lines
        if callouts:
            if detail_lines:
                lines.append("")
            lines += callouts
        lines.append("")

    # Swap plan — wholesale only; split runs use the routing-plan section.
    _swap_plan_lines = _render_md_wholesale_swap_plan(result)
    if _swap_plan_lines:
        lines += _swap_plan_lines

    # --- Cost by model (with total row) ---
    if result.cost_by_model:
        lines += [
            "## Cost by model",
            "",
            "| Model | Calls | Cost | % of total |",
            "|-------|------:|-----:|-----------:|",
        ]
        total_priced_calls = 0
        for model, cost in sorted(
            result.cost_by_model.items(), key=lambda kv: kv[1], reverse=True
        ):
            pct = (cost / result.total_cost * 100) if result.total_cost else Decimal("0")
            calls = result.calls_by_model.get(model, 0)
            total_priced_calls += calls
            lines.append(f"| {model} | {calls:,} | {_fmt_usd(cost)} | {float(pct):.1f}% |")
        lines.append(
            f"| **Total** | **{total_priced_calls:,}** | "
            f"**{_fmt_usd(result.total_cost)}** | **100.0%** |"
        )
        lines.append("")

    # --- Recommended swap ---
    if has_saving:
        baseline = _dominant_model(result)
        swap_label = (
            f"`{baseline}` → `{result.candidate_model}`"
            if baseline
            else f"`{result.candidate_model}`"
        )
        lines += ["## Recommended swap", "", swap_label, ""]

    lines += _data_quality_md(result)

    # Quality-measurement section before the caveats/methodology footer (empty /
    # byte-identical when no measure run ran).
    quality_lines = _quality_section_md(measure_result, limits=limits, result=result)
    if quality_lines:
        lines += ["", *quality_lines]

    # --- Caveats callout (only when a recommendation is present) ---
    # The dagger ties the after-swap figures above to this fine-print note,
    # styled as an official "before you switch" disclosure (honesty invariant).
    if result.candidate_model:
        _caveat = _quality_caveat_text(QUALITY_CAVEAT, measure_result)
        attribution = _get_attribution()
        if _caveat is not None:
            lines.append(f"> **† Before you switch:** {_caveat}")
            if attribution:
                lines.append(">")
                lines.append(f"> _Source: {attribution}_")
        elif attribution:
            lines.append(f"> _Source: {attribution}_")
        # Unrated-message family (findings #1 + #4) — same wording as every
        # other surface; its own callout(s) beneath the "before you switch" note.
        _unrated_lines = _md_unrated_family_lines(result, judged_models_from_measure(measure_result))
        if _unrated_lines:
            lines += [""] + _unrated_lines
        lines += ["", FUNNEL_LINE, ""]

    # --- Methodology / privacy footer ---
    lines += [
        "---",
        "",
        f"_{_methodology_note_text(measure_result)}_",
        "",
        "_methodology · tokencost · LiteLLM registry · "
        f"LMArena quality tiers, RouteLLM-style routing · {_methodology_tail_text(measure_result)}_",
        "",
    ]

    _atomic_write_text(output_path, "\n".join(lines))


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_HTML_CSS = """\
:root{
  --bg:#020202;--bg-2:#0a0a0a;--surface-1:#0E0E0E;--surface-2:#151515;--surface-3:#1C1C1C;
  --border:#2C2C2C;--ink:#F7F7F7;--ink-mute:#A1A1AA;--ink-dim:#6B6B72;--ink-faint:#3F3F46;
  --cyan:#00D1FF;--cyan-border:rgba(0,209,255,0.20);--green:#10B981;--red:#F87171;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{background:var(--bg)}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  font-feature-settings:"cv01","cv11","ss03","zero";
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility;
  background:var(--bg);color:var(--ink);line-height:1.55;
  min-height:100vh;padding-bottom:3rem;
}
header{border-bottom:1px solid var(--border);padding:18px 24px;margin-bottom:2rem}
.header-inner{max-width:860px;margin:0 auto;display:flex;align-items:center}
.wordmark{
  display:inline-flex;align-items:center;gap:8px;
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:18px;font-weight:700;letter-spacing:0.04em;color:var(--ink);
}
.brand-mark{width:16px;height:16px;color:var(--cyan);flex-shrink:0}
.container{max-width:860px;margin:0 auto;padding:0 24px}
.eyebrow{
  display:block;
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:11px;font-weight:500;letter-spacing:0.18em;text-transform:uppercase;
  color:var(--cyan);
  -webkit-font-smoothing:antialiased;text-rendering:geometricPrecision;
  margin-bottom:14px;
}
.card{
  background:var(--surface-1);border:1px solid var(--border);
  border-radius:10px;padding:24px 28px;margin-bottom:16px;
}
.saving-hero{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:3rem;font-weight:700;color:var(--green);line-height:1;margin-bottom:6px;
}
.saving-sub{font-size:.85rem;color:var(--ink-mute);margin-bottom:1rem}
.stat-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:.75rem;margin-top:.75rem;
}
.stat-label{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.68rem;text-transform:uppercase;letter-spacing:0.10em;
  color:var(--ink-dim);margin-bottom:.25rem;font-weight:500;
}
.stat-value{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:1rem;font-weight:600;color:var(--ink);font-variant-numeric:tabular-nums;
}
.stat-value.cyan{color:var(--cyan)}
.stat-value small{font-size:.78rem;font-weight:400;color:var(--ink-dim)}
.notice{color:var(--red)}
table{width:100%;border-collapse:collapse}
th{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.68rem;text-transform:uppercase;letter-spacing:0.10em;
  color:var(--ink-dim);padding:.5rem .75rem;
  border-bottom:1px solid var(--border);text-align:left;font-weight:500;
}
td{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.85rem;padding:.5rem .75rem;
  border-bottom:1px solid var(--surface-2);color:var(--ink-mute);
}
td.num{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
.model-name{color:var(--cyan)}
.projection-note{font-size:.8rem;color:var(--ink-dim);margin-top:.75rem}
/* Quality-tier disclosure: dim label + source tag; model names keep .model-name cyan. */
.qt-label{color:var(--ink-mute);font-weight:500}
.qt-src{color:var(--ink-dim)}
.caveat{font-size:.85rem;color:var(--ink-mute);line-height:1.6}
.attribution{font-size:.75rem;color:var(--ink-dim);margin-top:.5rem}
/* Amber caution inline note (stale pricing/quality table, --window override).
   Amber is reserved for caution across the report; the refresh-command/flag
   <code> stays in the surface's code styling and never wraps mid-token. */
.caution{color:#F59E0B}
.caution code{color:#F59E0B;background:rgba(245,158,11,0.10);border-color:rgba(245,158,11,0.30);white-space:nowrap}
.methodology{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.75rem;color:var(--ink-dim);line-height:1.6;
}
code{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.88em;background:var(--surface-3);
  padding:1px 6px;border-radius:4px;border:1px solid var(--border);
}
code{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.88em;background:var(--surface-3);
  padding:1px 6px;border-radius:4px;border:1px solid var(--border);
}
"""  # noqa: E501 — _ROUTING_PLAN_V1_CSS is concatenated below.

# Routing-plan + share-bar CSS shared by HTML v1's stat surface.  Kept as its own
# constant (appended to _HTML_CSS) so the routed/kept/already-optimal table reads
# in v1's own aesthetic while the proportional share bar matches v2's.
_ROUTING_PLAN_V1_CSS = """
/* Saving dollar amount beside the hero percent — the headline money win, green
   (green == money). */
.saving-money{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:1.05rem;font-weight:600;color:var(--green);margin-bottom:1rem;
  font-variant-numeric:tabular-nums;
}
/* Routing-plan table — full per-bucket parity (Calls, % calls, Model, Status,
   Cost) plus the Blended total row.  Uses v1's table aesthetic. */
.routing-plan{margin:1rem 0 1.25rem;overflow-x:auto}
.tbl-plan{table-layout:auto;width:100%;border-collapse:collapse}
.tbl-plan td.bucket{white-space:nowrap;color:var(--ink)}
.tbl-plan .row-total td{border-top:1px solid var(--border);color:var(--ink)}
.tbl-plan .badge{
  display:inline-block;padding:2px 8px;border-radius:999px;
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.66rem;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;
  color:var(--green);border:1px solid rgba(16,185,129,0.35);background:rgba(16,185,129,0.08);
  white-space:nowrap;
}
/* Per-bucket SHARE bar — a quiet horizontal fill showing the bucket's % of all
   analyzed calls; the % is always shown as text beside it (never colour-only). */
.tbl-plan .c-share{white-space:nowrap;vertical-align:middle}
.share-bar{
  display:inline-block;vertical-align:middle;width:3.4rem;height:7px;
  border-radius:999px;background:rgba(107,107,114,0.28);overflow:hidden;position:relative;
}
.share-bar .share-fill{
  position:absolute;top:0;left:0;bottom:0;width:var(--share,0%);
  background:var(--cyan,#00D1FF);border-radius:999px;
}
.share-pct{
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.8rem;color:var(--ink-mute);margin-left:8px;font-variant-numeric:tabular-nums;
}
"""

_HTML_CSS = _HTML_CSS + _ROUTING_PLAN_V1_CSS

# Candidates-considered table CSS for HTML v1.  Brings the multi-candidate block
# to PARITY with the cost-by-model + routing-plan tables on this surface: the
# bare ``table`` / ``th`` / ``td`` rules above already give cell padding, a
# header rule and row separators, so here we only (a) right-align + nowrap the
# numeric columns, (b) hold the model name in the surface's cyan, and (c) give
# each status badge the routing-plan pill treatment, recoloured per status
# (recommended = cyan emphasis, more-expensive = amber caution, the rest muted).
_CANDIDATES_TABLE_V1_CSS = """
.candidates-considered{margin:.25rem 0 0;overflow-x:auto}
.tbl-candidates{table-layout:auto;width:100%;border-collapse:collapse}
/* Right-align + nowrap the two numeric columns (the cost and the saving) so
   they read as clean tabular figures with a comfortable gutter from the model
   name on their left (the base td padding supplies the inter-column gap). */
.tbl-candidates th.num,.tbl-candidates td.num{
  text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;
}
.tbl-candidates td.c-model .model-name{color:var(--cyan)}
.tbl-candidates td.c-status,.tbl-candidates th.c-status{white-space:nowrap}
.tbl-candidates td.c-tier,.tbl-candidates th.c-tier{white-space:nowrap;opacity:0.8}
/* Status badge — same pill shape/typography as the routing-plan badge, with a
   per-status colour so the recommended candidate reads as the headline pick and
   a more-expensive one carries a quiet amber caution. */
.tbl-candidates .badge{
  display:inline-block;padding:2px 8px;border-radius:999px;
  font-family:'JetBrains Mono',ui-monospace,'Cascadia Code','SF Mono',monospace;
  font-size:.66rem;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;
  white-space:nowrap;border:1px solid var(--border);background:var(--surface-2);
  color:var(--ink-mute);
}
.tbl-candidates .badge-recommended{
  color:var(--cyan);border-color:rgba(0,209,255,0.35);background:rgba(0,209,255,0.08);
}
.tbl-candidates .badge-more-expensive{
  color:#F59E0B;border-color:rgba(245,158,11,0.35);background:rgba(245,158,11,0.10);
}
.tbl-candidates .badge-considered,
.tbl-candidates .badge-unpriced{color:var(--ink-dim)}
/* Caption beneath the block — the report's standard dim caption treatment. */
.candidates-caption{
  font-size:.8rem;color:var(--ink-dim);line-height:1.6;margin-top:.75rem;
}
/* Default-pool bullet legend — a <ul> standing in for the old prose
   paragraphs; strip default list markers/indent since each <li> supplies its
   own literal "·" bullet character, matching the terminal/Markdown surfaces. */
.candidates-legend{list-style:none;margin:.75rem 0 0;padding:0}
.candidates-legend li{margin:0 0 .25rem;padding-left:1rem;position:relative}
.candidates-legend li:last-child{margin-bottom:0}
.candidates-legend li::before{content:"\\00b7";position:absolute;left:0}
"""

_HTML_CSS = _HTML_CSS + _CANDIDATES_TABLE_V1_CSS


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>frugon — cost analysis</title>
<style>{css}</style>
</head>
<body>
<header>
<div class="header-inner">
<div class="wordmark">
<svg class="brand-mark" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<circle cx="12" cy="5" r="2.5" fill="currentColor"/>
<circle cx="6" cy="15.5" r="2.5" fill="currentColor"/>
<circle cx="18" cy="15.5" r="2.5" fill="currentColor"/>
</svg>
FRUGON
</div>
</div>
</header>
<div class="container">
{body}
</div>
</body>
</html>"""


def _html_share_cell(pct: float) -> str:
    """A SHARE table cell: the proportional fill bar + the % shown as text.

    Shared by the HTML v1 and v2 routing-plan tables so the share bar markup (and
    therefore the figure) is authored ONCE.  The % is always shown as TEXT beside
    the bar — never colour/width only — so it stays accessible.  The CSS classes
    (``.share-bar`` / ``.share-fill`` / ``.share-pct``) live in the shared
    ``_ROUTING_PLAN_SHARED_CSS`` appended to both stylesheets.
    """
    return (
        '<td class="c-share">'
        f'<span class="share-bar" style="--share:{pct:.1f}%">'
        '<span class="share-fill"></span></span>'
        f'<span class="share-pct">{pct:.1f}%</span>'
        "</td>"
    )


def _render_html_routing_plan(
    result: AnalysisResult,
    split: SplitRouting,
    fig: _SplitReportFigures,
    esc: Callable[[str], str],
) -> str:
    """Render the v1 routing-plan table — Bucket | Calls | % calls | Model | Status | Cost.

    Brings the v1 HTML surface to full information parity with the Markdown report
    and HTML v2: every analyzed call is accounted for across the routed / kept /
    already-optimal buckets, each carrying its per-bucket Cost and
    its share of all analyzed calls (the shared :func:`_call_share_pcts`,
    largest-remainder rounded to sum to exactly 100.0%), closed by a Blended total
    row.  Uses v1's own table aesthetic (the ``table`` / ``td.num`` / ``.badge``
    classes) and the shared share-bar cell, so the figures equal the other
    surfaces while the look stays v1's.
    """
    share_counts = [split.routed_count, split.kept_count]
    if fig.already_cheap > 0:
        share_counts.append(fig.already_cheap)
    shares = _call_share_pcts(share_counts)

    # Quality badge for the routed row — three cases (mirrors the terminal/MD logic):
    # 1. Candidate UNRATED — omit the badge; unrated caveat handles the disclosure.
    # 2. tier_drop <= 0 — show "same or better quality" (positive, non-risk claim).
    # 3. tier_drop >= 1 — show "within tolerance" (the existing band-estimate badge).
    if _is_unrated(split.candidate_model):
        _tol_badge = ""
    elif _is_equal_or_better_quality(result):
        _tol_badge = '<span class="badge">same or better quality</span>'
    else:
        _tol_badge = '<span class="badge">within tolerance</span>'

    rows = [
        "<tr>"
        '<td class="bucket">Routed &middot; easy</td>'
        f'<td class="num">{split.routed_count:,}</td>'
        f"{_html_share_cell(shares[0])}"
        f'<td><span class="model-name">{esc(split.candidate_model)}</span></td>'
        f"<td>{_tol_badge}</td>"
        f'<td class="num">{_fmt_usd(split.routed_cost)}</td>'
        "</tr>",
        "<tr>"
        '<td class="bucket">Kept &middot; hard</td>'
        f'<td class="num">{split.kept_count:,}</td>'
        f"{_html_share_cell(shares[1])}"
        f'<td><span class="model-name">{esc(split.baseline_model)}</span></td>'
        "<td></td>"
        f'<td class="num">{_fmt_usd(split.kept_cost)}</td>'
        "</tr>",
    ]
    if fig.already_cheap > 0:
        already_models = (
            esc(", ".join(fig.other_models)) if fig.other_models else "a cheaper model"
        )
        rows.append(
            "<tr>"
            '<td class="bucket">Keep &middot; already optimal</td>'
            f'<td class="num">{fig.already_cheap:,}</td>'
            f"{_html_share_cell(shares[2])}"
            f'<td><span class="model-name">{already_models}</span></td>'
            '<td><span class="badge">no action</span></td>'
            f'<td class="num">{_fmt_usd(fig.already_cheap_cost)}</td>'
            "</tr>"
        )
    rows.append(
        '<tr class="row-total">'
        "<td><strong>Blended</strong></td>"
        f'<td class="num"><strong>{result.priced_calls:,}</strong></td>'
        '<td class="num"><strong>100.0%</strong></td>'
        "<td>&mdash;</td><td></td>"
        f'<td class="num"><strong>{_fmt_usd(fig.blended)}</strong></td>'
        "</tr>"
    )
    return (
        '<div class="routing-plan"><table class="tbl-plan"><thead><tr>'
        "<th>Bucket</th><th style='text-align:right'>Calls</th>"
        "<th>% calls</th><th>Model</th><th>Status</th>"
        "<th style='text-align:right'>Cost</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_html_wholesale_swap_plan(
    result: AnalysisResult,
    esc: Callable[[str], str],
) -> str:
    """v1 swap-plan table for a wholesale run — Model | Calls | % calls | Current cost | Action.

    Mirrors the routing-plan table layout so wholesale 'Swap plan' and split
    'Routing plan' read as visual siblings.  Returns empty string when no
    candidate is set.
    """
    if not result.candidate_model or not result.cost_by_model:
        return ""
    candidate = result.candidate_model
    non_cand = sorted(
        [(m, c) for m, c in result.cost_by_model.items() if m != candidate],
        key=lambda mc: mc[1],
        reverse=True,
    )
    cand_cost = result.cost_by_model.get(candidate, Decimal("0"))
    ordered: list[tuple[str, Decimal]] = non_cand + [(candidate, cand_cost)]

    counts = [result.calls_by_model.get(m, 0) for m, _ in ordered]
    shares = _call_share_pcts(counts)

    rows = []
    for (model, cost), share in zip(ordered, shares, strict=True):
        calls = result.calls_by_model.get(model, 0)
        if model == candidate:
            action = f'<span class="badge">already on {esc(candidate)}</span>'
        else:
            action = f'→ swap to <span class="model-name">{esc(candidate)}</span>'
        rows.append(
            "<tr>"
            f'<td><span class="model-name">{esc(model)}</span></td>'
            f'<td class="num">{calls:,}</td>'
            f"{_html_share_cell(share)}"
            f'<td class="num">{_fmt_usd(cost)}</td>'
            f"<td>{action}</td>"
            "</tr>"
        )
    total_calls = sum(result.calls_by_model.get(m, 0) for m, _ in ordered)
    total_cost = sum((c for _, c in ordered), Decimal("0"))
    rows.append(
        '<tr class="row-total">'
        "<td><strong>Total</strong></td>"
        f'<td class="num"><strong>{total_calls:,}</strong></td>'
        '<td class="num"><strong>100.0%</strong></td>'
        f'<td class="num"><strong>{_fmt_usd(total_cost)}</strong></td>'
        "<td></td>"
        "</tr>"
    )
    return (
        '<div class="routing-plan"><table class="tbl-plan"><thead><tr>'
        "<th>Model</th><th style='text-align:right'>Calls</th>"
        "<th>% calls</th><th style='text-align:right'>Current cost</th>"
        "<th>Action</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_html_v1_split_body(
    result: AnalysisResult,
    split: SplitRouting,
    *,
    measure_result: MeasureResult | None = None,
    limits: PreviewLimits | None = None,
) -> str:
    """Build the v1 HTML body (cards) for the per-call split-routing headline.

    When *measure_result* is provided, a "Quality measurement" card is inserted
    before the methodology card; when None the output is byte-identical.
    *limits* carries the (possibly flag-overridden) report preview lengths.
    """
    esc = _html_escape.escape
    saving_pct = split.saving_pct
    assert saving_pct is not None  # guaranteed by _has_split
    # Every headline/stat figure reconciles to the FULL analyzed dataset
    # (identical to the terminal panel) via the shared helper — NOT the
    # baseline-only figure.  The hero percent, Current/Blended costs, and the
    # already-on-a-cheaper-model bucket all come from here.
    fig = _split_report_figures(result, split)
    projection_label = esc(_projection_label(result))
    parts: list[str] = []

    # Upper-bound (decision info) sits in the headline card directly under the
    # saving sub-line, BEFORE the freshness stats below it (Item 7 ordering).
    upper = _upper_bound_pct(result)
    upper_note = ""
    if upper is not None:
        upper_note = (
            '<p class="projection-note">Upper bound: moving every call to '
            f'<span class="model-name">{esc(result.candidate_model or "")}</span> '
            f"saves ~{float(upper):.1f}% &mdash; the aggressive end; the "
            f"~{float(fig.total_pct):.1f}% split shown is the conservative, "
            "quality-respecting recommendation (a full swap is a larger quality change).</p>"
        )

    # Routing-plan table — full information parity with the Markdown and HTML v2
    # surfaces: per-bucket Calls, % of all analyzed calls (with the shared share
    # bar), routed-to model, status, and the per-bucket Cost, then
    # a Blended total row.  Built from the SAME shared figures/helpers
    # (_split_report_figures, _call_share_pcts, _html_share_cell) so every number
    # equals the other surfaces.  Replaces the count-only stat-grid that omitted
    # the costs, shares, and totals (the v1 parity gap).
    plan_html = _render_html_routing_plan(result, split, fig, esc)

    # The saving DOLLAR amount beside the hero percent — the report's headline
    # money win, which the percent-only hero omitted.
    saving_sub_money = (
        f'<div class="saving-money">You save {_fmt_usd(fig.saved)}'
        f'{"/mo" if fig.projected else ""}</div>'
    )

    stats = '<div class="stat-grid">'
    stats += (
        '<div class="stat"><div class="stat-label">Current cost</div>'
        f'<div class="stat-value">{_fmt_usd(fig.current)} <small>total</small></div></div>'
        '<div class="stat"><div class="stat-label">Blended cost</div>'
        f'<div class="stat-value">{_fmt_usd(fig.blended)} <small>after routing</small></div></div>'
    )
    # Freshness stats: pricing AND quality last-synced (Item 1), each with its
    # staleness caution when the table is old (Item 2 — amber .caution note).
    if result.pricing_json_last_synced:
        pricing_note = _html_staleness_note(
            result.pricing_json_last_synced,
            _is_pricing_stale(result.pricing_json_last_synced, max_days=30),
            _PRICING_REFRESH_COMMAND,
        )
        stats += (
            '<div class="stat"><div class="stat-label">Pricing last synced</div>'
            '<div class="stat-value" style="color:var(--ink-dim);font-weight:400">'
            f"{esc(result.pricing_json_last_synced)}{pricing_note}</div></div>"
        )
    if result.quality_json_last_synced:
        quality_note = _html_staleness_note(
            result.quality_json_last_synced,
            _is_quality_stale(result.quality_json_last_synced, max_days=60),
            _QUALITY_REFRESH_COMMAND,
        )
        stats += (
            '<div class="stat"><div class="stat-label">Quality last synced</div>'
            '<div class="stat-value" style="color:var(--ink-dim);font-weight:400">'
            f"{esc(result.quality_json_last_synced)}{quality_note}</div></div>"
        )
    stats += "</div>"

    if result.window_days is not None:
        proj_note = (
            f'<p class="projection-note">Monthly projection based on a '
            f"<strong>{result.window_days}-day</strong> window (--window flag).</p>"
        )
    elif result.observed_span_days is not None:
        proj_note = (
            f'<p class="projection-note">Monthly projection based on '
            f"<strong>{result.observed_span_days:.1f} days</strong> of observed log timestamps.</p>"
        )
    else:
        proj_note = (
            '<p class="projection-note">Costs shown are totals across the analyzed calls. '
            "Pass <code>--window DAYS</code> to project to a monthly figure.</p>"
        )

    # Log span (Item 4) — earliest -> latest, the window the projection covers.
    log_span = _html_log_span_html(result)
    log_span_note = (
        f'<p class="projection-note">{log_span}</p>' if log_span else ""
    )
    # Window-vs-span caution (Item 3) — amber, near the projection note.
    window_caution = _html_window_caution(result)
    window_note = (
        f'<p class="projection-note">{window_caution}</p>' if window_caution else ""
    )

    # Quality-tier disclosure — the published LMArena class the routing moves
    # between, dim/neutral, directly under the routing stats.  Cyan model names,
    # plain labels (green stays money-only).
    tier_note = (
        f'<p class="projection-note">'
        f'{_html_quality_tier_line(split, model_class="model-name")}</p>'
    )

    # Order: the Quality-tier benchmark comparison rides directly under the
    # Upper-bound swap context (the decision area) and ABOVE the freshness
    # (Pricing/Quality last-synced) stats — the same grouping as the terminal and
    # Markdown surfaces.  The Log span — pure freshness metadata — stays below.
    parts.append(
        '<div class="card">'
        '<div class="eyebrow">Split routing</div>'
        f'<div class="saving-hero">{float(fig.total_pct):.1f}%</div>'
        f'<div class="saving-sub">estimated blended saving ({projection_label})</div>'
        + saving_sub_money
        + upper_note
        + plan_html
        + tier_note
        + stats
        + proj_note
        + log_span_note
        + window_note
        + "</div>"
    )

    # Candidates considered — multi-candidate transparency block, no-op when
    # the user passed <=1 candidate (string is empty so the surface is
    # byte-identical to before).  Wrapped in its own card so it reads as a peer
    # of the routing plan / by-model breakdown.
    _cand_html = _candidates_considered_html(
        result, esc, has_judge_section=_is_tier1(measure_result)
    )
    if _cand_html:
        parts.append(
            '<div class="card">'
            f'<div class="eyebrow">{esc(_candidates_header_title(result))}</div>'
            + _cand_html
            + "</div>"
        )

    # Per-model breakdown (same table markup as the wholesale v1 path), now with a
    # Total row so the per-model costs reconcile to the headline total — parity
    # with the Markdown and HTML v2 cost-by-model tables.
    if result.cost_by_model:
        rows = ""
        total_calls = 0
        for model, cost in sorted(result.cost_by_model.items(), key=lambda kv: kv[1], reverse=True):
            pct = (cost / result.total_cost * 100) if result.total_cost else Decimal("0")
            calls = result.calls_by_model.get(model, 0)
            total_calls += calls
            rows += (
                "<tr>"
                f'<td><span class="model-name">{esc(model)}</span></td>'
                f'<td class="num">{calls:,}</td>'
                f'<td class="num">{_fmt_usd(cost)}</td>'
                f'<td class="num">{float(pct):.1f}%</td>'
                "</tr>"
            )
        rows += (
            '<tr class="row-total">'
            "<td><strong>Total</strong></td>"
            f'<td class="num"><strong>{total_calls:,}</strong></td>'
            f'<td class="num"><strong>{_fmt_usd(result.total_cost)}</strong></td>'
            '<td class="num"><strong>100.0%</strong></td>'
            "</tr>"
        )
        parts.append(
            '<div class="card">'
            '<div class="eyebrow">By model</div>'
            "<table><thead><tr>"
            "<th>Model</th><th style='text-align:right'>Calls</th>"
            "<th style='text-align:right'>Cost</th><th style='text-align:right'>% of total</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            "</div>"
        )

    attribution = _get_attribution()
    attribution_html = (
        f'<p class="attribution">{esc(attribution)}</p>' if attribution else ""
    )
    _split_caveat_v1 = (
        SPLIT_CAVEAT_EQUAL_OR_BETTER
        if _is_equal_or_better_quality(result)
        else SPLIT_CAVEAT
    )
    _caveat = _quality_caveat_text(_split_caveat_v1, measure_result)
    _caveat_html = f'<p class="caveat">{esc(_caveat)}</p>' if _caveat else ""
    # Unrated-message family (findings #1 + #4) — same wording as every surface.
    _unrated_html = _html_unrated_family_html(result, judged_models_from_measure(measure_result))
    # When the caveat is dropped (Tier-1: a scored verdict already shows) and
    # there is no attribution either, the Quality card would be empty — skip it.
    if _caveat_html or _unrated_html or attribution_html:
        parts.append(
            '<div class="card">'
            '<div class="eyebrow">Quality</div>'
            + _caveat_html
            + _unrated_html
            + attribution_html
            + "</div>"
        )
    dq = _data_quality_html(result)
    if dq:
        parts.append(f'<div class="card">{dq}</div>')
    # Quality-measurement card before the methodology card (empty / byte-
    # identical when no measure run ran).
    quality_card = _quality_section_html(measure_result, style="v1", limits=limits, result=result)
    if quality_card:
        parts.append(quality_card)
    parts.append(
        '<div class="card">'
        '<div class="eyebrow">Methodology</div>'
        f'<p class="methodology">{esc(_methodology_note_text(measure_result))}</p>'
        "</div>"
    )
    parts.append(
        '<p class="methodology" style="margin-top:.5rem">'
        + esc(FUNNEL_LINE)
        + ' <a href="https://frugon.rodiun.io" style="color:var(--cyan);text-decoration:none">↗</a></p>'
    )
    return _HTML_TEMPLATE.format(
        css=_HTML_CSS + _quality_css(measure_result), body="\n".join(parts)
    )

def render_html(
    result: AnalysisResult,
    output_path: Path,
    *,
    measure_result: MeasureResult | None = None,
    limits: PreviewLimits | None = None,
) -> None:
    """Write a self-contained HTML cost report to *output_path*.

    The file is fully local — no external CDN, no font URLs, no analytics.
    All CSS is inlined in a <style> block. The quality caveat is included
    only when a routing recommendation is present (candidate_model is not None).

    When *measure_result* is provided, a "Quality measurement" card is appended
    mirroring the terminal's quality output; when None the output is
    byte-identical to before this parameter existed.
    """
    if _has_split(result):
        assert result.split is not None  # narrowed by _has_split
        _atomic_write_text(
            output_path,
            _render_html_v1_split_body(
                result, result.split, measure_result=measure_result, limits=limits
            ),
        )
        return

    body_parts: list[str] = []

    if result.priced_calls == 0:
        body_parts.append(
            '<div class="card">'
            '<div class="eyebrow notice">No priced calls</div>'
            f"<p>Analyzed <strong>{result.total_calls:,}</strong> records — "
            "none could be priced (unknown models or missing usage data).</p>"
            "<p class='projection-note'>"
            "Run <code>frugon pricing update</code> to refresh the pricing table, "
            "or check that your log records include a <code>model</code> field.</p>"
            "</div>"
        )
    else:
        # RECONCILIATION: derive the displayed saving% from the quantized amounts
        # that _fmt_usd prints, so the hero percent equals
        # round(printed_save / printed_current * 100, 1).
        # compute_saving_pct is kept for the gate check (> Decimal("0")) only.
        _raw_saving_pct = compute_saving_pct(result.total_cost, result.projected_cost)
        _, _, saving_pct = _reconciled_delta_pct(result.total_cost, result.projected_cost)
        projection_label = _html_escape.escape(_projection_label(result))

        # --- Headline card ---
        if result.candidate_model and _raw_saving_pct is not None and _raw_saving_pct > Decimal("0"):
            saving_str = f"{float(saving_pct):.1f}%"
            headline = (
                f'<div class="saving-hero">{_html_escape.escape(saving_str)}</div>'
                f'<div class="saving-sub">estimated saving ({projection_label})</div>'
            )
        else:
            headline = (
                '<div class="saving-sub" style="margin-bottom:.5rem">'
                "No cheaper candidate found within quality constraints.</div>"
            )

        stats = '<div class="stat-grid">'
        stats += (
            f'<div class="stat"><div class="stat-label">Current cost</div>'
            f'<div class="stat-value">{_fmt_usd(result.total_cost)}'
            f" <small>observed</small></div></div>"
        )

        if result.monthly_cost is not None:
            stats += (
                f'<div class="stat"><div class="stat-label">Monthly cost</div>'
                f'<div class="stat-value">{_fmt_usd(result.monthly_cost)}'
                f" <small>monthly projection</small></div></div>"
            )

        if result.candidate_model and _raw_saving_pct is not None and _raw_saving_pct > Decimal("0"):
            baseline = _dominant_model(result)
            swap_label = (
                f"{_html_escape.escape(baseline)} → {_html_escape.escape(result.candidate_model)}"
                if baseline
                else _html_escape.escape(result.candidate_model)
            )
            stats += (
                f'<div class="stat"><div class="stat-label">Projected cost</div>'
                f'<div class="stat-value">{_fmt_usd(result.projected_cost)}'
                f" <small>observed</small></div></div>"
                f'<div class="stat"><div class="stat-label">Recommended swap</div>'
                f'<div class="stat-value cyan">{swap_label}</div></div>'
            )
            if result.monthly_projected is not None:
                stats += (
                    f'<div class="stat"><div class="stat-label">Monthly projected</div>'
                    f'<div class="stat-value">{_fmt_usd(result.monthly_projected)}'
                    f" <small>monthly projection</small></div></div>"
                )

        unpriced = ""
        if result.unpriced_calls:
            unpriced = (
                f' <small style="color:var(--red)">'
                f"({result.unpriced_calls:,} unpriced)</small>"
            )
        stats += (
            f'<div class="stat"><div class="stat-label">Calls analyzed</div>'
            f"<div class='stat-value'>{result.priced_calls:,} priced{unpriced}</div></div>"
        )

        if result.pricing_json_last_synced:
            pricing_note = _html_staleness_note(
                result.pricing_json_last_synced,
                _is_pricing_stale(result.pricing_json_last_synced, max_days=30),
                _PRICING_REFRESH_COMMAND,
            )
            stats += (
                f'<div class="stat"><div class="stat-label">Pricing last synced</div>'
                f'<div class="stat-value" style="color:var(--ink-dim);font-weight:400">'
                f"{_html_escape.escape(result.pricing_json_last_synced)}{pricing_note}</div></div>"
            )
        if result.quality_json_last_synced:
            quality_note = _html_staleness_note(
                result.quality_json_last_synced,
                _is_quality_stale(result.quality_json_last_synced, max_days=60),
                _QUALITY_REFRESH_COMMAND,
            )
            stats += (
                f'<div class="stat"><div class="stat-label">Quality last synced</div>'
                f'<div class="stat-value" style="color:var(--ink-dim);font-weight:400">'
                f"{_html_escape.escape(result.quality_json_last_synced)}{quality_note}</div></div>"
            )
        stats += "</div>"

        if result.window_days is not None:
            proj_note = (
                f'<p class="projection-note">Monthly projection based on a '
                f"<strong>{result.window_days}-day</strong> window (--window flag). "
                f"Actual savings depend on your call distribution.</p>"
            )
        elif result.observed_span_days is not None:
            proj_note = (
                f'<p class="projection-note">Monthly projection based on '
                f"<strong>{result.observed_span_days:.1f} days</strong> "
                f"of observed log timestamps.</p>"
            )
        else:
            proj_note = (
                '<p class="projection-note">Costs shown are totals across the analyzed calls, '
                "not monthly projections. "
                "Pass <code>--window DAYS</code> to project to a monthly figure.</p>"
            )

        # Log span (Item 4) + window caution (Item 3) — the report is the
        # full view, so both disclosures ride beside the projection note.
        _log_span = _html_log_span_html(result)
        log_span_note = (
            f'<p class="projection-note">{_log_span}</p>' if _log_span else ""
        )
        _window_caution = _html_window_caution(result)
        window_note = (
            f'<p class="projection-note">{_window_caution}</p>' if _window_caution else ""
        )
        body_parts.append(
            '<div class="card">'
            '<div class="eyebrow">Cost analysis</div>'
            + headline
            + stats
            + proj_note
            + log_span_note
            + window_note
            + "</div>"
        )

        # Swap plan — wholesale only; split runs go through the routing-plan path.
        if result.candidate_model:
            _esc_ws = _html_escape.escape
            _swap_plan = _render_html_wholesale_swap_plan(result, _esc_ws)
            if _swap_plan:
                body_parts.append(
                    '<div class="card">'
                    '<div class="eyebrow">Swap plan</div>'
                    + _swap_plan
                    + "</div>"
                )

        # Candidates considered (multi-candidate transparency) — no-op when the
        # user passed <=1 candidate (returns empty string, no card emitted).
        _esc = _html_escape.escape
        _cand_html = _candidates_considered_html(
            result, _esc, has_judge_section=_is_tier1(measure_result)
        )
        if _cand_html:
            body_parts.append(
                '<div class="card">'
                f'<div class="eyebrow">{_esc(_candidates_header_title(result))}</div>'
                + _cand_html
                + "</div>"
            )

        # Per-model breakdown
        if result.cost_by_model:
            rows = ""
            for model, cost in sorted(
                result.cost_by_model.items(), key=lambda kv: kv[1], reverse=True
            ):
                pct = (
                    (cost / result.total_cost * 100)
                    if result.total_cost
                    else Decimal("0")
                )
                calls = result.calls_by_model.get(model, 0)
                rows += (
                    "<tr>"
                    f'<td><span class="model-name">{_html_escape.escape(model)}</span></td>'
                    f'<td class="num">{calls:,}</td>'
                    f'<td class="num">{_fmt_usd(cost)}</td>'
                    f'<td class="num">{float(pct):.1f}%</td>'
                    "</tr>"
                )

            body_parts.append(
                '<div class="card">'
                '<div class="eyebrow">By model</div>'
                "<table><thead><tr>"
                "<th>Model</th>"
                "<th style='text-align:right'>Calls</th>"
                "<th style='text-align:right'>Cost</th>"
                "<th style='text-align:right'>% of total</th>"
                f"</tr></thead><tbody>{rows}</tbody></table>"
                "</div>"
            )

        # Quality caveat — only when a routing recommendation is present
        if result.candidate_model:
            attribution = _get_attribution()
            attribution_html = (
                f'<p class="attribution">{_html_escape.escape(attribution)}</p>'
                if attribution
                else ""
            )
            _caveat = _quality_caveat_text(QUALITY_CAVEAT, measure_result)
            _caveat_html = (
                f'<p class="caveat">{_html_escape.escape(_caveat)}</p>'
                if _caveat
                else ""
            )
            # Unrated-message family (findings #1 + #4) — same wording as every
            # surface; here in the wholesale Quality card.
            _unrated_html = _html_unrated_family_html(result, judged_models_from_measure(measure_result))
            # Tier-1 drops the caveat (a verdict already shows); skip the card
            # when nothing (caveat nor attribution) remains to put in it.
            if _caveat_html or _unrated_html or attribution_html:
                body_parts.append(
                    '<div class="card">'
                    '<div class="eyebrow">Quality</div>'
                    + _caveat_html
                    + _unrated_html
                    + attribution_html
                    + "</div>"
                )

        dq = _data_quality_html(result)
        if dq:
            body_parts.append(f'<div class="card">{dq}</div>')

        # Quality-measurement card before the methodology card (empty / byte-
        # identical when no measure run ran).
        quality_card = _quality_section_html(measure_result, style="v1", limits=limits, result=result)
        if quality_card:
            body_parts.append(quality_card)

        body_parts.append(
            '<div class="card">'
            '<div class="eyebrow">Methodology</div>'
            f'<p class="methodology">{_html_escape.escape(_methodology_note_text(measure_result))}</p>'
            "</div>"
        )

        if result.candidate_model:
            body_parts.append(
                '<p class="methodology" style="margin-top:.5rem">'
                + _html_escape.escape(FUNNEL_LINE)
                + ' <a href="https://frugon.rodiun.io"'
                + ' style="color:var(--cyan);text-decoration:none">'
                + "↗</a>"
                + "</p>"
            )
    html = _HTML_TEMPLATE.format(
        css=_HTML_CSS + _quality_css(measure_result), body="\n".join(body_parts)
    )
    _atomic_write_text(output_path, html)


# ---------------------------------------------------------------------------
# HTML renderer — v2 (refined "engineering instrument" treatment)
# ---------------------------------------------------------------------------
#
# v2 elevates v1 into a premium, editorial single-column report:
#   * the saving is the hero — a large GREEN tabular-nums figure with a
#     before -> after frame and the projection caveat directly beneath;
#   * quiet hairline borders and generous vertical rhythm (no boxy cards);
#   * a cost-by-model table with a pure-CSS inline proportion bar per row
#     plus a total row;
#   * a recommended-swap treatment with a tasteful pill;
#   * a landing-page-style methodology footer with the privacy line prominent.
# All CSS is inlined; no external fonts, no CDN, no JS, no network.
# GREEN (--green) is reserved for the positive saving, matching the landing page.

_HTML_CSS_V2 = """\
:root{
  --bg:#040506;--panel:#0A0C0D;--hair:#1B1F22;--hair-soft:#13171A;
  --ink:#F7F7F7;--ink-mute:#A1A1AA;--ink-dim:#6B6B72;--ink-faint:#3F3F46;
  --cyan:#00D1FF;--cyan-dim:#5FE0FF;--green:#10B981;--green-dim:#6EE7B7;
  --bar:rgba(0,209,255,0.14);--bar-edge:rgba(0,209,255,0.30);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{background:var(--bg)}
body{
  font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility;font-variant-ligatures:none;
  background:
    radial-gradient(900px 420px at 50% -8%,rgba(0,209,255,0.05),transparent 70%),
    var(--bg);
  color:var(--ink);line-height:1.6;min-height:100vh;
  padding:0 28px 4rem;
}
.mono{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
}
.tnum{font-variant-numeric:tabular-nums}
.wrap{max-width:1080px;margin:0 auto}

/* Masthead */
.masthead{
  display:flex;align-items:center;justify-content:space-between;gap:16px;
  padding:22px 0 18px;border-bottom:1px solid var(--hair);margin-bottom:30px;
}
.wordmark{
  display:inline-flex;align-items:center;gap:9px;
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:15px;font-weight:700;letter-spacing:0.16em;color:var(--ink);
}
.brand-mark{width:15px;height:15px;color:var(--cyan);flex-shrink:0}
.masthead-tag{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;letter-spacing:0.05em;color:var(--ink-dim);
}

/* Section spine */
section{margin-bottom:30px}
.eyebrow{
  display:block;
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;
  color:var(--cyan);margin-bottom:14px;
  -webkit-font-smoothing:antialiased;
}

/* Above-the-fold two-column layout.
   Left rail: saving hero + before/after + key stats.
   Right rail: cost-by-model table + recommended swap.
   Collapses to a single column on narrow viewports. */
.fold{
  display:grid;
  grid-template-columns:minmax(0,1fr) minmax(0,1fr);
  gap:30px 42px;
  align-items:start;
}
.fold .col{min-width:0}
.fold section:last-child{margin-bottom:0}
/* The methodology block and footer always span the full width below. */
.below{margin-top:6px}
/* Routing plan spans the full content width below the fold so its five
   columns (BUCKET | CALLS | MODEL | STATUS | COST) and the
   COST figures have room to sit INSIDE the table at desktop width. */
.plan-full{margin-top:30px}
@media (max-width:720px){
  body{padding:0 18px 4rem}
  .fold{grid-template-columns:1fr;gap:0}
}

/* Hero — the saving */
.hero-figure{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:clamp(2.9rem,6vw,4.2rem);font-weight:700;line-height:0.98;
  letter-spacing:-0.02em;color:var(--green);font-variant-numeric:tabular-nums;
}
.hero-figure .pct{color:var(--green)}
.hero-lede{
  font-size:1.02rem;color:var(--ink);margin-top:12px;max-width:48ch;line-height:1.5;
}
.beforeafter{
  display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  margin-top:20px;padding-top:18px;border-top:1px solid var(--hair-soft);
}
.ba-block{display:flex;flex-direction:column;gap:4px}
.ba-label{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;letter-spacing:0.12em;text-transform:uppercase;color:var(--ink-dim);
}
.ba-value{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums;color:var(--ink);
}
.ba-value.after{color:var(--green)}
.ba-arrow{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:1.4rem;color:var(--ink-faint);align-self:center;padding:0 2px;
}
.hero-caveat{
  font-size:0.9rem;color:var(--ink-mute);margin-top:18px;line-height:1.55;max-width:52ch;
}
.no-rec{
  font-size:1.12rem;color:var(--ink);line-height:1.55;max-width:54ch;
}
.no-rec .accent{color:var(--cyan)}

/* Dagger footnote marker — links figures to the fine-print disclosure. */
.dagger{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.7em;color:var(--ink-dim);vertical-align:super;line-height:0;
  margin-left:2px;font-weight:400;
}

/* Comparison matrix — Current vs After recommended swap, across
   This sample / Monthly projection. A genuine 2x2 so the structure the
   six flat tiles hid reads instantly. */
.matrix{
  border:1px solid var(--hair);border-radius:10px;overflow:hidden;background:var(--panel);
}
.matrix table{width:100%;border-collapse:collapse}
.matrix th,.matrix td{
  padding:13px 16px;text-align:right;
  font-variant-numeric:tabular-nums;border-bottom:1px solid var(--hair-soft);
}
.matrix thead th{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;letter-spacing:0.10em;text-transform:uppercase;
  color:var(--ink-dim);font-weight:500;border-bottom:1px solid var(--hair);
}
.matrix thead th.row-head,.matrix tbody th.row-head{text-align:left}
.matrix tbody th.row-head{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;letter-spacing:0.06em;text-transform:uppercase;font-weight:600;color:var(--ink-mute);
}
.matrix tr:last-child th,.matrix tr:last-child td{border-bottom:none}
.matrix td.fig{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:1.06rem;font-weight:600;color:var(--ink);
}
/* After-swap row: positive treatment in green. */
.matrix tr.after th.row-head{color:var(--green-dim)}
.matrix tr.after td.fig{color:var(--green)}
.matrix td.empty{color:var(--ink-faint)}
/* Narrow-viewport reflow (<=640px). A 3-column money matrix cannot fit a
   320px column without clipping the rightmost (Monthly projection) figure —
   the .matrix overflow:hidden would shear it off. Below 640px we drop the
   grid and stack each scenario (Current / After recommended swap) as its own
   card; the two value columns become labelled rows inside it, the label
   supplied by each cell's data-label (the original column header). This reads
   cleanly in a single narrow column and can never force horizontal scroll. */
@media (max-width:640px){
  .matrix table,.matrix thead,.matrix tbody,
  .matrix tr,.matrix th,.matrix td{display:block;width:auto}
  /* The header row only labelled the now-inlined columns — retire it. */
  .matrix thead{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);clip-path:inset(50%);white-space:nowrap}
  /* Each scenario becomes a card, separated by the existing hairline. */
  .matrix tbody tr{padding:4px 0}
  .matrix tbody tr:last-child{border-bottom:none}
  .matrix tbody tr+tr{border-top:1px solid var(--hair-soft)}
  /* Scenario name: the card heading, left-aligned, full width. */
  .matrix tbody th.row-head{
    text-align:left;padding:12px 16px 6px;border-bottom:none;
  }
  /* Value cells: a compact label->value definition pairing. The label and its
     figure sit adjacent at the start of the row with a comfortable gap — NOT
     justify-content:space-between, which on a wide narrow card (~480-640px)
     pushed the figure to the far edge and opened a large void that read as a
     layout accident. The label column has a fixed em-width floor so the two
     figures (sample / monthly) align into a tidy vertical column across the
     Current and After rows; the figure then sits just to the right of its
     label, visually associated at every viewport from 320 to 640px.

     The label->value gap is a viewport-responsive clamp: ~18px at the 320px
     floor (6vw == 19.2px at 320, clamped up to 18 min), growing to a 30px cap
     by ~500px to spend the horizontal slack the narrow card carries on wider
     phones (360/540) for more breathing room. It can never force a horizontal
     scroll: the label column is flex:0 1 auto (shrinks first), and even at the
     320px floor the 18px gap + 150px label floor sit well inside the ~284px
     content box. The desktop table (>=880px) is unaffected — it never stacks. */
  .matrix td.fig,.matrix td.empty{
    display:flex;align-items:baseline;justify-content:flex-start;
    gap:clamp(18px,6vw,30px);
    text-align:left;padding:6px 16px;border-bottom:none;
    font-size:1.02rem;
  }
  .matrix tbody tr td:last-child{padding-bottom:12px}
  .matrix td.fig::before,.matrix td.empty::before{
    content:attr(data-label);
    /* flex floor in em (tracks the 12px mono) sized to the longest label,
       "This sample (7-day)"; shrinkable as a last resort so a 320px box can
       never overflow. */
    flex:0 1 auto;min-width:12.5em;
    font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
    font-size:12px;font-weight:500;letter-spacing:0.06em;text-transform:uppercase;
    color:var(--ink-dim);text-align:left;
  }
}
/* Quality badge on the routed row (mirrors the landing-hero badge style). */
.badge{
  display:inline-block;margin-left:8px;padding:2px 8px;border-radius:999px;
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;
  color:var(--green);border:1px solid rgba(16,185,129,0.35);background:rgba(16,185,129,0.08);
  vertical-align:middle;
}
.route-to{color:var(--cyan);font-weight:600}
.route-keep{color:var(--ink-mute)}
.route-dash{color:var(--ink-faint)}

/* Routing-plan table column law (full-width, six columns).
   The plan owns its own full-width section below the fold (.plan-full) so it has
   room for six distinct columns — BUCKET, CALLS, % CALLS, MODEL, STATUS, COST —
   each on its own track. The model name and the decision badge never share a
   cell, so they can never collide (a reported "Keep · already o10000 gpt-4o"
   overlap).

   table-layout:AUTO (not fixed): the earlier fixed layout left MODEL as the only
   un-sized column, so it absorbed ALL the spare desktop width and opened a wide
   empty band between the (left-aligned, short) model name and the STATUS column.
   Auto layout sizes every column to its content and packs them left with even
   inter-column spacing, so the dead band cannot form; the SHARE column fills the
   reclaimed width with a useful per-bucket %-of-calls bar. The nowrap guards
   below keep CALLS / % CALLS / COST / STATUS-badge on one line, and overflow-wrap
   on the model name lets only an extreme id wrap inside its own cell — so nothing
   bleeds the container even though the layout is content-driven. */
.tbl-plan{table-layout:auto;width:100%;max-width:100%;border-collapse:collapse}
.tbl-plan .c-calls{white-space:nowrap}
/* SHARE column: the proportional %-of-calls bar + its % text, both on one line.
   Sized to comfortably hold the bar and a three-figure "100.0%". */
.tbl-plan .c-share{white-space:nowrap;width:8.5rem}
.tbl-plan th.c-share{text-align:left}
/* STATUS holds the badge phrase on one line — nowrap prevents wrapping. */
.tbl-plan .c-status{white-space:nowrap}
/* COST holds the widest figure _fmt_usd can emit (a five-figure monthly
   "$50,000.00"); nowrap keeps it on one line, right-aligned. */
.tbl-plan .c-cost{white-space:nowrap}
/* Bucket label: a single deliberate line ("Routed · easy", "Keep · already
   optimal") — never a squashed wrap. */
.tbl-plan td.bucket,.tbl-plan th.c-bucket{white-space:nowrap}
/* MODEL cell: takes the natural remaining width and may wrap within its own
   column for an extreme id (overflow-wrap), never overflowing the cell. */
.tbl-plan .c-model{min-width:0}
.tbl-plan .c-model .route-to,
.tbl-plan .c-model .route-keep{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.92rem;line-height:1.4;overflow-wrap:anywhere;
}
/* STATUS cell: the badge sits on its own, no longer crammed beside the model
   name. It never wraps mid-phrase (the STATUS track is sized to hold it). */
.tbl-plan .c-status .badge{margin-left:0;white-space:nowrap}
/* Per-bucket SHARE bar — a subtle horizontal fill showing the bucket's % of all
   analyzed calls. The track is a quiet --ink-dim hairline; the fill is cyan,
   width driven by the --share custom property (the same % shown as text beside
   it, so the figure is never colour-only — accessible). */
.tbl-plan .c-share{vertical-align:middle}
.share-bar{
  display:inline-block;vertical-align:middle;width:3.4rem;height:7px;
  border-radius:999px;background:rgba(107,107,114,0.28);
  overflow:hidden;position:relative;
}
.share-bar .share-fill{
  position:absolute;top:0;left:0;bottom:0;width:var(--share,0%);
  background:var(--cyan,#00D1FF);border-radius:999px;
}
.share-pct{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.82rem;color:var(--ink-mute,#A1A1AA);margin-left:8px;
  font-variant-numeric:tabular-nums;
}
/* Saving delta — the hero number, grounded. */
.delta{
  display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;
  margin-top:14px;padding:13px 16px;border:1px solid var(--hair-soft);border-radius:9px;
  background:linear-gradient(0deg,rgba(52,211,153,0.05),rgba(52,211,153,0.05)),var(--panel);
}
.delta-label{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;letter-spacing:0.10em;text-transform:uppercase;color:var(--ink-dim);
}
.delta-value{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:1.12rem;font-weight:700;color:var(--green);font-variant-numeric:tabular-nums;
}
.delta-pct{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.9rem;color:var(--green-dim);font-variant-numeric:tabular-nums;
}
/* Metadata line — calls analyzed + pricing snapshot, demoted to quiet meta. */
.meta-line{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;color:var(--ink-dim);line-height:1.7;margin-top:16px;letter-spacing:0.01em;
  /* Item F — controlled formatting: a meta line wraps and never bleeds the
     container, even if a single fragment (e.g. a synced date + staleness note)
     runs long. The calls-priced count and synced dates already ride separate
     lines (see _html_v2_meta_lines). */
  overflow-wrap:anywhere;
}
.meta-line .sep{color:var(--ink-faint);padding:0 8px}
.meta-line .warn{color:var(--cyan-dim)}
/* Quality-tier disclosure: dim label + source tag; model names keep .route-to cyan. */
.meta-line .qt-label{color:var(--ink-mute);font-weight:500}
.meta-line .qt-src{color:var(--ink-dim)}
/* Amber caution inline note (stale pricing/quality table, --window override).
   Amber is the report's single caution colour; the <code> flag/command keeps
   the surface's mono code styling tinted amber and never wraps mid-token. */
.caution{color:#F59E0B}
.caution code{font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;font-size:0.92em;color:#F59E0B;white-space:nowrap}

/* Single-row matrix for the no-candidate state (Current only). */
.matrix.single td.fig{color:var(--ink)}

/* Fine-print disclosure — official small print, financial-disclosure style.
   Muted but >=12px and AA-legible; carries the unverified-quality caveat. */
.fineprint{
  /* No own rule: .fineprint is the first child of .foot, which already
     draws the single hairline above it. A second border-top here produced
     a stacked double rule. */
  display:flex;gap:9px;align-items:flex-start;
}
.fineprint .mark{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:13px;color:var(--ink-dim);flex-shrink:0;line-height:1.6;
}
/* The fine print always wraps inside the footer/container. An earlier desktop
   treatment forced the caveat onto a single non-breaking line above an 880px
   media query; that width was sized for the shorter single-candidate caveat,
   so the longer split caveat ("…run --measure to sample real outputs before
   you switch.") overran the container right edge at desktop width (a reported
   overflow). Wrapping is the correct law for both caveats: it can never force
   horizontal overflow at any viewport, and min-width:0 lets the flex body
   shrink to the available track rather than dictating an over-wide min-size. */
.fineprint .body{
  font-size:13px;color:var(--ink-dim);line-height:1.65;
  min-width:0;overflow-wrap:break-word;
}
.fineprint .body code{font-size:0.92em;white-space:nowrap}
.swap-note code{white-space:nowrap}

/* Cost-by-model table */
.tbl{width:100%;border-collapse:collapse}
.tbl th{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;letter-spacing:0.10em;text-transform:uppercase;
  color:var(--ink-dim);font-weight:500;text-align:left;
  padding:0 14px 12px;border-bottom:1px solid var(--hair);
}
.tbl th.r,.tbl td.r{text-align:right}
.tbl td{
  position:relative;padding:13px 14px;border-bottom:1px solid var(--hair-soft);
  font-size:0.95rem;color:var(--ink-mute);
}
.tbl tr:last-child td{border-bottom:none}
.tbl td.num{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums;color:var(--ink);
}
.tbl .model{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.92rem;color:var(--cyan);position:relative;z-index:1;
}
/* pure-CSS proportion bar behind the model cell */
.bar-cell{position:relative;overflow:hidden}
.bar-cell .bar{
  position:absolute;top:0;left:0;bottom:0;
  background:var(--bar);border-right:1px solid var(--bar-edge);
  z-index:0;
}
.tbl .total td{
  border-top:1px solid var(--hair);border-bottom:none;
  padding-top:14px;color:var(--ink);font-weight:600;
}
.tbl .total .lbl{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;letter-spacing:0.10em;text-transform:uppercase;color:var(--ink-dim);
}
/* Narrow-viewport gutter relief for the cost-by-model table. Its four columns
   plus 14px cell padding have a min-content width (~312px) that just exceeds a
   320px phone's content box (320 - 36px body padding = 284px), forcing a ~10px
   horizontal scroll. We trim the cell gutters to 9px below 640px — recovering
   ~40px of min-content so the table reflows within 320px. Placed AFTER the base
   .tbl rules so the longhand padding-left/right win on equal specificity; the
   desktop table (>640px) keeps its roomier 14px gutters untouched. */
@media (max-width:640px){
  .tbl th,.tbl td{padding-left:9px;padding-right:9px}
  /* Routing-plan table on mobile: the section is full-width, so the six columns
     (BUCKET / CALLS / % CALLS / MODEL / STATUS / COST) must reflow within a
     ~284px phone content box without a horizontal scroll. The desktop layout is
     already table-layout:auto; here we let BUCKET / STATUS wrap, CALLS / COST
     shrink to their figures, and the SHARE column collapse to its compact bar +
     % so it never dictates an over-wide track. The COST figure
     ("$496.72") is allowed to wrap so its own min-content is one segment. */
  .tbl-plan td.bucket,.tbl-plan th.c-bucket{width:auto;white-space:normal}
  .tbl-plan .c-status{width:auto}
  .tbl-plan .c-calls{width:1%}
  .tbl-plan .c-cost{width:1%;white-space:normal}
  .tbl-plan .c-share{width:1%}
  .tbl-plan .c-status .badge{white-space:normal}
  /* The share bar narrows so its column never crowds the cyan MODEL column on a
     phone; the % stays full-size text beside it (accessible, never colour-only). */
  .tbl-plan .share-bar{width:2.2rem}
  /* Trim the routing-plan gutters one more notch (6px, vs the 9px the cost-by-
     model table keeps) so the six columns + the COST figure fit the
     ~284px content box of a 320px phone with no horizontal scroll. The model
     name eases to 0.85rem (~13.6px — still well above the 12px legibility floor)
     to give the cyan MODEL column the room it needs without starving COST.
     All header labels and the CALLS figure keep their size. */
  .tbl-plan th,.tbl-plan td{padding-left:6px;padding-right:6px}
  .tbl-plan .c-model .route-to,.tbl-plan .c-model .route-keep{font-size:0.85rem}
}

/* Recommended swap */
.swap{
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;
}
.pill{
  display:inline-flex;align-items:center;gap:10px;
  border:1px solid var(--hair);border-radius:999px;
  padding:9px 16px;background:var(--panel);
}
.pill .from{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.92rem;color:var(--ink-mute);
}
.pill .to{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.92rem;color:var(--cyan);font-weight:600;
}
.pill .arrow{color:var(--ink-faint);font-family:ui-monospace,monospace}
.swap-note{font-size:0.9rem;color:var(--ink-mute);line-height:1.6;max-width:60ch}

/* Quality / methodology prose */
.prose{font-size:0.95rem;color:var(--ink-mute);line-height:1.7;max-width:64ch}
.prose .src{display:block;font-size:0.85rem;color:var(--ink-dim);margin-top:10px}
.note{font-size:0.9rem;color:var(--ink-dim);line-height:1.65;margin-top:14px;max-width:64ch}

/* Footer (always full width at the bottom) */
.foot{
  border-top:1px solid var(--hair);margin-top:38px;padding-top:24px;
}
.foot .privacy{
  font-size:1rem;color:var(--ink);font-weight:500;line-height:1.55;margin-bottom:14px;
}
.foot .privacy .em{color:var(--green)}
.foot .meta{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:12px;color:var(--ink-dim);line-height:1.9;letter-spacing:0.02em;
}
.foot .cta{font-size:0.95rem;color:var(--ink-mute);margin-top:18px;line-height:1.6}
.foot .cta a{color:var(--cyan);text-decoration:none;font-weight:600}
.foot .cta a:hover{text-decoration:underline}
code{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.88em;background:var(--panel);
  padding:2px 6px;border-radius:5px;border:1px solid var(--hair);color:var(--ink-mute);
}

/* Candidates-considered table — PARITY with the cost-by-model (.tbl) and the
   routing-plan (.tbl-plan) tables on this surface.  The table carries the
   shared .tbl class so it inherits the same cell padding (13px 14px), header
   rule and hairline row separators; here we add the candidates-specific bits
   the base .tbl rules do not cover on the v2 surface:
     * v2 right-aligns numeric cells via a .r class, but the candidates table
       marks them .num — so re-assert text-align:right + nowrap here;
     * the model name uses .model-name (v2's own model cell class is .model);
     * the status badge gets a per-status colour, mirroring the routing-plan
       badge vocabulary (recommended = cyan, more-expensive = amber, the rest
       muted) — and drops the routing-plan badge's left margin since it is the
       only thing in its STATUS cell. */
.candidates-considered{margin:.25rem 0 0;overflow-x:auto}
.tbl-candidates{table-layout:auto}
.tbl-candidates th.num,.tbl-candidates td.num{
  text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--ink);
}
.tbl-candidates td.c-model .model-name{
  font-family:ui-monospace,'JetBrains Mono','Cascadia Code','SF Mono',Menlo,Consolas,monospace;
  font-size:0.92rem;color:var(--cyan);
}
.tbl-candidates td.c-status,.tbl-candidates th.c-status{white-space:nowrap}
.tbl-candidates td.c-tier,.tbl-candidates th.c-tier{white-space:nowrap;color:var(--ink-mute)}
.tbl-candidates .badge{
  margin-left:0;color:var(--ink-mute);
  border:1px solid var(--hair);background:var(--panel);
}
.tbl-candidates .badge-recommended{
  color:var(--cyan);border-color:rgba(0,209,255,0.35);background:rgba(0,209,255,0.08);
}
.tbl-candidates .badge-more-expensive{
  color:#F59E0B;border-color:rgba(245,158,11,0.35);background:rgba(245,158,11,0.10);
}
.tbl-candidates .badge-considered,
.tbl-candidates .badge-unpriced{color:var(--ink-dim)}
/* Caption beneath the block — v2's standard dim caption treatment. */
.candidates-caption{font-size:13px;color:var(--ink-dim);line-height:1.65;margin-top:14px}
/* Default-pool bullet legend — see v1's identical rule for rationale. */
.candidates-legend{list-style:none;margin:14px 0 0;padding:0}
.candidates-legend li{margin:0 0 4px;padding-left:16px;position:relative}
.candidates-legend li:last-child{margin-bottom:0}
.candidates-legend li::before{content:"\\00b7";position:absolute;left:0}
"""

_HTML_TEMPLATE_V2 = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>frugon — cost analysis</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
<div class="masthead">
<div class="wordmark">
<svg class="brand-mark" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<circle cx="12" cy="5" r="2.5" fill="currentColor"/>
<circle cx="6" cy="15.5" r="2.5" fill="currentColor"/>
<circle cx="18" cy="15.5" r="2.5" fill="currentColor"/>
</svg>
FRUGON
</div>
<div class="masthead-tag">cost analysis</div>
</div>
{body}
</div>
</body>
</html>"""


def _v2_cost_by_model_section(result: AnalysisResult) -> str:
    """Build the v2 cost-by-model table section (inline proportion bars + total)."""
    esc = _html_escape.escape
    if not result.cost_by_model:
        return ""
    rows: list[str] = []
    total_priced_calls = 0
    for model, cost in sorted(result.cost_by_model.items(), key=lambda kv: kv[1], reverse=True):
        pct = (cost / result.total_cost * 100) if result.total_cost else Decimal("0")
        calls = result.calls_by_model.get(model, 0)
        total_priced_calls += calls
        rows.append(
            "<tr>"
            f'<td class="bar-cell"><span class="bar" style="width:{float(pct):.1f}%"></span>'
            f'<span class="model">{esc(model)}</span></td>'
            f'<td class="num r tnum">{calls:,}</td>'
            f'<td class="num r tnum">{_fmt_usd(cost)}</td>'
            f'<td class="num r tnum">{float(pct):.1f}%</td>'
            "</tr>"
        )
    rows.append(
        '<tr class="total"><td class="lbl">Total</td>'
        f'<td class="num r tnum">{total_priced_calls:,}</td>'
        f'<td class="num r tnum">{_fmt_usd(result.total_cost)}</td>'
        '<td class="num r tnum">100.0%</td></tr>'
    )
    return (
        "<section>"
        '<div class="eyebrow">Cost by model</div>'
        '<table class="tbl"><thead><tr>'
        '<th>Model</th><th class="r">Calls</th><th class="r">Cost</th><th class="r">% of total</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</section>"
    )


def _render_html_v2_split_body(
    result: AnalysisResult,
    split: SplitRouting,
    *,
    measure_result: MeasureResult | None = None,
    limits: PreviewLimits | None = None,
) -> str:
    """Build the v2 HTML body for the per-call split-routing headline.

    Mirrors the landing-page scoreboard: a green saving hero, a routed/kept/blended
    routing plan, the cost-by-model breakdown, and the privacy footer.  Self-
    contained — no network, no JS.  GREEN (#10B981) is reserved for the saving.

    When *measure_result* is provided, a "Quality measurement" section is
    inserted before the footer; when None the output is byte-identical.
    """
    esc = _html_escape.escape
    saving_pct = split.saving_pct
    assert saving_pct is not None  # guaranteed by _has_split
    # Every headline/routing figure reconciles to the FULL analyzed dataset
    # (identical to the terminal panel) via the shared helper — NOT the
    # baseline-only split.saving_pct.  Current is the TOTAL spend, the saving is
    # over that total, and the routing plan carries the already-on-a-cheaper
    # bucket so the buckets sum to every analyzed call.
    fig = _split_report_figures(result, split)
    pct_str = f"{float(fig.total_pct):.1f}%"
    projection_label = esc(_projection_label(result))

    # Quality phrase for the hero lede and routed-row badge — three cases:
    # 1. Candidate UNRATED — omit; unrated caveat handles the disclosure.
    # 2. tier_drop <= 0 (same or better) — use the stronger positive phrase.
    # 3. tier_drop >= 1 (genuine step-down) — use "within tolerance" as before.
    _routed_unrated = _is_unrated(split.candidate_model)
    if _routed_unrated:
        _hero_tol = ""
        _tol_badge_v2 = ""
    elif _is_equal_or_better_quality(result):
        _hero_tol = "same or better quality "
        _tol_badge_v2 = '<span class="badge">same or better quality</span>'
    else:
        _hero_tol = "within tolerance "
        _tol_badge_v2 = '<span class="badge">within tolerance</span>'

    # The hero's "before → after" dagger and "see note below" pointer must only
    # appear when the fineprint footnote they reference actually renders.  The
    # footnote is gated on _quality_caveat_text(...) is not None — which is None
    # on the Tier-1 (--judge) path (the report carries a scored verdict instead),
    # so on that path the dagger would dangle with no referent.  Gate the marker
    # and the "see note below" pointer on the SAME predicate.
    _split_caveat_v2_hero = (
        SPLIT_CAVEAT_EQUAL_OR_BETTER
        if _is_equal_or_better_quality(result)
        else SPLIT_CAVEAT
    )
    _has_fineprint = _quality_caveat_text(_split_caveat_v2_hero, measure_result) is not None
    _hero_dagger = '<span class="dagger">&dagger;</span>' if _has_fineprint else ""
    _hero_estimate_note = (
        "List-price estimate &mdash; see note below."
        if _has_fineprint
        else "List-price estimate."
    )

    # Money figures use the shared _fmt_usd (2 dp for amounts ≥ $0.01, finer only
    # for sub-cent values) so every figure reads IDENTICALLY across the terminal
    # panel, the Markdown report, and both HTML styles — e.g. "$496.72".  The
    # displayed saving and percent reconcile from these rounded figures
    # (Current − New = SAVING).  The COST column is sized for the widest such
    # figure (see the .c-cost CSS note), so a before/after pair can never overrun
    # its track.
    unit = "/mo" if fig.projected else ""
    current_val = _fmt_usd(fig.current)
    blended_val = _fmt_usd(fig.blended)
    delta_amt = fig.saved

    left_col: list[str] = []
    right_col: list[str] = []

    # Upper bound (decision info) rides under the hero caveat, BEFORE the
    # right-rail freshness meta (Item 7).  Same gate/figure as the terminal.
    _ub = _upper_bound_pct(result)
    _upper = ""
    if _ub is not None:
        _upper = (
            '<p class="hero-caveat">Upper bound: moving every call to '
            f'<span class="route-to">{esc(result.candidate_model or "")}</span> '
            f"saves ~{float(_ub):.1f}% &mdash; the aggressive end; the "
            f"~{float(fig.total_pct):.1f}% split shown is the conservative, "
            "quality-respecting recommendation.</p>"
        )

    # --- Hero: the blended saving ---
    left_col.append(
        "<section>"
        '<div class="eyebrow">Bottom line</div>'
        f'<div class="hero-figure"><span class="pct">{esc(pct_str)}</span> saved</div>'
        f'<p class="hero-lede">Route the <strong>{split.routed_count:,}</strong> simple '
        f"{esc(split.baseline_model)} calls to <strong>{esc(split.candidate_model)}</strong> "
        f"{_hero_tol}and keep the <strong>{split.kept_count:,}</strong> hard ones on "
        f"<strong>{esc(split.baseline_model)}</strong>.</p>"
        '<div class="beforeafter">'
        '<div class="ba-block"><span class="ba-label">Current</span>'
        f'<span class="ba-value tnum">{current_val}{unit}</span></div>'
        '<span class="ba-arrow">&rarr;</span>'
        '<div class="ba-block"><span class="ba-label">Blended</span>'
        f'<span class="ba-value after tnum">{blended_val}{unit}'
        f'{_hero_dagger}</span></div>'
        "</div>"
        '<p class="hero-caveat">'
        f"<span class='mono' style='color:var(--ink-dim)'>{projection_label}.</span> "
        f"{_hero_estimate_note}</p>"
        + _upper
        + "</section>"
    )

    # --- Cost by model (left rail, beneath the hero) ---
    # --- Cost by model (right rail, beside the hero) ---
    right_col.append(_v2_cost_by_model_section(result))

    parts: list[str] = [
        '<div class="fold">'
        f'<div class="col">{"".join(left_col)}</div>'
        f'<div class="col">{"".join(right_col)}</div>'
        "</div>"
    ]

    # --- Routing plan (full width below the fold) ------------------------------
    # Five distinct columns - Bucket | Calls | Model | Status | Cost - so the
    # bucket label, the routed model name, and the decision badge each own a
    # column and can never collide.  Costs use the shared _fmt_usd (2 dp for
    # amounts ≥ $0.01, the Cost-by-model formatter) and the section spans the full
    # content width so the figures ("$496.72") sit INSIDE the table rather than
    # overrunning a half-rail track.
    unit_pct = f"{float(fig.total_pct):.1f}%"

    # Per-bucket share of total analyzed calls.  The SHARE column reclaims the
    # space freed by rebalancing the column widths (the MODEL/STATUS dead band):
    # a thin proportional fill bar + the % as TEXT (accessible — never colour
    # only).  Shares are largest-remainder rounded so the non-blended buckets sum
    # to EXACTLY 100.0% (see _call_share_pcts), reconciling with the 100%-of-calls
    # Blended total row.  Bucket order matches the row order below: routed, kept,
    # then already-optimal when present.
    _share_counts = [split.routed_count, split.kept_count]
    if fig.already_cheap > 0:
        _share_counts.append(fig.already_cheap)
    _shares = _call_share_pcts(_share_counts)

    def _share_cell(pct: float) -> str:
        """A SHARE cell: a proportional fill bar with the % shown as text."""
        return (
            '<td class="c-share">'
            '<span class="share-bar" '
            f'style="--share:{pct:.1f}%"><span class="share-fill"></span></span>'
            f'<span class="share-pct tnum">{pct:.1f}%</span>'
            "</td>"
        )

    plan = [
        '<section class="below plan-full">',
        '<div class="eyebrow">Routing plan</div>',
        '<table class="tbl tbl-plan"><thead><tr>'
        '<th class="c-bucket">Bucket</th><th class="r c-calls">Calls</th>'
        '<th class="c-share">% calls</th>'
        '<th class="c-model">Model</th><th class="c-status">Status</th>'
        '<th class="r c-cost">Cost</th>'
        "</tr></thead><tbody>",
        "<tr>"
        '<td class="bucket">Routed &middot; easy</td>'
        f'<td class="num r tnum c-calls">{split.routed_count:,}</td>'
        f"{_share_cell(_shares[0])}"
        f'<td class="c-model"><span class="route-to">{esc(split.candidate_model)}</span></td>'
        f'<td class="c-status">{_tol_badge_v2}</td>'
        f'<td class="num r tnum c-cost">{_fmt_usd(split.routed_cost)}</td>'
        "</tr>",
        "<tr>"
        '<td class="bucket">Kept &middot; hard</td>'
        f'<td class="num r tnum c-calls">{split.kept_count:,}</td>'
        f"{_share_cell(_shares[1])}"
        f'<td class="c-model"><span class="route-keep">{esc(split.baseline_model)}</span></td>'
        '<td class="c-status"></td>'
        f'<td class="num r tnum c-cost">{_fmt_usd(split.kept_cost)}</td>'
        "</tr>",
    ]
    if fig.already_cheap > 0:
        _already_models = esc(", ".join(fig.other_models)) if fig.other_models else "a cheaper model"
        plan.append(
            "<tr>"
            '<td class="bucket">Keep &middot; already optimal</td>'
            f'<td class="num r tnum c-calls">{fig.already_cheap:,}</td>'
            f"{_share_cell(_shares[2])}"
            f'<td class="c-model"><span class="route-keep">{_already_models}</span></td>'
            '<td class="c-status"><span class="badge">no action</span></td>'
            f'<td class="num r tnum c-cost">{_fmt_usd(fig.already_cheap_cost)}</td>'
            "</tr>"
        )
    plan += [
        '<tr class="total">'
        '<td class="lbl bucket">Blended</td>'
        f'<td class="num r tnum c-calls">{result.priced_calls:,}</td>'
        '<td class="num r tnum c-share">100.0%</td>'
        '<td class="c-model"><span class="route-dash">&mdash;</span></td>'
        '<td class="c-status"></td>'
        f'<td class="num r tnum c-cost">{_fmt_usd(fig.blended)}</td>'
        "</tr>",
        "</tbody></table>",
        '<div class="delta">'
        '<span class="delta-label">You save</span>'
        f'<span class="delta-value tnum">{_fmt_usd(delta_amt)}{unit}</span>'
        f'<span class="delta-pct tnum">&minus;{esc(unit_pct)}</span>'
        "</div>",
    ]
    # Quality-tier disclosure — the published LMArena class the routing moves
    # between, on its own dim meta-line under the routing plan (alongside the
    # freshness lines).  Cyan model names (.route-to), plain labels.
    plan.append(
        f'<p class="meta-line">{_html_quality_tier_line(split, model_class="route-to")}</p>'
    )
    plan += _html_v2_meta_lines(result)
    _log_span = _html_log_span_html(result)
    if _log_span:
        plan.append(f'<p class="meta-line">{_log_span}</p>')
    _window_caution = _html_window_caution(result)
    if _window_caution:
        plan.append(f'<p class="meta-line">{_window_caution}</p>')
    plan.append("</section>")
    parts.append("\n".join(plan))

    # Candidates considered (multi-candidate transparency) — no-op when the
    # user passed <=1 candidate.  Sits as its own full-width below-the-fold
    # section above the methodology.
    _cand_html = _candidates_considered_html(
        result, esc, has_judge_section=_is_tier1(measure_result)
    )
    if _cand_html:
        parts.append(
            '<section class="below">'
            f'<div class="eyebrow">{esc(_candidates_header_title(result))}</div>'
            + _cand_html
            + "</section>"
        )

    dq = _data_quality_html(result)
    if dq:
        parts.append(f'<section class="below">{dq}</section>')

    # --- Methodology (full width, below the fold) ---
    attribution = _get_attribution()
    src_html = f'<span class="src">{esc(attribution)}</span>' if attribution else ""
    parts.append(
        '<section class="below">'
        '<div class="eyebrow">Methodology</div>'
        f'<p class="prose">{esc(_methodology_note_text(measure_result))}{src_html}</p>'
        "</section>"
    )

    # Quality-measurement section (full width, below the fold) before the
    # footer — empty / byte-identical when no measure run ran.
    quality_section = _quality_section_html(measure_result, style="v2", limits=limits, result=result)
    if quality_section:
        parts.append(quality_section)

    # --- Footer: fine-print split caveat + privacy + funnel ---
    foot = ['<div class="foot">']
    _split_caveat_v2 = (
        SPLIT_CAVEAT_EQUAL_OR_BETTER
        if _is_equal_or_better_quality(result)
        else SPLIT_CAVEAT
    )
    _caveat = _quality_caveat_text(_split_caveat_v2, measure_result)
    if _caveat is not None:
        foot.append(
            '<div class="fineprint">'
            '<span class="mark">&dagger;</span>'
            f'<span class="body">{esc(_caveat).replace("--measure", "<code>--measure</code>").replace("--judge", "<code>--judge</code>")}</span>'
            "</div>"
        )
    # Unrated-message family (findings #1 + #4) — same wording as every surface.
    _unrated_html = _html_unrated_family_html(result, judged_models_from_measure(measure_result))
    if _unrated_html:
        foot.append(_unrated_html)
    foot += [
        f'<p class="privacy">{_privacy_html(measure_result)}</p>',
        '<p class="meta">methodology &middot; tokencost &middot; LiteLLM registry '
        f"&middot; LMArena quality tiers, RouteLLM-style routing &middot; {esc(_methodology_tail_text(measure_result))}</p>",
        '<p class="cta">'
        + esc(FUNNEL_LINE).replace(
            esc("→ https://frugon.rodiun.io"),
            '&rarr; <a href="https://frugon.rodiun.io">frugon.rodiun.io</a>',
        )
        + "</p>",
        "</div>",
    ]
    parts.append("\n".join(foot))
    return "\n".join(parts)


def _render_html_v2_wholesale_swap_plan(
    result: AnalysisResult,
    esc: Callable[[str], str],
) -> str:
    """Full-width v2 swap-plan section for a wholesale run.

    Mirrors the split routing-plan table in HTML v2: below-the-fold section
    with eyebrow 'Swap plan', share-bar cells, and route-to/route-keep spans.
    Returns empty string when no candidate is set.
    """
    if not result.candidate_model or not result.cost_by_model:
        return ""
    candidate = result.candidate_model
    non_cand = sorted(
        [(m, c) for m, c in result.cost_by_model.items() if m != candidate],
        key=lambda mc: mc[1],
        reverse=True,
    )
    cand_cost = result.cost_by_model.get(candidate, Decimal("0"))
    ordered: list[tuple[str, Decimal]] = non_cand + [(candidate, cand_cost)]

    counts = [result.calls_by_model.get(m, 0) for m, _ in ordered]
    shares = _call_share_pcts(counts)

    def _share_cell(pct: float) -> str:
        return (
            '<td class="c-share">'
            '<span class="share-bar" '
            f'style="--share:{pct:.1f}%"><span class="share-fill"></span></span>'
            f'<span class="share-pct tnum">{pct:.1f}%</span>'
            "</td>"
        )

    rows: list[str] = []
    for (model, cost), share in zip(ordered, shares, strict=True):
        calls = result.calls_by_model.get(model, 0)
        if model == candidate:
            status = '<span class="badge">already on target</span>'
        else:
            status = f'→ swap to <span class="route-to">{esc(candidate)}</span>'
        rows.append(
            "<tr>"
            f'<td class="c-bucket"><span class="route-keep">{esc(model)}</span></td>'
            f'<td class="num r tnum c-calls">{calls:,}</td>'
            f"{_share_cell(share)}"
            f'<td class="num r tnum c-cost">{_fmt_usd(cost)}</td>'
            f'<td class="c-status">{status}</td>'
            "</tr>"
        )
    total_calls = sum(result.calls_by_model.get(m, 0) for m, _ in ordered)
    total_cost = sum((c for _, c in ordered), Decimal("0"))
    rows.append(
        '<tr class="total">'
        '<td class="lbl c-bucket">Total</td>'
        f'<td class="num r tnum c-calls">{total_calls:,}</td>'
        '<td class="num r tnum c-share">100.0%</td>'
        f'<td class="num r tnum c-cost">{_fmt_usd(total_cost)}</td>'
        '<td class="c-status"></td>'
        "</tr>"
    )
    return (
        '<section class="below plan-full">'
        '<div class="eyebrow">Swap plan</div>'
        '<table class="tbl tbl-plan"><thead><tr>'
        '<th class="c-bucket">Model</th><th class="r c-calls">Calls</th>'
        '<th class="c-share">% calls</th>'
        '<th class="r c-cost">Current cost</th><th class="c-status">Action</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</section>"
    )


def render_html_v2(
    result: AnalysisResult,
    output_path: Path,
    *,
    measure_result: MeasureResult | None = None,
    limits: PreviewLimits | None = None,
) -> None:
    """Write a self-contained v2 ("engineering instrument") HTML report.

    Same data contract as :func:`render_html`; refined editorial treatment.
    The file is fully local — no external CDN, no font URLs, no analytics,
    no JS. All CSS is inlined in a ``<style>`` block. GREEN is reserved for
    the positive saving figure (matching the landing page). The quality
    caveat, projection labels, CC-BY attribution, funnel line and privacy
    footer all carry over from v1 unchanged (honesty invariant).
    """
    esc = _html_escape.escape
    parts: list[str] = []

    # --- No priced calls guard ---
    if result.priced_calls == 0:
        parts.append(
            "<section>"
            '<div class="eyebrow">No priced calls</div>'
            f"<p class='no-rec'>Analyzed <strong>{result.total_calls:,}</strong> records — "
            "none could be priced (unknown models or missing usage data).</p>"
            "<p class='note'>"
            "Run <code>frugon pricing update</code> to refresh the pricing table, "
            "or check that your log records include a <code>model</code> field.</p>"
            "</section>"
            "<div class='foot'>"
            "<p class='privacy'>"
            "<span class='em'>No LLM calls. No network. No data leaves your machine.</span>"
            "</p>"
            f"<p class='meta'>{esc(_METHODOLOGY_NOTE)}</p>"
            "</div>"
        )
        html = _HTML_TEMPLATE_V2.format(css=_HTML_CSS_V2, body="\n".join(parts))
        _atomic_write_text(output_path, html)
        return

    # --- Per-call split routing is the headline when available ---
    if _has_split(result):
        assert result.split is not None  # narrowed by _has_split
        body = _render_html_v2_split_body(
            result, result.split, measure_result=measure_result, limits=limits
        )
        html = _HTML_TEMPLATE_V2.format(
            css=_HTML_CSS_V2 + _quality_css(measure_result), body=body
        )
        _atomic_write_text(output_path, html)
        return

    # RECONCILIATION: gate on the raw percent (so a positive but sub-display-
    # precision saving does not vanish), then derive the DISPLAYED percent and
    # delta from quantized amounts — the same pair _fmt_usd will print.
    # Prefer the monthly axis (headline basis); fall back to the sample axis.
    _raw_saving_pct = compute_saving_pct(result.total_cost, result.projected_cost)
    if result.monthly_cost is not None and result.monthly_projected is not None:
        _, _, saving_pct = _reconciled_delta_pct(
            result.monthly_cost, result.monthly_projected
        )
        _delta_unit_v2 = "/mo"
    else:
        _, _, saving_pct = _reconciled_delta_pct(
            result.total_cost, result.projected_cost
        )
        _delta_unit_v2 = ""
    projection_label = esc(_projection_label(result))
    has_saving = (
        result.candidate_model is not None
        and _raw_saving_pct is not None
        and _raw_saving_pct > Decimal("0")
    )

    # The after-swap dagger markers (hero + the What-we-found matrix) point at the
    # footer fineprint footnote, which is gated on _quality_caveat_text(...) is not
    # None.  That caveat is None on the Tier-1 (--judge) path — the report carries
    # a scored verdict instead — so on that path the daggers would dangle with no
    # referent.  Gate every after-swap marker on the SAME predicate, and downgrade
    # the hero's "see note below" pointer to a plain "List-price estimate." when
    # the footnote is absent.
    _wholesale_has_fineprint = (
        _quality_caveat_text(QUALITY_CAVEAT, measure_result) is not None
    )
    _ws_dagger = (
        '<span class="dagger">&dagger;</span>' if _wholesale_has_fineprint else ""
    )
    _ws_estimate_note = (
        "List-price estimate &mdash; see note below."
        if _wholesale_has_fineprint
        else "List-price estimate."
    )

    # The above-the-fold sections are assembled into two balanced rails so
    # that the saving hero, the Current -> After comparison, the cost-by-model
    # breakdown and the recommended swap all land in the initial viewport on a
    # typical laptop without leaving a dead quadrant.
    #
    #   Left rail : hero (the saving) + cost-by-model breakdown
    #   Right rail: What-we-found matrix + "You save" delta + recommended swap
    #
    # The two tallest blocks (hero and the matrix) are split across the rails,
    # and each is paired with the lighter block that reads naturally beneath
    # it, so the columns land near the same height. ``left_col`` and
    # ``right_col`` are joined into a ``.fold`` grid below.
    left_col: list[str] = []
    right_col: list[str] = []

    # --- Hero: the saving ---
    hero = ['<section>', '<div class="eyebrow">Bottom line</div>']
    if has_saving:
        pct_str = f"{float(saving_pct):.1f}%"
        # Before -> after frame: prefer monthly figures when a projection exists.
        if result.monthly_cost is not None and result.monthly_projected is not None:
            before_val, after_val = (
                _fmt_usd(result.monthly_cost),
                _fmt_usd(result.monthly_projected),
            )
            unit = "/mo"
        else:
            before_val, after_val = (
                _fmt_usd(result.total_cost),
                _fmt_usd(result.projected_cost),
            )
            unit = ""
        hero.append(
            f'<div class="hero-figure"><span class="pct">{esc(pct_str)}</span> saved</div>'
        )
        hero.append(
            '<p class="hero-lede">Route these calls to the recommended swap and '
            "cut their cost by roughly "
            f"<strong>{esc(pct_str)}</strong>.</p>"
        )
        hero.append(
            '<div class="beforeafter">'
            '<div class="ba-block"><span class="ba-label">Current</span>'
            f'<span class="ba-value tnum">{before_val}{unit}</span></div>'
            '<span class="ba-arrow">&rarr;</span>'
            '<div class="ba-block"><span class="ba-label">After swap</span>'
            f'<span class="ba-value after tnum">{after_val}{unit}'
            f'{_ws_dagger}</span></div>'
            "</div>"
        )
        hero.append(
            '<p class="hero-caveat">'
            f"<span class='mono' style='color:var(--ink-dim)'>{projection_label}.</span> "
            f"{_ws_estimate_note}</p>"
        )
    else:
        hero.append(
            '<p class="no-rec"><span class="accent">No cheaper swap clears the quality bar.</span> '
            "Your current routing is already efficient "
            "for the models considered.</p>"
        )
    hero.append("</section>")
    left_col.append("\n".join(hero))

    # --- What we found: the Current -> After-swap comparison matrix ---
    # The data is fundamentally a 2x2 (Current / After swap) x (This sample /
    # Monthly projection). A real matrix makes that structure read instantly,
    # where six equal-weight tiles hid it. The after-swap row carries the
    # positive green treatment; the saving delta is surfaced directly beneath
    # so the hero percentage is grounded in concrete dollars.
    has_monthly = result.monthly_cost is not None
    # Column headers adapt to whether a monthly projection exists.
    sample_label = (
        f"This sample ({result.window_days}-day)"
        if result.window_days is not None
        else "This sample"
    )
    monthly_head = "<th>Monthly projection</th>" if has_monthly else ""

    found = ['<section>', '<div class="eyebrow">What we found</div>']
    found.append('<div class="matrix">')
    found.append(
        "<table><thead><tr>"
        '<th class="row-head"></th>'
        f"<th>{esc(sample_label)}</th>"
        f"{monthly_head}"
        "</tr></thead><tbody>"
    )
    # Current row. Each value cell carries a data-label (its column header) so
    # that under the <=640px reflow — where the table stacks and the thead is
    # retired — the figure still reads with its meaning ("This sample" /
    # "Monthly projection") inline. esc() guards the dynamic window label.
    current_monthly_cell = (
        f'<td class="fig tnum" data-label="Monthly projection">'
        f"{_fmt_usd(result.monthly_cost)}</td>"
        if result.monthly_cost is not None
        else ""
    )
    found.append(
        '<tr class="current">'
        '<th class="row-head">Current</th>'
        f'<td class="fig tnum" data-label="{esc(sample_label)}">'
        f"{_fmt_usd(result.total_cost)}</td>"
        f"{current_monthly_cell}"
        "</tr>"
    )
    # After-swap row — only when a saving exists. Edge: no candidate => row absent.
    if has_saving:
        after_monthly_cell = (
            f'<td class="fig tnum" data-label="Monthly projection">'
            f"{_fmt_usd(result.monthly_projected)}"
            f"{_ws_dagger}</td>"
            if result.monthly_projected is not None
            else '<td class="empty" data-label="Monthly projection">&mdash;</td>'
            if has_monthly
            else ""
        )
        found.append(
            '<tr class="after">'
            '<th class="row-head">After recommended swap</th>'
            f'<td class="fig tnum" data-label="{esc(sample_label)}">'
            f"{_fmt_usd(result.projected_cost)}"
            f"{_ws_dagger}</td>"
            f"{after_monthly_cell}"
            "</tr>"
        )
    found.append("</tbody></table>")
    found.append("</div>")  # .matrix

    # Saving delta — concrete dollars saved + percent. Prefer the monthly axis
    # (the figure that grounds the hero); fall back to the sample axis.
    # RECONCILIATION: _reconciled_delta_pct already quantized both amounts and
    # derived the percent from the quantized pair; reuse those to compute
    # delta_amt so SAVING == printed Current − printed After exactly.
    if has_saving:
        if result.monthly_cost is not None and result.monthly_projected is not None:
            _cur_q, _proj_q, _ = _reconciled_delta_pct(
                result.monthly_cost, result.monthly_projected
            )
        else:
            _cur_q, _proj_q, _ = _reconciled_delta_pct(
                result.total_cost, result.projected_cost
            )
        delta_amt = _cur_q - _proj_q
        found.append(
            '<div class="delta">'
            '<span class="delta-label">You save</span>'
            f'<span class="delta-value tnum">{_fmt_usd(delta_amt)}{_delta_unit_v2}</span>'
            f'<span class="delta-pct tnum">&minus;{float(saving_pct):.1f}%</span>'
            "</div>"
        )

    # Metadata line — calls analyzed + pricing snapshot, demoted to quiet meta
    # rather than equal-weight tiles (they are not part of the cost comparison).
    # Calls-priced line + synced-dates line (Item E — synced dates on their own
    # line so a large priced-call count can never push them past the edge).
    found += _html_v2_meta_lines(result)
    # Log span (Item 4) + window caution (Item 3) — full-view disclosures in
    # the same quiet meta register beneath the comparison matrix.
    _log_span = _html_log_span_html(result)
    if _log_span:
        found.append(f'<p class="meta-line">{_log_span}</p>')
    _window_caution = _html_window_caution(result)
    if _window_caution:
        found.append(f'<p class="meta-line">{_window_caution}</p>')
    found.append("</section>")
    right_col.append("\n".join(found))

    # --- Cost by model (with inline proportion bars + total row) ---
    if result.cost_by_model:
        rows: list[str] = []
        total_priced_calls = 0
        for model, cost in sorted(
            result.cost_by_model.items(), key=lambda kv: kv[1], reverse=True
        ):
            pct = (cost / result.total_cost * 100) if result.total_cost else Decimal("0")
            calls = result.calls_by_model.get(model, 0)
            total_priced_calls += calls
            bar_w = f"{float(pct):.1f}"
            rows.append(
                "<tr>"
                f'<td class="bar-cell">'
                f'<span class="bar" style="width:{bar_w}%"></span>'
                f'<span class="model">{esc(model)}</span></td>'
                f'<td class="num r tnum">{calls:,}</td>'
                f'<td class="num r tnum">{_fmt_usd(cost)}</td>'
                f'<td class="num r tnum">{float(pct):.1f}%</td>'
                "</tr>"
            )
        rows.append(
            '<tr class="total">'
            '<td class="lbl">Total</td>'
            f'<td class="num r tnum">{total_priced_calls:,}</td>'
            f'<td class="num r tnum">{_fmt_usd(result.total_cost)}</td>'
            '<td class="num r tnum">100.0%</td>'
            "</tr>"
        )
        left_col.append(
            "<section>"
            '<div class="eyebrow">Cost by model</div>'
            '<table class="tbl"><thead><tr>'
            "<th>Model</th>"
            '<th class="r">Calls</th>'
            '<th class="r">Cost</th>'
            '<th class="r">% of total</th>'
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            "</section>"
        )

    # --- Recommended swap ---
    if has_saving:
        baseline = _dominant_model(result)
        if baseline:
            pill = (
                '<div class="pill">'
                f'<span class="from">{esc(baseline)}</span>'
                '<span class="arrow">&rarr;</span>'
                f'<span class="to">{esc(result.candidate_model or "")}</span>'
                "</div>"
            )
        else:
            pill = (
                '<div class="pill">'
                f'<span class="to">{esc(result.candidate_model or "")}</span>'
                "</div>"
            )
        # The swap-note's "verify with --measure" nudge + dagger point at the
        # footer fine-print; once a verdict exists (Tier-1) both the nudge and
        # the dagger are dropped so nothing points at a caveat that no longer
        # shows. Tier-0 keeps a measure-aware "--judge for a scored verdict".
        if measure_result is None:
            swap_note = (
                '<p class="swap-note">List-price estimate. '
                "Verify quality with <code>--measure</code> before switching"
                '<span class="dagger">&dagger;</span>.</p>'
            )
        elif _is_tier1(measure_result):
            swap_note = (
                '<p class="swap-note">List-price estimate; '
                "quality measured below.</p>"
            )
        else:
            swap_note = (
                '<p class="swap-note">List-price estimate. '
                "Raw samples below — run <code>--judge</code> for a scored "
                "verdict.</p>"
            )
        right_col.append(
            "<section>"
            '<div class="eyebrow">Recommended swap</div>'
            '<div class="swap">'
            + pill
            + swap_note
            + "</div>"
            "</section>"
        )

    # --- Assemble the above-the-fold two-column grid ---
    # Left rail carries the hero (saving + before/after) and the cost-by-model
    # breakdown beneath it; right rail carries the What-we-found matrix with the
    # "You save" delta and the recommended swap beneath it. Splitting the two
    # tallest blocks across the rails keeps the columns near the same height so
    # there is no dead quadrant. The grid collapses to a single column under
    # ~720px (see .fold media query in the CSS).
    parts.append(
        '<div class="fold">'
        f'<div class="col">{"".join(left_col)}</div>'
        f'<div class="col">{"".join(right_col)}</div>'
        "</div>"
    )

    # Swap plan — wholesale only; split runs go through the routing-plan path.
    _ws_plan = _render_html_v2_wholesale_swap_plan(result, esc)
    if _ws_plan:
        parts.append(_ws_plan)

    _dq = _data_quality_html(result)
    if _dq:
        parts.append(f'<section class="below">{_dq}</section>')

    # --- Quality / attribution (full width, below the fold) ---
    if result.candidate_model:
        attribution = _get_attribution()
        src_html = (
            f'<span class="src">{esc(attribution)}</span>' if attribution else ""
        )
        parts.append(
            '<section class="below">'
            '<div class="eyebrow">Methodology</div>'
            f'<p class="prose">{esc(_methodology_note_text(measure_result))}{src_html}</p>'
            "</section>"
        )
    else:
        parts.append(
            '<section class="below">'
            '<div class="eyebrow">Methodology</div>'
            f'<p class="prose">{esc(_methodology_note_text(measure_result))}</p>'
            "</section>"
        )

    # Quality-measurement section (full width, below the fold) before the
    # footer — empty / byte-identical when no measure run ran.
    quality_section = _quality_section_html(measure_result, style="v2", limits=limits, result=result)
    if quality_section:
        parts.append(quality_section)

    # --- Footer: privacy + sources + funnel (landing-page language) ---
    foot = ['<div class="foot">']
    # Fine-print disclosure — the quality caveat, styled as official small
    # print (like a financial disclosure) rather than inline prose. The dagger
    # links it to the after-swap figures above. Present whenever a routing
    # recommendation is shown (honesty invariant: never dropped or hidden).
    if result.candidate_model:
        _caveat = _quality_caveat_text(QUALITY_CAVEAT, measure_result)
        if _caveat is not None:
            foot.append(
                '<div class="fineprint">'
                '<span class="mark">&dagger;</span>'
                # `--measure` / `--judge` are wrapped in <code> (matching the
                # swap-note) so the CLI flag never wraps mid-token.
                f'<span class="body">{esc(_caveat).replace("--measure", "<code>--measure</code>").replace("--judge", "<code>--judge</code>")}</span>'
                "</div>"
            )
        # Unrated-message family (findings #1 + #4) — same wording as every surface.
        _unrated_html = _html_unrated_family_html(result, judged_models_from_measure(measure_result))
        if _unrated_html:
            foot.append(_unrated_html)
    foot.append(f'<p class="privacy">{_privacy_html(measure_result)}</p>')
    foot.append(
        '<p class="meta">methodology &middot; tokencost &middot; LiteLLM registry '
        f"&middot; LMArena quality tiers, RouteLLM-style routing &middot; {esc(_methodology_tail_text(measure_result))}</p>"
    )
    if result.candidate_model:
        foot.append(
            '<p class="cta">'
            + esc(FUNNEL_LINE).replace(
                esc("→ https://frugon.rodiun.io"),
                '&rarr; <a href="https://frugon.rodiun.io">frugon.rodiun.io</a>',
            )
            + "</p>"
        )
    foot.append("</div>")
    parts.append("\n".join(foot))

    html = _HTML_TEMPLATE_V2.format(
        css=_HTML_CSS_V2 + _quality_css(measure_result), body="\n".join(parts)
    )
    _atomic_write_text(output_path, html)


# ---------------------------------------------------------------------------
# `frugon models` — discovery listing of the local pricing table
# ---------------------------------------------------------------------------


def _fmt_per_million(cost_per_token: Decimal) -> str:
    """Format a per-token cost as a per-1M-token USD price for the listing.

    The pricing table stores per-token figures; per-1M tokens is the unit users
    recognise from provider pricing pages.  Whole-dollar prices read as ``$10``;
    sub-dollar prices keep enough precision to stay distinct (``$0.15``) without
    a wall of trailing zeros, and a genuinely free model reads ``$0``.
    """
    per_million = cost_per_token * Decimal("1000000")
    if per_million == 0:
        return "$0"
    if per_million >= Decimal("10"):
        return f"${per_million.quantize(Decimal('1')):,}"
    if per_million >= Decimal("1"):
        return f"${float(per_million):.2f}"
    # Sub-$1/1M: keep up to four significant places so cheap models stay distinct.
    return f"${float(per_million):.4g}".replace("$$", "$")


def render_models_terminal(
    rows: list[PricedModelRow], query: str | None = None
) -> None:
    """List the models in the local pricing table (the ``frugon models`` view).

    Borderless, design-consistent table in the same language as "Cost by model":
    cyan model name, right-aligned per-1M input/output prices, and the quality
    tier label when the model is rated ("—" when not).  A dim count footer echoes
    the total and the active query.  *rows* is expected pre-sorted by name.
    """
    from frugon.quality import tier_name

    table = Table(
        title="Models frugon can price (USD per 1M tokens)",
        title_justify="left",
        title_style="dim",
        box=None,
        show_header=True,
        header_style="dim",
        pad_edge=False,
        padding=(0, 2, 0, 0),
    )
    table.add_column("Model", style=BRAND_CYAN, no_wrap=True)
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Quality", justify="left", style="dim", no_wrap=True)

    for row in rows:
        label = tier_name(row.quality_tier) or "—"
        table.add_row(
            row.model,
            _fmt_per_million(row.input_cost_per_token),
            _fmt_per_million(row.output_cost_per_token),
            label,
        )

    rprint(Padding(table, (1, 0, 0, 2)))

    footer = Text(f"{len(rows):,} models", style="dim")
    if query:
        footer.append(" · query ", style="dim")
        footer.append(f'"{query}"', style=BRAND_CYAN)
    rprint(Padding(footer, (1, 0, 0, 2)))


def render_models_empty(query: str) -> None:
    """Render the clean no-match line for ``frugon models QUERY`` (no traceback)."""
    msg = Text("no models match ", style="dim")
    msg.append(f'"{query}"', style=BRAND_CYAN)
    msg.append(" — try a broader term.", style="dim")
    rprint(Padding(msg, (1, 0, 0, 2)))

