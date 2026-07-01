"""frugon CLI — entry point for all commands.

Two top-level commands:
  capture   Start the local passive logger (shim); your app points its base URL here.
  analyze   Read logs locally and show cost + routing recommendations.

Sub-app:
  pricing update   Refresh the bundled pricing table from the LiteLLM registry.

Privacy guarantee (printed on first run and in every help epilog):
  Your data never leaves your machine.
  Your keys go straight to your own providers. Nothing reaches us.
"""

from __future__ import annotations

import pathlib
import sys
from typing import TYPE_CHECKING

import typer
from rich import print as rprint
from rich.panel import Panel

from frugon import __version__

# Planned-call count above which a --measure run shows its cost estimate and, on
# a TTY, asks to proceed.  Re-used from the measure module so the CLI gate and
# the estimate logic share ONE threshold (no drift).  A default 10-sample
# single-candidate judge run is exactly 30 calls, so it stays under the gate and
# small runs remain frictionless.
from frugon.measure import _ESTIMATE_CALL_THRESHOLD as _MEASURE_CONFIRM_THRESHOLD

# A plain string constant — importing it does NOT pull in LiteLLM (measure.py
# imports litellm lazily, only inside _import_litellm), so the fast cost-only
# path stays light.  Needed at module load to default the --judge-model option.
from frugon.measure import DEFAULT_JUDGE_MODEL

if TYPE_CHECKING:
    from frugon.measure import MeasureEstimate, MeasureResult


def _resolve_judge_model(
    judge_model_flag: str | None, log_models: list[str]
) -> tuple[str | None, bool]:
    """Resolve the LLM judge for a ``--judge`` run, key-aware at the fallback.

    Precedence (highest first):
      1. an explicit ``--judge-model`` value (the user's stated intent);
      2. the highest quality-tier model present in *log_models* — the models the
         user's log actually contains, so they already have a key for it
         (``best_judge_from_log``); else
      3. the highest quality-tier rated+priced model whose provider key IS present
         in the environment (``best_judge_for_available_keys``) — so a user who
         holds only an Anthropic / Gemini / DeepSeek / OpenRouter key gets a judge
         they can actually reach, NOT a hard OpenAI default they cannot; else
      4. ``DEFAULT_JUDGE_MODEL`` (gpt-4o) ONLY when the OpenAI key it requires is
         actually present; else
      5. ``None`` — no judge is resolvable from the present keys.  The CLI then
         renders the clean fail-fast panel directing the user to set a provider
         key or pass ``--judge-model``.

    Returns ``(resolved_model_or_None, from_log)`` where *from_log* is True ONLY
    in case 2 — when the judge was auto-selected as the user's highest-tier logged
    model.  The CLI threads *from_log* into ``run_measure`` so the methodology note
    can honestly describe the judge as "your highest-tier model" in that case (and
    reserve "(independent)" for a genuinely external judge).

    The helpers are imported lazily so the fast cost-only path (which never
    resolves a judge) does not pay the cost/measure module import at CLI load.
    """
    if judge_model_flag is not None:
        return judge_model_flag, False
    import os

    from frugon.cost import best_judge_from_log
    from frugon.measure import (
        _required_key_for_model,
        best_judge_for_available_keys,
    )

    log_best = best_judge_from_log(log_models)
    if log_best is not None:
        return log_best, True

    # No rated model in the user's log: pick the best rated+priced model whose
    # provider key is present, instead of hard-defaulting to OpenAI.
    key_aware = best_judge_for_available_keys()
    if key_aware is not None:
        return key_aware, False

    # Last resort: the OpenAI default, but ONLY when its key is actually present —
    # never demand an OpenAI key from a user who does not hold one.
    default_key = _required_key_for_model(DEFAULT_JUDGE_MODEL)
    if default_key is not None and os.environ.get(default_key):
        return DEFAULT_JUDGE_MODEL, False

    return None, False

# ---------------------------------------------------------------------------
# Privacy line — single canonical string, referenced in tests
# ---------------------------------------------------------------------------
PRIVACY_LINE = (
    "Your data never leaves your machine. "
    "Your keys go straight to your own providers. "
    "Nothing reaches us."
)

# Record count above which analyze prints a one-line "this may take a moment"
# heads-up on the stderr progress channel.  Not a cap and not a hard limit — just
# a courtesy.  Sized well above the comfortable range: analysis is fast to ~100k
# records and the ~56k bundled demo prices in a few seconds, so the notice
# only fires for genuinely large logs where a short wait is expected.
_LARGE_LOG_NOTICE_THRESHOLD = 200_000


# ---------------------------------------------------------------------------
# Friendly, framed messages for the two EXPECTED --measure prerequisite errors.
#
# A missing optional install or a missing API key is an actionable user
# condition, not a bug — so it is rendered as a calm amber panel with the exact
# fix command, NOT a Python traceback.  Full tracebacks are reserved for
# genuinely unexpected failures.
# ---------------------------------------------------------------------------
def _render_missing_extra() -> None:
    """Render the friendly panel shown when the ``[measure]`` extra is absent."""
    rprint("")  # breathing room between the prompt/output above and the panel
    rprint(
        Panel(
            "[bold]--measure needs LiteLLM[/bold] (the optional [cyan]measure[/cyan] extra).\n\n"
            "Install it:\n"
            "  [cyan]uv tool install 'frugon\\[measure]' --force[/cyan]\n"
            "  [cyan]pip install 'frugon\\[measure]'[/cyan]\n\n"
            "[dim]Use the first line if you installed frugon with uv tool "
            "install (the recommended way), the pip line if you used pip. "
            "Then re-run your command — the cost analysis itself needs no "
            "install, only --measure / --judge do.[/dim]",
            title="[bold yellow]One install needed[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )
    rprint("")  # breathing room before the shell prompt returns


def _env_set_hint(var: str) -> str:
    """Return the shell-appropriate command to set an env var, for the host OS.

    Windows users are almost always in PowerShell (or cmd), where the Unix
    ``export`` is a "term not recognized" error — so show the PowerShell form
    there and ``export`` on POSIX shells.  Both forms are session-scoped, exactly
    like ``export`` itself; persisting the key is left to the user's own profile.
    """
    if sys.platform == "win32":
        return f'$env:{var}="<your-key>"'
    return f'export {var}="<your-key>"'


def _render_missing_key(exc: Exception, measured_models: list[str] | None = None) -> None:
    """Render the friendly panel shown when a required provider key is absent.

    *exc* is the :class:`~frugon.measure.MissingProviderKeyError`; its
    ``missing_vars`` attribute names the environment variables to set.

    *measured_models*, when given, is the set of models ``--measure`` will
    actually sample — the baseline plus the candidate frugon recommends (the
    split-routing target the headline saving is computed against, e.g.
    ``gpt-4o-mini``).  Naming them in the panel makes it explicit that the key
    being requested is for the precise switch the tool proposes — not for some
    separately auto-selected model — so the user can see *why* the key is needed.
    """
    missing_vars: list[str] = list(getattr(exc, "missing_vars", []) or [])
    if missing_vars:
        if len(missing_vars) == 1:
            need_line = (
                "[bold]--measure needs your AI provider's API key "
                f"([cyan]{missing_vars[0]}[/cyan]).[/bold]"
            )
        else:
            joined = ", ".join(f"[cyan]{v}[/cyan]" for v in missing_vars)
            need_line = (
                f"[bold]--measure needs your AI provider API keys ({joined}).[/bold]"
            )
        set_lines = "\n".join(
            f"  [cyan]{_env_set_hint(var)}[/cyan]" for var in missing_vars
        )
    else:  # Defensive fallback — should not happen, but never render a blank panel.
        need_line = "[bold]--measure needs your AI provider's API key.[/bold]"
        set_lines = f"  [cyan]{_env_set_hint('OPENAI_API_KEY')}[/cyan]"
    # "Did you mean…?" — nudge the user who set a typo'd variable (e.g. OPEN_API_KEY).
    suggestions: dict[str, str] = dict(getattr(exc, "suggestions", {}) or {})
    hint = ""
    for expected, near in suggestions.items():
        hint = (
            f"\n\n[yellow]Did you mean [bold]{expected}[/bold]? "
            f"You have [bold]{near}[/bold] set — check the spelling.[/yellow]"
        )
        break  # one hint is enough; the rest follow the same fix
    # Name the models --measure will sample so the requested key is tied to the
    # actual switch frugon recommends (deduped, order-preserving).
    measured_line = ""
    if measured_models:
        seen: set[str] = set()
        ordered: list[str] = []
        for m in measured_models:
            if m and m not in seen:
                seen.add(m)
                ordered.append(m)
        if ordered:
            joined_models = ", ".join(f"[cyan]{m}[/cyan]" for m in ordered)
            measured_line = f"\n\n[dim]--measure will sample:[/dim] {joined_models}"
    rprint("")  # breathing room between the prompt/output above and the panel
    rprint(
        Panel(
            f"{need_line}\n\n"
            "Set it:\n"
            f"{set_lines}"
            f"{hint}"
            f"{measured_line}\n\n"
            "[dim]The call goes to your own provider — never to Frugon.[/dim]",
            title="[bold yellow]Provider key needed[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )
    rprint("")  # breathing room before the shell prompt returns


def _render_unknown_model(exc: Exception) -> None:
    """Render the friendly panel shown when a model name is unresolvable.

    Fires from the same fail-fast --measure pre-flight that surfaces missing
    provider keys: if the user typo'd ``--candidates gpt-5.3`` (or named a
    judge model frugon cannot resolve), they see one clean amber panel —
    listing every bad name plus the closest pricing-table neighbours — and a
    config-error exit code, never a Python traceback and never a wasted
    provider call.

    *exc* is the :class:`~frugon.measure.UnknownModelError`; its
    ``unknown_models`` attribute carries the ordered list of
    ``(bad_name, [suggestions])`` pairs.
    """
    unknown: list[tuple[str, list[str]]] = list(
        getattr(exc, "unknown_models", []) or []
    )
    if not unknown:  # Defensive — should never happen; never render a blank panel.
        unknown = [("(unknown)", [])]

    lines: list[str] = []
    if len(unknown) == 1:
        lines.append("[bold]--measure can't find this model:[/bold]")
    else:
        lines.append("[bold]--measure can't find these models:[/bold]")
    for bad, suggestions in unknown:
        if suggestions:
            joined = ", ".join(f"[cyan]{s}[/cyan]" for s in suggestions)
            lines.append(f"  [cyan]{bad}[/cyan]   did you mean: {joined}?")
        else:
            lines.append(
                f"  [cyan]{bad}[/cyan]   [dim](no close match in the pricing table)[/dim]"
            )

    body = "\n".join(lines)
    rprint("")  # breathing room between the prompt/output above and the panel
    rprint(
        Panel(
            f"{body}\n\n"
            "[dim]Pass a known model name, or run "
            "[cyan]frugon models[/cyan] to see what's available.[/dim]",
            title="[bold yellow]Unknown model name[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )
    rprint("")  # breathing room before the shell prompt returns


def _render_no_judge_resolvable() -> None:
    """Render the panel shown when ``--judge`` can reach no judge with present keys.

    Fired when no logged model is rated AND no rated+priced model's provider key
    is present in the environment (and the OpenAI default's key is also absent).
    Directs the user to set a provider key OR name a judge explicitly — never a
    silent OpenAI demand, never a traceback.
    """
    rprint("")  # breathing room between the prompt/output above and the panel
    rprint(
        Panel(
            "[bold]--judge could not pick a judge model your keys can reach.[/bold]\n\n"
            "None of your logged models are quality-rated, and no rated model's "
            "provider key is set in your environment.\n\n"
            "Fix it either way:\n"
            f"  [cyan]{_env_set_hint('ANTHROPIC_API_KEY')}[/cyan]  (or OPENAI / GEMINI / …)\n"
            "  [cyan]frugon analyze … --judge --judge-model <model you have a key for>[/cyan]\n\n"
            "[dim]OpenRouter users: pass [/dim][cyan]--judge-model[/cyan][dim] "
            "explicitly (e.g. openrouter/anthropic/claude-3.5-sonnet).[/dim]\n"
            "[dim]The judge call goes to your own provider — never to Frugon.[/dim]",
            title="[bold yellow]No judge available[/bold yellow]",
            border_style="yellow",
            expand=False,
        )
    )
    rprint("")  # breathing room before the shell prompt returns


def _stdin_is_tty() -> bool:
    """Return True when stdin is an interactive terminal.

    A thin wrapper around ``sys.stdin.isatty()`` so the pre-run confirm gate has
    ONE patchable seam — the CliRunner swaps ``sys.stdin`` for a non-TTY stream
    during a test invocation, so tests monkeypatch THIS function to drive the TTY
    branch deterministically.  Guards against a stdin with no ``isatty`` (returns
    False — the safe, non-interactive default that never hangs).
    """
    isatty = getattr(sys.stdin, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def _render_measure_estimate(estimate: MeasureEstimate) -> None:
    """Print the pre-run cost estimate line for a large --measure run.

    Renders the planned-call arithmetic so the total VISIBLY adds up, e.g.
    (judge on)::

        About to make 150 provider calls: 100 to sample (50 prompts × 2 models:
        baseline + 1 candidate) + 50 to judge (50 prompts × 1 candidate).
        Estimated cost ~$0.90 on your keys.

    The sampling component is ``n_prompts × (1 baseline + n_candidates)`` models
    and the judging component is ``n_prompts × n_candidates``, so a reader can
    check both summands against the headline ``planned_calls`` total.  The two
    legs are introduced by a COLON and the cost by a PERIOD (no dashes, which
    read as minus signs next to the ``×``/``+``/numbers).  When ``--judge`` is
    off there is only the sampling leg — and stating "N calls: N to sample"
    would repeat the same number twice — so the no-judge form drops the colon
    breakdown and states the sampling models inline::

        About to make 100 provider calls (50 prompts × 2 models:
        baseline + 1 candidate). Estimated cost ~$0.65 on your keys.

    Singular/plural is handled for both prompts and candidates ("1 candidate" /
    "3 candidates", "1 prompt" / "50 prompts").

    The COUNT carries no ``~`` — it is the exact planned number of calls; only
    the dollar figure is an estimate (marked ``~``/"Estimated cost").  The counts
    come from the estimate's OWN ``n_prompts``/``n_candidates`` (capped to the
    available records), NOT the requested ``--samples``, so the arithmetic always
    reconciles to ``planned_calls`` even when the log held fewer records than
    asked for.  When NO target model could be priced the cost is replaced with
    "Estimated cost unavailable for <models>" but the call count is still shown.
    The whole line is dim — it is guidance, not a verdict.
    """
    n_prompts = estimate.n_prompts
    n_candidates = estimate.n_candidates
    use_judge = estimate.use_judge

    def _plural(count: int, noun: str) -> str:
        return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"

    candidate_phrase = (
        "baseline + 1 candidate"
        if n_candidates == 1
        else f"baseline + {n_candidates:,} candidates"
    )

    # Sampling component: n_prompts × (1 baseline + n_candidates) models.  The
    # parenthetical "(N prompts × M models: …)" is shared by both forms.
    sample_calls = n_prompts * (1 + n_candidates)
    n_models = 1 + n_candidates
    sample_paren = (
        f"({_plural(n_prompts, 'prompt')} × {_plural(n_models, 'model')}: "
        f"{candidate_phrase})"
    )

    if use_judge:
        # Two legs that must visibly reconcile (sample + judge = planned_calls).
        # Introduce them with a colon; "N to sample" carries the sample count so
        # the arithmetic adds up against the headline total.
        judge_calls = n_prompts * n_candidates
        plan_clause = (
            f"{estimate.planned_calls:,} provider calls: "
            f"{sample_calls:,} to sample {sample_paren}"
            f" + {judge_calls:,} to judge "
            f"({_plural(n_prompts, 'prompt')} × {_plural(n_candidates, 'candidate')})"
        )
    else:
        # Only the sampling leg — sample_calls == planned_calls.  Stating both
        # ("N calls: N to sample") would repeat the number, so state the models
        # inline without the colon breakdown.
        plan_clause = f"{estimate.planned_calls:,} provider calls {sample_paren}"

    if estimate.estimated_cost is not None:
        cost_clause = f"Estimated cost ~${float(estimate.estimated_cost):.2f} on your keys"
    else:
        # No target model was priceable — be honest: show the call count, not a
        # fabricated dollar figure.
        joined = ", ".join(estimate.unpriced_models) or "the target model(s)"
        cost_clause = f"Estimated cost unavailable for {joined}"
    rprint(f"\n[dim]About to make {plan_clause}. {cost_clause}.[/dim]")


# ---------------------------------------------------------------------------
# Root app
# ---------------------------------------------------------------------------
app = typer.Typer(
    name="frugon",
    help=(
        "Free, local, open-source LLM cost analyzer.\n\n"
        "Point frugon at your LLM call logs and see — on your machine — how much "
        "you'd save by switching or routing models.\n\n"
        f"[dim]{PRIVACY_LINE}[/dim]"
    ),
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
    epilog=f"[dim]{PRIVACY_LINE}[/dim]",
)

# ---------------------------------------------------------------------------
# pricing sub-app
# ---------------------------------------------------------------------------
pricing_app = typer.Typer(
    name="pricing",
    help="Manage the local pricing table.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(pricing_app, name="pricing")

# ---------------------------------------------------------------------------
# quality sub-app
# ---------------------------------------------------------------------------
quality_app = typer.Typer(
    name="quality",
    help="Manage the local quality tier table (LMArena-backed).",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
app.add_typer(quality_app, name="quality")


# ---------------------------------------------------------------------------
# --version callback
# ---------------------------------------------------------------------------
def _version_callback(value: bool) -> None:
    if value:
        rprint(f"[bold cyan]frugon[/bold cyan] {__version__}")
        raise typer.Exit()


@app.callback()
def _root_callback(
    version: bool | None = typer.Option(  # noqa: B008
        None,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """frugon — free, local LLM cost analyzer."""


# ---------------------------------------------------------------------------
# capture command
# ---------------------------------------------------------------------------
@app.command()
def capture(
    port: int = typer.Option(  # noqa: B008
        8787,
        "--port",
        "-p",
        help="Local port the passive logger listens on.",
        show_default=True,
    ),
    out: pathlib.Path = typer.Option(  # noqa: B008
        pathlib.Path("capture.jsonl"),
        "--out",
        help="Path to the output JSONL file.",
        show_default=True,
    ),
    upstream: str | None = typer.Option(  # noqa: B008
        None,
        "--upstream",
        help=(
            "Upstream base URL to forward calls to. "
            "Defaults to OPENAI_BASE_URL env var or https://api.openai.com."
        ),
        show_default=False,
    ),
    quiet: bool = typer.Option(  # noqa: B008
        False,
        "--quiet",
        "-q",
        help=(
            "Suppress all output after the startup panel. "
            "Use when backgrounding (nohup, systemd, CI)."
        ),
        show_default=False,
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        "-v",
        help=(
            "Print one line per captured call (timestamp, model, token counts). "
            "Useful for verifying the shim is recording on first run."
        ),
        show_default=False,
    ),
    allow_insecure: bool = typer.Option(  # noqa: B008
        False,
        "--allow-insecure",
        "--allow-insecure-upstream",
        help=(
            "Allow plain http:// to non-localhost upstream hosts. "
            "Use only for local LAN proxies (e.g. Ollama on 192.168.x.x). "
            "Never use against untrusted networks."
        ),
        show_default=False,
    ),
    proxy: str | None = typer.Option(  # noqa: B008
        None,
        "--proxy",
        help=(
            "Route upstream calls through this proxy URL (e.g. http://127.0.0.1:8080). "
            "By default frugon sends requests straight to your provider and ignores any "
            "HTTP_PROXY/HTTPS_PROXY environment, so your API key never passes through a "
            "third-party proxy. Set this only to opt in knowingly."
        ),
        show_default=False,
    ),
) -> None:
    """Start the local passive logger.

    Point your app's LLM base URL at [cyan]http://127.0.0.1:<PORT>/v1/...[/cyan].
    Each call is forwarded to your real provider and saved to [cyan]--out[/cyan].
    Nothing is sent to any frugon endpoint.

    [dim]Your data never leaves your machine. Your keys go straight to your own providers.
    Nothing reaches us.[/dim]
    """
    if quiet and verbose:
        rprint("[red]Error: --quiet and --verbose are mutually exclusive.[/red]")
        raise typer.Exit(code=1)

    verbosity = "quiet" if quiet else ("verbose" if verbose else "normal")

    from frugon.capture import run_capture

    rprint(
        Panel(
            f"[bold green]Listening on[/bold green]  [cyan]http://127.0.0.1:{port}[/cyan]\n"
            f"[bold green]Logging to[/bold green]    [cyan]{out}[/cyan]\n\n"
            f"Point your app's base URL at "
            f"[cyan]http://127.0.0.1:{port}/v1/...[/cyan]\n"
            "Calls are forwarded to your provider and saved locally. "
            "Nothing reaches us.\n\n"
            f"[dim]{PRIVACY_LINE}[/dim]\n\n"
            "[dim]Press Ctrl+C to stop.[/dim]",
            title="[bold]frugon capture[/bold]",
            border_style="green",
        )
    )

    run_capture(
        port=port,
        out_path=out,
        upstream=upstream,
        verbosity=verbosity,
        allow_insecure_upstream=allow_insecure,
        proxy=proxy,
    )


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------
@app.command()
def analyze(
    logs: pathlib.Path | None = typer.Argument(  # noqa: B008
        None,
        help="Path to a JSONL log file (OpenAI-compatible request/response format).",
    ),
    measure: bool = typer.Option(  # noqa: B008
        False,
        "--measure",
        help=(
            "Sample real prompts through candidate models using your own API keys. "
            "Calls go to your own providers — never to us. "
            "Requires the optional [cyan]frugon\\[measure][/cyan] extra "
            "([cyan]pip install 'frugon\\[measure]'[/cyan]) and the relevant provider "
            "API key (e.g. [cyan]OPENAI_API_KEY[/cyan])."
        ),
        show_default=True,
    ),
    samples: int = typer.Option(  # noqa: B008
        10,
        "--samples",
        help=(
            "Number of prompts to sample when --measure is set. "
            "5 = quick glance · 10 = default · 25-30 = confident before switching."
        ),
        show_default=True,
        min=1,
    ),
    concurrency: int = typer.Option(  # noqa: B008
        5,
        "--concurrency",
        help=(
            "Max concurrent provider calls per stage (sampling fans across "
            "providers; judging is capped to protect the single judge endpoint). "
            "Default 5."
        ),
        show_default=True,
        min=1,
    ),
    judge: bool = typer.Option(  # noqa: B008
        False,
        "--judge",
        help=(
            "Use an LLM-as-judge to score candidate outputs (Tier-1) instead of "
            "leaving you to eyeball the raw side-by-side outputs. For each sampled "
            "prompt a judge model decides whether the candidate's "
            "output is better / worse / tied vs your current model, and the run "
            "reports a win/loss/tie tally. By default the judge is the "
            "highest quality-tier model in YOUR log (so you already have a key "
            "for it); override with --judge-model, or it falls back to the best "
            "rated model your keys can reach when none of your logged models are "
            "rated. To "
            "remove position bias, the two outputs are shown to the judge in a "
            "randomised A/B order (seeded, so a run stays reproducible). "
            "Requires --measure (so also the [cyan]frugon\\[measure][/cyan] extra "
            "and a provider API key). Judge calls use your own API keys."
        ),
        show_default=True,
    ),
    judge_model: str | None = typer.Option(  # noqa: B008
        None,
        "--judge-model",
        help=(
            "Model to use as the LLM judge when --judge is set. "
            "When omitted, frugon picks the highest quality-tier model that "
            "appears in YOUR OWN log (so you already have a key for it); if none "
            "of your logged models carry a quality rating, it falls back to the "
            "best rated model whose provider key you actually have set. Pick any "
            "model you have a key for; if the judge you choose is the same as a "
            "model it scores, frugon prints a caution that the verdict may be "
            "self-biased."
        ),
        show_default=False,
    ),
    candidates: str | None = typer.Option(  # noqa: B008
        None,
        "--candidates",
        help=(
            "Comma-separated list of candidate model names to evaluate, "
            "e.g. 'gpt-4o-mini,claude-3-haiku-20240307'."
        ),
        show_default=False,
    ),
    window: int | None = typer.Option(  # noqa: B008
        None,
        "--window",
        help=(
            "Size of the observation window in days.  "
            "When provided, frugon projects costs to a 30-day month and "
            "discloses the window size.  "
            "When omitted and timestamps are present, the real span is used.  "
            "When omitted and timestamps are absent, no projection is made."
        ),
        min=1,
        show_default=False,
    ),
    report: pathlib.Path | None = typer.Option(  # noqa: B008
        None,
        "--report",
        help=(
            "Write a shareable report to this path. The extension picks the "
            "format: [cyan]NAME.md[/cyan] = Markdown only, [cyan]NAME.html[/cyan] "
            "= HTML only (styled per --report-style). Give a NAME with no "
            "extension to get every format in one pass: "
            "[cyan]NAME.md[/cyan], [cyan]NAME.v1.html[/cyan], "
            "[cyan]NAME.v2.html[/cyan]. Overrides [cyan]FRUGON_REPORT_PATH[/cyan]. "
            "When omitted, [cyan]FRUGON_REPORT_PATH[/cyan] is used if set; with "
            "neither, output stays terminal-only."
        ),
        show_default=False,
    ),
    no_report: bool = typer.Option(  # noqa: B008
        False,
        "--no-report",
        help=(
            "Force terminal-only output for this run, even when "
            "[cyan]FRUGON_REPORT_PATH[/cyan] is set. The escape hatch that beats "
            "both --report and the env var."
        ),
        show_default=False,
    ),
    report_style: str | None = typer.Option(  # noqa: B008
        None,
        "--report-style",
        help=(
            "HTML report design: v2 = refined editorial layout (default); "
            "v1 = classic. Applies to HTML reports only; Markdown has a single "
            "canonical layout. Ignored when --report is a no-extension prefix "
            "(every HTML style is emitted then).  [default: v2]"
        ),
        show_default=False,
    ),
    preview_chars: int | None = typer.Option(  # noqa: B008
        None,
        "--preview-chars",
        help=(
            "Override the per-prompt OUTPUT preview length (characters) in the "
            "--measure quality sample, on the terminal AND in reports. Prompt "
            "previews scale proportionally to keep each surface's prompt:output "
            "ratio. Minimum 20. Display-only — providers always receive the full "
            "text. Mutually exclusive with --no-truncate."
        ),
        show_default=False,
        min=20,
    ),
    no_truncate: bool = typer.Option(  # noqa: B008
        False,
        "--no-truncate",
        help=(
            "Show the --measure quality-sample previews in FULL (no … truncation) "
            "on the terminal and in reports. Display-only — providers always "
            "receive the full text. Mutually exclusive with --preview-chars."
        ),
        show_default=False,
    ),
    yes: bool = typer.Option(  # noqa: B008
        False,
        "--yes",
        "-y",
        help=(
            "Skip the pre-run confirmation prompt for a large --measure run "
            "(>30 planned provider calls). The cost estimate is still shown."
        ),
        show_default=False,
    ),
    demo: bool = typer.Option(  # noqa: B008
        False,
        "--demo",
        help="Run analysis on the bundled sample log file.",
        show_default=False,
        hidden=False,
    ),
    wholesale: bool = typer.Option(  # noqa: B008
        False,
        "--wholesale",
        help=(
            "Recommend a single wholesale model swap (every call moves to one "
            "candidate) instead of the default per-call routed/kept split."
        ),
        show_default=False,
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        "-v",
        help=(
            "Show supporting detail beneath the recommendation: the wholesale "
            "upper bound, the easy/hard heuristic, and automated routing."
        ),
        show_default=False,
    ),
    no_progress: bool = typer.Option(  # noqa: B008
        False,
        "--no-progress",
        help=(
            "Disable the live progress feedback (spinner, bar, checkpoints). "
            "Progress already auto-disables when stderr is not a terminal or "
            "NO_COLOR is set; this forces it off even in an interactive shell."
        ),
        show_default=False,
    ),
) -> None:
    """Analyze LLM call logs and show cost + routing recommendations.

    Fully local — no LLM calls required for cost analysis.
    Add [cyan]--measure[/cyan] to sample candidate models using your own API keys.

    Examples:

      frugon analyze ./logs.jsonl

      frugon analyze ./logs.jsonl --candidates gpt-4o-mini,claude-3-haiku-20240307

      frugon analyze ./logs.jsonl --window 7

      frugon analyze --demo

    Scale: analysis runs entirely on your machine and is comfortable well past
    100k records — the bundled ~56,100-record demo prices in a few seconds.
    Very large logs (>200k records) may take a little longer; frugon prints a
    one-line heads-up and a live progress bar so you can see it working. There is
    no hard limit.

    Environment:

      [cyan]FRUGON_REPORT_PATH[/cyan]   A saved default report path. When set and
      no [cyan]--report[/cyan] is passed, frugon writes the report there (a bare
      name with no extension emits every format: NAME.md, NAME.v1.html,
      NAME.v2.html). [cyan]--report[/cyan] overrides it; [cyan]--no-report[/cyan]
      disables it for one run. Opt-in: with neither set, output stays
      terminal-only.

    [dim]Your data never leaves your machine. Your keys go straight to your own providers.
    Nothing reaches us.[/dim]
    """
    from frugon._progress import Stopwatch, progress_reporter

    # --preview-chars and --no-truncate are mutually exclusive: one sets a finite
    # cap, the other removes the cap entirely — combining them is contradictory.
    # Fail clean BEFORE any work (parse, pricing, network) so the user fixes the
    # invocation immediately rather than after a long run.
    if no_truncate and preview_chars is not None:
        rprint(
            "[red]Error: --preview-chars and --no-truncate are mutually exclusive.[/red]"
        )
        raise typer.Exit(code=1)

    # Resolve the two per-surface preview limits ONCE from the display flags
    # (terminal for render_quality_terminal, report for the written report).  With
    # neither flag set these are the historical defaults, so default rendering is
    # byte-identical to before.  Display-only: nothing here changes what is sent
    # to providers — _call_model / the judge always receive the full prompt.
    from frugon.report import resolve_preview_limits

    terminal_limits, report_limits = resolve_preview_limits(
        preview_chars=preview_chars, no_truncate=no_truncate
    )

    # Resolve path — --demo loads the bundled sample file
    if demo:
        sample_path = pathlib.Path(__file__).parent / "data" / "sample_logs.jsonl.gz"
        if not sample_path.exists():
            rprint(
                "[red]Demo sample file not found. "
                "Run [bold]python scripts/gen_sample_logs.py[/bold] to regenerate it.[/red]"
            )
            raise typer.Exit(code=1)
        target_path = sample_path
    else:
        if logs is None:
            rprint(
                "[red]Provide a log file path, or use [bold]--demo[/bold] to analyze the bundled sample.[/red]"
            )
            raise typer.Exit(code=1)
        if not logs.exists():
            rprint(f"[red]Log file not found: {logs}[/red]")
            raise typer.Exit(code=1)
        target_path = logs

    # Parse candidate list
    candidate_list: list[str] | None = None
    if candidates:
        candidate_list = [c.strip() for c in candidates.split(",") if c.strip()]

    # --- Fail-fast prerequisite check for --measure / --judge -----------------
    # A measure run depends on the [measure] extra (LiteLLM) and the relevant
    # provider API keys.  Verifying those AFTER the full cost analysis means the
    # user waits through a long parse of a large log only to be told they are
    # missing a one-line install or an env var.  So we check up front — using a
    # cheap distinct-model scan (no token counting, no pricing) to learn which
    # models will be measured — and exit immediately on any failure.  The
    # in-run_measure key pre-flight remains as a backstop.
    if measure:
        # Everything in this block — the cost-module import (~0.6s), the
        # distinct-model scan of the log, and the LiteLLM import (~5s) — runs
        # BEFORE the main read/pricing phases, so each piece is covered by a
        # spinner; otherwise --measure sits silent for seconds before the first
        # animation frame (plain analyze imports none of this, which is why
        # only --measure had the gap).
        with progress_reporter(no_progress=no_progress) as pre_progress:
            scan_failure: str | None = None
            _distinct_models: list[str] = []
            _dominant_model: str | None = None
            with pre_progress.spinner("Preparing --measure…"):
                from frugon.cost import scan_models
                from frugon.measure import (
                    MissingProviderKeyError,
                    UnknownModelError,
                    verify_measure_prerequisites,
                )

                try:
                    _distinct_models, _dominant_model = scan_models(target_path)
                except UnicodeDecodeError:
                    scan_failure = (
                        f"Could not read {target_path}: the file is not valid "
                        "UTF-8. frugon reads UTF-8 JSONL logs."
                    )
                except OSError as exc:
                    scan_failure = (
                        f"Could not read {target_path}: {exc.strerror or exc}."
                    )
            if scan_failure is not None:
                rprint(f"[red]{scan_failure}[/red]")
                raise typer.Exit(code=1)

        # Models that will actually be measured up front:
        #   * the auto-detected baseline — the dominant model in the log, which
        #     is always measured and is cheap to know, so its key is verified
        #     here; and
        #   * the explicit --candidates, when the user named them (their stated
        #     intent), so a typo or missing key surfaces immediately; OR, when
        #     no --candidates are given, the split-routing's RECOMMENDED
        #     candidate — the very model the headline saving is computed against
        #     (e.g. gpt-4o-mini).  That recommendation is selected offline from
        #     the dominant baseline + the built-in pool by exactly the same
        #     select_easy_target() the full analysis uses, so it is cheap to
        #     know here and is the candidate --measure will actually sample.
        #     Verifying its key up front means the pre-check names the real
        #     switch frugon recommends — not a separately auto-selected model.
        # --judge adds the judge model, which is always invoked when set.
        precheck_models: list[str] = []
        if _dominant_model:
            precheck_models.append(_dominant_model)
        if candidate_list:
            precheck_models.extend(candidate_list)
        elif _dominant_model and not wholesale:
            # No explicit --candidates and split routing is active (the default):
            # the default measured candidate is the split recommendation.  Derive
            # it from the same offline selector the analysis uses so the key we
            # verify here is the key the sampled model needs.  None when no rated,
            # priced, strictly-cheaper candidate exists — the backstop covers
            # that residual case.
            from frugon.cost import _DEMO_CANDIDATES, _ROUTING_CANDIDATES
            from frugon.routing import select_easy_target

            _active_pool = _DEMO_CANDIDATES if demo else _ROUTING_CANDIDATES
            recommended = select_easy_target(_dominant_model, _active_pool)
            if recommended:
                precheck_models.append(recommended)
        if judge:
            # Resolve the judge BEFORE the key pre-check so the key we verify up
            # front is the key the judge run_measure will ACTUALLY invoke needs.
            # Resolution order (see _resolve_judge_model): flag > log-best >
            # best key-reachable rated model > OpenAI default (only if its key is
            # present) > None.  _distinct_models is the cheap distinct-model scan
            # of the same log, so the log-best judge resolved here matches the one
            # resolved later from result.cost_by_model.
            resolved_judge, _ = _resolve_judge_model(
                judge_model, _distinct_models
            )
            if resolved_judge is None:
                # No judge is reachable with the present keys — fail fast with a
                # clean panel (set a provider key or pass --judge-model), never a
                # traceback and never a silent OpenAI demand.
                _render_no_judge_resolvable()
                raise typer.Exit(code=1)
            precheck_models.append(resolved_judge)

        try:
            # Importing LiteLLM (the measure engine) is a heavy ~5s import — cover
            # it with a spinner so --measure shows immediate feedback instead of a
            # silent pause before the read phase begins (plain analyze never loads
            # LiteLLM, which is why only --measure had the gap).
            with progress_reporter(no_progress=no_progress) as pre_progress:
                with pre_progress.spinner("Loading the measure engine…"):
                    verify_measure_prerequisites(precheck_models)
        except ImportError:
            _render_missing_extra()
            raise typer.Exit(code=1) from None
        except UnknownModelError as exc:
            # Config error (typo / unknown name) — exit code 2 separates it from
            # a real run failure (code 1) so scripts / CI can branch on intent.
            _render_unknown_model(exc)
            raise typer.Exit(code=2) from None
        except MissingProviderKeyError as exc:
            _render_missing_key(exc, measured_models=precheck_models)
            raise typer.Exit(code=1) from None

    # Run analysis — fail loud with a friendly message on read/encoding errors
    # rather than surfacing a raw Python traceback (§4 fail-loud, but human).
    #
    # Live progress (stderr only): a spinner covers the read/parse phase (record
    # count unknown), then a determinate bar covers the pricing/tokenizing pass
    # (the slow part) driven by a per-record callback, with persisted ✓
    # checkpoints as each phase completes.  When stderr is not a TTY, NO_COLOR
    # is set, or --no-progress is passed, progress_reporter yields a no-op and
    # nothing is rendered.  The analysis RESULT stays entirely on stdout.
    #
    # The records read here are reused by the --measure path below so the log is
    # parsed exactly once (and both paths agree on the skipped count).
    with progress_reporter(no_progress=no_progress) as progress:
        try:
            with progress.spinner("Reading logs…"):
                # Import the analysis engine INSIDE the spinner so its ~0.6s
                # module load is covered by an already-spinning indicator
                # instead of leaving a dead, silent gap before any feedback.
                # frugon.report is loaded here too (used for the terminal
                # render below) for the same reason; both are function-local,
                # so they stay in scope for the rest of analyze().
                from frugon.cost import analyze_records, iter_records
                from frugon.report import render_terminal

                analysis_records, analysis_skipped = iter_records(target_path)
        except UnicodeDecodeError:
            rprint(
                f"[red]Could not read {target_path}: the file is not valid UTF-8. "
                "frugon reads UTF-8 JSONL logs.[/red]"
            )
            raise typer.Exit(code=1) from None
        except OSError as exc:
            rprint(f"[red]Could not read {target_path}: {exc.strerror or exc}.[/red]")
            raise typer.Exit(code=1) from None

        progress.checkpoint(f"Read {len(analysis_records):,} records")

        # Gentle, non-blocking heads-up for a very large log.  Not a cap and not
        # a warning — just a courtesy so a user who points frugon at a huge log
        # knows the pricing pass may take a moment rather than wondering if it
        # has stalled.  Stderr-only via the progress channel, so piped / CI /
        # --no-progress runs stay silent.
        if len(analysis_records) > _LARGE_LOG_NOTICE_THRESHOLD:
            progress.notice(
                f"{len(analysis_records):,} records — this may take a moment."
            )

        # When --demo is active and no explicit --candidates were supplied, pin
        # the demo pool so the demo recommendation and its dollar figures stay
        # numerically stable as _ROUTING_CANDIDATES evolves over time.
        _effective_candidates: list[str] | None = candidate_list
        if demo and candidate_list is None:
            from frugon.cost import _DEMO_CANDIDATES

            _effective_candidates = _DEMO_CANDIDATES

        with progress.bar("Pricing", total=len(analysis_records)) as pricing_task, Stopwatch() as sw:
            result = analyze_records(
                analysis_records,
                window_days=window,
                candidates=_effective_candidates,
                skipped_malformed=analysis_skipped,
                split_routing=not wholesale,
                progress_cb=lambda done, total: pricing_task.advance(1),
            )
        progress.checkpoint(f"Priced in {sw.elapsed:.1f}s")
        if result.split is not None or result.candidate_model is not None:
            progress.checkpoint("Routed")

    # Pricing/quality staleness is now folded into the report's Accounting block:
    # the Prices and Quality rows annotate themselves amber (with the refresh
    # command) when their tables are stale, so there is no stray pre-report warning
    # line here any more (it lived outside the designed blocks).  See
    # report._render_freshness_rows.

    # One blank line of breathing room ABOVE the result section, emitted on
    # stdout so it is present in EVERY mode — progress on, --no-progress, or
    # piped — not only when the ✓ checkpoint trail renders.  (It previously
    # lived on the stderr progress channel via progress.blank(), so it vanished
    # under --no-progress / non-TTY, leaving the panel flush against the prompt.)
    rprint("")

    # Render terminal report -- suppress the "quality unverified" caveat when --measure replaces it.
    # Both headlines \u2014 the per-call split routing and the wholesale single-model
    # swap \u2014 are now fully self-contained in render_terminal: each renders its own
    # panel, accounting, swap/route lines, unrated-tier cautions, privacy line, and
    # (under --verbose) supporting notes in ONE shared design language.  There are
    # no path-specific post-render disclosure lines to print here any more.
    render_terminal(
        result,
        suppress_caveat=measure,
        verbose=verbose,
        # The terminal candidates-considered caption points "below" at the
        # judge tally ONLY when a Tier-1 (--measure --judge) section will
        # actually render beneath it; otherwise it offers the command.
        has_judge_section=measure and judge,
        # Models this run will judge — the EXPLICIT --candidates under
        # --measure --judge, whose quality verdict renders in the section below.
        # Makes the unrated-message family measurement-aware: a model verified by
        # that verdict gets no contradictory "run --measure to confirm" caveat.
        # (The default candidate — when no --candidates is passed — is the
        # split's RATED recommendation, never an unrated model, so it never
        # appears in the unrated family; the explicit list is the precise set.)
        judged_models=(
            frozenset(candidate_list)
            if (measure and judge and candidate_list)
            else frozenset()
        ),
    )

    # Pool notice — shown whenever a recommendation was made and the default pool
    # was used (not an explicit --candidates run).  Tells the user where prices
    # come from and how to refresh, without blocking output.
    # When --demo is active with no explicit --candidates, _DEMO_CANDIDATES is
    # passed so used_default_pool is False, but we still want the notice.
    _show_pool_notice = (
        result.split is not None or result.candidate_model is not None
    ) and (result.used_default_pool or (demo and candidate_list is None))
    if _show_pool_notice:
        import datetime as _dt

        from frugon.pricing import is_pricing_stale

        _pls = result.pricing_json_last_synced
        _stale_suffix = ""
        if _pls and is_pricing_stale(_pls):
            _days_old = (_dt.date.today() - _dt.date.fromisoformat(_pls)).days
            _stale_suffix = f" — {_days_old} days old; run `frugon update`"
        _date_str = _pls or "unknown"
        rprint(
            f"[dim]Recommendations use a curated set of current top models across "
            f"providers, drawn from OpenRouter usage rankings. Prices synced {_date_str} "
            f"from the LiteLLM registry. Run `frugon update` for the full live roster."
            f"{_stale_suffix}[/dim]"
        )

    # --- --measure: sample real prompts (must run BEFORE the report is written
    # so the report can carry the quality verdict) ---------------------------
    #
    # Ordering invariant (the bug this fixes): when BOTH --report and --measure
    # are set, the report MUST be written AFTER run_measure completes, with the
    # resulting MeasureResult passed to the renderer — otherwise the report
    # lands on disk before the judge has even run and can never carry the
    # verdict.  So the measure block runs first and captures measure_result; the
    # report write happens once, afterwards, below.
    #
    # The TERMINAL output order is unchanged: the analysis panel was already
    # rendered above (render_terminal), and the quality sample renders here, in
    # the same place it always did — only the *report write* moves after this
    # block.  A --measure run that hits a skip/error path still leaves
    # measure_result = None, and the report (if any) is written without a
    # quality section, exactly as a --report-without---measure run would be.
    measure_result: MeasureResult | None = None
    if measure:
        from frugon._progress import progress_reporter
        from frugon.measure import MissingProviderKeyError, run_measure
        from frugon.report import render_quality_terminal

        # Reuse the records already parsed above (the log is read exactly once).
        # They come from the same shared iter_records helper analyze used, so the
        # measure sample agrees with the analysis on which lines were dropped —
        # no silent shrinkage of the quality sample (§4 fail-loud).
        measure_records, measure_skipped = analysis_records, analysis_skipped
        if measure_skipped:
            rprint(
                f"[yellow]{measure_skipped} record(s) skipped (malformed) "
                "before sampling.[/yellow]"
            )

        # Resolve the candidate(s) to measure.  Either skip condition leaves
        # measure_result = None and falls through to the report write below
        # (a --report run still produces its cost report, just without a quality
        # section — identical to a --report-without---measure run).
        measure_candidates: list[str] = []
        if not result.cost_by_model:
            rprint("[yellow]--measure skipped: no priced calls found.[/yellow]")
        else:
            current_model = max(
                result.cost_by_model, key=lambda m: result.cost_by_model[m]
            )
            # Default candidate (no explicit --candidates): measure the model the
            # split-routing actually RECOMMENDS — result.split.candidate_model,
            # the model the headline saving is computed against — so --measure
            # verifies the switch the tool proposes, not a separately
            # auto-selected wholesale model.  Fall back to the wholesale
            # candidate only when there is no split (e.g. --wholesale).  Explicit
            # --candidates always win (the user's stated intent).
            default_candidate: str | None = None
            if result.split is not None and result.split.candidate_model:
                default_candidate = result.split.candidate_model
            elif result.candidate_model:
                default_candidate = result.candidate_model
            measure_candidates = (
                candidate_list
                if candidate_list
                else ([default_candidate] if default_candidate else [])
            )
            if not measure_candidates:
                rprint(
                    "[yellow]--measure skipped: no candidate model to evaluate.[/yellow]"
                )

        if measure_candidates:
            # Resolve the judge ONCE here (flag > log-best > best key-reachable >
            # OpenAI default if its key is present > None), from the SAME log the
            # analysis priced, so the pre-run estimate, the confirm prompt, and
            # run_measure all agree on which judge will run.
            resolved_judge_model, judge_is_from_log = _resolve_judge_model(
                judge_model, list(result.cost_by_model.keys())
            )
            if judge and resolved_judge_model is None:
                # The up-front pre-check normally catches this, but the baseline
                # the full analysis selected can differ from the cheap scan — so
                # re-guard here and fail fast with the same clean panel.
                _render_no_judge_resolvable()
                raise typer.Exit(code=1)

            # --- Pre-run estimate + confirm for a big run -------------------
            # Before any sampling call, project the planned call count and the
            # dollar cost (from the sampled records' own token counts × the
            # pricing table).  Above the threshold we always SHOW the estimate;
            # on a TTY we additionally ask to proceed (skippable with --yes); in
            # a pipe / CI we never prompt (would hang) — we print and proceed.
            # At or below the threshold a small run stays frictionless: nothing
            # extra is printed.
            from frugon.measure import estimate_measure_cost

            estimate = estimate_measure_cost(
                measure_records,
                current_model=current_model,
                candidates=measure_candidates,
                n_samples=samples,
                use_judge=judge,
                judge_model=resolved_judge_model if judge else None,
            )
            if estimate.planned_calls > _MEASURE_CONFIRM_THRESHOLD:
                _render_measure_estimate(estimate)
                if not yes and _stdin_is_tty():
                    if not typer.confirm("Proceed?", default=False):
                        rprint("[yellow]Aborted — no provider calls made.[/yellow]")
                        raise typer.Exit(code=0)

            # Backstop: the up-front pre-check above already verified the extra
            # and the keys, but run_measure re-checks against the precise
            # baseline the full analysis selected.  Render the same clean, framed
            # messages here (never a traceback) for the two EXPECTED, actionable
            # conditions.
            #
            # Live progress (stderr only): a per-prompt indicator — "Sampling
            # prompt 3/5 · gpt-4o-mini" — advances once per sampled prompt (NOT
            # per provider call), so it counts the same PROMPTS as the "N
            # prompt(s)" result header, with a parallel "Judging 3/5" counter
            # when --judge is set.  Gated identically to the analyze bar; a no-op
            # under non-TTY / NO_COLOR / --no-progress.  These callbacks only
            # animate the indicator; they never touch the network call or the
            # synthesis, so the fail-fast pre-check and the verdict are
            # unaffected.
            try:
                with progress_reporter(no_progress=no_progress) as mprogress:
                    # Progress counts PROMPTS, the same unit as the "N prompt(s)"
                    # header — so the live "Sampling prompt 3/5" / "Judging 3/5"
                    # counters never disagree with the result header.  Each
                    # prompt still hits the baseline plus every candidate
                    # underneath; only the displayed unit is the prompt.  The
                    # callbacks fire once per prompt, so each .step() advances the
                    # counter by exactly one prompt.
                    n_prompts = min(samples, len(measure_records))
                    judge_total = n_prompts if judge else 0
                    # The blank line that separates the Quality-sample section
                    # from the Prices line is printed HERE — before the live
                    # sampling counter starts — so the gap exists while the
                    # counter animates, not only once the section renders.
                    # render_quality_terminal deliberately prints no leading
                    # blank for this reason; the stdout sequence is unchanged.
                    rprint("")
                    # resolved_judge_model / judge_is_from_log were resolved once
                    # above (shared with the pre-run estimate), so the judge
                    # sampled here is exactly the one the estimate priced and
                    # whose key the up-front pre-check verified.
                    with (
                        mprogress.counter("Sampling prompt", total=n_prompts) as sample_counter,
                        mprogress.counter("Judging", total=judge_total) as judge_counter,
                    ):
                        measure_result = run_measure(
                            measure_records,
                            current_model=current_model,
                            candidates=measure_candidates,
                            n_samples=samples,
                            use_judge=judge,
                            # When --judge is set, resolved_judge_model is a real
                            # model (the None case fails fast above); when it is
                            # not set, run_measure ignores judge_model, so the
                            # DEFAULT placeholder keeps the type str without effect.
                            judge_model=resolved_judge_model or DEFAULT_JUDGE_MODEL,
                            judge_from_log=judge_is_from_log,
                            concurrency=concurrency,
                            sample_cb=lambda done, total, label: sample_counter.step(label),
                            judge_cb=lambda done, total, label: judge_counter.step(label),
                        )
                render_quality_terminal(
                    measure_result,
                    verbose=verbose,
                    limits=terminal_limits,
                    result=result,
                )
            except ImportError:
                _render_missing_extra()
                raise typer.Exit(code=1) from None
            except MissingProviderKeyError as exc:
                _render_missing_key(exc)
                raise typer.Exit(code=1) from None

    # --report: write shareable artifact(s) — AFTER --measure so the measured
    # quality verdict (if any) is carried into the report.  When --measure was not
    # set, was skipped, or produced no result, measure_result is None and the
    # report is byte-identical to a report-only run.
    #
    # Target precedence (opt-in): the explicit --report flag wins; else the saved
    # FRUGON_REPORT_PATH env var (mirrors the API-key env-var pattern); else no
    # report at all.  --no-report is the escape hatch that forces terminal-only
    # for this run even when the env var is set, so it beats both.  With NO flag
    # and NO env var, NOTHING is written — the default stays terminal-only and a
    # bare run never sprays files.
    #
    # The extension on the resolved target decides the format(s): a .md / .html
    # value writes that single format; a value with NO recognised extension is a
    # PREFIX that emits the full set (NAME.md, NAME.v1.html, NAME.v2.html) from
    # this ONE analysis pass.  All formats render from the same in-memory result —
    # report.write_reports is the single mapping point, so the formats never drift
    # and nothing is recomputed.
    import os

    report_target: pathlib.Path | None
    if no_report:
        report_target = None
    elif report is not None:
        report_target = report
    else:
        env_path = os.environ.get("FRUGON_REPORT_PATH")
        report_target = pathlib.Path(env_path) if env_path else None

    if report_target is not None:
        from frugon.report import write_reports

        # --report-style defaults to the v2 editorial layout.  A bare default
        # (report_style is None) is distinguishable from an explicit choice, which
        # lets the Markdown notice below fire only when the user really asked for
        # v2 — never on a default report.
        explicit_v2 = report_style is not None and report_style.lower() == "v2"
        style = (report_style or "v2").lower()
        if style not in ("v1", "v2"):
            rprint(
                f"[yellow]Unknown report style '{report_style}' -- use 'v1' or 'v2'. "
                "Falling back to 'v2'.[/yellow]"
            )
            style = "v2"
            explicit_v2 = False

        # --report-style governs the HTML design only; Markdown has a single
        # canonical layout (v1 and v2 render identically).  Make the flag honest:
        # when v2 was EXPLICITLY requested for a SINGLE Markdown target, surface a
        # dim notice rather than silently ignoring it.  In the prefix case the
        # style is ignored entirely (every HTML style is emitted), so the notice
        # would be misleading there and is not shown.
        if explicit_v2 and report_target.suffix.lower() == ".md":
            rprint(
                "[dim]--report-style v2 styles HTML reports; "
                "Markdown has a single canonical layout.[/dim]"
            )

        written = write_reports(
            result,
            report_target,
            report_style=style,
            measure_result=measure_result,
            limits=report_limits,
        )
        # One confirmation line per file actually written, in write order, each
        # resolved to an absolute path so the user knows exactly where it landed.
        for path in written:
            resolved = pathlib.Path(path).resolve()
            rprint(f"\n[dim]Report written to[/dim] [cyan]{resolved}[/cyan]")


# ---------------------------------------------------------------------------
# models command -- discover the names --candidates accepts
# ---------------------------------------------------------------------------
@app.command()
def models(
    query: str | None = typer.Argument(  # noqa: B008
        None,
        metavar="[QUERY]",
        help="Filter to model names containing this text (case-insensitive).",
    ),
) -> None:
    """List the model names frugon can price -- the names [cyan]--candidates[/cyan] accepts.

    Reads the local pricing table only (the same table [cyan]--candidates[/cyan]
    resolves against), so every name shown is exactly a name you can pass to
    [cyan]--candidates[/cyan]. Optionally filter by a case-insensitive substring.

    [dim]Pure local read -- no network, no account, nothing sent about your usage.[/dim]
    """
    from frugon.pricing import list_priced_models
    from frugon.report import render_models_empty, render_models_terminal

    rows = list_priced_models(query)
    if not rows:
        if query:
            # A valid query that simply matched nothing is not an error -- show a
            # clean hint and exit 0 (no traceback), consistent with the calm tone
            # of the other commands' informational output.
            render_models_empty(query)
            return
        # An empty pricing table is the only way to reach here with no query --
        # surface it plainly and point at the refresh path.
        rprint(
            "[yellow]No models in the local pricing table.[/yellow] "
            "Run [cyan]frugon pricing update[/cyan] to populate it."
        )
        raise typer.Exit(code=1)

    render_models_terminal(rows, query)


# ---------------------------------------------------------------------------
# pricing update sub-command
# ---------------------------------------------------------------------------
@pricing_app.command(name="update")
def pricing_update() -> None:
    """Refresh the local pricing table from the LiteLLM registry.

    Downloads [cyan]model_prices_and_context_window.json[/cyan] and stores it locally.
    No account required. No data sent about your usage.

    [dim]Your data never leaves your machine.[/dim]
    """
    from datetime import date

    from frugon._progress import progress_reporter
    from frugon.pricing import (
        _LITELLM_REGISTRY_URL,
        _PRICING_JSON,
        PricingUpdateError,
        fetch_and_update_pricing,
    )

    # Live progress (stderr only): a spinner reassures the user the network
    # fetch is in flight.  A no-op under non-TTY / NO_COLOR.  The success/error
    # result lines stay on stdout, unchanged.
    try:
        with progress_reporter(no_progress=False) as progress:
            with progress.spinner("Updating pricing table…"):
                result = fetch_and_update_pricing(
                    registry_url=_LITELLM_REGISTRY_URL,
                    output_path=_PRICING_JSON,
                    today_date_str=date.today().isoformat(),
                )
    except PricingUpdateError as exc:
        rprint(f"[red]pricing update failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    rprint(
        f"[bold green]\u2713[/bold green] pricing.json updated -- "
        f"[bold]{result['models_synced']}[/bold] models, "
        f"dated [cyan]{date.today().isoformat()}[/cyan]."
    )
    rprint(f"[dim]{PRIVACY_LINE}[/dim]")


# ---------------------------------------------------------------------------
# quality update sub-command
# ---------------------------------------------------------------------------
@quality_app.command(name="update")
def quality_update() -> None:
    """Refresh the local quality tier table from the LMArena leaderboard.

    Downloads the [cyan]lmarena-ai/leaderboard-dataset[/cyan] (CC-BY-4.0) via the
    Hugging Face datasets API and bins Arena scores into quality tiers.
    No account required. No data sent about your usage.

    [dim]Quality tiers from LMArena (lmarena-ai/leaderboard-dataset, CC-BY-4.0).[/dim]
    [dim]Your data never leaves your machine.[/dim]
    """
    from datetime import date

    from frugon._progress import progress_reporter
    from frugon.quality import (
        _HF_BASE_URL,
        _QUALITY_JSON,
        QualityUpdateError,
        fetch_and_update_quality,
    )

    # Live progress (stderr only): a spinner reassures the user the network
    # fetch is in flight.  A no-op under non-TTY / NO_COLOR.  The success/error
    # result lines stay on stdout, unchanged.
    try:
        with progress_reporter(no_progress=False) as progress:
            with progress.spinner("Updating quality table\u2026"):
                result = fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=_QUALITY_JSON,
                    today_date_str=date.today().isoformat(),
                )
    except QualityUpdateError as exc:
        rprint(f"[red]quality update failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    rprint(
        f"[bold green]\u2713[/bold green] quality.json updated -- "
        f"[bold]{result['models_synced']}[/bold] models, "
        f"dated [cyan]{date.today().isoformat()}[/cyan]."
    )
    rprint("[dim]Quality tiers from LMArena (lmarena-ai/leaderboard-dataset, CC-BY-4.0).[/dim]")
    rprint(f"[dim]{PRIVACY_LINE}[/dim]")


# ---------------------------------------------------------------------------
# update command — runs pricing update AND quality update in one step
# ---------------------------------------------------------------------------
@app.command()
def update() -> None:
    """Refresh both the pricing table and quality tier table from upstream sources.

    Equivalent to running [cyan]frugon pricing update[/cyan] followed by
    [cyan]frugon quality update[/cyan] in one step.

    No account required. No data sent about your usage.

    [dim]Your data never leaves your machine.[/dim]
    """
    from datetime import date

    from frugon._progress import progress_reporter
    from frugon.pricing import (
        _LITELLM_REGISTRY_URL,
        _PRICING_JSON,
        PricingUpdateError,
        fetch_and_update_pricing,
    )
    from frugon.quality import (
        _HF_BASE_URL,
        _QUALITY_JSON,
        QualityUpdateError,
        fetch_and_update_quality,
    )

    today = date.today().isoformat()

    # --- Pricing ---
    try:
        with progress_reporter(no_progress=False) as progress:
            with progress.spinner("Updating pricing table…"):
                pricing_result = fetch_and_update_pricing(
                    registry_url=_LITELLM_REGISTRY_URL,
                    output_path=_PRICING_JSON,
                    today_date_str=today,
                )
    except PricingUpdateError as exc:
        rprint(f"[red]pricing update failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    rprint(
        f"[bold green]✓[/bold green] pricing.json updated — "
        f"[bold]{pricing_result['models_synced']}[/bold] models, "
        f"dated [cyan]{today}[/cyan]."
    )

    # --- Quality ---
    try:
        with progress_reporter(no_progress=False) as progress:
            with progress.spinner("Updating quality table…"):
                quality_result = fetch_and_update_quality(
                    hf_base_url=_HF_BASE_URL,
                    output_path=_QUALITY_JSON,
                    today_date_str=today,
                )
    except QualityUpdateError as exc:
        rprint(f"[red]quality update failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    rprint(
        f"[bold green]✓[/bold green] quality.json updated — "
        f"[bold]{quality_result['models_synced']}[/bold] models, "
        f"dated [cyan]{today}[/cyan]."
    )
    rprint(f"[dim]{PRIVACY_LINE}[/dim]")


# ---------------------------------------------------------------------------
# Console script entry point
# ---------------------------------------------------------------------------


def _force_utf8_streams() -> None:
    """Make stdout/stderr encode as UTF-8 regardless of the OS console default.

    Windows consoles default to a legacy code page (commonly cp1252) that cannot
    encode the glyphs frugon prints — the ``->`` arrow in routing splits, the
    middle dot and minus sign in summaries — so a plain ``analyze`` run dies
    mid-render with ``UnicodeEncodeError``.  CI never catches this: piped/captured
    output is UTF-8 by default, and only an interactive legacy console triggers
    it.  Reconfiguring to UTF-8 is the cross-platform contract (project rule §7) —
    modern terminals render it correctly and every glyph is encodable.  The
    ``errors="backslashreplace"`` backstop makes that absolute: even a malformed
    lone surrogate degrades to an escape sequence rather than raising, so output
    encoding can never be the reason a run dies.

    Defensive by design: ``reconfigure`` exists on 3.7+ ``TextIOWrapper`` but the
    stream may be a plain buffer (pytest capture, redirected pipe) without it, and
    reconfigure can raise on a detached stream.  Both are swallowed — output
    encoding must never be the reason a run fails.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            pass


def main() -> None:
    """Entry point for the frugon console script.

    Wraps the Typer app so that typer.Exit(code=N) reliably propagates to the
    OS exit code on all Typer + Click version combinations.  Click's
    standalone_mode raises SystemExit internally; we re-exit explicitly to
    ensure the code is not swallowed by any intermediary wrapper.
    """
    _force_utf8_streams()
    try:
        app()
    except SystemExit as exc:
        sys.exit(exc.code if exc.code is not None else 0)
