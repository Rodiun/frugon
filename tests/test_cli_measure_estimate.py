"""CLI surface tests for the pre-run estimate + confirm gate (Feature 3).

A large ``--measure`` run (>30 planned provider calls) shows a cost estimate
before sampling.  On a TTY it additionally asks ``Proceed?`` (skippable with
``--yes``); in a pipe / CI it prints the estimate and proceeds (never hangs).
A small run (≤30 calls) stays frictionless — nothing extra is printed.

All tests stub the measure engine so NO provider call is made; the estimate is
real (computed from the log records' own token counts).
"""

from __future__ import annotations

import json
import pathlib
import re
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from frugon import cli, measure
from frugon.cli import app
from frugon.measure import MeasureEstimate, MeasureResult

from .conftest import help_text

runner = CliRunner()

_WIDE_ENV = {"COLUMNS": "200", "TERM": "dumb", "OPENAI_API_KEY": "sk-test-not-real"}


def _estimate_line(output: str) -> str:
    """Return the single ``About to make …`` pre-run ESTIMATE line from *output*.

    The em-dash ban these tests assert is a property of the ESTIMATE line only —
    its separators are colon/period, never an em-dash (which reads as a minus
    sign next to the ``×``/``+``/digits).  It is NOT a property of the whole
    output: the cost-analysis report that prints ABOVE the estimate legitimately
    carries an em-dash in its dim "Upper bound … saves ~X% — run with --verbose
    for detail" note (a long-standing, intentional separator in the report body).
    Pinning the no-em-dash check to this one line mirrors how
    ``test_large_run_tty_decline_aborts`` already isolates the estimate by its
    exact substring instead of scanning the whole panel.
    """
    for line in output.splitlines():
        if "About to make" in line:
            return line
    raise AssertionError(f"no 'About to make …' estimate line in output:\n{output}")


def _priced_row(model: str = "gpt-4o", prompt: str = "classify: spam?") -> dict[str, object]:
    return {
        "model": model,
        "request": {"messages": [{"role": "user", "content": prompt}]},
        "response": {"choices": [{"message": {"content": "no"}}]},
        "usage": {"prompt_tokens": 20, "completion_tokens": 3},
    }


def _write_log(path: pathlib.Path, rows: int) -> pathlib.Path:
    # sample_records dedups by prompt content, so each row needs a unique
    # prompt for the estimate's planned-call arithmetic to reflect the user's
    # ``--samples`` instead of collapsing to the unique-prompt count.  These
    # tests project costs for a real-shape log, where every record IS distinct.
    data = [_priced_row(prompt=f"classify: spam? id={i}") for i in range(rows)]
    path.write_text("\n".join(json.dumps(r) for r in data) + "\n", encoding="utf-8")
    return path


def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the [measure] prerequisites + engine so no provider call is made."""
    monkeypatch.setattr(measure, "_import_litellm", lambda: object())
    monkeypatch.setattr(measure, "verify_measure_prerequisites", lambda *_a, **_k: None)
    monkeypatch.setattr("frugon.report.render_quality_terminal", lambda *_a, **_k: None)

    def _fake_run_measure(*_a: object, **_k: object) -> MeasureResult:
        return MeasureResult(
            samples_requested=1,
            samples_taken=1,
            current_model="gpt-4o",
            candidates=["gpt-4o-mini"],
            comparisons=[],
        )

    monkeypatch.setattr(measure, "run_measure", _fake_run_measure)


# ---------------------------------------------------------------------------
# ≤30 calls — silent
# ---------------------------------------------------------------------------


def test_small_run_prints_no_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _stub_engine(monkeypatch)
    # 6 records, 1 candidate, no judge → min(10,6)×2 = 12 ≤ 30 → frictionless.
    log = _write_log(tmp_path / "log.jsonl", rows=6)
    result = runner.invoke(
        app,
        ["analyze", str(log), "--measure", "--candidates", "gpt-4o-mini", "--no-progress"],
        env=_WIDE_ENV,
    )
    assert result.exit_code == 0, result.output
    assert "About to make" not in result.output


# ---------------------------------------------------------------------------
# >30 calls — non-TTY prints estimate and proceeds (no prompt, no hang)
# ---------------------------------------------------------------------------


def test_large_run_non_tty_prints_estimate_and_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _stub_engine(monkeypatch)
    # 20 records × 2 candidates = min(20,20)×3 = 60 planned calls > 30.  The
    # CliRunner stdin is NOT a TTY, so it must print + proceed, never prompt.
    log = _write_log(tmp_path / "log.jsonl", rows=20)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--measure",
            "--candidates",
            "gpt-4o-mini,gpt-4o",
            "--samples",
            "20",
            "--no-progress",
        ],
        env=_WIDE_ENV,
    )
    assert result.exit_code == 0, result.output
    # No-judge form: the count is EXACT (no ``~``) and the sampling models are
    # stated INLINE in parens — no redundant "60 calls: 60 to sample" — then the
    # cost in its own sentence (period separator, never an em-dash).
    assert (
        "About to make 60 provider calls "
        "(20 prompts × 3 models: baseline + 2 candidates). "
        "Estimated cost ~$"
    ) in result.output
    assert "on your keys." in result.output
    # No "to sample" leg-label in the no-judge form — the models are inline.
    assert "to sample" not in result.output
    # No judge in this run → no judge clause.
    assert "to judge" not in result.output
    # ESTIMATE-LINE separators are colon/period, never an em-dash (reads as a
    # minus sign).  Scoped to the estimate line: the report body above it
    # legitimately carries one in its dim "Upper bound … —" note.
    assert "—" not in _estimate_line(result.output)
    # No confirm prompt in non-TTY mode.
    assert "Proceed?" not in result.output


def test_large_run_estimate_names_judge_when_judge_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _stub_engine(monkeypatch)
    # 20 records, 1 candidate, judge ON → 20×(1+1) + 20×1 = 60 calls.
    log = _write_log(tmp_path / "log.jsonl", rows=20)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--measure",
            "--judge",
            "--candidates",
            "gpt-4o-mini",
            "--samples",
            "20",
            "--no-progress",
        ],
        env=_WIDE_ENV,
    )
    assert result.exit_code == 0, result.output
    # Judge form: exact count (no ``~``) + a colon, then both legs, then the
    # worst-case pointwise-check clause, then the cost as its own sentence.
    # Both summands reconcile to 60: 40 sample + 20 judge.  The check clause is
    # "up to" — 20 prompts × (1 baseline + 1 candidate) = 40 — since it only
    # fires on an actual TIE, unknowable before the run.  The longer sentence
    # now sits right at the 200-column test width, so Rich may word-wrap it
    # (a real newline replacing the space it broke at) — collapse whitespace
    # runs before matching so the assertion is robust to where exactly that
    # wrap lands, without weakening what it actually checks.
    flat = re.sub(r"\s+", " ", result.output)
    assert (
        "About to make 60 provider calls: "
        "40 to sample (20 prompts × 2 models: baseline + 1 candidate)"
        " + 20 to judge (20 prompts × 1 candidate), "
        "up to 40 more to check ties for shared failure. "
        "Estimated cost ~$"
    ) in flat
    # The default judge model (gpt-4.1) is priced against real pricing.json, so
    # the cost clause carries the matching dollar CEILING: base estimate PLUS
    # the worst-case pointwise-check cost, stated "up to ~$Y" alongside the
    # base "~$X" — not asserting exact digits here (that is the job of the
    # unit-level estimate_measure_cost/render tests), only that both figures
    # and the reconciling wording are present.
    assert "on your keys, up to ~$" in flat
    assert "if every judged pair ties." in flat
    # ESTIMATE-LINE separators are colon/period only — never an em-dash (reads as
    # a minus sign).  Scoped to the estimate line: the report body above it
    # legitimately carries one in its dim "Upper bound … —" note.
    assert "—" not in _estimate_line(result.output)


# ---------------------------------------------------------------------------
# >30 calls — TTY prompts; decline aborts, accept proceeds; --yes skips
# ---------------------------------------------------------------------------


def test_large_run_tty_decline_aborts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _stub_engine(monkeypatch)
    # Force the TTY branch and a declined confirmation.
    monkeypatch.setattr("frugon.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr("typer.confirm", lambda *_a, **_k: False)

    log = _write_log(tmp_path / "log.jsonl", rows=20)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--measure",
            "--candidates",
            "gpt-4o-mini,gpt-4o",
            "--samples",
            "20",
            "--no-progress",
        ],
        env=_WIDE_ENV,
    )
    # Clean abort (exit 0) — no provider call made.
    assert result.exit_code == 0, result.output
    # No-judge form: exact count, models inline, no em-dash IN THE ESTIMATE LINE
    # (the abort message below legitimately uses one, so we pin the estimate by
    # its exact substring rather than scanning the whole output here).
    assert (
        "About to make 60 provider calls "
        "(20 prompts × 3 models: baseline + 2 candidates)."
    ) in result.output
    assert "Aborted" in result.output


def test_large_run_tty_accept_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _stub_engine(monkeypatch)
    confirmed: list[bool] = []

    def _confirm(*_a: object, **_k: object) -> bool:
        confirmed.append(True)
        return True

    monkeypatch.setattr("frugon.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr("typer.confirm", _confirm)

    log = _write_log(tmp_path / "log.jsonl", rows=20)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--measure",
            "--candidates",
            "gpt-4o-mini,gpt-4o",
            "--samples",
            "20",
            "--no-progress",
        ],
        env=_WIDE_ENV,
    )
    assert result.exit_code == 0, result.output
    assert confirmed == [True]  # the prompt was shown and accepted
    assert "Aborted" not in result.output


def test_large_run_yes_flag_skips_prompt_on_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    _stub_engine(monkeypatch)
    prompted: list[bool] = []
    monkeypatch.setattr("frugon.cli._stdin_is_tty", lambda: True)
    monkeypatch.setattr(
        "typer.confirm", lambda *_a, **_k: prompted.append(True) or True
    )

    log = _write_log(tmp_path / "log.jsonl", rows=20)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--measure",
            "--yes",
            "--candidates",
            "gpt-4o-mini,gpt-4o",
            "--samples",
            "20",
            "--no-progress",
        ],
        env=_WIDE_ENV,
    )
    assert result.exit_code == 0, result.output
    # --yes shows the estimate but NEVER calls typer.confirm.  No-judge form:
    # exact count, models inline, no em-dash in the ESTIMATE LINE (the report body
    # above it legitimately carries one in its dim "Upper bound … —" note).
    assert (
        "About to make 60 provider calls "
        "(20 prompts × 3 models: baseline + 2 candidates)."
    ) in result.output
    assert "—" not in _estimate_line(result.output)
    assert prompted == []


# ---------------------------------------------------------------------------
# --help advertises the new flags
# ---------------------------------------------------------------------------


def test_help_advertises_yes_and_preview_flags() -> None:
    out = help_text("analyze")
    assert "--yes" in out
    assert "--preview-chars" in out
    assert "--no-truncate" in out


# ---------------------------------------------------------------------------
# Estimate-string shape — direct renderer unit tests (singular + unpriced legs).
#
# These exercise ``_render_measure_estimate`` straight from a ``MeasureEstimate``
# so the singular grammar and the "cost unpriceable" honesty branch are pinned
# without depending on the pricing table.  Every assertion forbids the em-dash
# (which was reported as reading like a minus sign next to the × / + / digits)
# and checks the colon/period separators + arithmetic reconciliation.
# ---------------------------------------------------------------------------


def _render(estimate: MeasureEstimate, monkeypatch: pytest.MonkeyPatch) -> str:
    """Render one estimate and return the line with Rich markup stripped."""
    captured: list[str] = []
    monkeypatch.setattr(cli, "rprint", lambda s="": captured.append(str(s)))
    cli._render_measure_estimate(estimate)
    assert len(captured) == 1
    return captured[0].replace("[dim]", "").replace("[/dim]", "").strip()


def test_render_estimate_singular_prompt_and_candidate_with_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1 prompt × 1 candidate + judge → 1×(1+1) sample + 1×1 judge = 3.
    line = _render(
        MeasureEstimate(
            planned_calls=3,
            estimated_cost=Decimal("0.01"),
            unpriced_models=[],
            n_prompts=1,
            n_candidates=1,
            use_judge=True,
        ),
        monkeypatch,
    )
    assert line == (
        "About to make 3 provider calls: "
        "2 to sample (1 prompt × 2 models: baseline + 1 candidate)"
        " + 1 to judge (1 prompt × 1 candidate). "
        "Estimated cost ~$0.01 on your keys."
    )
    assert "—" not in line  # no minus-sign-looking separator
    # Singular grammar — neither noun is pluralised.
    assert "prompts" not in line
    assert "candidates" not in line


def test_render_estimate_unpriced_models_with_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No target model priceable → the cost sentence is replaced honestly, but the
    # exact call count + the colon breakdown still render (and still reconcile).
    line = _render(
        MeasureEstimate(
            planned_calls=150,
            estimated_cost=None,
            unpriced_models=["weird-model", "also-weird"],
            n_prompts=50,
            n_candidates=1,
            use_judge=True,
        ),
        monkeypatch,
    )
    assert line == (
        "About to make 150 provider calls: "
        "100 to sample (50 prompts × 2 models: baseline + 1 candidate)"
        " + 50 to judge (50 prompts × 1 candidate). "
        "Estimated cost unavailable for weird-model, also-weird."
    )
    assert "—" not in line


def test_render_estimate_unpriced_models_no_judge_uses_inline_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No-judge + unpriceable: inline models (no colon breakdown) AND the honest
    # cost-unavailable sentence.
    line = _render(
        MeasureEstimate(
            planned_calls=100,
            estimated_cost=None,
            unpriced_models=["weird-model"],
            n_prompts=50,
            n_candidates=1,
            use_judge=False,
        ),
        monkeypatch,
    )
    assert line == (
        "About to make 100 provider calls "
        "(50 prompts × 2 models: baseline + 1 candidate). "
        "Estimated cost unavailable for weird-model."
    )
    assert "—" not in line
    # No-judge inline form omits both leg labels.
    assert "to sample" not in line
    assert "to judge" not in line
