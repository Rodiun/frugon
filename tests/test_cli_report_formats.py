"""Tests for multi-format report output and the FRUGON_REPORT_PATH env var.

Two features, one analysis pass:

  * F1 — the extension on ``--report`` chooses the format.  ``NAME.md`` writes
    Markdown only; ``NAME.html`` writes HTML only (styled per ``--report-style``);
    a ``NAME`` with NO recognised extension is a PREFIX that emits the full set —
    ``NAME.md``, ``NAME.v1.html``, ``NAME.v2.html`` — all rendered from the SAME
    in-memory result (no recompute, no extra provider call).

  * F2 — ``FRUGON_REPORT_PATH`` is a saved default report path (opt-in, mirrors
    the API-key env-var pattern).  ``--report`` overrides it; ``--no-report``
    disables it for one run; with neither set the run is terminal-only.

The prefix-vs-single content-parity is proven directly: a prefix run's
``NAME.md`` / ``NAME.v1.html`` / ``NAME.v2.html`` are byte-identical to the
single-format renders of the same result, so the one-pass fan-out can never
drift from the per-format path.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from typer.testing import CliRunner

from frugon.cli import app
from frugon.cost import analyze_records, iter_records
from frugon.report import (
    render_html,
    render_html_v2,
    render_markdown_v2,
    report_paths_for,
    write_reports,
)

runner = CliRunner()

_TERMINAL_ENV = {"COLUMNS": "200", "TERM": "dumb", "NO_COLOR": "1"}


def _priced_row(model: str) -> dict[str, object]:
    return {
        "model": model,
        "request": {"messages": [{"role": "user", "content": "classify: spam?"}]},
        "response": {"choices": [{"message": {"content": "no"}}]},
        "usage": {"prompt_tokens": 20, "completion_tokens": 3},
    }


def _write_log(path: pathlib.Path) -> pathlib.Path:
    rows = [_priced_row("gpt-4-turbo") for _ in range(6)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _analyze(log: pathlib.Path):  # type: ignore[no-untyped-def]
    records, skipped = iter_records(log)
    return analyze_records(list(records), skipped_malformed=skipped, split_routing=True)


# ---------------------------------------------------------------------------
# report_paths_for — pure path arithmetic behind the heuristic
# ---------------------------------------------------------------------------


def test_report_paths_for_md_target_is_itself(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "r.md"
    assert report_paths_for(target) == [target]


def test_report_paths_for_html_target_is_itself(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "r.html"
    assert report_paths_for(target) == [target]


def test_report_paths_for_prefix_emits_three_formats(tmp_path: pathlib.Path) -> None:
    prefix = tmp_path / "r"
    assert report_paths_for(prefix) == [
        tmp_path / "r.md",
        tmp_path / "r.v1.html",
        tmp_path / "r.v2.html",
    ]


def test_report_paths_for_dotted_prefix_appends_not_replaces(
    tmp_path: pathlib.Path,
) -> None:
    """A prefix that itself contains a dot (no recognised ext) gains the format
    suffix rather than having its text replaced."""
    prefix = tmp_path / "report.2026"
    assert report_paths_for(prefix) == [
        tmp_path / "report.2026.md",
        tmp_path / "report.2026.v1.html",
        tmp_path / "report.2026.v2.html",
    ]


# ---------------------------------------------------------------------------
# write_reports — one pass, every requested format, content parity
# ---------------------------------------------------------------------------


def test_write_reports_md_only(tmp_path: pathlib.Path) -> None:
    result = _analyze(_write_log(tmp_path / "log.jsonl"))
    target = tmp_path / "out.md"
    written = write_reports(result, target)
    assert written == [target]
    assert target.exists()
    assert not (tmp_path / "out.v1.html").exists()
    assert not (tmp_path / "out.v2.html").exists()


def test_write_reports_html_only(tmp_path: pathlib.Path) -> None:
    result = _analyze(_write_log(tmp_path / "log.jsonl"))
    target = tmp_path / "out.html"
    written = write_reports(result, target)
    assert written == [target]
    assert target.exists()
    assert not (tmp_path / "out.md").exists()


def test_write_reports_prefix_emits_full_set(tmp_path: pathlib.Path) -> None:
    result = _analyze(_write_log(tmp_path / "log.jsonl"))
    prefix = tmp_path / "out"
    written = write_reports(result, prefix)
    assert written == [
        tmp_path / "out.md",
        tmp_path / "out.v1.html",
        tmp_path / "out.v2.html",
    ]
    for p in written:
        assert p.exists()


def test_prefix_outputs_byte_identical_to_single_format(tmp_path: pathlib.Path) -> None:
    """The one-pass prefix fan-out matches the per-format single renders exactly.

    Proves md/html-v1/html-v2 from the prefix path are byte-for-byte what the
    single-format renderers produce for the SAME result — the formats can never
    silently drift from the dedicated render path.
    """
    result = _analyze(_write_log(tmp_path / "log.jsonl"))

    # One-pass prefix set.
    write_reports(result, tmp_path / "multi")

    # Per-format single renders of the same result.
    render_markdown_v2(result, tmp_path / "single.md")
    render_html(result, tmp_path / "single.v1.html")
    render_html_v2(result, tmp_path / "single.v2.html")

    assert (tmp_path / "multi.md").read_bytes() == (tmp_path / "single.md").read_bytes()
    assert (tmp_path / "multi.v1.html").read_bytes() == (
        tmp_path / "single.v1.html"
    ).read_bytes()
    assert (tmp_path / "multi.v2.html").read_bytes() == (
        tmp_path / "single.v2.html"
    ).read_bytes()


def test_html_target_honours_report_style(tmp_path: pathlib.Path) -> None:
    """A single .html target renders v1 vs v2 per the report_style argument."""
    result = _analyze(_write_log(tmp_path / "log.jsonl"))

    write_reports(result, tmp_path / "v1.html", report_style="v1")
    write_reports(result, tmp_path / "v2.html", report_style="v2")
    render_html(result, tmp_path / "ref_v1.html")
    render_html_v2(result, tmp_path / "ref_v2.html")

    assert (tmp_path / "v1.html").read_bytes() == (tmp_path / "ref_v1.html").read_bytes()
    assert (tmp_path / "v2.html").read_bytes() == (tmp_path / "ref_v2.html").read_bytes()


# ---------------------------------------------------------------------------
# CLI F1 — the extension heuristic through the real command
# ---------------------------------------------------------------------------


def test_cli_prefix_writes_all_three_and_reports_each(tmp_path: pathlib.Path) -> None:
    prefix = tmp_path / "r"
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(prefix), "--no-progress"],
        env=_TERMINAL_ENV,
    )
    assert res.exit_code == 0, res.output
    for suffix in (".md", ".v1.html", ".v2.html"):
        assert (tmp_path / f"r{suffix}").exists()
    # One "Report written to" line per file actually written.
    assert res.output.count("Report written to") == 3


def test_cli_md_target_writes_only_md(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "r.md"
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(target), "--no-progress"],
        env=_TERMINAL_ENV,
    )
    assert res.exit_code == 0, res.output
    assert target.exists()
    assert not (tmp_path / "r.v1.html").exists()
    assert res.output.count("Report written to") == 1


def test_cli_html_target_writes_only_html(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "r.html"
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(target), "--no-progress"],
        env=_TERMINAL_ENV,
    )
    assert res.exit_code == 0, res.output
    assert target.exists()
    assert not (tmp_path / "r.md").exists()
    assert res.output.count("Report written to") == 1


# ---------------------------------------------------------------------------
# CLI F2 — FRUGON_REPORT_PATH precedence matrix
# ---------------------------------------------------------------------------


def test_cli_env_var_writes_full_set_when_no_flag(tmp_path: pathlib.Path) -> None:
    prefix = tmp_path / "saved"
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--no-progress"],
        env={**_TERMINAL_ENV, "FRUGON_REPORT_PATH": str(prefix)},
    )
    assert res.exit_code == 0, res.output
    for suffix in (".md", ".v1.html", ".v2.html"):
        assert (tmp_path / f"saved{suffix}").exists()


def test_cli_report_flag_overrides_env_var(tmp_path: pathlib.Path) -> None:
    env_prefix = tmp_path / "saved"
    flag_target = tmp_path / "other.md"
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(flag_target), "--no-progress"],
        env={**_TERMINAL_ENV, "FRUGON_REPORT_PATH": str(env_prefix)},
    )
    assert res.exit_code == 0, res.output
    assert flag_target.exists()
    # The env-var prefix produced nothing — the flag won.
    assert not (tmp_path / "saved.md").exists()
    assert not (tmp_path / "saved.v1.html").exists()


def test_cli_no_report_beats_env_var(tmp_path: pathlib.Path) -> None:
    prefix = tmp_path / "saved"
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--no-report", "--no-progress"],
        env={**_TERMINAL_ENV, "FRUGON_REPORT_PATH": str(prefix)},
    )
    assert res.exit_code == 0, res.output
    assert "Report written to" not in res.output
    assert list(tmp_path.glob("saved*")) == []


def test_cli_no_env_no_flag_writes_nothing(tmp_path: pathlib.Path) -> None:
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--no-progress"],
        env=_TERMINAL_ENV,
    )
    assert res.exit_code == 0, res.output
    assert "Report written to" not in res.output
    # No files anywhere under tmp_path — the opt-in default never sprays files.
    assert list(tmp_path.iterdir()) == []


def test_cli_no_report_beats_explicit_report_flag(tmp_path: pathlib.Path) -> None:
    """--no-report is the strongest signal: it beats an explicit --report too."""
    target = tmp_path / "r.md"
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(target), "--no-report", "--no-progress"],
        env=_TERMINAL_ENV,
    )
    assert res.exit_code == 0, res.output
    assert not target.exists()
    assert "Report written to" not in res.output


@pytest.mark.parametrize("suffix", [".md", ".v1.html", ".v2.html"])
def test_cli_env_var_full_set_matches_single_renders(
    tmp_path: pathlib.Path, suffix: str
) -> None:
    """The env-var prefix path produces the same bytes as the dedicated renders."""
    prefix = tmp_path / "saved"
    res = runner.invoke(
        app,
        ["analyze", "--demo", "--no-progress"],
        env={**_TERMINAL_ENV, "FRUGON_REPORT_PATH": str(prefix)},
    )
    assert res.exit_code == 0, res.output

    import frugon

    assert frugon.__file__ is not None
    sample = pathlib.Path(frugon.__file__).parent / "data" / "sample_logs.jsonl.gz"
    records, skipped = iter_records(sample)
    result = analyze_records(
        list(records), skipped_malformed=skipped, split_routing=True
    )
    renderer = {
        ".md": render_markdown_v2,
        ".v1.html": render_html,
        ".v2.html": render_html_v2,
    }[suffix]
    ref = tmp_path / f"ref{suffix}"
    renderer(result, ref)
    assert (tmp_path / f"saved{suffix}").read_bytes() == ref.read_bytes()
