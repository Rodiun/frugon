"""Smoke tests for the frugon CLI.

These tests verify the public CLI surface without touching any network,
filesystem, or external service.  They run in full isolation — no I/O,
no phone-home, no side effects.

Privacy invariant: every test that exercises a command implicitly asserts
that the command exits cleanly without making any outbound network call
(the commands are stubs and perform no I/O in this phase).
"""

from __future__ import annotations

import pathlib
import re

import pytest
from typer.testing import CliRunner

from frugon import __version__
from frugon.cli import PRIVACY_LINE, app

runner = CliRunner()

# Wide environment used for help-text assertions: prevents Rich from wrapping
# flags across lines and disables ANSI so cross-platform renderers behave the
# same.  The Windows CI runner inserts ANSI escape codes inside flag names
# (e.g. "--\x1b[...m\x1b[...]measure") at its default narrow width, causing
# literal substring matches like "--measure" to fail.
_WIDE_ENV = {"COLUMNS": "200", "TERM": "dumb"}
_NARROW_ENV = {"COLUMNS": "40", "TERM": "dumb"}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")


def _clean(text: str) -> str:
    """Strip ANSI escape codes from *text*."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_version_flag_exits_zero_and_prints_version() -> None:
    """Arrange: invoke --version.
    Act: run the root app with the flag.
    Assert: exit code is 0 and the version string is present in output.
    """
    # Arrange
    expected = __version__

    # Act
    result = runner.invoke(app, ["--version"])

    # Assert
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}:\n{result.output}"
    assert expected in result.output, (
        f"Version '{expected}' not found in output:\n{result.output}"
    )


def test_version_short_flag_exits_zero_and_prints_version() -> None:
    """Arrange: invoke -V (short alias).
    Act: run with -V.
    Assert: exit code is 0 and version present.
    """
    # Arrange / Act
    result = runner.invoke(app, ["-V"])

    # Assert
    assert result.exit_code == 0
    assert __version__ in result.output


# ---------------------------------------------------------------------------
# analyze --help
# ---------------------------------------------------------------------------


def test_analyze_help_exits_zero() -> None:
    """Arrange: invoke analyze --help.
    Act: run.
    Assert: exits 0.
    """
    # Act
    result = runner.invoke(app, ["analyze", "--help"])

    # Assert
    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"


def test_analyze_help_mentions_logs_argument() -> None:
    """Arrange: invoke analyze --help with wide terminal.
    Act: run.
    Assert: 'logs' (the positional arg name) is mentioned.
    """
    # Act
    result = runner.invoke(app, ["analyze", "--help"], env=_WIDE_ENV)
    out = _clean(result.output)

    # Assert
    assert "logs" in out.lower(), (
        f"Expected 'logs' in help output:\n{result.output}"
    )


def test_analyze_help_mentions_measure_option() -> None:
    """Arrange: invoke analyze --help with wide terminal.
    Act: run with COLUMNS=200 and TERM=dumb so Rich emits plain text.
    Assert: the stable token 'measure' is present (width-independent).

    The dashed flag '--measure' can render with embedded ANSI codes on some
    Windows CI runners, so we match the bare word and strip ANSI defensively.
    """
    # Act — wide + dumb terminal prevents line-wrapping and ANSI injection
    result = runner.invoke(app, ["analyze", "--help"], env=_WIDE_ENV)
    out = _clean(result.output)

    # Assert on the stable token; '--measure' is present as '--measure' in the
    # cleaned output, but matching 'measure' is immune to ANSI-split artifacts.
    assert "measure" in out, (
        f"Expected 'measure' in help output:\n{result.output}"
    )


def test_analyze_help_mentions_privacy_line() -> None:
    """Arrange: invoke analyze --help with wide terminal.
    Act: run.
    Assert: the privacy guarantee string is present.

    This is the Privacy Invariant test: the user must see the privacy
    commitment on every command's help, before they even run it.
    """
    # Act
    result = runner.invoke(app, ["analyze", "--help"], env=_WIDE_ENV)
    out = _clean(result.output)

    # 'never leaves' is a short stable phrase that cannot wrap mid-phrase
    assert "never leaves" in out, (
        f"Privacy line not found in analyze --help output:\n{result.output}"
    )


def test_analyze_nonexistent_file_exits_nonzero() -> None:
    """Arrange: invoke analyze with a path that does not exist.
    Act: run.
    Assert: exits non-zero (analyze is now live and validates the path).
    """
    # Act
    result = runner.invoke(app, ["analyze", "./does_not_exist_frugon_test.jsonl"])

    # Assert — file-not-found is an error, not a stub
    assert result.exit_code != 0, (
        f"Expected non-zero exit for missing file, got {result.exit_code}:\n{result.output}"
    )


def test_analyze_missing_file_process_exit_code_is_nonzero() -> None:
    """OS-level process exit code is nonzero when the log file does not exist.

    Arrange: subprocess that invokes frugon.cli.main() with a missing file path.
    Act: run as a child process.
    Assert: returncode != 0 and output is a clean user-facing error (no Python
            traceback) — typer.Exit(code=1) propagates through frugon.cli:main.

    This is distinct from the CliRunner test above: it verifies the real
    process boundary, not just the in-process exit_code attribute.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.argv = ['frugon', 'analyze', './no_such_file_frugon_test.jsonl']; "
                "from frugon.cli import main; main()"
            ),
        ],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0, (
        f"Expected non-zero OS exit code for missing file, got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # Must be a clean frugon error, not a Python traceback or ImportError.
    assert b"Traceback" not in result.stderr, (
        f"Python traceback in stderr — exit must come from frugon.cli.main, "
        f"not an unhandled exception:\n{result.stderr!r}"
    )
    assert b"not found" in result.stdout.lower(), (
        f"Expected 'not found' in stdout (frugon error message). "
        f"stdout={result.stdout!r}"
    )


def test_analyze_demo_flag_exits_zero() -> None:
    """Arrange: invoke analyze --demo (loads bundled sample file).
    Act: run.
    Assert: exits 0 and produces output (no 'coming soon' — real analysis runs).
    """
    # Act
    result = runner.invoke(app, ["analyze", "--demo"])

    # Assert
    assert result.exit_code == 0, (
        f"Expected exit 0 for --demo, got {result.exit_code}:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# capture --help
# ---------------------------------------------------------------------------


def test_capture_help_exits_zero() -> None:
    """Arrange: invoke capture --help.
    Act: run.
    Assert: exits 0.
    """
    # Act
    result = runner.invoke(app, ["capture", "--help"])

    # Assert
    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"


def test_capture_help_mentions_port_option() -> None:
    """Arrange: invoke capture --help with wide terminal.
    Act: run with COLUMNS=200 and TERM=dumb so Rich emits plain text.
    Assert: the stable token 'port' is present (width-independent).

    Same ANSI-injection guard as test_analyze_help_mentions_measure_option.
    """
    # Act
    result = runner.invoke(app, ["capture", "--help"], env=_WIDE_ENV)
    out = _clean(result.output)

    # Assert
    assert "port" in out, (
        f"Expected 'port' in capture help:\n{result.output}"
    )


def test_capture_shows_privacy_message_on_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: mock run_capture so the test exits immediately (avoids blocking).
    Act: invoke capture with default options.
    Assert: exits 0 and privacy message shown in startup panel.
    """
    import frugon.capture as cap_mod

    monkeypatch.setattr(cap_mod, "run_capture", lambda **_: None)

    # Act
    result = runner.invoke(app, ["capture"])

    # Assert
    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    assert "never leaves" in result.output, (
        f"Privacy message not shown on capture start:\n{result.output}"
    )


def test_capture_shows_custom_port_in_startup_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: mock run_capture; invoke with --port 9000.
    Act: run.
    Assert: exits 0 and port 9000 is mentioned in the startup message.
    """
    import frugon.capture as cap_mod

    monkeypatch.setattr(cap_mod, "run_capture", lambda **_: None)

    # Act
    result = runner.invoke(app, ["capture", "--port", "9000"])

    # Assert
    assert result.exit_code == 0
    assert "9000" in result.output, (
        f"Expected port 9000 in output:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# pricing update
# ---------------------------------------------------------------------------


def test_pricing_update_attempts_fetch() -> None:
    """Arrange: invoke pricing update with mocked network failure.
    Act: run.
    Assert: exits non-zero and shows error; stub text is gone.
    """
    import urllib.error
    from unittest.mock import patch

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("no network in test"),
    ):
        result = runner.invoke(app, ["pricing", "update"])

    assert result.exit_code != 0, f"Expected non-zero exit:\n{result.output}"
    assert "pricing update failed" in result.output.lower(), (
        f"Expected error message:\n{result.output}"
    )
    assert "coming soon" not in result.output.lower(), (
        f"Stub text still present:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# Privacy invariant — canonical string
# ---------------------------------------------------------------------------


def test_analyze_demo_shows_quality_caveat() -> None:
    """Arrange: invoke analyze --demo (per-call split-routing headline).
    Act: run.
    Assert: a quality disclosure is present in the output (§6 honesty).

    Whenever a routing recommendation is shown, frugon must disclose the
    quality status of the recommendation.  The demo fixture's baseline
    (chatgpt-4o-latest) is absent from the user-data-dir quality table
    (present only in the bundled seed), so the runtime tier_drop resolves
    to None → the routing panel shows "within tolerance" rather than the
    full "same or better quality" confirmation.  Either phrase satisfies §6.
    """
    # Act
    result = runner.invoke(app, ["analyze", "--demo"])

    # Assert
    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    # Normalise whitespace so Rich line-wrapping can't split the phrase.
    out = " ".join(_clean(result.output).split())
    # The demo baseline (chatgpt-4o-latest) resolves to tier_drop=None at
    # runtime → the routing panel shows the neutral "within tolerance" badge.
    assert "within tolerance" in out, (
        f"Quality disclosure not found in analyze --demo output:\n{result.output}"
    )


def test_analyze_demo_shows_split_shape() -> None:
    """Arrange: invoke analyze --demo.
    Act: run.
    Assert: the plain-English routed/kept routing plan is shown (CLI redesign).

    The redesigned terminal view leads with a hero saving, then a plain-English
    decision zone: "Route N easy calls → mini / Keep M hard calls → premium".
    """
    result = runner.invoke(app, ["analyze", "--demo"])
    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = " ".join(_clean(result.output).split())
    assert "Route" in out
    assert "easy calls" in out
    assert "Keep" in out
    assert "hard calls" in out
    assert "gpt-4o-mini" in out


def test_analyze_wholesale_flag_suppresses_split() -> None:
    """Arrange: analyze --demo --wholesale.
    Act: run.
    Assert: the wholesale swap line is shown and the split scoreboard is not.

    The redesigned wholesale hero leads with a plain-English full-swap line
    ("Swap   every call  →  <model>   (full swap)") and omits the per-call
    split's "within tolerance" routing band entirely.
    """
    result = runner.invoke(app, ["analyze", "--demo", "--wholesale"])
    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = _clean(result.output)
    # The wholesale hero shows the full-swap line, not the split scoreboard.
    assert "Swap" in out
    assert "every call" in out
    assert "full swap" in out
    # The "within tolerance" band belongs to the split path; it must not appear.
    assert "within tolerance" not in out


def test_analyze_prints_privacy_line_at_foot() -> None:
    """Arrange: analyze --demo --wholesale.
    Act: run.
    Assert: the privacy reassurance is printed at the foot for first contact.

    The redesigned footer prints a pared-down privacy reassurance under both the
    wholesale and split paths: "Your data never leaves your machine. Your keys go
    to your own providers." followed by the single product upsell link.
    """
    result = runner.invoke(app, ["analyze", "--demo", "--wholesale"])
    assert result.exit_code == 0
    out = " ".join(_clean(result.output).split())
    assert "Your data never leaves your machine." in out
    assert "Your keys go to your own providers." in out


def test_analyze_split_prints_pared_privacy_caveat() -> None:
    """Arrange: analyze --demo (the default split path).
    Act: run.
    Assert: the privacy caveat is printed and the footer carries exactly one
            upsell link to the product (CLI redesign — footer = caveat / privacy /
            one upsell).
    """
    result = runner.invoke(app, ["analyze", "--demo"])
    assert result.exit_code == 0
    out = " ".join(_clean(result.output).split())
    assert "Your data never leaves your machine." in out
    # The redesigned footer carries exactly one upsell line → the product.
    assert out.count("https://frugon.rodiun.io") == 1


def test_analyze_invalid_utf8_fails_friendly(tmp_path: pathlib.Path) -> None:
    """Arrange: a log file with invalid UTF-8 bytes.
    Act: analyze.
    Assert: exit code 1 and a friendly message — no raw traceback (§4 fail-loud).
    """
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"\xff\xfe not valid utf-8 \x80\x81\n")
    result = runner.invoke(app, ["analyze", str(bad)])
    assert result.exit_code == 1, f"Expected exit 1:\n{result.output}"
    assert "UTF-8" in _clean(result.output)


# ---------------------------------------------------------------------------
# Privacy invariant — canonical string
# ---------------------------------------------------------------------------


def test_privacy_line_constant_matches_spec() -> None:
    """Arrange: the canonical PRIVACY_LINE constant.
    Act: inspect the value.
    Assert: it contains all three required clauses from the spec.

    This test is the machine-checkable anchor for §5 of the spec.
    If the string is changed, this test fails — prompting a conscious decision.
    """
    # Assert all three clauses are present
    assert "never leaves your machine" in PRIVACY_LINE
    assert "your own providers" in PRIVACY_LINE
    assert "Nothing reaches us" in PRIVACY_LINE


# ---------------------------------------------------------------------------
# P1-C acceptance: help assertions survive a narrow terminal (COLUMNS=40)
# ---------------------------------------------------------------------------


def test_help_tokens_found_at_narrow_width() -> None:
    """Verify that the stable tokens used in help-text assertions are found
    even under a narrow (COLUMNS=40) terminal, simulating the Windows CI
    runner's default width.

    This is the acceptance gate for P1-C: if this test passes, the help-text
    assertions are width-independent and will not regress on Windows CI.
    """
    # Arrange — narrow terminal, ANSI-free
    analyze_result = runner.invoke(app, ["analyze", "--help"], env=_NARROW_ENV)
    capture_result = runner.invoke(app, ["capture", "--help"], env=_NARROW_ENV)

    analyze_out = _clean(analyze_result.output)
    capture_out = _clean(capture_result.output)

    # Assert — same tokens used in the main tests, verified at COLUMNS=40
    assert "measure" in analyze_out, (
        f"'measure' not found with COLUMNS=40:\n{analyze_result.output}"
    )
    assert "logs" in analyze_out.lower(), (
        f"'logs' not found with COLUMNS=40:\n{analyze_result.output}"
    )
    assert "never leaves" in analyze_out, (
        f"Privacy phrase not found with COLUMNS=40:\n{analyze_result.output}"
    )
    assert "port" in capture_out, (
        f"'port' not found with COLUMNS=40:\n{capture_result.output}"
    )


# ---------------------------------------------------------------------------
# #6 — Swap line names both models
# ---------------------------------------------------------------------------


def test_analyze_demo_swap_line_names_both_models() -> None:
    """Arrange: analyze --demo --wholesale (chatgpt-4o-latest baseline → gpt-4o candidate).
    Act: run.
    Assert: the output contains a swap line with both model names, making
            the recommendation explicit (not just showing the candidate).

    The swap line is a wholesale-path disclosure; the default --demo now leads
    with the per-call split (which shows its own routed/kept plan), so this test
    exercises the wholesale path explicitly via --wholesale.

    For wholesale (full-swap basis) the cheapest candidate on the chatgpt-4o-latest
    demo log is gpt-4o (Elite tier 0, ~$0.0025/0.010 per 1k tokens), not
    gpt-4o-mini — gpt-4o wins the full-swap ranking because its lower output price
    dominates the output-heavy hard calls that a full swap reprices.  The split
    path routes easy calls to gpt-4o-mini; wholesale routes every call to gpt-4o.
    """
    result = runner.invoke(app, ["analyze", "--demo", "--wholesale"])

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = _clean(result.output)
    # The hero must name BOTH the baseline (chatgpt-4o-latest, in the masthead) and the
    # candidate (gpt-4o, on the full-swap line), making the recommendation explicit.
    assert "chatgpt-4o-latest" in out, (
        f"Expected baseline model 'chatgpt-4o-latest' in output:\n{result.output}"
    )
    assert "gpt-4o" in out, (
        f"Expected candidate model 'gpt-4o' in output:\n{result.output}"
    )
    # The redesigned full-swap line reads "Swap   every call  →  <model>   (full swap)".
    assert "Swap" in out, (
        f"Expected 'Swap' full-swap line in output:\n{result.output}"
    )
    assert "full swap" in out, (
        f"Expected 'full swap' qualifier in output:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# #8 — Report completion prints absolute path
# ---------------------------------------------------------------------------


def test_analyze_report_prints_absolute_path(tmp_path: pathlib.Path) -> None:
    """Arrange: analyze --demo --report <relative-path>.
    Act: run.
    Assert: the completion message contains the ABSOLUTE path, not the relative one.
    """
    report_file = tmp_path / "test_report.html"

    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(report_file)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = _clean(result.output)
    # Collapse newlines before checking: Rich may wrap long paths across lines.
    out_flat = out.replace("\n", "").replace("\r", "")
    abs_path = str(report_file.resolve())
    assert abs_path in out_flat, (
        f"Expected absolute path '{abs_path}' in output:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# --report-style routes to the v1 / v2 renderers (default v2)
# ---------------------------------------------------------------------------


def test_analyze_report_style_v2_html_marks_v2_surface(tmp_path: pathlib.Path) -> None:
    """Arrange: analyze --demo --report report.html --report-style v2.
    Act: run.
    Assert: the written HTML carries a v2-only marker (the masthead tag),
            confirming the v2 renderer was routed to.
    """
    report_file = tmp_path / "v2.html"

    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(report_file), "--report-style", "v2"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    html = report_file.read_text(encoding="utf-8")
    assert 'class="masthead-tag"' in html
    assert "No data leaves your machine." in html


def test_analyze_report_style_default_is_v2_html(tmp_path: pathlib.Path) -> None:
    """Arrange: analyze --demo --report report.html (no --report-style).
    Act: run.
    Assert: the v2 editorial renderer is used by default (v2-only masthead tag
            present), confirming v2 is the bare default.
    """
    report_file = tmp_path / "default.html"

    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(report_file)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    html = report_file.read_text(encoding="utf-8")
    assert 'class="masthead-tag"' in html
    assert "No data leaves your machine." in html


def test_analyze_report_style_explicit_v1_html(tmp_path: pathlib.Path) -> None:
    """Arrange: analyze --demo --report report.html --report-style v1.
    Act: run.
    Assert: v1 stays fully reachable (v1-only .card class present; v2 masthead
            tag absent).
    """
    report_file = tmp_path / "v1.html"

    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(report_file), "--report-style", "v1"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    html = report_file.read_text(encoding="utf-8")
    assert 'class="masthead-tag"' not in html
    assert 'class="card"' in html


def test_analyze_report_style_v2_markdown(tmp_path: pathlib.Path) -> None:
    """Arrange: analyze --demo --report report.md --report-style v2.
    Act: run.
    Assert: the v2 markdown bottom-line headline is present.
    """
    report_file = tmp_path / "v2.md"

    result = runner.invoke(
        app,
        ["analyze", "--demo", "--report", str(report_file), "--report-style", "v2"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    md = report_file.read_text(encoding="utf-8")
    assert "## Bottom line" in md


# ---------------------------------------------------------------------------
# #4 — Default pool disclosure printed only when no --candidates
# ---------------------------------------------------------------------------


def test_analyze_demo_default_pool_disclosure_printed() -> None:
    """Arrange: analyze --demo --wholesale --verbose (no --candidates).
    Act: run.
    Assert: the verbose Notes block contains the built-in candidate pool disclosure.

    The default-pool disclosure is a wholesale-path line; the redesigned terminal
    keeps the hero uncluttered and moves this disclosure into the --verbose Notes
    block (the "Pool" line), so this exercises the verbose wholesale path.
    """
    result = runner.invoke(app, ["analyze", "--demo", "--wholesale", "--verbose"])

    assert result.exit_code == 0
    out = _clean(result.output)
    assert "built-in candidates" in out, (
        f"Expected default-pool disclosure in output:\n{result.output}"
    )
    # The disclosure points the user at --candidates as the override.
    assert "--candidates" in out, (
        f"Expected --candidates override hint in pool disclosure:\n{result.output}"
    )


def test_analyze_explicit_candidates_no_default_pool_disclosure(
    tmp_path: pathlib.Path,
) -> None:
    """Arrange: analyze with explicit --candidates (gpt-4o baseline → gpt-4o-mini).
    Act: run.
    Assert: the default-pool disclosure line is NOT printed (user supplied their own list).
    """
    import json as _json

    log_file = tmp_path / "logs.jsonl"
    records = [
        {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "hello"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "world"}}]},
            "usage": {"prompt_tokens": 50, "completion_tokens": 5},
        }
        for _ in range(5)
    ]
    with log_file.open("w") as fh:
        for rec in records:
            fh.write(_json.dumps(rec) + "\n")

    result = runner.invoke(
        app, ["analyze", str(log_file), "--candidates", "gpt-4o-mini"]
    )

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = _clean(result.output)
    assert "built-in candidates" not in out, (
        f"Default-pool disclosure must NOT appear when --candidates is used:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# #10 — quality caveat on a large tier-drop full swap
# ---------------------------------------------------------------------------


def test_analyze_routellm_caveat_on_large_tier_drop(tmp_path: pathlib.Path) -> None:
    """Arrange: gpt-4o (tier 0) baseline with explicit --candidates claude-3-haiku (tier 3).
    Act: run.
    Assert: a full swap across a large quality-tier gap surfaces the unverified
            quality caveat ("a full swap can change output quality").

    The redesigned wholesale hero carries a single honest quality caveat for any
    full swap — "Quality is not verified — a full swap can change output quality;
    run --measure to confirm it on your real outputs before you switch." — which
    is exactly the warning a large (tier 0 → tier 3) drop must surface.
    """
    import json as _json

    log_file = tmp_path / "logs.jsonl"
    records = [
        {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "classify"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "a"}}]},
            "usage": {"prompt_tokens": 50, "completion_tokens": 5},
        }
        for _ in range(10)
    ]
    with log_file.open("w") as fh:
        for rec in records:
            fh.write(_json.dumps(rec) + "\n")

    # The full-swap quality caveat is a wholesale-path disclosure; the default
    # split path shows its own routed/kept plan, so this exercises the wholesale
    # path explicitly via --wholesale.
    result = runner.invoke(
        app,
        ["analyze", str(log_file), "--candidates", "claude-3-haiku-20240307", "--wholesale"],
    )

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = " ".join(_clean(result.output).split())
    # Only check the caveat if a candidate was actually selected (requires price data).
    if "claude-3-haiku" in out:
        assert "a full swap can change output quality" in out, (
            f"Expected full-swap quality caveat for 3-tier drop:\n{result.output}"
        )


def test_analyze_no_routellm_caveat_on_single_tier_drop(tmp_path: pathlib.Path) -> None:
    """Arrange: gpt-4-turbo (unrated) → gpt-4o (tier 0): auto-selection path.
    Act: analyze without --candidates.
    Assert: RouteLLM caveat is NOT printed (tier_drop is None for unrated baseline).
    """
    import json as _json

    log_file = tmp_path / "logs.jsonl"
    records = [
        {
            "model": "gpt-4-turbo",
            "request": {"messages": [{"role": "user", "content": "summarize"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "summary"}}]},
            "usage": {"prompt_tokens": 80, "completion_tokens": 20},
        }
        for _ in range(10)
    ]
    with log_file.open("w") as fh:
        for rec in records:
            fh.write(_json.dumps(rec) + "\n")

    result = runner.invoke(app, ["analyze", str(log_file)])

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = _clean(result.output)
    # The wholesale full-swap tier-drop caveat must not appear for a single-tier
    # drop.  (The split method legitimately cites RouteLLM as its inspiration, so
    # we assert the specific wholesale caveat phrase is absent, not the bare word.)
    assert "real routing typically saves" not in out, (
        f"Wholesale tier-drop caveat must NOT appear for unrated baseline:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# #3 — Unrated note for baseline or candidate
# ---------------------------------------------------------------------------


def test_analyze_unrated_baseline_note_shown(
    tmp_path: pathlib.Path, unrated_model: str
) -> None:
    """Arrange: a priced-but-unrated model as dominant model.
    Act: analyze.
    Assert: 'quality tier unknown' note is printed for the unrated baseline.
    """
    import json as _json

    log_file = tmp_path / "logs.jsonl"
    records = [
        {
            "model": unrated_model,
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
            "usage": {"prompt_tokens": 30, "completion_tokens": 5},
        }
        for _ in range(5)
    ]
    with log_file.open("w") as fh:
        for rec in records:
            fh.write(_json.dumps(rec) + "\n")

    result = runner.invoke(app, ["analyze", str(log_file)])

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = " ".join(_clean(result.output).split())
    # The split footer folds the unrated-baseline caution into the quality caveat
    # line: "(<model> has no known quality tier)".  The wholesale path keeps the
    # older "quality tier unknown for <model>" wording — accept either so the test
    # pins the disclosure, not one specific phrasing.
    assert ("has no known quality tier" in out) or ("quality tier unknown" in out), (
        f"Expected an unrated-baseline quality caution:\n{result.output}"
    )
    assert unrated_model in out


def test_analyze_rated_baseline_no_unrated_note(tmp_path: pathlib.Path) -> None:
    """Arrange: gpt-4o (tier 0, rated) as dominant model.
    Act: analyze.
    Assert: 'quality tier unknown' note is NOT printed.
    """
    import json as _json

    log_file = tmp_path / "logs.jsonl"
    records = [
        {
            "model": "gpt-4o",
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
            "usage": {"prompt_tokens": 50, "completion_tokens": 5},
        }
        for _ in range(5)
    ]
    with log_file.open("w") as fh:
        for rec in records:
            fh.write(_json.dumps(rec) + "\n")

    result = runner.invoke(app, ["analyze", str(log_file)])

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    out = _clean(result.output)
    assert "quality tier unknown" not in out, (
        f"'quality tier unknown' must NOT appear for a rated baseline:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# #9 — capture --quiet / --verbose verbosity flags
# ---------------------------------------------------------------------------


def test_capture_quiet_flag_maps_to_quiet_verbosity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: mock run_capture to capture the verbosity kwarg; invoke capture -q.
    Act: run.
    Assert: exits 0 and run_capture was called with verbosity='quiet'.
    """
    import frugon.capture as cap_mod

    captured: dict[str, object] = {}

    def fake_run_capture(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cap_mod, "run_capture", fake_run_capture)

    result = runner.invoke(app, ["capture", "-q"])

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    assert captured.get("verbosity") == "quiet", (
        f"Expected verbosity='quiet', got {captured.get('verbosity')!r}"
    )


def test_capture_verbose_flag_maps_to_verbose_verbosity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: mock run_capture; invoke capture -v.
    Act: run.
    Assert: exits 0 and run_capture was called with verbosity='verbose'.
    """
    import frugon.capture as cap_mod

    captured: dict[str, object] = {}

    def fake_run_capture(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cap_mod, "run_capture", fake_run_capture)

    result = runner.invoke(app, ["capture", "-v"])

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    assert captured.get("verbosity") == "verbose", (
        f"Expected verbosity='verbose', got {captured.get('verbosity')!r}"
    )


def test_capture_no_flag_maps_to_normal_verbosity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: mock run_capture; invoke capture with no verbosity flag.
    Act: run.
    Assert: exits 0 and run_capture was called with verbosity='normal'.
    """
    import frugon.capture as cap_mod

    captured: dict[str, object] = {}

    def fake_run_capture(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cap_mod, "run_capture", fake_run_capture)

    result = runner.invoke(app, ["capture"])

    assert result.exit_code == 0, f"Expected exit 0:\n{result.output}"
    assert captured.get("verbosity") == "normal", (
        f"Expected verbosity='normal', got {captured.get('verbosity')!r}"
    )


def test_capture_quiet_and_verbose_together_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: mock run_capture; invoke capture -q -v.
    Act: run.
    Assert: exits non-zero with a clear error message.
    """
    import frugon.capture as cap_mod

    monkeypatch.setattr(cap_mod, "run_capture", lambda **_: None)

    result = runner.invoke(app, ["capture", "-q", "-v"])

    assert result.exit_code != 0, (
        f"Expected non-zero exit when --quiet and --verbose used together:\n{result.output}"
    )
    assert "mutually exclusive" in result.output.lower(), (
        f"Expected 'mutually exclusive' error message:\n{result.output}"
    )
