"""Tests for the cross-platform, provider-agnostic 'provider key needed' hint.

The ``--measure`` key panel must (a) name the key generically as an AI-provider
key rather than demanding OpenAI specifically, and (b) print the shell command
that actually works on the host OS — PowerShell's ``$env:`` on Windows, not the
Unix ``export`` that errors there.
"""

from __future__ import annotations

import sys

import pytest

import frugon.cli as cli


def test_env_set_hint_windows_uses_powershell_syntax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: pretend we're on Windows.
    monkeypatch.setattr(sys, "platform", "win32")

    # Act
    hint = cli._env_set_hint("OPENAI_API_KEY")

    # Assert: PowerShell form, quoted so the value is a literal (an unquoted
    # bareword after `=` is parsed as a command in PowerShell) — never the Unix
    # `export` that errors in PowerShell.  The "<your-key>" placeholder makes it
    # obvious the surrounding quotes must be kept when pasting a real key.
    assert hint == '$env:OPENAI_API_KEY="<your-key>"'
    assert "export" not in hint


def test_env_set_hint_posix_uses_export(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr(sys, "platform", "linux")

    # Act / Assert: quoted so "<your-key>"'s angle brackets are literal in bash
    # (they would otherwise be redirection operators), and consistent with the
    # PowerShell form.
    assert cli._env_set_hint("OPENAI_API_KEY") == 'export OPENAI_API_KEY="<your-key>"'


def test_missing_key_panel_is_provider_agnostic_and_platform_aware(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange: a missing-key error naming OPENAI_API_KEY, rendered on Windows.
    monkeypatch.setattr(sys, "platform", "win32")

    class _Exc(Exception):
        missing_vars = ["OPENAI_API_KEY"]
        suggestions: dict[str, str] = {}

    # Act
    cli._render_missing_key(_Exc(), measured_models=["gpt-4o-mini"])
    out = capsys.readouterr().out

    # Assert: generic "AI provider" framing (not an OpenAI-only demand), the
    # specific key still named, and the PowerShell set syntax — never `export`.
    assert "provider" in out.lower()
    assert "OPENAI_API_KEY" in out
    assert "$env:" in out
    assert "export " not in out
