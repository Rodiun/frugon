"""Privacy invariant tests — prove analyze is 100% local.

Monkeypatches socket.socket.connect and socket.create_connection to raise on
any outbound connection attempt.  The full analyze path must run successfully
(exit 0) with the network layer denied, proving zero network calls are made
during cost analysis.

The local-only privacy invariant: cost analysis is fully local — no LLM, no
network.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module-level imports — intentionally before the socket-deny fixture.
#
# If tokencost or any frugon module makes a network call during import or
# lazy initialisation, it happens here (before the deny fixture activates).
# The fixture then seals the network at test runtime: any call that occurs
# during the analysis itself is what this suite is designed to catch.
# ---------------------------------------------------------------------------
import tokencost  # noqa: F401 — ensures tokencost is initialised now
from typer.testing import CliRunner

from frugon.cli import app

# ---------------------------------------------------------------------------
# Fixture: deny all outbound socket connections
# ---------------------------------------------------------------------------


@pytest.fixture
def deny_sockets(monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[type-arg]
    """Raise AssertionError on any outbound socket connect or DNS lookup.

    Covers both direct socket.socket.connect calls and the convenience
    socket.create_connection helper used by http.client and similar, plus
    socket.getaddrinfo so DNS resolution attempts are blocked too.
    """

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Network call detected during analysis — analyze must be 100% local. "
            f"Call args: {args!r}"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


# ---------------------------------------------------------------------------
# Local-only privacy invariant tests
# ---------------------------------------------------------------------------


def test_deny_sockets_blocks_connect_helpers_and_dns(deny_sockets: None) -> None:
    """Arrange: deny all outbound socket paths.
    Act/Assert: direct connects, create_connection, and getaddrinfo all fail.
    """
    with pytest.raises(AssertionError):
        socket.socket().connect(("example.com", 443))

    with pytest.raises(AssertionError):
        socket.create_connection(("example.com", 443))

    with pytest.raises(AssertionError):
        socket.getaddrinfo("example.com", 443)


def test_analyze_demo_makes_no_network_calls(deny_sockets: None) -> None:
    """Arrange: deny all outbound socket connections.
    Act: run frugon analyze --demo (full engine path with bundled sample file).
    Assert: exits 0 — the complete analysis ran locally with the network denied.

    This is the local-only privacy invariant test.  If any part of the cost
    engine, tokenizer, or pricing lookup makes a runtime network call, the
    deny fixture raises AssertionError and the test fails with a clear message.
    """
    # Act
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "--demo"])

    # Assert
    assert result.exit_code == 0, (
        f"analyze --demo exited {result.exit_code} with sockets denied.\n"
        "This indicates a network call was attempted (or another error).\n"
        f"Output:\n{result.output}"
        + (f"\nException: {result.exception}" if result.exception else "")
    )


def test_analyze_file_makes_no_network_calls(deny_sockets: None, tmp_path: Path) -> None:
    """Arrange: a minimal JSONL log file; deny all outbound sockets.
    Act: run frugon analyze <file>.
    Assert: exits 0 — full local analysis with the network denied.

    Uses an explicit usage block so the tokenizer count path is exercised
    through the pricing lookup only, not via tiktoken network calls.
    """
    # Arrange — minimal log with an explicit usage block
    log_file = tmp_path / "test_logs.jsonl"
    record = {
        "model": "gpt-4o",
        "request": {"messages": [{"role": "user", "content": "summarize this document"}]},
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "A concise summary."}}]
        },
        "usage": {"prompt_tokens": 15, "completion_tokens": 5},
    }
    log_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    # Act
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(log_file)])

    # Assert
    assert result.exit_code == 0, (
        f"analyze <file> exited {result.exit_code} with sockets denied.\n"
        "Network call detected or other error.\n"
        f"Output:\n{result.output}"
        + (f"\nException: {result.exception}" if result.exception else "")
    )


def test_analyze_output_does_not_leak_raw_prompt_content(tmp_path: Path) -> None:
    """Arrange: a log whose message + completion carry unique sentinel strings.
    Act: run frugon analyze <file>.
    Assert: neither sentinel appears anywhere in stdout/stderr.

    Network denial proves nothing leaves over the wire; this proves the OTHER
    exfil channel — the rendered output — never echoes raw log-record content.
    The report shows model names, token counts, and costs; it must NEVER print
    the user's prompt or completion text (the local-only privacy invariant).
    """
    prompt_sentinel = "ZZQ_PROMPT_SECRET_7f3a"
    completion_sentinel = "ZZQ_COMPLETION_SECRET_b91c"
    log_file = tmp_path / "secret_logs.jsonl"
    record = {
        "model": "gpt-4o",
        "request": {"messages": [{"role": "user", "content": prompt_sentinel}]},
        "response": {"choices": [{"message": {"role": "assistant", "content": completion_sentinel}}]},
        "usage": {"prompt_tokens": 15, "completion_tokens": 5},
    }
    log_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(log_file)])

    assert result.exit_code == 0, f"analyze exited {result.exit_code}:\n{result.output}"
    assert prompt_sentinel not in result.output, (
        "Raw prompt content leaked into analyze output — the report must never "
        "echo the user's message text."
    )
    assert completion_sentinel not in result.output, (
        "Raw completion content leaked into analyze output."
    )
