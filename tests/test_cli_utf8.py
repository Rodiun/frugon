"""Regression tests for cross-platform console encoding (project rule §7).

frugon prints non-ASCII glyphs in its summaries — the ``->`` routing arrow, the
middle dot, the minus sign.  On a legacy Windows console (cp1252) those glyphs
are unencodable, so an ``analyze`` run died mid-render with ``UnicodeEncodeError``.
``cli._force_utf8_streams`` reconfigures stdout/stderr to UTF-8 at entry so every
glyph encodes on every platform.  CI piping is UTF-8 by default and never caught
this; these tests reproduce the legacy-console condition explicitly.
"""

from __future__ import annotations

import io
import sys

import pytest

import frugon.cli as cli

# The exact glyphs the analyze summary renders that cp1252 cannot encode.
_UNENCODABLE_GLYPHS = "→·−"


def test_cp1252_console_cannot_encode_routing_arrow_documents_the_bug() -> None:
    # A legacy Windows console encodes its output with cp1252, which cannot
    # represent these glyphs — encoding raises, the failure mode the fix prevents.
    with pytest.raises(UnicodeEncodeError):
        _UNENCODABLE_GLYPHS.encode("cp1252")


def test_force_utf8_streams_switches_cp1252_stdout_to_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: point stdout/stderr at a cp1252-encoded stream.
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    assert sys.stdout.encoding.lower() in ("cp1252", "windows-1252")

    # Act
    cli._force_utf8_streams()

    # Assert: encoding flipped, and the previously-fatal glyphs now write clean.
    assert sys.stdout.encoding.lower() == "utf-8"
    sys.stdout.write(f"Route 36,100 easy calls {_UNENCODABLE_GLYPHS} gpt-4o-mini")
    sys.stdout.flush()
    assert _UNENCODABLE_GLYPHS.encode("utf-8") in raw.getvalue()


def test_force_utf8_streams_tolerates_stream_without_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: a stream with no reconfigure (pytest capture / redirected pipe).
    class _NoReconfigure:
        encoding = "cp1252"

        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdout", _NoReconfigure())
    monkeypatch.setattr(sys, "stderr", _NoReconfigure())

    # Act / Assert: absence of reconfigure must never be fatal.
    cli._force_utf8_streams()


def test_force_utf8_streams_swallows_reconfigure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: a stream whose reconfigure raises (e.g. detached buffer).
    class _RaisingReconfigure:
        encoding = "cp1252"

        def reconfigure(self, **_kwargs: object) -> None:
            raise ValueError("underlying buffer has been detached")

        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stdout", _RaisingReconfigure())
    monkeypatch.setattr(sys, "stderr", _RaisingReconfigure())

    # Act / Assert: a reconfigure error must be swallowed, never propagated.
    cli._force_utf8_streams()
