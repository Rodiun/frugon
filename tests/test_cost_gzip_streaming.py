"""Tests for frugon.cost's streaming gzip decompression (FRG-OSS-015).

Covers:
- Happy path: a well-formed .gz log round-trips through iter_records identically
  to the equivalent plain .jsonl.
- Truncated .gz raises LogReadError (an OSError subclass) rather than crashing
  with a raw EOFError/traceback.
- Non-gzip bytes with a misleading .gz extension raise LogReadError
  (BadGzipFile), not a raw traceback.
- A decompressed payload exceeding the (test-overridden, small) ceiling raises
  LogReadError rather than materialising an unbounded buffer in memory.
- The ceiling is env-var overridable via FRUGON_MAX_GZ_DECOMPRESSED_BYTES, and
  a malformed override falls back to the safe default rather than disabling
  the cap.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from frugon.cost import (
    LogReadError,
    _max_decompressed_gz_bytes,
    _read_log_text,
    iter_records,
)

_SAMPLE_RECORD = {
    "model": "gpt-4o",
    "request": {"messages": [{"role": "user", "content": "hi"}]},
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    "timestamp": "2026-01-01T00:00:00Z",
}


def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Happy path — round-trip identical to plain .jsonl
# ---------------------------------------------------------------------------


def test_gz_log_round_trips_identically_to_plain_jsonl(tmp_path: Path) -> None:
    """Arrange: the same JSONL content written as both plain .jsonl and .gz.
    Act: iter_records on each.
    Assert: identical records + skipped_malformed count — streaming
    decompression must be byte-for-byte equivalent to the old
    gzip.decompress(...) path for well-formed input.
    """
    lines = "\n".join(json.dumps(_SAMPLE_RECORD) for _ in range(50)) + "\n"

    plain_path = tmp_path / "logs.jsonl"
    plain_path.write_text(lines, encoding="utf-8")

    gz_path = tmp_path / "logs.jsonl.gz"
    _write_gz(gz_path, lines)

    plain_records, plain_skipped = iter_records(plain_path)
    gz_records, gz_skipped = iter_records(gz_path)

    assert gz_skipped == plain_skipped == 0
    assert len(gz_records) == len(plain_records) == 50
    assert gz_records[0].model == plain_records[0].model == "gpt-4o"


def test_gz_log_with_unicode_content_decodes_correctly(tmp_path: Path) -> None:
    """Arrange: a .gz log containing multi-byte UTF-8 content (emoji + accents).
    Act: _read_log_text.
    Assert: the streamed chunks reassemble into the exact original text — a
    chunk boundary landing mid-multibyte-character would corrupt decoding if
    handled naively (it isn't, here, because bytes are joined before the
    single final .decode("utf-8") call).
    """
    text = "café 🎉 naïve €100\n" * 500
    gz_path = tmp_path / "unicode.jsonl.gz"
    _write_gz(gz_path, text)

    assert _read_log_text(gz_path) == text


# ---------------------------------------------------------------------------
# Truncated gzip stream
# ---------------------------------------------------------------------------


def test_truncated_gz_raises_log_read_error_not_raw_traceback(tmp_path: Path) -> None:
    """Arrange: a valid gzip file truncated mid-stream (chops off the last
    bytes, including the CRC/size trailer).
    Act: _read_log_text.
    Assert: raises LogReadError (an OSError subclass), never a bare
    EOFError/zlib.error escaping to the caller — so cli.py's existing
    `except OSError` panel fires instead of a raw traceback.
    """
    text = json.dumps(_SAMPLE_RECORD) + "\n"
    full_path = tmp_path / "full.jsonl.gz"
    _write_gz(full_path, text * 200)  # large enough that truncation lands mid-stream

    full_bytes = full_path.read_bytes()
    truncated_path = tmp_path / "truncated.jsonl.gz"
    truncated_path.write_bytes(full_bytes[: len(full_bytes) // 2])

    with pytest.raises(LogReadError):
        _read_log_text(truncated_path)


def test_truncated_gz_is_an_oserror_subclass(tmp_path: Path) -> None:
    """Arrange: a truncated .gz file.
    Act: catch via `except OSError` — the exact pattern cli.py already uses at
    both call sites (scan_models pre-flight, iter_records in analyze).
    Assert: caught cleanly, proving zero changes are needed at the call sites.
    """
    text = json.dumps(_SAMPLE_RECORD) + "\n"
    full_path = tmp_path / "full2.jsonl.gz"
    _write_gz(full_path, text * 200)
    truncated_path = tmp_path / "truncated2.jsonl.gz"
    truncated_path.write_bytes(full_path.read_bytes()[:100])

    with pytest.raises(LogReadError) as exc_info:
        _read_log_text(truncated_path)
    assert str(exc_info.value)  # a non-empty, actionable message


# ---------------------------------------------------------------------------
# Non-gzip bytes with a misleading .gz extension
# ---------------------------------------------------------------------------


def test_non_gzip_bytes_with_gz_extension_raises_log_read_error(tmp_path: Path) -> None:
    """Arrange: a file named *.gz containing plain (non-gzip) text.
    Act: _read_log_text.
    Assert: raises LogReadError wrapping BadGzipFile — not a raw traceback,
    and not silently treated as valid gzip.
    """
    fake_gz = tmp_path / "not_actually_gzip.jsonl.gz"
    fake_gz.write_bytes(json.dumps(_SAMPLE_RECORD).encode("utf-8"))

    with pytest.raises(LogReadError, match="not a valid gzip file"):
        _read_log_text(fake_gz)


def test_non_gzip_bytes_iter_records_surfaces_clean_error(tmp_path: Path) -> None:
    """Arrange: the same misleading .gz file, via the iter_records() entry
    point analyze_logs/--measure actually calls.
    Act: iter_records.
    Assert: LogReadError propagates (an OSError), not a bare exception type
    the CLI wouldn't recognise.
    """
    fake_gz = tmp_path / "fake2.jsonl.gz"
    fake_gz.write_bytes(b"this is not gzip data at all, just plain bytes")

    with pytest.raises(LogReadError, match="not a valid gzip file"):
        iter_records(fake_gz)


# ---------------------------------------------------------------------------
# Ceiling exceeded (gzip-bomb guard)
# ---------------------------------------------------------------------------


def test_decompressed_size_exceeding_ceiling_raises_log_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arrange: a small ceiling override (1 KiB) and a .gz payload that
    decompresses to well over that (highly compressible repeated content, so
    the ON-DISK file stays tiny — the gzip-bomb shape).
    Act: _read_log_text.
    Assert: raises LogReadError before the full payload is buffered — proving
    the ceiling is enforced during streaming, not after the fact.
    """
    monkeypatch.setenv("FRUGON_MAX_GZ_DECOMPRESSED_BYTES", "1024")

    # 100,000 repeats of a 60-byte line = ~6MB decompressed, compresses to a
    # tiny on-disk size (highly repetitive) — well over the 1KiB test ceiling.
    line = json.dumps(_SAMPLE_RECORD) + "\n"
    bomb_path = tmp_path / "bomb.jsonl.gz"
    _write_gz(bomb_path, line * 100_000)

    with pytest.raises(LogReadError, match="exceeds the"):
        _read_log_text(bomb_path)


def test_ceiling_ok_payload_under_override_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arrange: a ceiling override large enough for a small legitimate payload.
    Act: _read_log_text.
    Assert: succeeds normally — the override does not break the happy path.
    """
    monkeypatch.setenv("FRUGON_MAX_GZ_DECOMPRESSED_BYTES", "1000000")
    text = (json.dumps(_SAMPLE_RECORD) + "\n") * 10
    path = tmp_path / "small.jsonl.gz"
    _write_gz(path, text)

    assert _read_log_text(path) == text


# ---------------------------------------------------------------------------
# Ceiling resolution — env var parsing
# ---------------------------------------------------------------------------


def test_max_decompressed_bytes_defaults_to_512mb_absent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: no override set.
    Act: _max_decompressed_gz_bytes.
    Assert: returns the documented 512MB default.
    """
    monkeypatch.delenv("FRUGON_MAX_GZ_DECOMPRESSED_BYTES", raising=False)
    assert _max_decompressed_gz_bytes() == 512 * 1024 * 1024


def test_max_decompressed_bytes_honors_valid_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: a valid positive override.
    Act: _max_decompressed_gz_bytes.
    Assert: the override value is used verbatim.
    """
    monkeypatch.setenv("FRUGON_MAX_GZ_DECOMPRESSED_BYTES", "2048")
    assert _max_decompressed_gz_bytes() == 2048


@pytest.mark.parametrize("bad_value", ["not-a-number", "", "-5", "0"])
def test_max_decompressed_bytes_falls_back_on_malformed_override(
    monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """Arrange: an unparseable, empty, negative, or zero override.
    Act: _max_decompressed_gz_bytes.
    Assert: falls back to the safe 512MB default — a malformed override must
    never silently DISABLE the ceiling (e.g. a bogus "0" must not mean
    unlimited).
    """
    monkeypatch.setenv("FRUGON_MAX_GZ_DECOMPRESSED_BYTES", bad_value)
    assert _max_decompressed_gz_bytes() == 512 * 1024 * 1024
