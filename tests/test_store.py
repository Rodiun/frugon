"""Tests for frugon._store — shared persistence helpers.

Covers:
  - seed_if_missing: copies seed on first run; no-op when user file exists;
    silently ignores OSError
  - load_json_or_empty: reads user file; falls back to seed; returns empty dict
    on parse errors or missing files
  - atomic_write_json: temp-then-rename; no leftover .tmp; creates parent dirs;
    raises OSError and cleans up on write failure
  - validate_fetch_url: rejects non-HTTPS; rejects unknown hosts; passes valid URLs
"""

from __future__ import annotations

import email.message
import json
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from frugon._store import fetch_url_with_retry

# ---------------------------------------------------------------------------
# seed_if_missing
# ---------------------------------------------------------------------------


class TestSeedIfMissing:
    """seed_if_missing copies seed to user path on first run only."""

    def test_copies_seed_when_user_file_absent(self, tmp_path: Path) -> None:
        """Arrange: user file absent, seed present.
        Act: seed_if_missing.
        Assert: user file created with same content as seed.
        """
        from frugon._store import seed_if_missing

        seed = tmp_path / "seed.json"
        user = tmp_path / "user" / "data.json"
        seed.write_text('{"_v": 1, "model": 0}', encoding="utf-8")

        seed_if_missing(user, seed)

        assert user.exists()
        assert json.loads(user.read_text(encoding="utf-8")) == {"_v": 1, "model": 0}

    def test_no_op_when_user_file_exists(self, tmp_path: Path) -> None:
        """Arrange: user file already present with distinct content.
        Act: seed_if_missing.
        Assert: user file unchanged.
        """
        from frugon._store import seed_if_missing

        seed = tmp_path / "seed.json"
        user = tmp_path / "data.json"
        seed.write_text('{"source": "seed"}', encoding="utf-8")
        user.write_text('{"source": "user"}', encoding="utf-8")

        seed_if_missing(user, seed)

        assert json.loads(user.read_text(encoding="utf-8")) == {"source": "user"}

    def test_warns_but_does_not_raise_on_oserror(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Arrange: seed exists but the copy hits OSError (unwritable data dir).
        Act: seed_if_missing.
        Assert: no exception raised AND a stderr warning is emitted (fail-loud,
                not silent) so the unwritable dir surfaces here, not later as
                empty tables.
        """
        from frugon._store import seed_if_missing

        seed = tmp_path / "seed.json"
        seed.write_text('{"ok": true}', encoding="utf-8")
        user = tmp_path / "data.json"

        with patch("shutil.copy2", side_effect=OSError("permission denied")):
            seed_if_missing(user, seed)  # must not raise

        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "could not seed" in err

    def test_idempotent_on_second_call(self, tmp_path: Path) -> None:
        """Calling seed_if_missing twice does not overwrite the user file."""
        from frugon._store import seed_if_missing

        seed = tmp_path / "seed.json"
        user = tmp_path / "data.json"
        seed.write_text('{"v": "seed"}', encoding="utf-8")
        user.write_text('{"v": "user"}', encoding="utf-8")

        seed_if_missing(user, seed)
        seed_if_missing(user, seed)

        assert json.loads(user.read_text(encoding="utf-8")) == {"v": "user"}


# ---------------------------------------------------------------------------
# load_json_or_empty
# ---------------------------------------------------------------------------


class TestLoadJsonOrEmpty:
    """load_json_or_empty reads user file, falls back to seed, or returns {}."""

    def test_reads_user_file_when_present(self, tmp_path: Path) -> None:
        """Arrange: user file present with known content.
        Act: load_json_or_empty.
        Assert: returns the user file's content.
        """
        from frugon._store import load_json_or_empty

        user = tmp_path / "user.json"
        seed = tmp_path / "seed.json"
        user.write_text('{"_v": "user"}', encoding="utf-8")
        seed.write_text('{"_v": "seed"}', encoding="utf-8")

        result = load_json_or_empty(user, seed)
        assert result == {"_v": "user"}

    def test_falls_back_to_seed_when_user_absent(self, tmp_path: Path) -> None:
        """Arrange: user file absent; seed present.
        Act: load_json_or_empty.
        Assert: returns the seed's content.
        """
        from frugon._store import load_json_or_empty

        user = tmp_path / "user.json"
        seed = tmp_path / "seed.json"
        seed.write_text('{"_v": "seed"}', encoding="utf-8")

        result = load_json_or_empty(user, seed)
        assert result == {"_v": "seed"}

    def test_returns_empty_when_both_absent(self, tmp_path: Path) -> None:
        """Arrange: neither user nor seed exist.
        Act: load_json_or_empty.
        Assert: returns empty dict.
        """
        from frugon._store import load_json_or_empty

        result = load_json_or_empty(tmp_path / "u.json", tmp_path / "s.json")
        assert result == {}

    def test_returns_empty_on_malformed_user_json(self, tmp_path: Path) -> None:
        """Arrange: user file is malformed JSON.
        Act: load_json_or_empty.
        Assert: returns empty dict without raising.
        """
        from frugon._store import load_json_or_empty

        user = tmp_path / "user.json"
        seed = tmp_path / "seed.json"
        user.write_text("{not valid json}", encoding="utf-8")
        seed.write_text('{"_v": "seed"}', encoding="utf-8")

        result = load_json_or_empty(user, seed)
        assert result == {}

    def test_returns_empty_on_non_dict_user_json(self, tmp_path: Path) -> None:
        """Arrange: user file contains a JSON array, not a dict.
        Act: load_json_or_empty.
        Assert: returns empty dict.
        """
        from frugon._store import load_json_or_empty

        user = tmp_path / "user.json"
        seed = tmp_path / "seed.json"
        user.write_text("[1, 2, 3]", encoding="utf-8")
        seed.write_text('{"_v": "seed"}', encoding="utf-8")

        result = load_json_or_empty(user, seed)
        assert result == {}


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    """atomic_write_json: temp-then-rename, indent=2, sort_keys support."""

    def test_writes_correct_content(self, tmp_path: Path) -> None:
        """Arrange: payload dict.
        Act: atomic_write_json.
        Assert: written file contains the same data.
        """
        from frugon._store import atomic_write_json

        out = tmp_path / "out.json"
        payload = {"_last_synced": "2026-01-01", "gpt-4o": 0}
        atomic_write_json(out, payload)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == payload

    def test_uses_indent_2(self, tmp_path: Path) -> None:
        """Output is pretty-printed with 2-space indent."""
        from frugon._store import atomic_write_json

        out = tmp_path / "out.json"
        atomic_write_json(out, {"a": 1})

        raw = out.read_text(encoding="utf-8")
        assert "  " in raw  # 2-space indent

    def test_sort_keys_false_by_default(self, tmp_path: Path) -> None:
        """Default: keys in insertion order, not alphabetical."""
        from frugon._store import atomic_write_json

        out = tmp_path / "out.json"
        atomic_write_json(out, {"z": 1, "a": 2})

        raw = out.read_text(encoding="utf-8")
        assert raw.index('"z"') < raw.index('"a"')

    def test_sort_keys_true_sorts(self, tmp_path: Path) -> None:
        """sort_keys=True produces alphabetically sorted output."""
        from frugon._store import atomic_write_json

        out = tmp_path / "out.json"
        atomic_write_json(out, {"z": 1, "a": 2}, sort_keys=True)

        raw = out.read_text(encoding="utf-8")
        assert raw.index('"a"') < raw.index('"z"')

    def test_no_tmp_file_on_success(self, tmp_path: Path) -> None:
        """No .tmp file remains after a successful write."""
        from frugon._store import atomic_write_json

        out = tmp_path / "out.json"
        atomic_write_json(out, {"ok": True})

        assert not list(tmp_path.glob("*.tmp"))

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        from frugon._store import atomic_write_json

        out = tmp_path / "nested" / "deep" / "out.json"
        atomic_write_json(out, {"x": 1})

        assert out.exists()

    def test_raises_oserror_and_removes_tmp_on_failure(self, tmp_path: Path) -> None:
        """On write failure, OSError is raised and the .tmp file is cleaned up."""
        from frugon._store import atomic_write_json

        out = tmp_path / "out.json"

        original_write_text = Path.write_text

        def failing_write_text(self: Path, *args: object, **kwargs: object) -> None:
            if self.suffix == ".tmp":
                self.write_bytes(b"partial")
                raise OSError("disk full")
            return original_write_text(self, *args, **kwargs)

        with patch.object(Path, "write_text", failing_write_text):
            with pytest.raises(OSError, match="disk full"):
                atomic_write_json(out, {"ok": True})

        assert not out.exists()
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# atomic_write_text (FRG-OSS-017)
# ---------------------------------------------------------------------------


class TestAtomicWriteText:
    """atomic_write_text: NamedTemporaryFile-then-os.replace, symlink-safe."""

    def test_writes_correct_content(self, tmp_path: Path) -> None:
        """Arrange: destination path, text content.
        Act: atomic_write_text.
        Assert: file contains exactly the written text.
        """
        from frugon._store import atomic_write_text

        out = tmp_path / "report.html"
        atomic_write_text(out, "<html>hello</html>\n")

        assert out.read_text(encoding="utf-8") == "<html>hello</html>\n"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        from frugon._store import atomic_write_text

        out = tmp_path / "nested" / "deep" / "out.md"
        atomic_write_text(out, "# report\n")

        assert out.exists()
        assert out.read_text(encoding="utf-8") == "# report\n"

    def test_no_tmp_file_remains_after_success(self, tmp_path: Path) -> None:
        """No temp artifact remains in the destination directory after a
        successful write — only the final file."""
        from frugon._store import atomic_write_text

        out = tmp_path / "out.html"
        atomic_write_text(out, "content")

        remaining = list(tmp_path.iterdir())
        assert remaining == [out]

    def test_overwrites_existing_regular_file(self, tmp_path: Path) -> None:
        """Arrange: an existing regular file at the destination.
        Act: atomic_write_text with new content.
        Assert: old content is fully replaced (not appended/merged).
        """
        from frugon._store import atomic_write_text

        out = tmp_path / "report.html"
        out.write_text("OLD CONTENT", encoding="utf-8")

        atomic_write_text(out, "NEW CONTENT")

        assert out.read_text(encoding="utf-8") == "NEW CONTENT"

    def test_preserves_lf_only_newlines_on_content_with_newlines(
        self, tmp_path: Path
    ) -> None:
        """The written bytes on disk must be LF-only (no \\r\\n translation),
        matching the historical write_text(..., newline="\\n") contract every
        call site in report.py relied on — a cross-platform artifact-byte
        guarantee (§7)."""
        from frugon._store import atomic_write_text

        out = tmp_path / "out.md"
        atomic_write_text(out, "line one\nline two\nline three\n")

        raw = out.read_bytes()
        assert b"\r\n" not in raw
        assert raw.count(b"\n") == 3

    def test_raises_oserror_and_removes_tmp_on_failure(self, tmp_path: Path) -> None:
        """On write failure, OSError propagates and no temp file survives."""
        import os as _os

        from frugon._store import atomic_write_text

        out = tmp_path / "out.html"

        def failing_fdopen(*args: object, **kwargs: object) -> Any:
            raise OSError("disk full")

        with patch.object(_os, "fdopen", failing_fdopen):
            with pytest.raises(OSError, match="disk full"):
                atomic_write_text(out, "content")

        assert not out.exists()
        # No stray temp file (matching the `.{name}.*.tmp` naming pattern) left behind.
        assert not list(tmp_path.glob(f".{out.name}.*"))

    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="symlink semantics under test are POSIX-specific",
    )
    def test_does_not_follow_symlink_into_target_file(self, tmp_path: Path) -> None:
        """Arrange: *path* is a symlink pointing at an unrelated 'secret' file
        elsewhere on disk (simulating an attacker-planted symlink, or a stale
        leftover, at the report output path).
        Act: atomic_write_text(path, new_content).
        Assert: the symlink TARGET is left completely untouched — only the
        symlink itself is atomically replaced by a fresh regular file
        containing the new content.  This is the core regression for
        FRG-OSS-017: the OLD `path.write_text(...)` would have overwritten
        the secret file's contents in place.
        """
        from frugon._store import atomic_write_text

        secret = tmp_path / "secret.txt"
        secret.write_text("SENSITIVE ORIGINAL CONTENT", encoding="utf-8")

        report_path = tmp_path / "report.html"
        report_path.symlink_to(secret)

        atomic_write_text(report_path, "<html>new report</html>")

        # The symlink's original target is untouched.
        assert secret.read_text(encoding="utf-8") == "SENSITIVE ORIGINAL CONTENT"
        # report_path is now a fresh regular file (the symlink was replaced).
        assert not report_path.is_symlink()
        assert report_path.read_text(encoding="utf-8") == "<html>new report</html>"

    @pytest.mark.skipif(
        __import__("sys").platform.startswith("win"),
        reason="symlink semantics under test are POSIX-specific",
    )
    def test_atomic_replace_happy_path_on_posix(self, tmp_path: Path) -> None:
        """Arrange: destination does not yet exist.
        Act: atomic_write_text.
        Assert: destination is a regular file (not a symlink) with the exact
        content, and no temp file remains — the ordinary non-symlink happy
        path on POSIX.
        """
        from frugon._store import atomic_write_text

        out = tmp_path / "fresh.html"
        atomic_write_text(out, "<html>fresh</html>")

        assert out.is_file()
        assert not out.is_symlink()
        assert out.read_text(encoding="utf-8") == "<html>fresh</html>"
        assert list(tmp_path.iterdir()) == [out]


# ---------------------------------------------------------------------------
# validate_fetch_url
# ---------------------------------------------------------------------------


class TestValidateFetchUrl:
    """validate_fetch_url enforces HTTPS and allowlist."""

    _ALLOWED: frozenset[str] = frozenset({"raw.githubusercontent.com", "datasets-server.huggingface.co"})

    def test_valid_github_url_passes(self) -> None:
        """https://raw.githubusercontent.com/... passes validation."""
        from frugon._store import validate_fetch_url

        validate_fetch_url(
            "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices.json",
            self._ALLOWED,
        )  # must not raise

    def test_valid_huggingface_url_passes(self) -> None:
        """https://datasets-server.huggingface.co/... passes validation."""
        from frugon._store import validate_fetch_url

        validate_fetch_url(
            "https://datasets-server.huggingface.co/rows?dataset=lmarena-ai",
            self._ALLOWED,
        )  # must not raise

    def test_http_url_raises_value_error(self) -> None:
        """http:// URL raises ValueError mentioning HTTPS."""
        from frugon._store import validate_fetch_url

        with pytest.raises(ValueError, match="HTTPS"):
            validate_fetch_url(
                "http://raw.githubusercontent.com/BerriAI/litellm/main/prices.json",
                self._ALLOWED,
            )

    def test_unknown_host_raises_value_error(self) -> None:
        """HTTPS URL with disallowed host raises ValueError."""
        from frugon._store import validate_fetch_url

        with pytest.raises(ValueError, match="allowed"):
            validate_fetch_url("https://evil.example.com/prices.json", self._ALLOWED)

    def test_file_scheme_raises_value_error(self) -> None:
        """file:// URL raises ValueError."""
        from frugon._store import validate_fetch_url

        with pytest.raises(ValueError, match="HTTPS"):
            validate_fetch_url("file:///etc/passwd", self._ALLOWED)

    def test_custom_single_host_allowlist(self) -> None:
        """A single-host allowlist accepts only that host."""
        from frugon._store import validate_fetch_url

        allowed = frozenset({"api.example.com"})
        validate_fetch_url("https://api.example.com/data.json", allowed)
        with pytest.raises(ValueError, match="allowed"):
            validate_fetch_url("https://other.example.com/data.json", allowed)


# ---------------------------------------------------------------------------
# fetch_url_with_retry — the shared resilient-fetch primitive
#
# Direct contract tests for the single retry primitive backing BOTH the quality
# (LMArena) and pricing (LiteLLM registry) sync paths, so its contract is pinned
# independently of either caller.
# ---------------------------------------------------------------------------

_UA = "frugon-test/0.0 (+https://example.invalid)"


class _Resp:
    """Minimal urlopen context-manager stand-in returning a fixed body."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self, *args: object) -> bytes:
        return self._data

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://host.invalid/x", code, "err", hdrs, None)  # type: ignore[arg-type]


def _boom(exc: Exception) -> Exception:
    """Default on_failure: a sentinel domain error tagged with the exc type."""
    return RuntimeError(f"failed:{type(exc).__name__}")


class TestFetchUrlWithRetry:
    """User-Agent sent; 429/5xx/transient retried with bounded backoff; 4xx not."""

    def test_sends_user_agent_and_returns_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[Any] = []

        def fake(req: Any, *a: object, **k: object) -> _Resp:
            captured.append(req)
            return _Resp(b"OK")

        monkeypatch.setattr("urllib.request.urlopen", fake)
        body = fetch_url_with_retry(
            "https://host.invalid/x", user_agent=_UA, max_bytes=99, on_failure=_boom
        )
        assert body == b"OK"
        assert captured[0].get_header("User-agent") == _UA

    def test_retries_429_then_succeeds_exact_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("time.sleep", lambda *a: None)
        calls = {"n": 0}

        def fake(req: Any, *a: object, **k: object) -> _Resp:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(429)
            return _Resp(b"OK")

        monkeypatch.setattr("urllib.request.urlopen", fake)
        assert (
            fetch_url_with_retry(
                "https://host.invalid/x", user_agent=_UA, max_bytes=99, on_failure=_boom
            )
            == b"OK"
        )
        assert calls["n"] == 2

    def test_retry_after_overrides_computed_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
        calls = {"n": 0}

        def fake(req: Any, *a: object, **k: object) -> _Resp:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(429, retry_after="7")
            return _Resp(b"OK")

        monkeypatch.setattr("urllib.request.urlopen", fake)
        fetch_url_with_retry(
            "https://host.invalid/x", user_agent=_UA, max_bytes=99, backoff_base=1.0, on_failure=_boom
        )
        assert slept == [7.0]  # Retry-After (7s) honoured over backoff_base*2**0 (1.0)

    def test_non_numeric_retry_after_falls_back_to_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
        calls = {"n": 0}

        def fake(req: Any, *a: object, **k: object) -> _Resp:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(503, retry_after="soon")  # unparseable
            return _Resp(b"OK")

        monkeypatch.setattr("urllib.request.urlopen", fake)
        fetch_url_with_retry(
            "https://host.invalid/x", user_agent=_UA, max_bytes=99, backoff_base=2.0, on_failure=_boom
        )
        assert slept == [2.0]  # falls back to backoff_base*2**0

    def test_4xx_not_retried_and_on_failure_gets_exact_exc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("time.sleep", lambda *a: None)
        calls = {"n": 0}
        seen: list[Exception] = []
        err = _http_error(404)

        def fake(req: Any, *a: object, **k: object) -> _Resp:
            calls["n"] += 1
            raise err

        def on_fail(exc: Exception) -> Exception:
            seen.append(exc)
            return RuntimeError("permanent")

        monkeypatch.setattr("urllib.request.urlopen", fake)
        with pytest.raises(RuntimeError, match="permanent"):
            fetch_url_with_retry(
                "https://host.invalid/x", user_agent=_UA, max_bytes=99, on_failure=on_fail
            )
        assert calls["n"] == 1  # 4xx is permanent — not retried
        assert seen == [err]  # on_failure received the exact triggering exception

    def test_5xx_exhausts_bounded_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("time.sleep", lambda *a: None)
        calls = {"n": 0}

        def fake(req: Any, *a: object, **k: object) -> _Resp:
            calls["n"] += 1
            raise _http_error(500)

        monkeypatch.setattr("urllib.request.urlopen", fake)
        with pytest.raises(RuntimeError):
            fetch_url_with_retry(
                "https://host.invalid/x", user_agent=_UA, max_bytes=99, max_retries=3, on_failure=_boom
            )
        assert calls["n"] == 4  # 3 retries + 1 initial = bounded, not infinite

    def test_transient_urlerror_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("time.sleep", lambda *a: None)
        calls = {"n": 0}

        def fake(req: Any, *a: object, **k: object) -> _Resp:
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError("connection reset")
            return _Resp(b"OK")

        monkeypatch.setattr("urllib.request.urlopen", fake)
        assert (
            fetch_url_with_retry(
                "https://host.invalid/x", user_agent=_UA, max_bytes=99, on_failure=_boom
            )
            == b"OK"
        )
        assert calls["n"] == 2

    def test_reads_at_most_max_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen_n: list[int] = []

        class _R:
            def read(self, n: int = -1) -> bytes:
                seen_n.append(n)
                return b"x" * (n if n and n > 0 else 5)

            def __enter__(self) -> _R:
                return self

            def __exit__(self, *a: object) -> None:
                pass

        monkeypatch.setattr("urllib.request.urlopen", lambda req, *a, **k: _R())
        fetch_url_with_retry(
            "https://host.invalid/x", user_agent=_UA, max_bytes=42, on_failure=_boom
        )
        assert seen_n == [42]  # body is read capped at max_bytes
