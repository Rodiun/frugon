"""frugon._store — shared persistence helpers for pricing, quality, and report.

Provides atomic JSON/text writes, first-run seeding, and fetch-URL validation
used by pricing.py, quality.py, and report.py to eliminate code duplication.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def seed_if_missing(user_path: Path, seed_path: Path) -> None:
    """Copy *seed_path* to *user_path* if *user_path* does not yet exist.

    Best-effort: the tool never fails on startup due to a permissions issue in
    the data directory.  But the failure is no longer silent — it emits a
    one-line stderr warning so an unwritable data dir surfaces here rather than
    only later as mysteriously empty tables (§4 fail-loud).  Callers fall back
    to the bundled seed via load_json_or_empty.
    """
    if user_path.exists():
        return
    try:
        user_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_path, user_path)
    except OSError as exc:
        print(
            f"frugon: WARNING could not seed {user_path} ({exc}); "
            "using the bundled data instead.",
            file=sys.stderr,
        )


def load_json_or_empty(user_path: Path, seed_path: Path) -> dict[str, Any]:
    """Load JSON from *user_path*, falling back to *seed_path* if absent.

    Returns an empty dict on any I/O or parse error so callers degrade
    gracefully without raising.
    """
    if user_path.exists():
        read_path = user_path
    elif seed_path.exists():
        read_path = seed_path
    else:
        return {}
    try:
        with read_path.open(encoding="utf-8") as fh:
            raw: Any = json.load(fh)
        if not isinstance(raw, dict):
            return {}
        return raw
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    sort_keys: bool = False,
    trailing_newline: bool = False,
) -> None:
    """Write *payload* to *path* via a temp-then-replace atomic operation.

    Creates parent directories as needed.  Raises OSError on failure;
    callers that need a domain-specific error type should wrap with ``except
    OSError``.  No .tmp file is left on success; any .tmp is removed on
    failure before re-raising.

    When *trailing_newline* is True, a ``\\n`` is appended after the JSON
    text.  Use this for seed files that must end with a newline so that the
    on-disk form is the writer's fixed point (a subsequent write that changes
    only one value produces a one-line diff rather than a whole-file reformat).
    Default is False so every existing caller is byte-for-byte unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    text = json.dumps(payload, indent=2, sort_keys=sort_keys)
    if trailing_newline:
        text += "\n"
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write *text* to *path* via a NamedTemporaryFile-then-``os.replace`` swap.

    Symlink-safety (FRG-OSS-017): a naive ``path.write_text(...)`` FOLLOWS a
    symlink at *path* and overwrites whatever the symlink points to — if
    *path* is a symlink planted by an attacker (or left over from a prior,
    unrelated file) pointing at e.g. ``~/.ssh/authorized_keys``, a plain
    write_text call corrupts that target, not just "the report file".  This
    helper never opens *path* directly for writing: it creates a new,
    exclusively-named temp file IN THE SAME DIRECTORY (so the final
    ``os.replace`` is on the same filesystem — no cross-device rename
    failure), writes the content there, then atomically replaces *path* with
    it.  ``os.replace`` on a symlink target REPLACES THE SYMLINK ITSELF (it
    does not dereference and write through it) — so even if *path* already is
    a symlink, only the link is swapped for a fresh regular file; the
    original symlink target is left completely untouched.

    Creates parent directories as needed.  Raises OSError on failure; the
    temp file is removed before re-raising so no stray temp file survives a
    failed write.  newline="" is NOT passed here (unlike the raw write_text
    calls this replaces) because every call site already produces
    ``\\n``-only content and Python's default text-mode write on the temp
    file's file object performs no newline translation when the string
    itself contains only ``\\n`` — this preserves the existing
    byte-for-byte LF-only artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    fd_wrapped = False
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fd_wrapped = True
            fh.write(text)
        os.replace(tmp_name, path)
    except OSError:
        # If os.fdopen() itself raised, the raw fd from mkstemp was never
        # wrapped (and so never closed by the `with` block's __exit__) — an
        # open handle on the temp file blocks its own deletion on Windows.
        # Close it defensively before attempting cleanup so the unlink below
        # can actually succeed on every platform.
        if not fd_wrapped:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def validate_fetch_url(url: str, allowed_hosts: frozenset[str]) -> None:
    """Raise ValueError if *url* is not HTTPS or its host is not in *allowed_hosts*.

    Prevents accidental or adversarial redirects to non-HTTPS endpoints and
    limits outbound update fetches to the known upstream hosts.
    """
    if not url.startswith("https://"):
        raise ValueError(f"Update URL must use HTTPS; got: {url!r}")
    host = urlsplit(url).hostname or ""
    if host not in allowed_hosts:
        raise ValueError(
            f"Update URL host {host!r} is not in the allowed list "
            f"{sorted(allowed_hosts)!r}"
        )


def fetch_url_with_retry(
    url: str,
    *,
    user_agent: str,
    max_bytes: int,
    timeout: int = 30,
    max_retries: int = 4,
    backoff_base: float = 1.0,
    on_failure: Callable[[Exception], Exception],
) -> bytes:
    """Fetch *url* with bounded retry on transient failures, returning the body.

    Sends an explicit ``User-Agent`` (some hosts reject the default urllib agent
    with a 5xx).  Retries on HTTP 429, HTTP 5xx, and transient
    ``(URLError, OSError)`` with exponential backoff (``backoff_base * 2**attempt``
    seconds).  When a 429/5xx carries a ``Retry-After`` header (integer seconds),
    that value overrides the computed backoff.  A 4xx other than 429 is a
    permanent client error and is NOT retried.

    Budget: *max_retries* retries after the initial attempt, i.e. at most
    ``max_retries + 1`` total requests.  Reads at most *max_bytes* of the body.

    On exhaustion of the retry budget OR a non-retryable error, the supplied
    *on_failure* callable is invoked with the triggering exception and its return
    value is raised — letting each caller produce its own domain exception and
    message (e.g. distinguishing an HTTP failure from a network failure).

    Args:
        url: Absolute URL to fetch (caller validates host/scheme beforehand).
        user_agent: Value for the outbound ``User-Agent`` header.
        max_bytes: Maximum number of body bytes to read.
        timeout: Per-request socket timeout in seconds.
        max_retries: Retries allowed after the initial attempt.
        backoff_base: Base backoff in seconds; doubles each attempt.
        on_failure: Maps the triggering exception to the domain exception to raise.

    Returns:
        The response body, capped at *max_bytes*.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):  # attempt 0 = first try
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": user_agent}),
                timeout=timeout,
            ) as resp:
                return resp.read(max_bytes)  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            # 429 (rate limit) and 5xx (transient server errors) are retryable;
            # other 4xx (client errors, e.g. 404) are permanent and are not.
            if exc.code == 429 or exc.code >= 500:
                last_exc = exc
                if attempt < max_retries:
                    # Check the headers object's PRESENCE, not truthiness:
                    # http.client.HTTPMessage defines __len__, so a present-but-
                    # empty headers object is falsy — `if exc.headers` would then
                    # wrongly skip an existing Retry-After. `is not None` is correct.
                    retry_after_raw: Any = (
                        exc.headers.get("Retry-After") if exc.headers is not None else None
                    )
                    try:
                        wait = float(retry_after_raw) if retry_after_raw is not None else None
                    except (ValueError, TypeError):
                        wait = None
                    if wait is None:
                        wait = backoff_base * (2**attempt)
                    time.sleep(wait)
                    continue
                # Exhausted retries on a retryable status.
                raise on_failure(exc) from exc
            # Non-retryable HTTP error (4xx client error).
            raise on_failure(exc) from exc
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_base * (2**attempt))
                continue
            raise on_failure(exc) from exc

    # Unreachable, but satisfies type-checkers: the loop always raises or returns.
    assert last_exc is not None
    raise on_failure(last_exc) from last_exc
