"""frugon capture — local passive OpenAI-compatible logger.

Starts a local HTTP server that:
  1. Accepts POST /v1/chat/completions and /v1/completions
  2. Forwards each request unchanged to the configured upstream
  3. Writes one canonical JSONL record per call to a local file
  4. Returns the upstream response verbatim to the caller

Privacy: the shim makes no calls to any frugon endpoint.
All traffic flows between the user's app and their own provider.
"""

from __future__ import annotations

import datetime
import http.server
import json
import os
import pathlib
import socketserver
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import IO, Any, cast

_DEFAULT_UPSTREAM = "https://api.openai.com"
_CAPTURE_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})

# Hard cap applied to both inbound request bodies and upstream response bodies.
# Requests exceeding this limit receive 413; upstream responses are truncated.
_MAX_BODY = 32 * 1024 * 1024  # 32 MiB

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Headers stripped from requests that follow a cross-origin redirect.
_SENSITIVE_REDIRECT_HEADERS = frozenset({
    "authorization",
    "cookie",
    "proxy-authorization",
})

# Default connect+read timeout applied to every upstream request (seconds).
# Pass upstream_timeout=None to disable (not recommended); 0 means instant fail.
_DEFAULT_UPSTREAM_TIMEOUT: float = 60.0


def _validate_upstream(upstream: str, *, allow_insecure_upstream: bool = False) -> None:
    """Raise ValueError if the upstream URL is unsafe.

    Accepted:
    - ``https://`` with any host.
    - ``http://`` with a localhost-equivalent host (127.0.0.1, ::1, localhost).
    - ``http://`` with any host when *allow_insecure_upstream* is True.

    Rejected:
    - Any non-http/https scheme (file://, ftp://, data:, …).
    - ``http://`` to a non-localhost host without the explicit opt-in flag.
    """
    parsed = urllib.parse.urlsplit(upstream)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"upstream scheme {parsed.scheme!r} is not allowed; use http or https"
        )
    if parsed.scheme == "http":
        host = parsed.hostname or ""
        if host not in _LOCALHOST_HOSTS and not allow_insecure_upstream:
            raise ValueError(
                f"upstream {upstream!r} uses plain http for a non-localhost host. "
                "Use https:// or pass allow_insecure_upstream=True (local testing only)."
            )


def _origins_differ(url1: str, url2: str) -> bool:
    """Return True when url1 and url2 have different scheme, host, or port."""
    p1 = urllib.parse.urlsplit(url1)
    p2 = urllib.parse.urlsplit(url2)
    return (p1.scheme.lower(), p1.hostname, p1.port) != (
        p2.scheme.lower(),
        p2.hostname,
        p2.port,
    )


class _CredentialStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drops Authorization/Cookie headers when a redirect crosses origins.

    Same-origin redirects (same scheme + host + port) forward credentials
    unchanged.  Cross-origin redirects have those headers stripped before the
    new request is sent so that a developer's API key cannot leak to an
    unintended host via a provider-issued redirect.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        if _origins_differ(req.full_url, newurl):
            for header in list(new_req.headers.keys()):
                if header.lower() in _SENSITIVE_REDIRECT_HEADERS:
                    new_req.remove_header(header.capitalize())
        return new_req


def _build_restricted_opener() -> urllib.request.OpenerDirector:
    """Return an OpenerDirector that handles ONLY http:// and https://.

    Unlike urllib.request.build_opener(), this does NOT register the default
    FileHandler / FTPHandler / DataHandler. Any non-HTTP scheme falls through
    to UnknownHandler, which raises URLError('unknown url type: <scheme>').
    Defence-in-depth below _validate_upstream.
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        _CredentialStrippingRedirectHandler(),
        urllib.request.HTTPErrorProcessor(),
        urllib.request.UnknownHandler(),
    ):
        opener.add_handler(handler)
    return opener


# ---------------------------------------------------------------------------
# Feedback stream
# ---------------------------------------------------------------------------

_VERBOSITY_VALUES = frozenset({"normal", "quiet", "verbose"})


class _FeedbackStream:
    """Emit per-call feedback to stdout at the configured verbosity level.

    Verbosity levels:
    - "quiet":   no per-call output after startup.
    - "normal":  a single in-place updating counter line using carriage-return
                 rewrite (no scroll). Degrades to silence when stdout is not a
                 TTY, because ``\\r`` rewrites corrupt log files.
    - "verbose": one newline-terminated line per call with timestamp, model,
                 token counts, and upstream HTTP status.
    """

    def __init__(self, verbosity: str) -> None:
        if verbosity not in _VERBOSITY_VALUES:
            raise ValueError(
                f"verbosity must be one of {sorted(_VERBOSITY_VALUES)!r}; got {verbosity!r}"
            )
        self._verbosity = verbosity
        self._count = 0
        self._lock = threading.Lock()

    @property
    def verbose(self) -> bool:
        """True when per-call verbose diagnostics should be emitted."""
        return self._verbosity == "verbose"

    def on_call(self, record: dict[str, Any], status: int) -> None:
        """Emit one feedback update for a completed captured call.

        Parameters
        ----------
        record:
            The canonical JSONL record built by ``_build_record``.
        status:
            The HTTP status code returned by the upstream provider.
        """
        if self._verbosity == "quiet":
            return

        model: str = record.get("model") or "unknown"
        usage: dict[str, Any] = record.get("usage") or {}
        total_tokens: int = int(usage.get("total_tokens") or 0)
        timestamp: str = record.get("timestamp") or ""

        with self._lock:
            self._count += 1
            count = self._count

        if self._verbosity == "verbose":
            prompt_tok: int = int(usage.get("prompt_tokens") or 0)
            completion_tok: int = int(usage.get("completion_tokens") or 0)
            sys.stdout.write(
                f"{timestamp}  {model}  prompt={prompt_tok} completion={completion_tok}"
                f"  status={status}\n"
            )
            sys.stdout.flush()
            return

        # "normal" — in-place counter, TTY-aware.
        # The \r + \x1b[K (carriage return + clear-to-end-of-line) rewrites the
        # current line in place.  Off-TTY we stay silent (a \r would corrupt a
        # redirected logfile).  On a TTY, modern Windows terminals (Windows
        # Terminal, PowerShell 7, VS Code, WSL) honour the ANSI sequence; on a
        # legacy conhost without VT processing the worst case is a stray "[K"
        # glyph, never a crash — and Python ≥3.6 enables VT on Windows by default.
        if not sys.stdout.isatty():
            return

        line = f"\r\x1b[K⦿ {count} calls captured · last: {model} ({total_tokens} tok)"
        sys.stdout.write(line)
        sys.stdout.flush()


def _now_utc() -> str:
    """Return current UTC time as an ISO 8601 string with Z suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_record(
    request_data: dict[str, Any],
    response_data: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Build a canonical JSONL record from request and response data.

    The shape matches what ``frugon analyze`` ingests:
      model, request.messages, response, usage, timestamp
    """
    return {
        "model": request_data.get("model", ""),
        "request": {"messages": request_data.get("messages", [])},
        "response": response_data,
        "usage": response_data.get("usage", {}),
        "timestamp": timestamp,
    }


class _CaptureHandler(http.server.BaseHTTPRequestHandler):
    """Forward-and-log request handler for the capture shim."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # suppress default stderr output

    def _drain_body(self) -> None:
        """Consume the declared request body before an early-return response.

        Closing a connection with unread bytes still in the socket buffer makes
        macOS and Windows emit a TCP RST, which the client surfaces as
        RemoteDisconnected / ConnectionResetError instead of reading our status
        line.  Draining first lets the client read a clean response.  No-op when
        no valid Content-Length is declared (chunked / malformed — those rarer
        error paths may still RST).  Bounded by _MAX_BODY so a draining read can
        never be turned into a memory-exhaustion vector.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (ValueError, TypeError):
            return
        remaining = min(length, _MAX_BODY)
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def do_POST(self) -> None:
        if self.path not in _CAPTURE_PATHS:
            # Drain the body first: a client POSTing to an unknown path would
            # otherwise get a TCP RST (RemoteDisconnected) on macOS / Windows
            # when we close the connection with its body still unread.
            self._drain_body()
            self.send_response(404)
            self.end_headers()
            return

        # Chunked transfer has no Content-Length; reading 0 bytes would forward
        # an empty request and silently log a useless record.  Reject it clearly
        # rather than corrupt the user's traffic (411 Length Required).  A chunked
        # body cannot be drained via Content-Length, so this rarer path may RST.
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self.send_response(411)
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except (ValueError, TypeError):
            # Malformed Content-Length: length is untrustworthy, so we cannot
            # safely drain; this rarer path may RST on macOS / Windows.
            self.send_response(400)
            self.end_headers()
            return
        if length > _MAX_BODY:
            # Deliberately NOT drained: reading an oversized body would defeat
            # the _MAX_BODY cap, so an over-limit client may observe a RST.
            self.send_response(413)
            self.end_headers()
            return
        request_body = self.rfile.read(length)

        srv = cast(CaptureServer, self.server)
        upstream_url = srv.upstream.rstrip("/") + self.path

        req = urllib.request.Request(upstream_url, data=request_body, method="POST")
        req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
        auth = self.headers.get("Authorization")
        if auth:
            req.add_header("Authorization", auth)

        status: int
        response_body: bytes
        content_type: str

        try:
            with srv._opener.open(req, timeout=srv._upstream_timeout) as resp:
                status = resp.status
                response_body = resp.read(_MAX_BODY)
                content_type = resp.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            status = exc.code
            response_body = exc.read(_MAX_BODY)
            content_type = "application/json"
        except TimeoutError:
            # urllib wraps connect-phase timeouts in URLError, but response-read
            # timeouts escape as bare TimeoutError — catch both paths.
            self.send_response(504)
            self.end_headers()
            return
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                self.send_response(504)
                self.end_headers()
                return
            self.send_response(502)
            self.end_headers()
            return

        body_unparseable = False
        try:
            request_data: dict[str, Any] = json.loads(request_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_data = {}
            body_unparseable = True

        try:
            response_data: dict[str, Any] = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_data = {}
            body_unparseable = True

        # Trace silently-dropped bodies at --verbose so the downstream skip is
        # never invisible (§4 fail-loud): a record with an unparseable body
        # produces a near-empty record that `analyze` later skips.
        if body_unparseable and srv.feedback.verbose:
            print(
                "frugon capture: 1 call captured with an unparseable body "
                "(it will not contribute usable token counts).",
                file=sys.stderr,
            )

        record = _build_record(request_data, response_data, _now_utc())
        # A logging failure must NEVER break the user's own traffic: isolate the
        # write so the upstream response is still returned even if the append
        # fails (disk full, path revoked mid-run).
        try:
            srv.write_record(record)
        except OSError as exc:
            print(
                f"frugon capture: WARNING could not log call to disk: {exc}. "
                "Your response was still returned; the call was not saved.",
                file=sys.stderr,
            )
        srv.feedback.on_call(record, status)

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


class CaptureServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded local OpenAI-compatible logger.

    Accepts requests on *port*, writes canonical JSONL to *out_path*, forwards
    each call to *upstream*, and returns the upstream response verbatim.
    """

    allow_reuse_address = True
    daemon_threads = True  # request threads do not block interpreter exit

    def __init__(
        self,
        port: int,
        out_path: pathlib.Path,
        upstream: str,
        *,
        host: str = "127.0.0.1",
        verbosity: str = "normal",
        allow_insecure_upstream: bool = False,
        upstream_timeout: float | None = _DEFAULT_UPSTREAM_TIMEOUT,
    ) -> None:
        _validate_upstream(upstream, allow_insecure_upstream=allow_insecure_upstream)
        # Validate output path writability before binding the port so errors
        # surface immediately with a clear message rather than mid-startup.
        try:
            with out_path.open("a", encoding="utf-8", newline=""):
                pass
        except OSError as exc:
            raise OSError(f"cannot write to output path {out_path}: {exc}") from exc
        super().__init__((host, port), _CaptureHandler)
        self.out_path = out_path
        self.upstream = upstream
        self._upstream_timeout = upstream_timeout
        self._opener: urllib.request.OpenerDirector = _build_restricted_opener()
        self.feedback = _FeedbackStream(verbosity)
        # newline="" disables universal-newline translation so the literal
        # "\n" written below reaches disk unchanged on every OS (no \r\n on
        # Windows), keeping the JSONL artifact byte-identical across platforms.
        self._out_file: IO[str] = out_path.open("a", encoding="utf-8", newline="")
        self._file_lock = threading.Lock()

    def write_record(self, record: dict[str, Any]) -> None:
        """Append one JSON-serialised record + newline to the output file."""
        line = json.dumps(record, ensure_ascii=False)
        with self._file_lock:
            self._out_file.write(line + "\n")
            self._out_file.flush()

    def close_output(self) -> None:
        """Flush and close the output file. Safe to call multiple times."""
        with self._file_lock:
            try:
                self._out_file.flush()
                self._out_file.close()
            except (OSError, ValueError):
                pass


def run_capture(
    port: int = 8787,
    out_path: pathlib.Path = pathlib.Path("capture.jsonl"),
    upstream: str | None = None,
    *,
    verbosity: str = "normal",
    allow_insecure_upstream: bool = False,
    upstream_timeout: float | None = _DEFAULT_UPSTREAM_TIMEOUT,
) -> None:
    """Start the capture server and block until interrupted (Ctrl+C).

    Parameters
    ----------
    port:
        Local port to listen on (default 8787).
    out_path:
        File path where captured JSONL records are appended.
    upstream:
        Base URL of the upstream OpenAI-compatible API. Falls back to the
        ``OPENAI_BASE_URL`` environment variable, then ``https://api.openai.com``.
    verbosity:
        Feedback level for captured calls:

        - ``"normal"`` (default) — a single in-place updating counter line
          (carriage-return rewrite). Degrades to silence when stdout is not a
          TTY to avoid corrupting log files.
        - ``"quiet"`` — no per-call output after startup.
        - ``"verbose"`` — one newline-terminated line per call with timestamp,
          model, token counts, and upstream HTTP status.
    allow_insecure_upstream:
        Allow plain ``http://`` to non-localhost hosts. Intended for local
        development proxies only (e.g. Ollama on a LAN address).
    upstream_timeout:
        Connect+read timeout in seconds for upstream requests. Hung upstreams
        are cut at this deadline and the caller receives a 504. Pass ``None``
        to disable the timeout (not recommended). Note: ``0`` means
        non-blocking (instant fail), not disabled. Default: 60 s.
    """
    if upstream is None:
        upstream = os.environ.get("OPENAI_BASE_URL", _DEFAULT_UPSTREAM)

    try:
        server = CaptureServer(
            port=port,
            out_path=out_path,
            upstream=upstream,
            verbosity=verbosity,
            allow_insecure_upstream=allow_insecure_upstream,
            upstream_timeout=upstream_timeout,
        )
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(1)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.close_output()
