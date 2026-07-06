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
import signal
import socketserver
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from types import FrameType
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


def _build_restricted_opener(proxy: str | None = None) -> urllib.request.OpenerDirector:
    """Return an OpenerDirector that handles ONLY http:// and https://.

    Unlike urllib.request.build_opener(), this does NOT register the default
    FileHandler / FTPHandler / DataHandler. Any non-HTTP scheme falls through
    to UnknownHandler, which raises URLError('unknown url type: <scheme>').
    Defence-in-depth below _validate_upstream.

    Proxy policy (privacy invariant): by default NO proxy is used. An empty
    ``ProxyHandler({})`` deliberately ignores the ambient ``HTTP_PROXY`` /
    ``HTTPS_PROXY`` environment, so the developer's request — and its
    ``Authorization`` header — goes straight to their own provider, never
    silently through a third-party proxy. Pass an explicit *proxy* URL (the
    ``--proxy`` flag) to opt in to routing through it knowingly.
    """
    proxies: dict[str, str] = {}
    if proxy:
        # Mirror _validate_upstream's scheme allowlist: an explicit proxy must be
        # http(s):// (urllib cannot route SOCKS/file/etc. anyway). Reject up front
        # with a clear message instead of failing deep inside urllib later.
        scheme = urllib.parse.urlsplit(proxy).scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"proxy scheme {scheme or '(empty)'!r} is not allowed; "
                "use an http:// or https:// proxy URL"
            )
        proxies = {"http": proxy, "https": proxy}
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(proxies),
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

    def _reject_json(self, *, status: int, message: str) -> None:
        """Send a JSON error body ``{"error": {"message": ...}}`` at *status*.

        Used for request-shape rejections (e.g. unsupported ``stream: true``)
        where the client benefits from an actionable message rather than a
        bare status code with an empty body.  The request body must already
        be fully read by the caller before calling this (no draining is
        performed here).
        """
        payload = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path not in _CAPTURE_PATHS:
            # Drain the body first: a client POSTing to an unknown path would
            # otherwise get a TCP RST (RemoteDisconnected) on macOS / Windows
            # when we close the connection with its body still unread.
            self._drain_body()
            cast(CaptureServer, self.server).warn_unknown_path_once(self.path)
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

        # Reject streaming (SSE) requests explicitly (FRG-OSS-018) rather than
        # silently breaking them.  capture's do_POST reads the FULL upstream
        # response body via `resp.read(_MAX_BODY)` before ever writing
        # anything back to the client — the opposite of SSE's incremental
        # chunk-as-you-go contract.  Forwarding a `"stream": true` request
        # today would make the caller wait for the ENTIRE completion to
        # buffer, then receive it as one lump instead of a token stream:  a
        # silent behavioural break, not a clean failure.  A malformed/
        # non-JSON body is deliberately NOT rejected here — that is handled
        # by the existing best-effort JSON parse below, which already
        # degrades to an empty record rather than blocking the call.  A
        # syntactically-valid-but-non-object JSON body (a bare array, string,
        # number, bool, or null — e.g. `"stream"` or `[1,2,3]`) parses without
        # raising JSONDecodeError, so it must be explicitly excluded before
        # calling ``.get`` on it: an OpenAI-shaped request is always a JSON
        # object, so anything else is simply not a candidate for the
        # stream-flag check, not a crash.
        try:
            _parsed_body: Any = json.loads(request_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            _parsed_body = {}
        _stream_check: dict[str, Any] = _parsed_body if isinstance(_parsed_body, dict) else {}
        if _stream_check.get("stream") is True:
            self._reject_json(
                status=400,
                message=(
                    "frugon capture does not support streaming (SSE) responses yet. "
                    'Remove "stream": true (or set it to false) from your request, '
                    "or call the provider directly for streaming calls."
                ),
            )
            return

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

        # A syntactically-valid-but-non-object JSON body (a bare array, string,
        # number, bool, or null) parses without raising JSONDecodeError, but
        # is not a dict — treating it as one crashes _build_record's `.get()`
        # calls with an unhandled AttributeError.  An OpenAI-shaped
        # request/response is always a JSON object, so anything else is
        # exactly as "unparseable" for our purposes as invalid JSON: it
        # degrades to an empty record rather than crashing the handler thread.
        body_unparseable = False
        try:
            _request_parsed: Any = json.loads(request_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            _request_parsed = None
        if isinstance(_request_parsed, dict):
            request_data: dict[str, Any] = _request_parsed
        else:
            request_data = {}
            body_unparseable = True

        try:
            _response_parsed: Any = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            _response_parsed = None
        if isinstance(_response_parsed, dict):
            response_data: dict[str, Any] = _response_parsed
        else:
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
        proxy: str | None = None,
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
        self._opener: urllib.request.OpenerDirector = _build_restricted_opener(proxy)
        self.feedback = _FeedbackStream(verbosity)
        # newline="" disables universal-newline translation so the literal
        # "\n" written below reaches disk unchanged on every OS (no \r\n on
        # Windows), keeping the JSONL artifact byte-identical across platforms.
        self._out_file: IO[str] = out_path.open("a", encoding="utf-8", newline="")
        self._file_lock = threading.Lock()
        # Privacy hardening (FRG-OSS-019): capture.jsonl holds the FULL prompt
        # and completion text of every captured call — as sensitive as an API
        # key or a database dump.  Restrict to owner-read/write (0o600) on
        # every start, whether the file was just created (umask may have left
        # it group/world-readable) OR already existed from a prior run (a
        # loose mode from before this hardening landed must also be tightened,
        # not just left as-is).  os.chmod is a POSIX-only permission model —
        # Windows has no equivalent bit pattern, so this is a no-op there
        # (guarded by hasattr, matching the codebase's existing Windows-guard
        # convention); Windows users get the amber caution line in the
        # startup panel instead (see cli.py's `capture` command).
        if hasattr(os, "chmod"):
            try:
                os.chmod(out_path, 0o600)
            except OSError:
                # Best-effort: an unusual filesystem (network share, some
                # Docker bind-mount overlays) may reject chmod even though the
                # preceding open() succeeded.  Never let a permissions
                # tightening failure block capture from starting — the
                # caution line in the startup panel still applies regardless.
                pass
        # One-time-per-path unmatched-request tracking (FRG-OSS-045): a client
        # POSTing to a path outside _CAPTURE_PATHS (e.g. Anthropic's
        # /v1/messages, OpenAI's /v1/responses) gets a 404 with ZERO other
        # signal today — easy to misread as "capture is broken" rather than
        # "this provider/endpoint shape isn't supported yet".  Guarded by a
        # lock because ThreadingMixIn dispatches concurrent requests onto
        # separate handler threads that all share this one server instance.
        self._warned_unknown_paths: set[str] = set()
        self._warned_unknown_paths_lock = threading.Lock()

    def warn_unknown_path_once(self, path: str) -> None:
        """Emit a one-time stderr warning for an unmatched capture path.

        No-op on every call after the first for a given *path* (tracked for
        the lifetime of this server instance) — a client hammering the same
        wrong endpoint must not flood stderr with a repeated warning.
        """
        with self._warned_unknown_paths_lock:
            if path in self._warned_unknown_paths:
                return
            self._warned_unknown_paths.add(path)
        print(
            f"frugon capture: WARNING received a request for {path!r}, which "
            "frugon capture does not recognise. Supported paths: "
            f"{sorted(_CAPTURE_PATHS)}.",
            file=sys.stderr,
        )

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
    proxy: str | None = None,
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
    proxy:
        Optional proxy URL to route upstream calls through. By default no proxy
        is used and the ambient HTTP(S)_PROXY environment is ignored, so the
        user's API key goes straight to their provider — never through a third
        party. Set this to opt in to a proxy knowingly.
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
            proxy=proxy,
        )
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(1)

    # SIGTERM handling (FRG-OSS-043): before this, `kill <pid>` / systemd stop
    # / a container orchestrator's shutdown signal terminated the process
    # WITHOUT ever running the `finally:` block below — the in-flight
    # capture.jsonl file handle was never flushed/closed, risking a truncated
    # final line.  POSIX only: Windows has no SIGTERM delivery model
    # equivalent to POSIX (a Windows `taskkill` maps closer to SIGKILL, which
    # no handler can intercept on any OS) — guarded by hasattr so this is a
    # clean no-op there rather than an AttributeError.
    #
    # The handler must NOT call server.shutdown() directly: shutdown() blocks
    # until serve_forever()'s poll loop observes the stop request and exits —
    # but signal handlers run ON THE THREAD THEY INTERRUPT, which here is the
    # very same thread currently blocked inside serve_forever().  Calling
    # shutdown() synchronously from the handler would deadlock (the thread
    # would be waiting on itself).  Spawning shutdown() on a short-lived
    # background thread lets the handler return immediately, so
    # serve_forever()'s loop is free to notice the stop request and exit.
    if hasattr(signal, "SIGTERM"):

        def _handle_sigterm(signum: int, frame: FrameType | None) -> None:
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.close_output()
