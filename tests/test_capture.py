"""Tests for frugon capture — local passive OpenAI-compatible logger.

Covers:
- _build_record unit tests (canonical JSONL shape)
- Integration: records canonical JSONL line for a mocked upstream
- Forwarding transparency (status + body unchanged)
- Authorization header pass-through
- Privacy invariant (no egress to Rodiun/Frugon hosts)
- Graceful shutdown flushes the file
- Unknown path returns 404
- UTF-8 / unicode content handled correctly
- Round-trip: captured file passes through frugon analyze
"""

from __future__ import annotations

import contextlib
import http.client
import http.server
import json
import pathlib
import signal
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Generator
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import frugon.capture as capture_mod
from frugon.capture import (
    CaptureServer,
    _build_record,
    _build_restricted_opener,
    _validate_upstream,
    run_capture,
)
from frugon.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_SAMPLE_REQUEST: dict[str, Any] = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Say hello."}],
}

_SAMPLE_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello! How can I help you?"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
}

# ---------------------------------------------------------------------------
# Deterministic server lifecycle helper
# ---------------------------------------------------------------------------
#
# Design rationale — why the old sleep-based approach was flaky
# ─────────────────────────────────────────────────────────────
# The old code used time.sleep(0.02) after thread.start() to "let serve_forever
# enter the select loop."  Two independent races resulted:
#
#   Race A — readiness: serve_forever has not yet called selector.register()
#   when the test sends its first request.  The connection sits in the kernel
#   backlog and is accepted on the *next* poll interval (up to 0.5 s later),
#   making the test slower but not wrong.  However on a loaded CI runner the
#   20 ms sleep sometimes expired before the thread was even scheduled,
#   meaning the connection arrived before any accept() was ever called.
#
#   Race B — teardown: server.shutdown() was called immediately after _post()
#   returned.  _post() finishes reading the response (resp.read()), but with
#   CaptureServer(daemon_threads=True) the handler thread is NOT joined by
#   server_close() (socketserver._Threads.append silently skips daemon threads).
#   shutdown() signals the select loop to stop; server_close() joins no threads;
#   the handler may still be executing wfile.write() when the server's socket is
#   closed — the OS sends a TCP RST and the client (or a later assertion) sees
#   RemoteDisconnected or ConnectionResetError.
#
# Fix
# ───
# running_server() provides three guarantees before yielding control:
#
#   1. Readiness: serve_forever is confirmed running by probing the socket
#      with a real TCP connect+close (no sleep, no assumptions about scheduling).
#
#   2. Handler drain before shutdown: an in-flight counter (protected by a lock
#      and drained via a threading.Event) ensures every handler thread has
#      returned before shutdown() is called.
#
#   3. Clean socket release: shutdown() + thread.join() + server_close() run
#      unconditionally in the finally block.
#
# The handler-drain wrapper (below) is injected as a mixin so that EVERY
# subclass used in these tests — _StubServer, _Http4xxServer, CaptureServer
# itself, and the redirect / large-body servers — all gets the same guarantee
# without duplicating bookkeeping code.


class _DrainMixin:
    """Mixin that tracks in-flight request handlers and provides drain().

    Inject as the *first* base before the TCPServer/CaptureServer so that
    process_request_thread is called on the mixin before the real handler.

    Usage in tests: call server.drain() AFTER the last _post() call and
    BEFORE server.shutdown() to guarantee all handlers have finished writing.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._inflight_count = 0
        self._inflight_lock = threading.Lock()
        self._all_done = threading.Event()
        self._all_done.set()  # starts "done" (no in-flight requests)
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]

    def process_request_thread(  # type: ignore[override]
        self, request: Any, client_address: Any
    ) -> None:
        with self._inflight_lock:
            self._inflight_count += 1
            self._all_done.clear()
        try:
            super().process_request_thread(request, client_address)  # type: ignore[misc]
        finally:
            with self._inflight_lock:
                self._inflight_count -= 1
                if self._inflight_count == 0:
                    self._all_done.set()

    def drain(self, timeout: float = 5.0) -> None:
        """Block until all in-flight request handlers have returned.

        Call this after the last _post() call and before server.shutdown()
        to prevent the teardown RST race described in the module docstring.
        """
        if not self._all_done.wait(timeout=timeout):
            raise TimeoutError(
                f"server handlers did not complete within {timeout} s"
            )


class _DrainStubServer(_DrainMixin, socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Thread-per-request TCPServer with DrainMixin for deterministic teardown."""

    allow_reuse_address = True
    daemon_threads = True  # don't block interpreter exit

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # silence stray BrokenPipeError tracebacks at teardown


@contextlib.contextmanager
def running_server(
    server: Any,
    *,
    drain_before_shutdown: bool = True,
) -> Generator[Any, None, None]:
    """Context manager: start *server*, guarantee readiness, yield, then shut down cleanly.

    Args:
        server: Any socketserver.TCPServer instance (or subclass).
        drain_before_shutdown: if True (default), call server.drain() before
            shutdown() so all in-flight handlers finish writing their response
            before the server socket is closed.  Set False for servers that do
            not inherit from _DrainMixin (raw low-level tests).

    Guarantee 1 — readiness before yield:
        Probes the bound port with a real TCP connect/close loop (max 50 × 10 ms
        = 500 ms) rather than sleeping a fixed duration.  This is zero-race
        because a successful accept() proves serve_forever is in its select loop.

    Guarantee 2 — handler drain:
        Waits for all in-flight handlers to finish before calling shutdown()
        (requires the server to inherit _DrainMixin).

    Guarantee 3 — clean teardown:
        Calls shutdown(), joins the server thread (timeout 5 s), server_close().
    """
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Wait until serve_forever has entered its select loop and is accepting
    # connections.  We probe by connecting and immediately closing; this is
    # accepted into a handler thread (which exits immediately on close) only
    # once serve_forever is running.  At most 50 × 10 ms = 500 ms.
    host, port = server.server_address[:2]
    for _ in range(50):
        try:
            probe = socket.create_connection((host, port), timeout=0.05)
            probe.close()
            break
        except OSError:
            pass
    else:
        raise RuntimeError(
            f"server on {host}:{port} did not become ready within 500 ms"
        )

    try:
        yield server
    finally:
        if drain_before_shutdown and hasattr(server, "drain"):
            server.drain()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


# ---------------------------------------------------------------------------
# Stub upstream server
# ---------------------------------------------------------------------------


class _StubHandler(http.server.BaseHTTPRequestHandler):
    """Returns a configurable JSON response; records each received request."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        srv = cast("_StubServer", self.server)
        srv.received.append(
            {"path": self.path, "body": body, "headers": dict(self.headers)}
        )
        resp_bytes = json.dumps(srv.response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)
        self.wfile.flush()


class _StubServer(_DrainStubServer):
    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.response: dict[str, Any] = response
        self.received: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_upstream() -> Generator[tuple[str, _StubServer], None, None]:
    """Start a stub upstream on a free port; yield (base_url, server); shut down after."""
    server = _StubServer(_SAMPLE_RESPONSE)
    with running_server(server) as srv:
        port: int = srv.server_address[1]
        yield f"http://127.0.0.1:{port}", srv


@pytest.fixture
def capture_srv(
    tmp_path: pathlib.Path,
    stub_upstream: tuple[str, _StubServer],
) -> Generator[tuple[CaptureServer, int, pathlib.Path], None, None]:
    """Start a CaptureServer on a free port pointing at stub_upstream."""
    upstream_url, _ = stub_upstream
    out_file = tmp_path / "captured.jsonl"
    server = _DrainCaptureServer(port=0, out_path=out_file, upstream=upstream_url)
    cap_port: int = server.server_address[1]
    with running_server(server) as srv:
        try:
            yield srv, cap_port, out_file
        finally:
            srv.close_output()


# ---------------------------------------------------------------------------
# _DrainCaptureServer: CaptureServer + _DrainMixin for deterministic teardown
# ---------------------------------------------------------------------------


class _DrainCaptureServer(_DrainMixin, CaptureServer):
    """CaptureServer instrumented with the _DrainMixin handler-drain guarantee.

    The MRO is: _DrainCaptureServer → _DrainMixin → CaptureServer →
    ThreadingMixIn → TCPServer.  _DrainMixin.process_request_thread wraps
    CaptureServer's (via ThreadingMixIn), so every handler thread is counted.
    """


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _post(
    port: int, path: str, body: dict[str, Any], headers: dict[str, str] | None = None
) -> tuple[int, bytes]:
    """Send a POST to 127.0.0.1:<port><path>; return (status_code, response_body).

    Uses a fresh HTTPConnection per call with an explicit timeout.  The caller
    is responsible for draining in-flight handlers (via server.drain()) before
    calling server.shutdown() — see running_server().
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    data = json.dumps(body).encode("utf-8")
    req_headers: dict[str, str] = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    try:
        conn.request("POST", path, body=data, headers=req_headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests — _build_record
# ---------------------------------------------------------------------------


def test_build_record_canonical_shape_contains_required_keys() -> None:
    """Arrange: sample request, response, and timestamp.
    Act: call _build_record.
    Assert: all keys frugon analyze ingests are present with correct values.
    """
    # Arrange
    ts = "2025-01-01T00:00:00Z"

    # Act
    record = _build_record(_SAMPLE_REQUEST, _SAMPLE_RESPONSE, ts)

    # Assert — model, request.messages, response, usage, timestamp
    assert record["model"] == "gpt-4o"
    assert record["request"]["messages"] == _SAMPLE_REQUEST["messages"]
    assert record["response"] == _SAMPLE_RESPONSE
    assert record["usage"] == _SAMPLE_RESPONSE["usage"]
    assert record["timestamp"] == ts


def test_build_record_empty_inputs_produce_safe_defaults() -> None:
    """Arrange: empty dicts.
    Act: call _build_record.
    Assert: result has safe default values (no KeyError, no None in required fields).
    """
    # Act
    record = _build_record({}, {}, "2025-01-01T00:00:00Z")

    # Assert
    assert record["model"] == ""
    assert record["request"]["messages"] == []
    assert record["response"] == {}
    assert record["usage"] == {}


def test_build_record_usage_comes_from_response() -> None:
    """Arrange: response with usage block.
    Act: call _build_record.
    Assert: usage in record matches response.usage.
    """
    # Arrange
    response = {"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}

    # Act
    record = _build_record({}, response, "2025-01-01T00:00:00Z")

    # Assert
    assert record["usage"]["prompt_tokens"] == 5
    assert record["usage"]["completion_tokens"] == 3


# ---------------------------------------------------------------------------
# Integration tests — capture server
# ---------------------------------------------------------------------------


def test_capture_records_canonical_jsonl_line(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server pointing at stub upstream.
    Act: POST /v1/chat/completions.
    Assert: one canonical JSONL line written; all analyze-required fields present.
    """
    # Arrange
    srv, port, out_file = capture_srv

    # Act
    status, _ = _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()  # ensure handler has written to file before asserting

    # Assert — HTTP status
    assert status == 200

    # Assert — JSONL content
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, f"Expected 1 JSONL line, got {len(lines)}"

    record = json.loads(lines[0])
    assert record["model"] == "gpt-4o"
    assert isinstance(record["request"]["messages"], list)
    assert record["usage"]["prompt_tokens"] == 12
    assert record["usage"]["completion_tokens"] == 9
    assert "timestamp" in record
    assert record["timestamp"]


def test_capture_records_completions_endpoint(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server.
    Act: POST /v1/completions (legacy endpoint).
    Assert: record written; status 200.
    """
    # Arrange
    srv, port, out_file = capture_srv

    # Act
    status, _ = _post(port, "/v1/completions", _SAMPLE_REQUEST)
    srv.drain()

    # Assert
    assert status == 200
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_capture_forwarding_transparent_status_and_body(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: stub upstream returns a known response.
    Act: POST to capture server.
    Assert: client receives exactly the upstream response (status + body unchanged).
    """
    # Arrange
    _, port, _ = capture_srv

    # Act
    status, body = _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)

    # Assert
    assert status == 200
    assert json.loads(body.decode("utf-8")) == _SAMPLE_RESPONSE


def test_capture_forwarding_passes_authorization_header(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
    stub_upstream: tuple[str, _StubServer],
) -> None:
    """Arrange: capture server + stub upstream.
    Act: POST with Authorization: Bearer secret.
    Assert: stub upstream received the Authorization header unchanged.
    """
    # Arrange
    srv, port, _ = capture_srv
    _, stub = stub_upstream

    # Act
    _post(
        port,
        "/v1/chat/completions",
        _SAMPLE_REQUEST,
        headers={"Authorization": "Bearer test-key-xyz"},
    )
    srv.drain()

    # Assert
    assert stub.received, "Stub upstream received no requests"
    auth = stub.received[-1]["headers"].get("Authorization", "")
    assert auth == "Bearer test-key-xyz", f"Authorization not forwarded; got: {auth!r}"


def test_capture_privacy_no_egress_to_rodiun_frugon(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: track all outbound socket connections.
    Act: POST to capture server (which forwards to stub upstream on 127.0.0.1).
    Assert: no connection made to any Rodiun/Frugon host.

    This is the local-only privacy invariant for capture.
    """
    # Arrange — track connected hosts WITHOUT blocking them
    srv, port, _ = capture_srv
    connected_hosts: list[str] = []
    original_connect = socket.socket.connect

    def _tracking_connect(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address:
            connected_hosts.append(str(address[0]))
        original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", _tracking_connect)

    # Act
    _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()

    # Assert — no Rodiun or Frugon host in the egress list
    bad = [h for h in connected_hosts if "rodiun" in h.lower() or "frugon" in h.lower()]
    assert not bad, (
        f"Capture shim made an unexpected outbound connection: {bad}\n"
        "The capture shim must only connect to the configured upstream."
    )


def test_capture_outbound_body_is_exactly_the_client_request(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
    stub_upstream: tuple[str, _StubServer],
) -> None:
    """Arrange: capture server + stub upstream.
    Act: POST a known request.
    Assert: the body the upstream received is byte-for-byte the client request —
            proving the shim adds no synthesized telemetry payload (§5).

    Asserting the SHAPE (exact forwarded bytes), not just the host: a privacy
    promise of "sends nothing" must prove nothing extra is appended.
    """
    srv, port, _ = capture_srv
    _, stub = stub_upstream

    _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()

    assert stub.received, "Stub upstream received no requests"
    assert stub.received[-1]["body"] == json.dumps(_SAMPLE_REQUEST).encode("utf-8"), (
        "The forwarded body must be exactly the client request — no extra "
        "telemetry may be synthesized into the outbound payload."
    )


def test_capture_write_failure_still_returns_response(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: make write_record raise OSError (disk full / path revoked).
    Act: POST a request.
    Assert: the user still gets the upstream response (status 200); a logging
            failure never breaks the user's own traffic (§4 fail-loud).
    """
    srv, port, _ = capture_srv

    def _boom(_record: dict[str, Any]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(srv, "write_record", _boom)

    status, body = _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()
    assert status == 200, "Upstream response must still be returned despite a write failure"
    assert body, "Response body must be forwarded to the client"


def test_capture_graceful_shutdown_flushes_file(
    tmp_path: pathlib.Path,
    stub_upstream: tuple[str, _StubServer],
) -> None:
    """Arrange: capture server; POST a request.
    Act: shut down the server.
    Assert: JSONL file contains the complete record (flush occurred before close).
    """
    # Arrange
    upstream_url, _ = stub_upstream
    out_file = tmp_path / "shutdown_test.jsonl"
    server = _DrainCaptureServer(port=0, out_path=out_file, upstream=upstream_url)

    with running_server(server) as srv:
        cap_port: int = srv.server_address[1]

        # Act — POST then drain (running_server() shuts down in finally)
        _post(cap_port, "/v1/chat/completions", _SAMPLE_REQUEST)
        srv.drain()

    # running_server finalizer called shutdown/server_close; close file explicitly
    server.close_output()

    # Assert — file is a complete, parseable JSONL line (not truncated)
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, f"Expected 1 line after shutdown, got {len(lines)}"
    record = json.loads(lines[0])  # ValueError if truncated
    assert record["model"] == "gpt-4o"


def test_capture_unknown_path_returns_404(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server.
    Act: POST to an unrecognised path (with a body).
    Assert: 404 returned cross-platform (the handler drains the body before
    the early return, so macOS / Windows no longer RST); nothing written.
    """
    # Arrange
    srv, port, out_file = capture_srv

    # Act
    status, _ = _post(port, "/v1/unknown_endpoint", _SAMPLE_REQUEST)
    srv.drain()

    # Assert
    assert status == 404
    assert not out_file.exists() or out_file.read_text(encoding="utf-8").strip() == ""


def test_capture_unknown_path_emits_one_time_stderr_warning(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FRG-OSS-045: an unmatched path (e.g. Anthropic's /v1/messages) gets a
    one-time stderr warning naming the path and the supported paths — instead
    of silently 404ing with zero other signal.

    Arrange: capture server.
    Act: POST to /v1/messages (an unsupported provider path) TWICE.
    Assert: exactly ONE warning line is emitted (not two — the "once" in
    one-time), and it names both the unmatched path and the supported paths.
    """
    srv, port, out_file = capture_srv

    status1, _ = _post(port, "/v1/messages", _SAMPLE_REQUEST)
    status2, _ = _post(port, "/v1/messages", _SAMPLE_REQUEST)
    srv.drain()

    assert status1 == 404
    assert status2 == 404

    err = capsys.readouterr().err
    warning_lines = [line for line in err.splitlines() if "/v1/messages" in line]
    assert len(warning_lines) == 1, (
        f"expected exactly one warning for a repeated unmatched path, got "
        f"{len(warning_lines)}:\n{err}"
    )
    assert "/v1/chat/completions" in warning_lines[0]
    assert "/v1/completions" in warning_lines[0]


def test_capture_unknown_path_warning_is_per_distinct_path(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FRG-OSS-045: the one-time suppression is PER PATH, not global — a
    second, DIFFERENT unmatched path still gets its own warning.

    Arrange: capture server.
    Act: POST to two distinct unsupported paths.
    Assert: two distinct warning lines, one per path.
    """
    srv, port, out_file = capture_srv

    _post(port, "/v1/messages", _SAMPLE_REQUEST)
    _post(port, "/v1/responses", _SAMPLE_REQUEST)
    srv.drain()

    err = capsys.readouterr().err
    assert "/v1/messages" in err
    assert "/v1/responses" in err


def test_capture_multiple_requests_append_multiple_lines(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server.
    Act: POST three requests.
    Assert: three JSONL lines in the output file.
    """
    # Arrange
    srv, port, out_file = capture_srv

    # Act
    for _ in range(3):
        _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()

    # Assert
    lines = [ln for ln in out_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
    for line in lines:
        json.loads(line)  # each must be valid JSON


def test_capture_record_terminator_is_lf_only_on_every_os(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server.
    Act: POST two requests so the file has two record terminators.
    Assert: the on-disk bytes use LF ("\\n") terminators with no carriage
            returns — the canonical JSONL artifact is byte-identical across
            Linux, macOS and Windows.

    Without newline="" on the output handle, Windows text-mode writes would
    translate each "\\n" to "\\r\\n", making the public data-interchange file
    platform-dependent and leaving a trailing "\\r" on every record for any
    consumer that splits strictly on "\\n".
    """
    # Arrange
    srv, port, out_file = capture_srv

    # Act
    for _ in range(2):
        _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()

    # Assert — raw bytes carry no carriage returns; LF terminates each record
    raw = out_file.read_bytes()
    assert b"\r" not in raw, (
        "Captured JSONL must use LF-only record terminators on every OS; "
        f"found a carriage return in {raw!r}"
    )
    lf_count = raw.count(b"\n")
    assert lf_count == 2, (
        f"Expected two LF record terminators, got {lf_count} in {raw!r}"
    )


def test_capture_file_written_as_utf8_with_unicode_content(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: request with unicode content.
    Act: POST to capture server.
    Assert: output file is valid UTF-8 and preserves the unicode content.
    """
    # Arrange
    srv, port, out_file = capture_srv
    unicode_request: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Translate: 日本語テスト 🌸"}],
    }

    # Act
    _post(port, "/v1/chat/completions", unicode_request)
    srv.drain()

    # Assert — read with explicit utf-8; content round-trips
    content = out_file.read_text(encoding="utf-8")
    record = json.loads(content.splitlines()[0])
    messages = record["request"]["messages"]
    assert any("日本語" in m.get("content", "") for m in messages), (
        "Unicode content not preserved in captured JSONL record"
    )


def test_capture_output_uses_pathlib_path(tmp_path: pathlib.Path) -> None:
    """Assert: CaptureServer.out_path is a pathlib.Path (cross-platform guarantee)."""
    server = CaptureServer(
        port=0, out_path=tmp_path / "check.jsonl", upstream="http://127.0.0.1:1"
    )
    try:
        assert isinstance(server.out_path, pathlib.Path), (
            "CaptureServer.out_path must be a pathlib.Path for cross-platform portability"
        )
    finally:
        server.server_close()
        server.close_output()


# ---------------------------------------------------------------------------
# Round-trip: captured file → frugon analyze
# ---------------------------------------------------------------------------


def test_capture_roundtrip_with_analyze(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server.
    Act: POST a request (writes JSONL), then run frugon analyze on the captured file.
    Assert: analyze exits 0 and reports a cost figure — file is analyze-compatible.
    """
    # Arrange
    srv, port, out_file = capture_srv

    # Act — capture one call
    _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()

    # Act — analyze the captured file
    result = runner.invoke(app, ["analyze", str(out_file)])

    # Assert
    assert result.exit_code == 0, (
        f"frugon analyze exited {result.exit_code} on captured JSONL file.\n"
        f"Output:\n{result.output}"
        + (f"\nException: {result.exception}" if result.exception else "")
    )
    # A cost report must contain a dollar amount or token count
    assert "$" in result.output or "token" in result.output.lower(), (
        f"No cost figure found in analyze output:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# Error-path coverage (lines 87-94, 98-99, 103-104, 152-153, 162-173)
# ---------------------------------------------------------------------------


class _Http4xxHandler(http.server.BaseHTTPRequestHandler):
    """Upstream stub that returns 400 with a plain-text (non-JSON) body.

    The non-JSON body exercises the json.JSONDecodeError fallback for the
    response (lines 103-104 of capture.py) in the same request.
    """

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = b"Bad Request"
        self.send_response(400)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


class _Http4xxServer(_DrainStubServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Http4xxHandler)


@pytest.fixture
def http4xx_upstream() -> Generator[tuple[str, _Http4xxServer], None, None]:
    """Stub that always returns 400 with a non-JSON body."""
    server = _Http4xxServer()
    with running_server(server) as srv:
        port: int = srv.server_address[1]
        yield f"http://127.0.0.1:{port}", srv


def test_capture_upstream_http_error_forwarded_to_client(
    tmp_path: pathlib.Path,
    http4xx_upstream: tuple[str, _Http4xxServer],
) -> None:
    """Arrange: upstream always returns 400 with a non-JSON body.
    Act: POST to capture server.
    Assert: client receives 400; record written with empty response dict.

    Covers capture.py lines 87-90 (HTTPError handler) and 103-104
    (response body JSON decode fallback).
    """
    # Arrange
    upstream_url, _ = http4xx_upstream
    out_file = tmp_path / "http_error.jsonl"
    server = _DrainCaptureServer(port=0, out_path=out_file, upstream=upstream_url)
    cap_port: int = server.server_address[1]

    with running_server(server) as srv:
        # Act
        status, _ = _post(cap_port, "/v1/chat/completions", _SAMPLE_REQUEST)
        srv.drain()

    server.close_output()

    # Assert
    assert status == 400
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "JSONL record must be written even for upstream error responses"
    record = json.loads(lines[0])
    assert record["response"] == {}, "Non-JSON error body must produce empty response dict"


def test_capture_upstream_url_error_returns_502(tmp_path: pathlib.Path) -> None:
    """Arrange: capture server pointing at a port where nothing is listening.
    Act: POST to capture server.
    Assert: capture server returns 502; nothing written to JSONL.

    Covers capture.py lines 91-94 (URLError handler).
    """
    # Arrange — port 1 is almost certainly not listening on 127.0.0.1
    out_file = tmp_path / "url_error.jsonl"
    server = _DrainCaptureServer(port=0, out_path=out_file, upstream="http://127.0.0.1:1")
    cap_port: int = server.server_address[1]

    with running_server(server) as srv:
        # Act
        status, _ = _post(cap_port, "/v1/chat/completions", _SAMPLE_REQUEST)
        srv.drain()

    server.close_output()

    # Assert
    assert status == 502
    assert out_file.read_text(encoding="utf-8").strip() == "", (
        "No JSONL record should be written when the upstream is unreachable"
    )


def test_capture_malformed_request_body_produces_safe_defaults(
    tmp_path: pathlib.Path,
    stub_upstream: tuple[str, _StubServer],
) -> None:
    """Arrange: send a non-JSON POST body through capture server.
    Act: capture server forwards to stub upstream and writes a JSONL record.
    Assert: record uses empty fallbacks for model and messages.

    Covers capture.py lines 98-99 (request body JSON decode fallback).
    """
    # Arrange
    upstream_url, _ = stub_upstream
    out_file = tmp_path / "bad_req.jsonl"
    server = _DrainCaptureServer(port=0, out_path=out_file, upstream=upstream_url)
    cap_port: int = server.server_address[1]

    with running_server(server) as srv:
        # Act — send raw non-JSON bytes (bypasses the _post JSON-encoding helper)
        conn = http.client.HTTPConnection("127.0.0.1", cap_port, timeout=5)
        raw = b"not valid json!!"
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw,
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        srv.drain()

    server.close_output()

    # Assert
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["model"] == "", "Non-JSON request must fall back to empty model"
    assert record["request"]["messages"] == [], "Non-JSON request must fall back to empty messages"


def test_capture_close_output_idempotent(tmp_path: pathlib.Path) -> None:
    """Arrange: CaptureServer with an open output file.
    Act: call close_output() twice.
    Assert: second call does not raise — OSError/ValueError silenced.

    Covers capture.py lines 152-153 (exception guard in close_output).
    """
    # Arrange
    out_file = tmp_path / "idempotent.jsonl"
    server = CaptureServer(port=0, out_path=out_file, upstream="http://127.0.0.1:1")
    try:
        server.close_output()  # first call: flushes + closes the file
        server.close_output()  # second call: file already closed → silenced exception
    finally:
        server.server_close()


def test_run_capture_resolves_upstream_from_env_and_shuts_down(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: set OPENAI_BASE_URL; mock CaptureServer so serve_forever raises KeyboardInterrupt.
    Act: call run_capture(upstream=None).
    Assert: upstream resolved from env var; full shutdown sequence executed.

    Covers capture.py lines 162-173 (entire run_capture body including env-var branch,
    KeyboardInterrupt handling, and the finally shutdown).
    """
    import frugon.capture as cap_mod
    from frugon.capture import run_capture

    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-provider.test")
    captured: dict[str, Any] = {}

    class _MockServer:
        def __init__(self, port: int, out_path: pathlib.Path, upstream: str, **_: Any) -> None:
            captured["upstream"] = upstream

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            captured["shutdown"] = True

        def server_close(self) -> None:
            captured["server_close"] = True

        def close_output(self) -> None:
            captured["close_output"] = True

    monkeypatch.setattr(cap_mod, "CaptureServer", _MockServer)

    run_capture(port=0, out_path=tmp_path / "env.jsonl", upstream=None)

    assert captured.get("upstream") == "http://env-provider.test", (
        "run_capture must resolve upstream from OPENAI_BASE_URL env var"
    )
    assert captured.get("shutdown"), "run_capture must call server.shutdown() in finally"
    assert captured.get("server_close"), "run_capture must call server.server_close() in finally"
    assert captured.get("close_output"), "run_capture must call server.close_output() in finally"


def test_run_capture_with_explicit_upstream_skips_env_resolution(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: mock CaptureServer; pass explicit upstream string.
    Act: call run_capture(upstream='http://explicit.test').
    Assert: explicit value used; OPENAI_BASE_URL env var not consulted.

    Covers capture.py line 162 (False branch — upstream already provided).
    """
    import frugon.capture as cap_mod
    from frugon.capture import run_capture

    monkeypatch.setenv("OPENAI_BASE_URL", "http://should-not-be-used.test")
    captured: dict[str, Any] = {}

    class _MockServer:
        def __init__(self, port: int, out_path: pathlib.Path, upstream: str, **_: Any) -> None:
            captured["upstream"] = upstream

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            pass

        def server_close(self) -> None:
            pass

        def close_output(self) -> None:
            pass

    monkeypatch.setattr(cap_mod, "CaptureServer", _MockServer)

    run_capture(port=0, out_path=tmp_path / "explicit.jsonl", upstream="http://explicit.test")

    assert captured.get("upstream") == "http://explicit.test", (
        "Explicit upstream must take precedence over OPENAI_BASE_URL env var"
    )


# ---------------------------------------------------------------------------
# SIGTERM handling — FRG-OSS-043
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="SIGTERM delivery is POSIX-only"
)
def test_run_capture_registers_a_sigterm_handler(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arrange: mock CaptureServer whose serve_forever raises KeyboardInterrupt
    immediately (so run_capture returns without actually blocking).
    Act: call run_capture.
    Assert: the process's SIGTERM handler is no longer the default
    (SIG_DFL/SIG_IGN) after run_capture starts — a real handler was installed.
    """
    import frugon.capture as cap_mod
    from frugon.capture import run_capture

    class _MockServer:
        def __init__(self, **_: Any) -> None:
            pass

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def shutdown(self) -> None:
            pass

        def server_close(self) -> None:
            pass

        def close_output(self) -> None:
            pass

    monkeypatch.setattr(cap_mod, "CaptureServer", _MockServer)

    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        run_capture(port=0, out_path=tmp_path / "sigterm.jsonl")
        installed_handler = signal.getsignal(signal.SIGTERM)
        assert installed_handler not in (signal.SIG_DFL, signal.SIG_IGN), (
            "run_capture must install a real SIGTERM handler, not leave the "
            "default disposition in place"
        )
        assert callable(installed_handler)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="SIGTERM delivery is POSIX-only"
)
def test_sigterm_handler_triggers_clean_shutdown(tmp_path: pathlib.Path) -> None:
    """Arrange: a REAL `frugon capture` process (a genuine child process, not
    a thread in the test runner) bound to an ephemeral port.

    IMPORTANT: this runs run_capture in a SEPARATE PROCESS rather than a
    background thread of the test runner.  ``signal.signal()`` may only be
    called from the main thread of the main interpreter — calling
    run_capture() on a background *thread* (as an earlier draft of this test
    did) makes its `signal.signal(SIGTERM, ...)` call raise ValueError, which
    means NO handler gets installed, so the test's own `os.kill(getpid(),
    SIGTERM)` falls through to the DEFAULT disposition and kills the entire
    pytest process instead of the target.  A subprocess sidesteps this
    entirely: the target is a fully separate PID with the signal
    handler installed on ITS OWN main thread, and the parent test process is
    never at risk.

    Act: send the child process a real SIGTERM after it has had time to bind
    the port and install the handler.
    Assert: the child exits with code 0 (clean) within a generous timeout —
    the existing `finally:` cleanup ran to completion rather than the process
    being hard-killed mid-write.

    This is the true end-to-end regression: before FRG-OSS-043, a `kill <pid>`
    against a running `frugon capture` process terminated it WITHOUT ever
    running capture.py's finally: block (unflushed writes, no clean socket
    close, no default SIGTERM disposition override).
    """
    import subprocess

    out_path = tmp_path / "sigterm_e2e.jsonl"
    script = (
        "import sys; sys.path.insert(0, 'src'); "
        "import pathlib; from frugon.capture import run_capture; "
        f"run_capture(port=0, out_path=pathlib.Path(r'{out_path}'))"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Give the child time to bind the port and install the SIGTERM
        # handler before signalling — avoids a race where SIGTERM arrives
        # before signal.signal() has been called (which would fall through
        # to the default disposition, i.e. an immediate hard kill rather than
        # the clean shutdown path this test verifies).
        time.sleep(1.0)
        proc.send_signal(signal.SIGTERM)
        try:
            returncode = proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
            pytest.fail(
                "frugon capture subprocess did not exit within 5s of SIGTERM "
                "— the handler likely deadlocked or was never installed"
            )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)

    assert returncode == 0, (
        f"expected clean exit (0) after SIGTERM, got {returncode}. "
        f"stderr: {proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr else ''}"
    )
    assert out_path.exists()  # the output file handle was opened and closed cleanly


# ---------------------------------------------------------------------------
# SSRF — upstream scheme and host validation (P1)
# ---------------------------------------------------------------------------


def test_validate_upstream_rejects_file_scheme() -> None:
    """Arrange: file:// upstream URL.
    Act: _validate_upstream.
    Assert: ValueError raised naming the rejected scheme.
    """
    with pytest.raises(ValueError, match="file"):
        _validate_upstream("file:///etc/passwd")


def test_validate_upstream_rejects_ftp_scheme() -> None:
    """Arrange: ftp:// upstream URL.
    Act: _validate_upstream.
    Assert: ValueError raised naming the rejected scheme.
    """
    with pytest.raises(ValueError, match="ftp"):
        _validate_upstream("ftp://evil.example.com")


def test_validate_upstream_rejects_data_scheme() -> None:
    """Arrange: data: upstream URL.
    Act: _validate_upstream.
    Assert: ValueError raised naming the rejected scheme.
    """
    with pytest.raises(ValueError, match="data"):
        _validate_upstream("data:text/plain,hello")


def test_validate_upstream_rejects_http_non_localhost() -> None:
    """Arrange: http:// URL for a non-localhost host; no allow flag.
    Act: _validate_upstream.
    Assert: ValueError raised (plain http to external host is SSRF risk).
    """
    with pytest.raises(ValueError, match="non-localhost"):
        _validate_upstream("http://api.example.com")


def test_validate_upstream_allows_http_localhost_ip() -> None:
    """Arrange: http://127.0.0.1 (loopback IP).
    Act: _validate_upstream.
    Assert: no exception — loopback http is always permitted.
    """
    _validate_upstream("http://127.0.0.1:8080")


def test_validate_upstream_allows_http_localhost_name() -> None:
    """Arrange: http://localhost.
    Act: _validate_upstream.
    Assert: no exception.
    """
    _validate_upstream("http://localhost:8080")


def test_validate_upstream_allows_http_localhost_ipv6() -> None:
    """Arrange: http://[::1] (IPv6 loopback).
    Act: _validate_upstream.
    Assert: no exception.
    """
    _validate_upstream("http://[::1]:8080")


def test_validate_upstream_allows_https_external() -> None:
    """Arrange: https:// external host.
    Act: _validate_upstream.
    Assert: no exception — https is always safe.
    """
    _validate_upstream("https://api.openai.com")


def test_validate_upstream_allows_http_non_localhost_with_flag() -> None:
    """Arrange: http:// non-localhost URL; allow_insecure_upstream=True.
    Act: _validate_upstream.
    Assert: no exception — caller has explicitly opted in.
    """
    _validate_upstream("http://api.example.com", allow_insecure_upstream=True)


def test_capture_server_rejects_unsafe_upstream_at_init(tmp_path: pathlib.Path) -> None:
    """Arrange: file:// upstream.
    Act: CaptureServer.__init__.
    Assert: ValueError raised before any port is bound.
    """
    with pytest.raises(ValueError, match="file"):
        CaptureServer(port=0, out_path=tmp_path / "out.jsonl", upstream="file:///etc/passwd")


def test_capture_server_rejects_http_non_localhost_at_init(tmp_path: pathlib.Path) -> None:
    """Arrange: plain http non-localhost upstream; no allow flag.
    Act: CaptureServer.__init__.
    Assert: ValueError raised.
    """
    with pytest.raises(ValueError, match="non-localhost"):
        CaptureServer(port=0, out_path=tmp_path / "out.jsonl", upstream="http://external.example.com")


def test_capture_server_allows_insecure_with_flag(tmp_path: pathlib.Path) -> None:
    """Arrange: plain http non-localhost upstream; allow_insecure_upstream=True.
    Act: CaptureServer.__init__.
    Assert: server constructed without error; scheme permitted via flag.
    """
    server = CaptureServer(
        port=0,
        out_path=tmp_path / "allow.jsonl",
        upstream="http://192.168.1.1:11434",
        allow_insecure_upstream=True,
    )
    server.server_close()
    server.close_output()


# ---------------------------------------------------------------------------
# Request body cap — Content-Length > _MAX_BODY → 413; malformed → 400 (P2)
# ---------------------------------------------------------------------------


def test_capture_oversized_content_length_returns_413(
    stub_upstream: tuple[str, _StubServer],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: patch _MAX_BODY=100; send Content-Length=101 with no body.
    Act: server reads header, detects oversized before reading any body bytes.
    Assert: client receives HTTP 413.
    """
    monkeypatch.setattr(capture_mod, "_MAX_BODY", 100)
    upstream_url, _ = stub_upstream
    out_file = tmp_path / "cap413.jsonl"
    server = _DrainCaptureServer(port=0, out_path=out_file, upstream=upstream_url)
    cap_port: int = server.server_address[1]

    with running_server(server) as srv:
        sock = socket.create_connection(("127.0.0.1", cap_port), timeout=5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 101\r\n"
            b"\r\n"
        )
        resp_bytes = b""
        # Read until we have the status line or EOF
        sock.settimeout(5)
        try:
            while b"\r\n" not in resp_bytes:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp_bytes += chunk
        except OSError:
            pass
        sock.close()
        srv.drain()

    server.close_output()

    status_line = resp_bytes.split(b"\r\n")[0]
    assert b"413" in status_line, f"Expected HTTP 413, got: {status_line!r}"


def test_capture_malformed_content_length_returns_400(
    stub_upstream: tuple[str, _StubServer],
    tmp_path: pathlib.Path,
) -> None:
    """Arrange: send Content-Length with a non-integer value.
    Act: server cannot parse Content-Length.
    Assert: client receives HTTP 400.
    """
    upstream_url, _ = stub_upstream
    out_file = tmp_path / "cap400.jsonl"
    server = _DrainCaptureServer(port=0, out_path=out_file, upstream=upstream_url)
    cap_port: int = server.server_address[1]

    with running_server(server) as srv:
        sock = socket.create_connection(("127.0.0.1", cap_port), timeout=5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: not_a_number\r\n"
            b"\r\n"
            b"{}"
        )
        resp_bytes = b""
        sock.settimeout(5)
        try:
            while b"\r\n" not in resp_bytes:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp_bytes += chunk
        except OSError:
            pass
        sock.close()
        srv.drain()

    server.close_output()

    status_line = resp_bytes.split(b"\r\n")[0]
    assert b"400" in status_line, f"Expected HTTP 400, got: {status_line!r}"


# ---------------------------------------------------------------------------
# Response body cap — resp.read(_MAX_BODY) (P2)
# ---------------------------------------------------------------------------


class _LargeBodyHandler(http.server.BaseHTTPRequestHandler):
    """Upstream stub that returns a response body of configurable size."""

    body_size: int = 150

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = b"Z" * self.__class__.body_size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass


class _LargeBodyServer(_DrainStubServer):
    def __init__(self, body_size: int) -> None:
        _LargeBodyHandler.body_size = body_size
        super().__init__(("127.0.0.1", 0), _LargeBodyHandler)


def test_capture_response_body_capped_at_max_body(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: patch _MAX_BODY=100; upstream sends 150 bytes.
    Act: POST to capture server.
    Assert: client receives exactly 100 bytes — cap applied to upstream response.
    """
    monkeypatch.setattr(capture_mod, "_MAX_BODY", 100)

    large_server = _LargeBodyServer(150)
    large_port: int = large_server.server_address[1]
    upstream_url = f"http://127.0.0.1:{large_port}"

    out_file = tmp_path / "cap_resp.jsonl"
    cap_server = _DrainCaptureServer(port=0, out_path=out_file, upstream=upstream_url)
    cap_port: int = cap_server.server_address[1]

    with running_server(large_server):
        with running_server(cap_server) as srv:
            _, body = _post(cap_port, "/v1/chat/completions", _SAMPLE_REQUEST)
            srv.drain()

    cap_server.close_output()

    assert len(body) == 100, (
        f"Response body must be capped at _MAX_BODY=100 bytes; got {len(body)} bytes"
    )


# ---------------------------------------------------------------------------
# Output path validation — friendly error before port is bound (P2)
# ---------------------------------------------------------------------------


def test_capture_server_bad_out_path_raises(tmp_path: pathlib.Path) -> None:
    """Arrange: out_path whose parent directory does not exist.
    Act: CaptureServer.__init__.
    Assert: OSError raised with a message referencing the path — no port bound.
    """
    bad_path = tmp_path / "nonexistent_dir" / "out.jsonl"
    with pytest.raises(OSError, match="nonexistent_dir"):
        CaptureServer(port=0, out_path=bad_path, upstream="https://api.openai.com")


def test_run_capture_bad_out_path_exits_1(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Arrange: out_path whose parent directory does not exist.
    Act: run_capture.
    Assert: exits with code 1; stderr contains an error message.
    """
    bad_path = tmp_path / "nonexistent_dir" / "out.jsonl"
    with pytest.raises(SystemExit) as exc_info:
        run_capture(port=0, out_path=bad_path, upstream="https://api.openai.com")
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _build_restricted_opener — defence-in-depth transport guard (C2)
# ---------------------------------------------------------------------------


def test_restricted_opener_only_registers_http_https_unknown() -> None:
    """Arrange: call _build_restricted_opener().
    Act: inspect handle_open keys.
    Assert: exactly ['http', 'https', 'unknown'] — FileHandler / FTPHandler absent.
    """
    opener = _build_restricted_opener()
    assert sorted(opener.handle_open.keys()) == ["http", "https", "unknown"]


def test_restricted_opener_blocks_file_scheme() -> None:
    """Arrange: _build_restricted_opener().
    Act: open a file:// URL.
    Assert: URLError raised with 'unknown url type' — NOT a file read.
    """
    import urllib.error
    import urllib.request
    opener = _build_restricted_opener()
    with pytest.raises(urllib.error.URLError, match='unknown url type'):
        opener.open(urllib.request.Request("file:///etc/passwd"))


def test_restricted_opener_blocks_ftp_scheme() -> None:
    """Arrange: _build_restricted_opener().
    Act: open an ftp:// URL.
    Assert: URLError raised with 'unknown url type'.
    """
    import urllib.error
    import urllib.request
    opener = _build_restricted_opener()
    with pytest.raises(urllib.error.URLError, match='unknown url type'):
        opener.open(urllib.request.Request("ftp://evil.example.com/x"))


def test_restricted_opener_blocks_data_scheme() -> None:
    """Arrange: _build_restricted_opener().
    Act: open a data: URL.
    Assert: URLError raised with 'unknown url type'.
    """
    import urllib.error
    import urllib.request
    opener = _build_restricted_opener()
    with pytest.raises(urllib.error.URLError, match='unknown url type'):
        opener.open(urllib.request.Request("data:text/plain,hello"))


# ---------------------------------------------------------------------------
# Proxy policy — privacy invariant: ignore ambient HTTP(S)_PROXY (FRG-OSS-016)
# ---------------------------------------------------------------------------


def test_restricted_opener_ignores_ambient_proxy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: HTTP_PROXY / HTTPS_PROXY / ALL_PROXY set in the environment.
    Act: _build_restricted_opener() with no explicit proxy.
    Assert: the ProxyHandler carries NO proxies — the ambient env is ignored, so
            the developer's request (and its Authorization header) is never routed
            through a third-party proxy (privacy invariant, FRG-OSS-016).

    Fails-before / passes-after: the previous ``ProxyHandler()`` (no args) would
    pick up these env vars, so .proxies would be non-empty.
    """
    import urllib.request

    monkeypatch.setenv("HTTP_PROXY", "http://evil.proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://evil.proxy.example:8080")
    monkeypatch.setenv("ALL_PROXY", "http://evil.proxy.example:8080")

    opener = _build_restricted_opener()
    proxy_handlers = [
        h for h in opener.handlers if isinstance(h, urllib.request.ProxyHandler)
    ]
    # An empty ProxyHandler({}) registers no <proto>_open methods, so urllib's
    # add_handler does not even add it to opener.handlers — without a registered
    # ProxyHandler, connections go direct. The invariant either way: no registered
    # ProxyHandler carries a proxy derived from the ambient environment. (Old code
    # used ProxyHandler() which WOULD pick up HTTP_PROXY and register a non-empty
    # proxy handler here — so this fails-before / passes-after.)
    assert all(not h.proxies for h in proxy_handlers), (
        "the ambient HTTP(S)_PROXY environment leaked into the capture opener: "
        f"{[h.proxies for h in proxy_handlers]} — the key must go straight to the provider"
    )


def test_restricted_opener_uses_explicit_proxy_when_given() -> None:
    """Arrange: an explicit proxy URL (the --proxy opt-in).
    Act: _build_restricted_opener(proxy=...).
    Assert: the ProxyHandler routes http + https through exactly that proxy — an
            informed, explicit opt-in (FRG-OSS-016).
    """
    import urllib.request

    proxy = "http://127.0.0.1:8080"
    opener = _build_restricted_opener(proxy)
    proxy_handlers = [
        h for h in opener.handlers if isinstance(h, urllib.request.ProxyHandler)
    ]
    assert proxy_handlers, "opener must register a ProxyHandler"
    assert proxy_handlers[0].proxies == {"http": proxy, "https": proxy}


def test_capture_cli_threads_proxy_to_run_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: mock run_capture to record its kwargs.
    Act: invoke ``frugon capture --proxy <url>``.
    Assert: the proxy value is threaded CLI -> run_capture (wiring for FRG-OSS-016).
    """
    captured: dict[str, Any] = {}

    def _fake_run_capture(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(capture_mod, "run_capture", _fake_run_capture)
    result = runner.invoke(
        app, ["capture", "--proxy", "http://127.0.0.1:8080", "--port", "0"]
    )
    assert result.exit_code == 0, result.output
    assert captured.get("proxy") == "http://127.0.0.1:8080", (
        f"--proxy must thread through to run_capture; got {captured.get('proxy')!r}"
    )


def test_capture_cli_default_proxy_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: mock run_capture; invoke ``frugon capture`` WITHOUT --proxy.
    Act/Assert: the CLI threads proxy=None — the privacy default at the CLI seam
    (locks against a future regression that defaults --proxy to a real proxy).
    """
    captured: dict[str, Any] = {}

    def _fake_run_capture(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(capture_mod, "run_capture", _fake_run_capture)
    result = runner.invoke(app, ["capture", "--port", "0"])
    assert result.exit_code == 0, result.output
    assert captured.get("proxy") is None, (
        f"CLI default must thread proxy=None (privacy default); got {captured.get('proxy')!r}"
    )


# ---------------------------------------------------------------------------
# Streaming (SSE) rejection — FRG-OSS-018
# ---------------------------------------------------------------------------


def test_capture_rejects_stream_true_with_400(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server pointing at stub upstream.
    Act: POST a request with "stream": true.
    Assert: 400 response with a clear JSON error message; the request is
    NEVER forwarded upstream (capture cannot support incremental SSE — it
    would silently break streaming clients by buffering the whole response).
    """
    srv, port, out_file = capture_srv

    request = {**_SAMPLE_REQUEST, "stream": True}
    status, body = _post(port, "/v1/chat/completions", request)

    assert status == 400
    payload = json.loads(body)
    assert "stream" in payload["error"]["message"].lower()

    # Nothing was written to the output file — the call never completed.
    assert out_file.read_text(encoding="utf-8") == ""


def test_capture_stream_false_is_unaffected(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server pointing at stub upstream.
    Act: POST a request with "stream": false (explicit, common client default).
    Assert: normal 200 forwarding path — the rejection targets ONLY
    stream=true, not the mere presence of the key.
    """
    srv, port, out_file = capture_srv
    request = {**_SAMPLE_REQUEST, "stream": False}

    status, _ = _post(port, "/v1/chat/completions", request)
    srv.drain()

    assert status == 200
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_capture_stream_absent_is_unaffected(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Arrange: capture server pointing at stub upstream.
    Act: POST a request with no "stream" key at all (the common non-streaming
    request shape).
    Assert: normal 200 forwarding path, unchanged from before this feature.
    """
    srv, port, out_file = capture_srv

    status, _ = _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()

    assert status == 200
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_capture_non_object_json_body_does_not_crash_stream_check(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
) -> None:
    """Regression: a syntactically-valid JSON body that is NOT an object (a
    bare array, string, number, bool, or null) must not crash the stream-flag
    check added for FRG-OSS-018.

    ``json.loads(b"[1, 2, 3]")`` succeeds (it is valid JSON) and returns a
    ``list``, not a ``dict`` — calling ``.get("stream")`` on that list raises
    an unhandled ``AttributeError`` that the existing
    ``except (json.JSONDecodeError, UnicodeDecodeError)`` clause does NOT
    catch (a list is not a decode failure). Before the fix, this crashed the
    request-handling thread instead of degrading gracefully like every other
    non-OpenAI-shaped body the capture proxy already tolerates.

    Arrange: capture server.
    Act: POST a raw JSON array (not an object) as the request body.
    Assert: the connection does not crash/hang — a response is returned (the
    request is forwarded, since it is not `stream: true`, matching how an
    unparseable/malformed body is already tolerated elsewhere in do_POST) —
    AND the call still reaches _build_record and writes a (degraded, empty
    request_data) record rather than crashing before the write.
    """
    srv, port, out_file = capture_srv

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        body = b"[1, 2, 3]"
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        status = resp.status
        resp.read()
    finally:
        conn.close()
    srv.drain()

    # The key assertion: no crash. A non-stream-true body forwards normally.
    assert status == 200
    lines = [ln for ln in out_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    # request_data degraded to {} (a bare array is not an OpenAI request
    # object), so model/messages fall back to their _build_record defaults.
    assert record["model"] == ""
    assert record["request"]["messages"] == []


def test_capture_non_object_json_upstream_response_does_not_crash(
    capture_srv: tuple[_DrainCaptureServer, int, pathlib.Path],
    stub_upstream: tuple[str, _StubServer],
) -> None:
    """Regression: a syntactically-valid, non-object JSON UPSTREAM RESPONSE
    (e.g. an upstream returning a bare JSON string or array on an error path)
    must not crash ``_build_record`` the same way a non-object REQUEST body
    would — the response-side ``json.loads`` had the identical unguarded
    ``dict[str, Any]`` annotation as the request side.

    Arrange: stub upstream configured to return a non-dict JSON body.
    Act: POST a normal (object) request through capture.
    Assert: the call completes cleanly (200) and a record is still written,
    with response_data degraded to {} rather than crashing on `.get("usage")`.
    """
    srv, port, out_file = capture_srv
    _, stub = stub_upstream
    # dict[str, Any] is the stub's declared type, but this exercises the
    # SERVER's actual runtime behavior — json.dumps a bare list just as
    # readily as a dict, and the wire format is what capture.py must handle.
    stub.response = cast(Any, ["not", "an", "object"])

    status, _ = _post(port, "/v1/chat/completions", _SAMPLE_REQUEST)
    srv.drain()

    assert status == 200
    lines = [ln for ln in out_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    # response_data degraded to {} — usage falls back to _build_record's default.
    assert record["usage"] == {}
    assert record["response"] == {}


def test_capture_startup_panel_shows_sensitivity_caution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FRG-OSS-019: the capture startup panel must warn that capture.jsonl
    holds full prompts/completions and should be treated like a credential —
    on EVERY platform (this is the load-bearing signal on Windows, where the
    0o600 chmod below is a no-op).

    Arrange: mock run_capture so the CLI returns immediately.
    Act: invoke ``frugon capture``.
    Assert: the caution text appears in the startup panel output.
    """
    monkeypatch.setattr(capture_mod, "run_capture", lambda **kwargs: None)
    result = runner.invoke(
        app, ["capture", "--port", "0"], env={"COLUMNS": "200", "TERM": "dumb"}
    )
    assert result.exit_code == 0, result.output
    assert "treat it like a credential" in result.output


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX file-mode bits only"
)
def test_run_capture_sets_0o600_on_new_output_file(tmp_path: pathlib.Path) -> None:
    """FRG-OSS-019: a freshly created capture.jsonl is owner-read/write only.

    Arrange: an out_path that does not yet exist.
    Act: construct CaptureServer (which opens/creates the file).
    Assert: the file's POSIX mode bits are exactly 0o600 (no group/world
    access), regardless of the process umask.
    """
    import stat

    out_path = tmp_path / "capture.jsonl"
    server = CaptureServer(port=0, out_path=out_path, upstream="https://api.openai.com")
    try:
        mode = stat.S_IMODE(out_path.stat().st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
    finally:
        server.close_output()
        server.server_close()


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="POSIX file-mode bits only"
)
def test_run_capture_tightens_permissions_on_preexisting_loose_file(
    tmp_path: pathlib.Path,
) -> None:
    """FRG-OSS-019: an ALREADY-EXISTING capture.jsonl with loose permissions
    (e.g. from before this hardening landed, or a user's own umask) is
    tightened to 0o600 on the next `frugon capture` start — not just at
    initial creation.

    Arrange: pre-create out_path with permissive 0o644.
    Act: construct CaptureServer against that existing path.
    Assert: mode is tightened to 0o600.
    """
    import stat

    out_path = tmp_path / "capture.jsonl"
    out_path.write_text("", encoding="utf-8")
    out_path.chmod(0o644)
    assert stat.S_IMODE(out_path.stat().st_mode) == 0o644  # sanity: loose before

    server = CaptureServer(port=0, out_path=out_path, upstream="https://api.openai.com")
    try:
        mode = stat.S_IMODE(out_path.stat().st_mode)
        assert mode == 0o600, f"expected tightened 0o600, got {oct(mode)}"
    finally:
        server.close_output()
        server.server_close()


def test_restricted_opener_rejects_non_http_proxy_scheme() -> None:
    """Arrange: an explicit proxy with a non-http(s) scheme.
    Act: _build_restricted_opener(bad).
    Assert: ValueError up front (mirrors _validate_upstream's scheme allowlist),
            rather than a deep urllib failure later. urllib cannot route SOCKS/file
            proxies anyway, so rejecting them with a clear message is correct.
    """
    for bad in ("ftp://127.0.0.1:8080", "file:///etc/x", "socks5://127.0.0.1:1080"):
        with pytest.raises(ValueError, match="proxy scheme"):
            _build_restricted_opener(bad)


# ---------------------------------------------------------------------------
# Item 6 — cross-origin redirect must strip Authorization header
# ---------------------------------------------------------------------------


class _HeaderRecordingHandler(http.server.BaseHTTPRequestHandler):
    """Records headers for any HTTP method (GET or POST) without blocking."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _handle_any(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 0:
            self.rfile.read(length)
        cast("_HeaderRecordingServer", self.server).received.append(
            {"headers": dict(self.headers)}
        )
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    do_GET = _handle_any
    do_POST = _handle_any


class _HeaderRecordingServer(_DrainStubServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _HeaderRecordingHandler)
        self.received: list[dict[str, Any]] = []


class _CrossOriginRedirectHandler(http.server.BaseHTTPRequestHandler):
    """Returns a 302 redirect to the URL stored on the server."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        target = cast("_CrossOriginRedirectServer", self.server).redirect_target
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()
        self.wfile.flush()


class _CrossOriginRedirectServer(_DrainStubServer):
    def __init__(self, redirect_target: str) -> None:
        super().__init__(("127.0.0.1", 0), _CrossOriginRedirectHandler)
        self.redirect_target: str = redirect_target


def test_cross_origin_redirect_strips_authorization_header(
    tmp_path: pathlib.Path,
) -> None:
    """Arrange: upstream returns 302 to a different port (cross-origin redirect).
    Act: POST to capture server with Authorization: Bearer secret.
    Assert: the redirect target (different origin) does NOT receive Authorization.

    Proves: the developer's API key must not leak to an unintended host
    when the upstream provider issues a cross-origin redirect.
    """
    # Arrange — recording server is the redirect target
    rec_server = _HeaderRecordingServer()
    rec_port: int = rec_server.server_address[1]

    # Arrange — redirect server (different port → different origin)
    redirect_target = f"http://127.0.0.1:{rec_port}/v1/chat/completions"
    redir_server = _CrossOriginRedirectServer(redirect_target)
    redir_port: int = redir_server.server_address[1]

    # Arrange — capture server pointing at the redirect server
    out_file = tmp_path / "redirect_sec.jsonl"
    cap_server = _DrainCaptureServer(
        port=0,
        out_path=out_file,
        upstream=f"http://127.0.0.1:{redir_port}",
    )
    cap_port: int = cap_server.server_address[1]

    with running_server(rec_server):
        with running_server(redir_server):
            with running_server(cap_server) as srv:
                # Act — POST with a secret API key
                _post(
                    cap_port,
                    "/v1/chat/completions",
                    _SAMPLE_REQUEST,
                    headers={"Authorization": "Bearer api-key-must-not-leak"},
                )
                srv.drain()

    cap_server.close_output()

    # Assert — redirect target received at least one request
    assert rec_server.received, (
        "Recording server received no requests — redirect was not followed"
    )
    # Assert — none of those requests carried the Authorization header
    for req_info in rec_server.received:
        auth_present = any(
            k.lower() == "authorization" for k in req_info["headers"]
        )
        assert not auth_present, (
            f"Authorization header leaked to cross-origin redirect target. "
            f"Headers received: {list(req_info['headers'].keys())}"
        )


class _SameOriginRedirectHandler(http.server.BaseHTTPRequestHandler):
    """On POST: 302 to a different path on the same host:port.
    On GET at that path: records headers and returns 200."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        port = cast("_SameOriginRedirectServer", self.server).server_address[1]
        target = f"http://127.0.0.1:{port}/v1/redirect-final"
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()
        self.wfile.flush()

    def do_GET(self) -> None:
        cast("_SameOriginRedirectServer", self.server).received.append(
            {"headers": dict(self.headers)}
        )
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


class _SameOriginRedirectServer(_DrainStubServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _SameOriginRedirectHandler)
        self.received: list[dict[str, Any]] = []


def test_same_origin_redirect_preserves_authorization_header(
    tmp_path: pathlib.Path,
) -> None:
    """Arrange: upstream returns a 302 to a different path on the same host:port.
    Act: POST to capture server with Authorization: Bearer secret.
    Assert: the redirect target (same origin) receives the Authorization header.

    Proves P2: same-origin redirects must not strip credentials — guards against
    the over-strip regression where legitimate same-host paths lose the auth token.
    """
    # Arrange — single server: POST 302s to itself (same host:port, different path)
    redir_server = _SameOriginRedirectServer()
    redir_port: int = redir_server.server_address[1]

    out_file = tmp_path / "same_origin.jsonl"
    cap_server = _DrainCaptureServer(
        port=0,
        out_path=out_file,
        upstream=f"http://127.0.0.1:{redir_port}",
    )
    cap_port: int = cap_server.server_address[1]

    with running_server(redir_server):
        with running_server(cap_server) as srv:
            # Act
            _post(
                cap_port,
                "/v1/chat/completions",
                _SAMPLE_REQUEST,
                headers={"Authorization": "Bearer secret-same-origin"},
            )
            srv.drain()

    cap_server.close_output()

    # Assert — the same-origin target received Authorization
    assert redir_server.received, (
        "Same-origin redirect target received no requests — redirect was not followed"
    )
    for req_info in redir_server.received:
        auth_present = any(k.lower() == "authorization" for k in req_info["headers"])
        assert auth_present, (
            f"Authorization must not be stripped on a same-origin redirect. "
            f"Headers at redirect target: {list(req_info['headers'].keys())}"
        )


# ---------------------------------------------------------------------------
# Item 7 — upstream timeout must return 504, not hang
# ---------------------------------------------------------------------------


class _HangingHandler(http.server.BaseHTTPRequestHandler):
    """Hangs on every POST for long enough to trigger a short timeout."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        # Block longer than any test timeout; the handler thread is daemon,
        # so it is killed when the server is shut down.
        threading.Event().wait(timeout=10)
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _HangingServer(_DrainStubServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _HangingHandler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # silence tracebacks from threads still sleeping at teardown


def test_upstream_timeout_returns_504(tmp_path: pathlib.Path) -> None:
    """Arrange: upstream server hangs indefinitely; capture server has upstream_timeout=0.2.
    Act: POST to capture server.
    Assert: capture server returns HTTP 504 quickly instead of hanging.

    Proves: a slow/hung upstream must not block the local proxy
    indefinitely — the caller receives a clean 504 error.
    """
    # Arrange — hanging upstream (drain=False: handler sleeps 10 s, don't wait for it)
    hang_server = _HangingServer()
    hang_port: int = hang_server.server_address[1]

    # Arrange — capture server with a very short timeout
    out_file = tmp_path / "timeout_test.jsonl"
    cap_server = _DrainCaptureServer(
        port=0,
        out_path=out_file,
        upstream=f"http://127.0.0.1:{hang_port}",
        upstream_timeout=0.2,
    )
    cap_port: int = cap_server.server_address[1]

    with running_server(hang_server, drain_before_shutdown=False):
        with running_server(cap_server) as srv:
            # Act — POST; client timeout generously longer than upstream_timeout
            conn = http.client.HTTPConnection("127.0.0.1", cap_port, timeout=5)
            data = json.dumps(_SAMPLE_REQUEST).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=data,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            status = resp.status
            resp.read()
            conn.close()
            srv.drain()

    cap_server.close_output()

    # Assert — 504 returned, not a hang or 502
    assert status == 504, (
        f"Expected HTTP 504 Gateway Timeout from hung upstream, got {status}. "
        "The capture proxy must surface a clean timeout error to the caller."
    )


def test_upstream_timeout_none_forwards_normally(
    tmp_path: pathlib.Path,
    stub_upstream: tuple[str, _StubServer],
) -> None:
    """Arrange: CaptureServer with upstream_timeout=None (timeout disabled).
    Act: POST a normal request.
    Assert: upstream response forwarded successfully — None passes through unchanged.
    """
    upstream_url, _ = stub_upstream
    out_file = tmp_path / "timeout_none.jsonl"
    cap_server = _DrainCaptureServer(
        port=0,
        out_path=out_file,
        upstream=upstream_url,
        upstream_timeout=None,
    )
    cap_port: int = cap_server.server_address[1]

    with running_server(cap_server) as srv:
        status, body = _post(cap_port, "/v1/chat/completions", _SAMPLE_REQUEST)
        srv.drain()

    cap_server.close_output()

    assert status == 200
    assert json.loads(body.decode("utf-8")) == _SAMPLE_RESPONSE
