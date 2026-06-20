"""frugon live-progress helper — transient feedback on stderr, never stdout.

Why this module exists
----------------------
A first-time user who runs ``frugon analyze`` on a large log should never stare
at a silent terminal wondering whether the tool has hung.  This module supplies
a small, self-contained set of progress affordances — a spinner, a determinate
progress bar, and persisted phase checkpoints — that reassure the user while the
read / pricing pass runs.

The one hard rule
-----------------
**Every byte of progress chrome goes to a Rich ``Console(stderr=True)``.**  The
analysis RESULT (the panel, tables, footer, report-written line) stays on
stdout, untouched.  This keeps stdout byte-identical to today, which protects:

  * ``--report`` (the HTML/Markdown artifact is unaffected),
  * piping (``frugon analyze … | cat`` and ``> file`` see only the result),
  * the deterministic ``--demo`` (the gif/screenshot single source of truth), and
  * every existing stdout-asserting test.

Gating
------
Progress animates ONLY when **all** of the following hold:

  * stderr is a TTY (``sys.stderr.isatty()``), AND
  * ``NO_COLOR`` is not set in the environment, AND
  * progress was not explicitly disabled (the ``--no-progress`` flag).

Otherwise the helper is a complete no-op: no spinner, no bar, no checkpoints —
non-interactive / piped / CI runs stay clean.

Colour discipline
-----------------
Progress chrome is neutral / cyan.  Green is reserved for the saving headline in
the result, so it never appears here.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:  # pragma: no cover — typing only
    from rich.status import Status


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def progress_enabled(*, no_progress: bool) -> bool:
    """Return True iff live progress chrome should render.

    All three conditions must hold: stderr is a TTY, ``NO_COLOR`` is unset, and
    the caller did not pass ``--no-progress``.  Any one being false makes the
    helper a no-op (silent).  Centralised here so every call site shares one
    rule.
    """
    if no_progress:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(sys.stderr.isatty())
    except (ValueError, AttributeError):  # pragma: no cover — detached/odd stderr
        return False


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """A small reusable progress surface bound to a stderr console.

    Construct via :func:`progress_reporter` (a context manager) so the gating
    decision and console wiring happen in one place.  When ``enabled`` is False
    every method is a cheap no-op, so call sites stay branch-free.

    The spinner and bar are *transient* — they clear from the terminal when
    their phase ends.  Checkpoints (``checkpoint``) are *persisted*: each prints
    one dim line that stays on screen, leaving a short trail of completed phases
    (e.g. ``✓ Read 56,100 records``).  Keep the trail short and tasteful — a few
    lines, never a log dump.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        # A dedicated stderr console.  Even the checkpoint lines go here, never
        # stdout — stdout carries only the analysis result.
        self._console: Console | None = Console(stderr=True) if enabled else None

    # -- phase checkpoints (persisted) --------------------------------------
    def checkpoint(self, message: str) -> None:
        """Print a persisted ``✓`` checkpoint line on stderr (dim, neutral).

        No-op when disabled.  *message* should be terse, e.g.
        ``"Read 56,100 records"``.  The green checkmark is intentionally NOT
        used (green is reserved for the saving headline); the mark is rendered
        in neutral cyan to stay within the progress colour discipline.
        """
        if self._console is None:
            return
        self._console.print(f"[dim][cyan]✓[/cyan] {message}[/dim]")

    # -- informational notice (persisted) -----------------------------------
    def notice(self, message: str) -> None:
        """Print a one-line informational heads-up on stderr (dim, neutral).

        For a gentle, non-blocking aside — e.g. telling the user a very large log
        may take a moment.  It is NOT a warning and NOT a cap; it never changes
        what frugon does.  Stderr only, and a no-op when progress is disabled
        (non-TTY / NO_COLOR / --no-progress), so piped and CI runs stay silent.
        """
        if self._console is None:
            return
        self._console.print(f"[dim]{message}[/dim]")

    # -- blank separator (persisted) ----------------------------------------
    def blank(self) -> None:
        """Print one empty line on the stderr progress console.

        A tasteful, single blank that separates the persisted checkpoint trail
        from whatever the analysis result prints next on stdout — so a fresh
        run does not read as a wall of cramped lines.  Stderr only (never
        stdout, which carries the result), and a no-op when disabled (non-TTY /
        NO_COLOR / --no-progress), so piped and CI runs stay clean.
        """
        if self._console is None:
            return
        self._console.print()

    # -- spinner (transient, unknown total) ---------------------------------
    @contextmanager
    def spinner(self, message: str) -> Iterator[None]:
        """Show a transient spinner while an unbounded phase runs.

        Used for the read/parse phase where the record count is not yet known
        (``Reading logs…``).  Clears when the ``with`` block exits.  No-op when
        disabled.
        """
        if self._console is None:
            yield
            return
        status: Status = self._console.status(
            f"[cyan]{message}[/cyan]", spinner="dots", spinner_style="cyan"
        )
        with status:
            yield

    # -- determinate bar (transient, known total) ---------------------------
    @contextmanager
    def bar(self, message: str, total: int) -> Iterator[ProgressTask]:
        """Show a transient determinate progress bar for a bounded phase.

        Yields a :class:`ProgressTask` whose ``advance(n=1)`` the caller invokes
        per unit of work (e.g. once per priced record).  The bar shows the
        message, an ``n/total`` count, a bar, elapsed time, and ETA — the key
        reassurance on a big log.  Clears when the ``with`` block exits.

        When disabled (or *total* is non-positive) the yielded task's
        ``advance`` is a no-op, so the per-record callback stays cheap and the
        call site never branches.
        """
        if self._console is None or total <= 0:
            yield _NULL_TASK
            return
        progress = Progress(
            TextColumn("[cyan]{task.description}[/cyan]"),
            MofNCompleteColumn(),
            BarColumn(complete_style="cyan", finished_style="cyan"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=True,
        )
        with progress:
            task_id = progress.add_task(message, total=total)
            yield _RichProgressTask(progress, task_id)

    # -- counter (transient, n/total without a bar) -------------------------
    @contextmanager
    def counter(self, prefix: str, total: int) -> Iterator[StepCounter]:
        """Show a transient ``prefix n/total · <label>`` spinner line.

        Used for the per-prompt ``--measure`` / ``--judge`` indicator
        (``Sampling prompt 3/5 · gpt-4o-mini``).  Yields a :class:`StepCounter`;
        call ``step(label)`` as each prompt begins.  No-op when disabled.
        """
        if self._console is None or total <= 0:
            yield _NULL_COUNTER
            return
        progress = Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[cyan]{task.description}[/cyan]"),
            console=self._console,
            transient=True,
        )
        with progress:
            task_id = progress.add_task(prefix, total=total)
            yield _RichStepCounter(progress, task_id, prefix, total)


# ---------------------------------------------------------------------------
# Progress-task abstractions (advance per unit of work)
# ---------------------------------------------------------------------------


class ProgressTask:
    """Advance handle for a determinate bar.  Base class is the null no-op."""

    def advance(self, n: int = 1) -> None:  # noqa: D401 — simple verb
        """Advance the bar by *n* units.  No-op in the null implementation."""


class _RichProgressTask(ProgressTask):
    """A live advance handle backed by a Rich :class:`Progress` task."""

    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, n: int = 1) -> None:
        self._progress.advance(self._task_id, n)


_NULL_TASK = ProgressTask()


class StepCounter:
    """Step handle for an ``n/total · label`` counter.  Base is the null no-op."""

    def step(self, label: str = "") -> None:  # noqa: D401 — simple verb
        """Mark one step beginning, optionally labelled.  No-op in the null impl."""


class _RichStepCounter(StepCounter):
    """A live step handle backed by a Rich :class:`Progress` spinner task."""

    def __init__(self, progress: Progress, task_id: TaskID, prefix: str, total: int) -> None:
        self._progress = progress
        self._task_id = task_id
        self._prefix = prefix
        self._total = total
        self._done = 0

    def step(self, label: str = "") -> None:
        self._done += 1
        desc = f"{self._prefix} {self._done}/{self._total}"
        if label:
            desc = f"{desc} · {label}"
        self._progress.update(self._task_id, description=desc, completed=self._done - 1)


_NULL_COUNTER = StepCounter()


# ---------------------------------------------------------------------------
# Entry-point context manager
# ---------------------------------------------------------------------------


@contextmanager
def progress_reporter(*, no_progress: bool) -> Iterator[ProgressReporter]:
    """Yield a :class:`ProgressReporter`, gated by :func:`progress_enabled`.

    The single entry point for call sites: wrap a command's work in
    ``with progress_reporter(no_progress=no_progress) as progress:`` and use
    ``progress.spinner(...)`` / ``progress.bar(...)`` / ``progress.checkpoint(...)``.
    When gating says "off" the reporter is a no-op and renders nothing.
    """
    yield ProgressReporter(enabled=progress_enabled(no_progress=no_progress))


# ---------------------------------------------------------------------------
# Elapsed timing helper (used for the "Priced in 4.2s" checkpoint)
# ---------------------------------------------------------------------------


class Stopwatch:
    """A tiny monotonic stopwatch for phase-duration checkpoint lines.

    Usage::

        with Stopwatch() as sw:
            ... work ...
        reporter.checkpoint(f"Priced in {sw.elapsed:.1f}s")
    """

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start
