"""Tests for scripts/check-commit-attribution.sh.

Drives the guard against real throwaway git repos in tmp_path rather than
mocking git: the behaviour under test is git's own trailer parsing, identity
fields and revision ranges, which a mock would have to reimplement to prove
anything.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD = REPO_ROOT / "scripts" / "check-commit-attribution.sh"

ALLOWED_NAME = "Jarod-RH"
ALLOWED_EMAIL = "144908421+Jarod-RH@users.noreply.github.com"
ALLOWED_IDENTITY = f"{ALLOWED_NAME} <{ALLOWED_EMAIL}>"

BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"

ZERO_SHA = "0" * 40

# The commit that motivated the guard: a Co-authored-by trailer naming a tool,
# with a correct author and committer.
INCIDENT_SHA = "2f92dfe"

# Resolve bash to an absolute path rather than letting subprocess resolve the
# bare name. On Windows these differ: shutil.which finds Git Bash, but the
# CreateProcess search finds C:\Windows\System32\bash.exe (the WSL launcher),
# which cannot see "C:/Users/..." paths and fails with a misleading
# "No such file or directory".
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or BASH is None,
    reason="requires git and bash on PATH",
)


def _git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return str(result.stdout).strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repo with one clean commit by the allowed identity."""
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", ALLOWED_NAME)
    _git(path, "config", "user.email", ALLOWED_EMAIL)
    _git(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("seed\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-q", "-m", "chore: seed")
    return path


def _commit(
    repo: Path,
    message: str,
    *,
    name: str = ALLOWED_NAME,
    email: str = ALLOWED_EMAIL,
    committer_name: str | None = None,
    committer_email: str | None = None,
) -> str:
    """Create an empty commit with the given message and identities."""
    env = None
    if committer_name is not None or committer_email is not None:
        import os

        env = dict(os.environ)
        env["GIT_COMMITTER_NAME"] = committer_name or name
        env["GIT_COMMITTER_EMAIL"] = committer_email or email
    subprocess.run(
        [
            "git",
            "-c",
            f"user.name={name}",
            "-c",
            f"user.email={email}",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            message,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return _git(repo, "rev-parse", "HEAD")


def _run_guard(
    repo: Path,
    *args: str,
    stdin: str | None = None,
    raw_stdin: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the guard against ``repo``.

    stdin is passed as bytes so Python's text-mode newline translation cannot
    rewrite LF to CRLF on Windows; ``raw_stdin`` lets a test send exact bytes.
    ``as_posix`` because bash treats backslashes in a native Windows path as
    escapes.
    """
    assert BASH is not None  # guarded by pytestmark
    payload = raw_stdin if raw_stdin is not None else (stdin or "").encode("utf-8")
    proc = subprocess.run(
        [BASH, GUARD.as_posix(), *args],
        cwd=repo,
        capture_output=True,
        input=payload,
    )
    return subprocess.CompletedProcess(
        proc.args,
        proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )


def _short(repo: Path, sha: str) -> str:
    return _git(repo, "rev-parse", "--short", sha)


# ---------------------------------------------------------------------------
# Clean paths — the guard must not cry wolf.
# ---------------------------------------------------------------------------


def test_guard_clean_commit_by_allowed_identity_exits_zero(repo: Path) -> None:
    # Arrange
    _commit(repo, "feat: add a thing")

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all clean" in result.stdout


def test_guard_bot_authored_commit_exits_zero(repo: Path) -> None:
    """The sync workflows commit as github-actions[bot] and push to main.

    This is the author/committer axis of the real bot commits, which the trailer
    test below does not exercise.
    """
    # Arrange
    _commit(
        repo,
        "chore(pricing): refresh curated seed prices",
        name=BOT_NAME,
        email=BOT_EMAIL,
    )

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr


def test_guard_bot_author_with_github_web_ui_committer_exits_zero(repo: Path) -> None:
    """A sync PR merged through the web UI: bot author, "GitHub" committer."""
    # Arrange
    _commit(
        repo,
        "chore(quality): sync LMArena quality tiers (#4)",
        name=BOT_NAME,
        email=BOT_EMAIL,
        committer_name="GitHub",
        committer_email="noreply@github.com",
    )

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr


def test_guard_allowed_identity_coauthor_trailer_exits_zero(repo: Path) -> None:
    """The bot-raised sync PRs carry a Co-authored-by trailer for the maintainer."""
    # Arrange
    _commit(
        repo,
        f"chore(pricing): sync pricing table (#3)\n\nCo-authored-by: {ALLOWED_IDENTITY}\n",
    )

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr


def test_guard_merge_commit_by_allowed_identity_exits_zero(repo: Path) -> None:
    # Arrange
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "topic")
    _commit(repo, "feat: on a branch")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "chore: merge topic", "topic")

    # Act
    result = _run_guard(repo, f"{base}..HEAD")

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 commit(s) checked" in result.stdout


def test_guard_merge_commit_with_bad_trailer_on_parent_blocks(repo: Path) -> None:
    """A violation reachable through a merge is still being published."""
    # Arrange
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "topic")
    _commit(repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "chore: merge topic", "topic")

    # Act
    result = _run_guard(repo, f"{base}..HEAD")

    # Assert
    assert result.returncode != 0
    assert "cursoragent@cursor.com" in result.stdout


def test_guard_empty_range_exits_zero(repo: Path) -> None:
    # Arrange / Act
    result = _run_guard(repo, "HEAD..HEAD")

    # Assert
    assert result.returncode == 0
    assert "no new commits" in result.stdout


def test_guard_conventional_message_without_tracker_ids_passes(repo: Path) -> None:
    """Ordinary prose must not trip the message patterns."""
    # Arrange
    _commit(
        repo,
        "fix(routing): task ordering when the pool is empty\n\n"
        "The sprint of work here is unrelated to the role of the router.\n",
    )

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# The regression that motivated the guard.
# ---------------------------------------------------------------------------


def test_guard_tool_coauthor_trailer_blocks_and_names_sha_and_value(repo: Path) -> None:
    """Correct author and committer, but a tool's trailer on the Contributors list.

    This re-creation is the DURABLE proof of the guard's reason for existing: it
    is self-contained, so it survives the incident commit being rewritten out of
    history (at which point the real-history test below skips).
    """
    # Arrange
    sha = _commit(
        repo,
        "fix(cli): error when --judge is passed without --measure\n\n"
        "Co-authored-by: Cursor <cursoragent@cursor.com>\n",
    )

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0, "tool co-author trailer must block the push"
    assert _short(repo, sha) in result.stdout
    assert "cursoragent@cursor.com" in result.stdout
    assert "trailer" in result.stdout


def test_guard_real_incident_commit_blocks() -> None:
    """The guard against the actual commit in this repo's history, not a re-creation.

    Expected to skip once the trailer is rewritten out of history, or in a
    shallow clone or fork. The re-creation test above is what carries the proof
    forward; this one is a bonus while the evidence still exists.
    """
    # Arrange
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{INCIDENT_SHA}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if exists.returncode != 0:
        pytest.skip(
            f"{INCIDENT_SHA} not reachable — expected after the trailer is "
            "rewritten out of history, or in a shallow clone/fork. The "
            "tmp-repo re-creation test covers this behaviour durably."
        )

    # Act
    result = _run_guard(REPO_ROOT, f"{INCIDENT_SHA}~1..{INCIDENT_SHA}")

    # Assert
    assert result.returncode != 0
    assert "cursoragent@cursor.com" in result.stdout
    assert INCIDENT_SHA in result.stdout


def test_guard_coauthor_trailer_alternate_casing_blocks(repo: Path) -> None:
    """``Co-Authored-By:`` is the same trailer to git and to GitHub."""
    # Arrange
    _commit(repo, "feat: thing\n\nCo-Authored-By: Some Tool <bot@example.invalid>\n")

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "bot@example.invalid" in result.stdout


def test_guard_coauthor_line_outside_trailer_block_blocks(repo: Path) -> None:
    """A Co-authored-by line followed by prose is not a trailer to git.

    `git interpret-trailers` only recognises the final block, so the trailer
    parser reports nothing here. The raw-body scan is what catches it.
    """
    # Arrange
    _commit(
        repo,
        "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n\n"
        "A trailing prose paragraph that ends the message.\n",
    )

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "cursoragent@cursor.com" in result.stdout


def test_guard_canonical_trailer_reported_once(repo: Path) -> None:
    """Both sources see a canonical trailer; it must be recorded once."""
    # Arrange
    _commit(repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n")

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert result.stdout.count("Co-authored-by trailer: Cursor") == 1


def test_guard_multiple_trailers_on_one_commit_all_reported(repo: Path) -> None:
    # Arrange
    _commit(
        repo,
        "feat: thing\n\n"
        "Co-authored-by: One Tool <one@example.invalid>\n"
        "Co-authored-by: Two Tool <two@example.invalid>\n",
    )

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "one@example.invalid" in result.stdout
    assert "two@example.invalid" in result.stdout


# ---------------------------------------------------------------------------
# Allowlist semantics.
# ---------------------------------------------------------------------------


def test_guard_unknown_novel_tool_identity_blocks(repo: Path) -> None:
    """An identity named nowhere in the guard must still block.

    This is the allowlist/blocklist difference: a blocklist of known tool names
    passes this commit.
    """
    # Arrange
    novel = "Zephyr Autopilot <agent@zephyr-not-a-real-tool.invalid>"
    _commit(repo, f"feat: thing\n\nCo-authored-by: {novel}\n")
    guard_source = GUARD.read_text(encoding="utf-8")
    assert "zephyr" not in guard_source.lower(), "premise: identity unknown to the guard"

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "zephyr-not-a-real-tool.invalid" in result.stdout


def test_guard_foreign_author_blocks(repo: Path) -> None:
    # Arrange
    _commit(repo, "feat: thing", name="Some Agent", email="agent@example.invalid")

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "author" in result.stdout
    assert "agent@example.invalid" in result.stdout


def test_guard_foreign_committer_with_correct_author_blocks(repo: Path) -> None:
    """Committer is a separate attribution field; a correct author does not redeem it."""
    # Arrange
    _commit(
        repo,
        "feat: thing",
        committer_name="Rebase Tool",
        committer_email="tool@example.invalid",
    )

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "committer" in result.stdout
    assert "tool@example.invalid" in result.stdout


# ---------------------------------------------------------------------------
# Message-body hygiene.
# ---------------------------------------------------------------------------


def test_guard_internal_tracker_id_in_message_blocks(repo: Path) -> None:
    # Arrange
    _commit(repo, "feat: thing\n\nCloses BL-29-1234.\n")

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "BL-29-1234" in result.stdout
    assert "message body" in result.stdout


def test_guard_internal_process_metadata_in_message_blocks(repo: Path) -> None:
    # Arrange
    _commit(repo, "feat: thing\n\nSprint: SPR-42\nRole: Some Role\n")

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "message body" in result.stdout


# ---------------------------------------------------------------------------
# Fail-closed behaviour.
# ---------------------------------------------------------------------------


def test_guard_unresolvable_new_branch_range_fails_closed(repo: Path) -> None:
    """A git failure must block, not wave the push through.

    `set -e` does not apply inside a command substitution on an assignment RHS,
    so an unchecked `git rev-list` failure yields empty output and would read as
    "no commits to check".
    """
    # Arrange
    stdin = f"refs/heads/main {'deadbeef' * 5} refs/heads/main {ZERO_SHA}\n"

    # Act
    result = _run_guard(repo, stdin=stdin)

    # Assert
    assert result.returncode != 0
    assert "could not determine" in result.stdout
    assert "no new commits" not in result.stdout


def test_guard_unresolvable_update_range_fails_closed(repo: Path) -> None:
    """Same, on the ordinary <remote-sha>..<local-sha> path."""
    # Arrange
    head = _git(repo, "rev-parse", "HEAD")
    stdin = f"refs/heads/main {head} refs/heads/main {'cafebabe' * 5}\n"

    # Act
    result = _run_guard(repo, stdin=stdin)

    # Assert
    assert result.returncode != 0
    assert "could not determine" in result.stdout


def test_guard_unresolvable_explicit_range_fails_closed(repo: Path) -> None:
    # Arrange / Act
    result = _run_guard(repo, "no-such-ref..HEAD")

    # Assert
    assert result.returncode != 0
    assert "could not determine" in result.stdout


# ---------------------------------------------------------------------------
# Range resolution.
# ---------------------------------------------------------------------------


def test_guard_hook_stdin_branch_update_blocks_on_bad_commit(repo: Path) -> None:
    # Arrange
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n")
    head = _git(repo, "rev-parse", "HEAD")
    stdin = f"refs/heads/main {head} refs/heads/main {base}\n"

    # Act
    result = _run_guard(repo, stdin=stdin)

    # Assert
    assert result.returncode != 0
    assert "cursoragent@cursor.com" in result.stdout


def test_guard_hook_stdin_crlf_line_endings_still_resolves(repo: Path) -> None:
    """A CRLF-terminated ref line must not glue a carriage return to the last SHA."""
    # Arrange
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n")
    head = _git(repo, "rev-parse", "HEAD")
    raw = f"refs/heads/main {head} refs/heads/main {base}\r\n".encode()

    # Act
    result = _run_guard(repo, raw_stdin=raw)

    # Assert
    assert result.returncode != 0
    assert "could not determine" not in result.stdout
    assert "cursoragent@cursor.com" in result.stdout


def test_guard_hook_stdin_multiple_ref_lines_checks_every_ref(repo: Path) -> None:
    """Pushing two branches at once: a violation on either must block."""
    # Arrange
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "clean-branch")
    clean_head = _commit(repo, "feat: a clean thing")
    _git(repo, "checkout", "-q", "-b", "dirty-branch", base)
    dirty_head = _commit(
        repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n"
    )
    stdin = (
        f"refs/heads/clean-branch {clean_head} refs/heads/clean-branch {base}\n"
        f"refs/heads/dirty-branch {dirty_head} refs/heads/dirty-branch {base}\n"
    )

    # Act
    result = _run_guard(repo, stdin=stdin)

    # Assert
    assert result.returncode != 0
    assert "cursoragent@cursor.com" in result.stdout


def test_guard_hook_stdin_new_branch_zero_remote_sha_blocks_on_bad_commit(
    repo: Path,
) -> None:
    """New branch: remote-sha is all zeros, so there is no previous state to diff."""
    # Arrange
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n")
    head = _git(repo, "rev-parse", "HEAD")
    stdin = f"refs/heads/feature {head} refs/heads/feature {ZERO_SHA}\n"

    # Act
    result = _run_guard(repo, stdin=stdin)

    # Assert
    assert result.returncode != 0
    assert "cursoragent@cursor.com" in result.stdout


def test_guard_hook_stdin_new_branch_all_clean_exits_zero(repo: Path) -> None:
    """The new-branch path must not flag a clean history it walks to the root."""
    # Arrange
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feat: a clean thing")
    head = _git(repo, "rev-parse", "HEAD")
    stdin = f"refs/heads/feature {head} refs/heads/feature {ZERO_SHA}\n"

    # Act
    result = _run_guard(repo, stdin=stdin)

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr


def test_guard_hook_stdin_branch_deletion_is_skipped(repo: Path) -> None:
    """A delete (local-sha all zeros) publishes no commits."""
    # Arrange
    stdin = f"(delete) {ZERO_SHA} refs/heads/gone {_git(repo, 'rev-parse', 'HEAD')}\n"

    # Act
    result = _run_guard(repo, stdin=stdin)

    # Assert
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no new commits" in result.stdout


def test_guard_explicit_range_overrides_stdin(repo: Path) -> None:
    # Arrange
    _commit(repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n")

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD", stdin="")

    # Assert
    assert result.returncode != 0
    assert "cursoragent@cursor.com" in result.stdout


def test_guard_no_upstream_falls_back_to_unpushed_commits(repo: Path) -> None:
    """With no upstream and no remotes, every local commit is unpublished."""
    # Arrange
    _commit(repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n")

    # Act
    result = _run_guard(repo)

    # Assert
    assert result.returncode != 0
    assert "cursoragent@cursor.com" in result.stdout


# ---------------------------------------------------------------------------
# Packaging — git silently skips a non-executable hook.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "scripts/check-commit-attribution.sh",
        ".githooks/pre-push",
        "scripts/ci-local.sh",
    ],
)
def test_guard_shell_entrypoints_are_executable_in_the_index(path: str) -> None:
    """The mode must be recorded in the index (100755), not just on the filesystem.

    git skips a non-executable hook without a word, and the hook execs the
    script. This clone has core.fileMode=false, and a POSIX clone takes its mode
    from what was committed.
    """
    # Arrange / Act
    entry = subprocess.run(
        ["git", "ls-files", "-s", "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    # Assert
    assert entry.strip(), f"{path} is not tracked"
    assert entry.startswith("100755"), f"{path} must be executable, got: {entry.strip()}"


# ---------------------------------------------------------------------------
# Output contract.
# ---------------------------------------------------------------------------


def test_guard_failure_output_names_remediation_and_cost_asymmetry(repo: Path) -> None:
    # Arrange
    _commit(repo, "feat: thing\n\nCo-authored-by: Cursor <cursoragent@cursor.com>\n")

    # Act
    result = _run_guard(repo, "HEAD~1..HEAD")

    # Assert
    assert result.returncode != 0
    assert "git rebase" in result.stdout
    assert "force-push" in result.stdout
    assert ALLOWED_IDENTITY in result.stdout


def test_guard_is_wired_into_pre_push_hook_before_ci() -> None:
    """Attribution is instant; CI takes minutes. Fail fast, in that order."""
    # Arrange
    hook = (REPO_ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")

    # Assert
    assert "check-commit-attribution.sh" in hook
    assert "ci-local.sh" in hook
    assert hook.index("check-commit-attribution.sh") < hook.index("ci-local.sh")
