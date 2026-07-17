from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

# A pinned third-party action: owner/repo@<40-hex-sha>  # vX (version comment).
_SHA_PIN_RE = re.compile(r"@([0-9a-f]{40})\s*#\s*v?[\w./-]+")


def _setup_uv_sha(workflow: str) -> str | None:
    """Return the full commit SHA the workflow pins astral-sh/setup-uv to, or None."""
    m = re.search(r"astral-sh/setup-uv@([0-9a-f]{40})", workflow)
    return m.group(1) if m else None


def test_pytest_addopts_enforces_overall_coverage_floor() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]

    assert "--cov-fail-under=80" in addopts


def test_ci_uses_uv_for_install_lint_typecheck_and_tests() -> None:
    workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    # setup-uv must be pinned to a full 40-hex commit SHA (not a mutable tag),
    # with a human-readable version comment.  Parse it rather than hardcode the
    # literal so a deliberate bump updates one place and a mutable-tag regression
    # is caught here.
    sha = _setup_uv_sha(workflow)
    assert sha is not None, "ci.yml must pin astral-sh/setup-uv to a 40-hex commit SHA"
    assert re.search(rf"astral-sh/setup-uv@{sha}\s*#\s*v", workflow), (
        "the setup-uv SHA pin must carry a trailing '# vX' version comment"
    )
    assert "python-version: \"${{ matrix.python-version }}\"" in workflow
    assert "uv sync --extra dev --extra measure --frozen" in workflow
    assert "uv run ruff check ." in workflow
    assert "uv run mypy src" in workflow
    assert "uv run pytest" in workflow
    assert "actions/setup-python" not in workflow
    assert "python -m pip install" not in workflow
    assert "python -m pytest" not in workflow
    assert "python -m ruff" not in workflow
    assert "python -m mypy" not in workflow


def test_ci_runs_strict_coverage_gate_for_cost_math_modules() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "uv run pytest --cov-config=.coveragerc-strict" in workflow


def test_strict_coverage_config_gates_cost_math_triad_at_90_percent() -> None:
    strict_config = (ROOT / ".coveragerc-strict").read_text(encoding="utf-8")

    assert re.search(r"(?m)^fail_under = 90$", strict_config)
    assert "src/frugon/cost.py" in strict_config
    assert "src/frugon/pricing.py" in strict_config
    # routing.py is now a REAL module (the split-routing engine) — the strict
    # cost-math gate measures it; the earlier "phantom file" defect is resolved
    # by the file existing and being covered, not by deleting the reference.
    assert "src/frugon/routing.py" in strict_config
    assert (ROOT / "src" / "frugon" / "routing.py").exists()


def test_every_workflow_declares_a_top_level_permissions_block() -> None:
    """Supply-chain hardening (FRG-OSS-042): every workflow file under
    .github/workflows/ must declare an explicit top-level ``permissions:``
    block.  Without one, GITHUB_TOKEN defaults to the repository's configured
    default (which can be as broad as read/write on every scope) — an
    invisible, drifting privilege level.  A workflow-level block is required
    even when a job overrides it more narrowly (e.g. release.yml's publish
    job further restricts to id-token: write for Trusted Publishing).

    A top-level block is recognised as a ``permissions:`` line at column 0
    (not indented under a job), which distinguishes it from a job-level-only
    ``permissions:`` block (indented under `jobs: <name>:`).
    """
    workflow_files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert workflow_files, "expected at least one workflow file"

    top_level_permissions_re = re.compile(r"(?m)^permissions:\s*$")

    for path in workflow_files:
        text = path.read_text(encoding="utf-8")
        assert top_level_permissions_re.search(text), (
            f"{path.name}: missing a top-level `permissions:` block "
            "(GITHUB_TOKEN would fall back to the repo default scope)"
        )


def test_ci_triggers_on_pull_requests_from_any_source_branch() -> None:
    """FRG-OSS-044: ci.yml must run on pull requests targeting main regardless
    of which branch the PR comes from.

    ``pull_request:`` filtered by ``branches:`` matches on the PR's BASE
    (target) branch, not its head (source) branch — so
    ``pull_request: branches: ["main"]`` already covers a PR from ANY source
    branch as long as it targets main.  This test pins that behaviour
    explicitly so a future edit that narrows the trigger (e.g. accidentally
    scoping by head branch, or dropping the pull_request trigger entirely)
    is caught here rather than silently starving feature-branch PRs of CI.
    """
    workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    assert "push:" in workflow
    assert "pull_request:" in workflow
    # The pull_request trigger must not be scoped by a `types:` allowlist that
    # excludes the default (opened/synchronize/reopened) events, and must
    # target main via `branches:` — confirming both triggers fire on every
    # ordinary push-to-main and every PR aimed at main.
    pr_block_match = re.search(
        r"pull_request:\n(?P<body>(?:[ \t]+\S.*\n)+)", workflow
    )
    assert pr_block_match is not None, "pull_request: trigger block not found"
    pr_body = pr_block_match.group("body")
    assert 'branches: ["main"]' in pr_body or "branches:\n" in pr_body
    assert "types:" not in pr_body, (
        "pull_request trigger must not restrict event types away from the "
        "default (opened/synchronize/reopened)"
    )


def test_release_workflow_setup_uv_steps_use_distinct_cache_suffixes() -> None:
    """FRG-OSS-055: pushing a release tag triggers release.yml AND ci.yml
    concurrently against the same commit. release.yml's own "test" and
    "build" jobs also both run setup-uv. Without a distinct cache-suffix per
    job, every setup-uv step shares the same default cache key, so whichever
    job's cache-save loses the race logs a spurious "Failed to save: Unable
    to reserve cache" on every release run. Each setup-uv step in
    release.yml must carry its own ``cache-suffix`` input so no job depends
    on winning a race against another.
    """
    workflow = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")

    # One cache-suffix per astral-sh/setup-uv step, in step order.
    suffixes = re.findall(
        r"astral-sh/setup-uv@[0-9a-f]{40}[^\n]*\n(?:[ \t]+with:\n(?:[ \t]+\S.*\n)*)?",
        workflow,
    )
    assert len(suffixes) == 2, (
        f"expected exactly 2 astral-sh/setup-uv steps in release.yml, found {len(suffixes)}"
    )

    cache_suffixes = [
        m.group(1)
        for block in suffixes
        for m in [re.search(r'cache-suffix:\s*"([^"]+)"', block)]
        if m is not None
    ]
    assert len(cache_suffixes) == 2, (
        "both astral-sh/setup-uv steps in release.yml must set a "
        f"cache-suffix input; found {cache_suffixes!r} in:\n{workflow}"
    )
    assert len(set(cache_suffixes)) == 2, (
        f"release.yml's setup-uv cache-suffix values must be distinct, got {cache_suffixes!r}"
    )


def test_quality_sync_workflow_uses_patient_retry_profile() -> None:
    """quality-sync-retry-profile: the scheduled quality-sync.yml job must pass
    the patient SYNC_MAX_RETRIES / SYNC_BACKOFF_BASE profile to
    fetch_and_update_quality explicitly. Without this pin, a future edit could
    silently drop back to the CLI's snappy default (~15s total) -- decorative
    against the minutes-long HuggingFace dataset-server outages that actually
    redden this workflow.
    """
    workflow = (WORKFLOWS / "quality-sync.yml").read_text(encoding="utf-8")

    fetch_step_match = re.search(
        r"- name: Fetch new quality tiers\n(?P<body>(?:[ \t]+\S.*\n)+)", workflow
    )
    assert fetch_step_match is not None, (
        "quality-sync.yml: 'Fetch new quality tiers' step not found"
    )
    step_body = fetch_step_match.group("body")

    assert "SYNC_MAX_RETRIES" in step_body
    assert "SYNC_BACKOFF_BASE" in step_body
    assert "max_retries=SYNC_MAX_RETRIES" in step_body, (
        "the fetch step must pass max_retries=SYNC_MAX_RETRIES explicitly"
    )
    assert "backoff_base=SYNC_BACKOFF_BASE" in step_body, (
        "the fetch step must pass backoff_base=SYNC_BACKOFF_BASE explicitly"
    )

    # Pin the constants' actual values too, so a drift in quality.py itself
    # (not just the workflow wiring) is caught here: an outage-shaped budget
    # is ~7.75 minutes (15/30/60/120/240s), not the CLI's ~15s total.
    from frugon.quality import SYNC_BACKOFF_BASE, SYNC_MAX_RETRIES

    assert SYNC_MAX_RETRIES == 5
    assert SYNC_BACKOFF_BASE == 15.0


def test_release_and_sync_workflows_pin_actions_to_commit_shas() -> None:
    """Supply-chain hardening: the publish/sync workflows (which push to PyPI and
    open PRs) must pin every third-party action to a full 40-hex commit SHA, not a
    mutable tag.  These are the most security-sensitive paths in the repo (§8)."""
    ci_sha = _setup_uv_sha((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    assert ci_sha is not None

    for name in ("release.yml", "pricing-sync.yml", "quality-sync.yml"):
        path = WORKFLOWS / name
        if not path.exists():
            continue
        wf = path.read_text(encoding="utf-8")
        # setup-uv pinned to the SAME SHA as ci.yml.
        assert _setup_uv_sha(wf) == ci_sha, (
            f"{name}: astral-sh/setup-uv must be pinned to ci.yml's SHA {ci_sha}"
        )
        # No third-party action may be pinned to a bare mutable tag.  Allowed:
        # 40-hex SHA pins (with a # version comment) and first-party actions/*.
        for line in wf.splitlines():
            stripped = line.strip()
            if not stripped.startswith("uses:"):
                continue
            ref = stripped.split("uses:", 1)[1].strip()
            if ref.startswith("actions/") or ref.startswith("./"):
                continue  # GitHub-owned or local actions are trusted
            assert _SHA_PIN_RE.search(ref), (
                f"{name}: third-party action not SHA-pinned: {ref!r}"
            )
