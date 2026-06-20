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
