#!/usr/bin/env bash
# Run the full CI gate locally — mirrors .github/workflows/ci.yml exactly.
# The pre-push hook runs this automatically; you can also run it by hand:
#   ./scripts/ci-local.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "▶ uv sync (frozen)"
uv sync --extra dev --extra measure --frozen
echo "▶ ruff check"
uv run ruff check .
echo "▶ mypy"
uv run mypy src
echo "▶ pytest (full suite)"
uv run pytest
echo "▶ strict cost coverage"
uv run pytest --cov-config=.coveragerc-strict --cov-fail-under=90

echo "✓ Local CI gate passed — safe to push."
