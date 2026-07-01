# Contributing to Frugon

Thank you for your interest in Frugon. Contributions are welcome — bug reports, feature
requests, and pull requests all help.

---

## Getting started

**Prerequisites:** Python 3.10+, [pip](https://pip.pypa.io/) or [pipx](https://pipx.pypa.io/).

```bash
git clone https://github.com/Rodiun/frugon.git
cd frugon

# Create a virtual environment (any method works)
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows

# Install the project in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Running the checks

All checks must pass before a pull request is merged. These mirror the CI workflow (`.github/workflows/ci.yml`):

```bash
# Lint
ruff check .

# Type check
mypy src

# Tests (overall coverage ≥ 80%)
pytest

# Strict cost-math coverage gate (cost / pricing / routing must stay ≥ 90%)
pytest --cov-config=.coveragerc-strict --cov-fail-under=90
```

CI installs both the `dev` and `measure` extras (`uv sync --extra dev --extra measure --frozen`,
then `uv run <step>`) and runs every step on Ubuntu, macOS, and Windows against Python 3.10–3.13.
A PR is not merged until all 12 combinations are green.

---

## Code style

- **Python 3.10+** — type hints on every function signature.
- **`pathlib.Path`** for all filesystem paths — no `os.path` string concatenation.
- **`encoding="utf-8"`** on every `open()` call.
- No platform-specific shell-outs (`bash`, `/tmp`, POSIX-only commands).
- `ruff` is the formatter and linter. Run it before committing.

---

## Scope

Frugon is deliberately small, and the scope is locked. The user-facing surface is the
`capture` and `analyze` commands plus the local-table helpers `models`, `pricing update`,
`quality update`, and the combined `update` (refreshes both tables at once) — delivering **three capabilities**: cost analysis, quality
visibility, and routing recommendation.

The following are **out of scope** by design: gateway/proxy, live routing,
web UI, accounts, a database, a marketplace, eval-set management, and support
for more than two log formats. If your idea is on that list, please open a
discussion issue rather than a PR.

---

## Privacy

Frugon must never phone home. `analyze` is fully local — no network calls.
`--measure` calls only the user's own providers using the user's own keys.
Any PR that introduces an outbound call to a Rodiun / Frugon endpoint will be
rejected.

---

## Submitting a pull request

1. Fork the repository and create a branch: `git checkout -b fix/short-description`.
2. Make your change; add or update tests; ensure all checks pass.
3. Open a PR against `main` with a short description of what changed and why.

---

## Reporting issues

Open a [GitHub issue](https://github.com/Rodiun/frugon/issues). Include your OS,
Python version, and the full command + output that triggered the issue.

---

## Cutting a release

Tag the commit and push — the release workflow handles the rest:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

This triggers `.github/workflows/release.yml`, which builds the sdist and wheel,
publishes to PyPI via Trusted Publishing (no API token required), and creates a
GitHub Release with both artifacts attached.

---

## License

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE).
