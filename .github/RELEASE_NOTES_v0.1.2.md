# frugon v0.1.2

A patch release. The headline fix: `frugon --version` now reports the correct version.

## Fixed

- **`--version` (and the request `User-Agent`) now report the actual installed version.** They previously read a hardcoded string that wasn't kept in lockstep with the package version — so the 0.1.1 wheel reported `0.1.0`. The version is now derived from the installed package metadata, so it can never drift again.

## Housekeeping

- **Refreshed the bundled pricing and quality data.** The out-of-the-box pricing table now covers the full [LiteLLM](https://github.com/BerriAI/litellm) registry, and the quality tiers reflect the latest LMArena snapshot. You can refresh these yourself any time with `frugon pricing update` / `frugon quality update` — this just improves the defaults you get on a fresh install.
- Internal CI maintenance (GitHub Action runtime bumps; data-sync workflow fix).

## Install / upgrade

```bash
pipx upgrade frugon        # or:  pip install -U frugon  /  uvx frugon@latest
```
