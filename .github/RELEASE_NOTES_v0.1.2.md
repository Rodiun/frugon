# frugon v0.1.2

A patch release. The headline fix: `frugon --version` now reports the correct version. It also broadens out-of-the-box pricing coverage.

## Fixed

- **`--version` (and the request `User-Agent`) now report the actual installed version.** They previously read a hardcoded string that wasn't kept in lockstep with the package version — so the 0.1.1 wheel reported `0.1.0`. The version is now derived from the installed package metadata, so it can never drift again.

## Improved

- **Broader out-of-the-box price coverage.** More bare model names now resolve to a price automatically — including the DeepSeek family (`deepseek-r1`, `deepseek-chat`) and xAI Grok (`grok-2`, `grok-3`). Frugon now matches your model name against the pricing registry's provider-prefixed entries, so you don't have to spell out the full provider path to get a cost. Where a model's price genuinely differs across hosts (as many open-weight models do), Frugon still reports no price rather than guess.

## Housekeeping

- **Refreshed the bundled quality tiers** to the latest LMArena snapshot. You can refresh these yourself any time with `frugon quality update` — this just improves the defaults you get on a fresh install.
- Internal CI maintenance (GitHub Action runtime bumps; data-sync workflow fix).

## Install / upgrade

```bash
pipx upgrade frugon        # or:  pip install -U frugon  /  uvx frugon@latest
```
