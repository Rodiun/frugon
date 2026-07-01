# frugon v0.2.0

Recommendations now track the models developers actually use. The routing pool is a curated set of current top models — every one priced and quality-rated — and a new `frugon update` command keeps your pricing and quality tables current.

## Added

- **`frugon update`** — refresh the pricing and quality tables from their public sources in one step (the same as running `frugon pricing update` then `frugon quality update`). No account, and nothing about your usage is sent anywhere.
- **Curated recommendation roster.** The default candidate pool is now a hand-picked set of current top models across providers, drawn from OpenRouter usage rankings and intersected with LiteLLM pricing and LMArena quality tiers — so recommendations reflect what teams run today rather than a stale list. Every model in the pool is both priced and quality-rated.

## Improved

- **A clear note on where recommendations come from.** After a recommendation, frugon names the curated set, when its prices were last synced, and its sources — with a nudge to run `frugon update` if the bundled data is getting old. It never blocks output.
- **Broader out-of-the-box pricing** — several newly-listed models added to the bundled table.
- **Modernized `--demo`** — the built-in demo now reflects a current baseline and recommendation.

## Install / upgrade

```bash
uvx frugon@latest          # or:  uv tool upgrade frugon  /  pipx upgrade frugon  /  pip install -U frugon
```
