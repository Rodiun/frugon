# frugon v0.2.5

Kept promises: this release makes frugon's own claims about itself as honest as the numbers it reports. Flags that did nothing now say so, failures name their cause, and every displayed saving is a floor, never a round-up.

## Fixed

- **`--judge` and `--judge-model` fail fast when their prerequisite is missing.** Passing `--judge` without `--measure`, or `--judge-model` without `--judge`, used to run the analysis and silently ignore the flag. Both now error immediately with a clear message instead of quietly doing nothing.
- **Sampling failures during `--measure` name the cause.** A drained account or a bad key used to surface as one generic error. frugon now tells you whether it was quota exhaustion, an auth failure, or a rate limit, so you know what to fix. Provider error detail is included, with API keys and account identifiers redacted before anything is displayed or written to a report.
- **The site link renders once under `--verbose`, not twice.** The verbose Notes table carried a second copy of the footer link; the duplicate is gone.

## Improved

- **Saving percentages floor, never round up.** Displayed saving percentages used to round to one decimal, which could nudge a figure up across a boundary (49.96% becoming 50.0%), marginally overstating the saving. They now floor instead, so the displayed percentage is never higher than the computed saving. Cost increases keep standard rounding, so a downside is never understated either.
- **Local and unpriced candidate pools get an honest verdict.** Racing an all-local or otherwise unpriced pool used to render "no cheaper candidate found," reading like an evaluated-and-lost verdict when the cost race never ran. frugon now says plainly when a pool has no list price to race against, and never fabricates a price to project a saving. Mixed pools still race their priced candidates as before.

## Install / upgrade

```bash
uvx frugon@latest          # or:  uv tool upgrade frugon  /  pipx upgrade frugon  /  pip install -U frugon
```
