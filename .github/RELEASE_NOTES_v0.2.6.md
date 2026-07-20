# frugon v0.2.6

A verdict you can price. The judge already told you which model answered better; this release tells you what that verdict costs, and stops counting two failed answers as a draw.

## Added

- **Ties where neither model addressed the prompt are flagged, not counted as successes.** The judge compares two answers against each other, so two equally poor answers scored a tie, and a tie counted toward "equivalent or better." A cheap model that failed the same way as the baseline could look like a safe swap. frugon now runs a separate check asking whether each answer addressed the prompt at all, flags the ties where neither did, and excludes them from the success count. The pairwise comparison is unchanged and remains the preference signal; this only closes the shared-failure blind spot. The check's default is deliberately honest rather than suspicious: an ambiguous reply, or a transient fault that exhausts its retries, resolves to "addressed", so the flag under-reports shared failure rather than over-reports it.

- **Effective cost per judged success.** A new column alongside the quality tally divides the price you are already shown by the judged success rate, so a cheaper model that fails more often is visibly not cheaper per successful outcome. The success rate is the one printed beside it on the same row, in the form `8/10 equivalent or better`, and the price is the displayed rounded one, so you can recompute the figure from the printed row alone. It reads `n/a` rather than a misleading number in three cases: no price for the candidate, no verdicts at all, or zero judged successes.

## Improved

- **The pre-run estimate now prices the tie checks, not just counts them.** The shared-failure check adds judge calls, and the estimate disclosed how many but never what they cost, so the printed figure was not a true ceiling. It now shows both: the base estimate and the worst case if every judged pair ties. If the judge model has no list price, the ceiling is omitted rather than understated.

- **`--judge` warns when the sample size is too small to trust.** The default of 10 samples is noisy for judge comparisons, and a narrow verdict on 10 samples reads more confident than it is. Running `--judge` with fewer than 20 samples now prints one warning naming your sample count and pointing at the 25 to 30 range. The default is deliberately unchanged, since silently raising it would change what a scripted run costs without asking.

- **Failed shared-failure checks are surfaced, never silent.** If a check cannot complete after retries, the affected comparisons are marked with `~` and footnoted rather than quietly counted as successes, so a judge-side outage cannot inflate the success rate without saying so.

## Install / upgrade

```bash
uvx frugon@latest          # or:  uv tool upgrade frugon  /  pipx upgrade frugon  /  pip install -U frugon
```
