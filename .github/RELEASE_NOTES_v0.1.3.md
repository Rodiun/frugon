# frugon v0.1.3

Money is easier to read — and the savings now check out by hand. Every dollar amount displays at two decimal places, and the reported saving reconciles exactly with the figures shown.

## Improved

- **All dollar amounts display at 2 decimal places** (e.g. `$389.88` instead of `$389.8849`). Sub-cent amounts keep the precision they need — costs below $0.01 show 4 decimals, below $0.0001 show 6 — so a genuinely tiny cost never rounds away to `$0.00`.
- **Reported savings reconcile from the printed figures.** The saving amount and the percentage are derived from the rounded numbers you see, so `Current − New = SAVING` and `percent = saving / current` check out by hand — on every surface (the terminal panel, the Markdown report, and both HTML report styles).

## Install / upgrade

```bash
pipx upgrade frugon        # or:  pip install -U frugon  /  uvx frugon@latest
```
