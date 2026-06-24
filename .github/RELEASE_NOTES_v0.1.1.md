# frugon v0.1.1

A documentation and report-wording accuracy pass. **No behaviour, cost-math, or analysis changes** — the numbers are identical to v0.1.0; only documentation and report wording differ.

## What changed

- **Provenance, stated precisely.** The report's methodology line now reads **"LMArena quality tiers, RouteLLM-style routing"** — naming each source for exactly what it does: model-quality tiers come from the [LMArena] leaderboard; routing is a RouteLLM-style local heuristic (not a trained router); and the headline savings ranges (30–50%, up to ~85%) are anchored to [RouteLLM]'s published research. The easy/hard split caveat is worded to match.
- **Demo size stated accurately** as 56,100 calls (the bundled `--demo` set; was rounded to "50k" in `--help`).
- **Cross-platform timing wording** — the bundled demo "prices in a few seconds" (≈2s on WSL, ≈4s on Windows).
- **SECURITY.md** — single security contact (`security@rodiun.io`, plus GitHub private vulnerability reporting).

## Install / upgrade

```bash
pipx upgrade frugon        # or:  pip install -U frugon  /  uvx frugon@latest
```

---

Built by [Rodiun]. MIT licensed.

[LMArena]: https://lmarena.ai
[RouteLLM]: https://github.com/lm-sys/RouteLLM
[Rodiun]: https://rodiun.io
