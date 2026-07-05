# frugon v0.2.3

A timed comparison checkpoint, an upfront disclosure for the demo's `--measure` try-out, and a routing fix for Mistral's newest model.

## Improved

- **The "Compared" checkpoint now shows its own elapsed time**, and the live progress bar names the comparison stage while it runs. Ranking a couple dozen candidates against the baseline is real work — you'd see `✓ Compared 23 candidates` sit there with no indication of how long it took or that anything was happening. It now reads `✓ Compared 23 candidates in 0.9s`, and the bar itself switches from "Pricing" to "Comparing 23 candidates…" the moment pricing finishes, so the trail never goes quiet.
- **`frugon analyze --demo --measure` now tells you upfront why it only needs an OpenAI key.** The demo's recommendation is computed against the full model roster, same as a real run — but its `--measure` try-out samples a single pinned model so anyone can run it with just `OPENAI_API_KEY`, no multi-provider key hunt. That's now disclosed at the point it happens, not just in the fine print afterward, and the `--measure --help` text and README say the same thing.

## Fixed

- **`mistral-large-3` (and the whole bare `mistral-*` family) now routes correctly for `--measure`.** LiteLLM doesn't infer a provider from a bare Mistral model name the way it does for OpenAI or Anthropic — a `--measure` run against `mistral-large-3` was failing with a generic "bad request" instead of actually sampling the model. It now routes through `mistral/` automatically.

## Install / upgrade

```bash
uvx frugon@latest          # or:  uv tool upgrade frugon  /  pipx upgrade frugon  /  pip install -U frugon
```
