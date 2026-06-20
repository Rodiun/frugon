# frugon v0.1.0

> **Your data never leaves your machine. Your keys go straight to your own providers. Nothing reaches us.**

Free, local, open-source LLM cost analyzer. Point it at your real call logs and see — on your machine — how much you'd save by switching or routing models.

## Install

```bash
# One-shot (no install)
uvx frugon analyze ./logs.jsonl

# Permanent install
pipx install frugon
frugon analyze --demo            # bundled sample log, see it work in 5 seconds
frugon analyze ./logs.jsonl      # your real logs
```

Cross-platform: macOS, Linux, Windows · Python 3.10 / 3.11 / 3.12 / 3.13.

## What's in this release

**`frugon analyze`** — read OpenAI-compatible JSONL logs and produce a cost analysis. Counts tokens with [tokencost], prices with [LiteLLM's registry], picks a cheaper-than-baseline candidate model, and tells you the dollar saving. Cross-platform, fully local, no LLM calls, no network. Honest savings anchored to [RouteLLM/LMSYS] quality bands — we never inflate the number.

**`frugon capture`** — passive OpenAI-compatible logger. Point your app's base URL at `http://127.0.0.1:8787` for a day; it records every call to a local JSONL file in the canonical shape and forwards the request unchanged to the real upstream. No data goes anywhere but your local file and your existing upstream.

**`frugon pricing update`** — refresh the bundled pricing table from the [LiteLLM model_prices_and_context_window.json] registry. Atomic write, JSON shape validation, weekly GitHub Actions sync.

**`frugon analyze --measure`** *(optional `[measure]` extra)* — sample real prompts through candidate models using **your own** API keys. Calls go straight to your providers (OpenAI / Anthropic / etc.) — never to us. Two tiers: side-by-side diffs (human judge) or LLM-as-judge win/loss/tie tallies.

**`frugon analyze --report file.html|file.md`** — shareable single-page report. Self-contained HTML with inline CSS (deep indigo + cyan + silver), or clean Markdown. The viral surface someone shows their boss.

## Realistic savings

Anchored to [RouteLLM] / [LMSYS] research bands:

| Traffic mix | Typical saving |
|---|---|
| General mixed traffic | 30–50% |
| Easy / repetitive (MT-Bench) | up to ~85% |
| Hard tasks (MMLU) | ~30% |

**Your actual number comes from your logs.** Frugon shows what the math says for your data.

## Privacy guarantees (tested as code, not promised in prose)

- **Cost analysis is fully local.** No LLM, no network, no telemetry.
- **`capture` never sends data anywhere but your configured upstream.** Asserted at the socket layer in tests — any future regression that introduces a side-channel HTTP client breaks CI.
- **`--measure` calls only the user's own providers with the user's own keys.** Keys are never logged, never persisted, never sent anywhere but the provider. Asserted by a defense-in-depth fixture patching `socket.socket` / `socket.create_connection` / `socket.getaddrinfo`.
- **The CLI collects nothing.** Open source — anyone can verify.

## Quality

- 209 tests, 88% overall coverage, >90% on the cost-math triad (`cost.py` / `pricing.py` / `routing.py`).
- CI green on 3 OS × 4 Python (ubuntu / macos / windows × 3.10 / 3.11 / 3.12 / 3.13).
- All commits reviewer-Opus gated; cost-math changes get an extra `§2a` Opus pass.
- ruff + mypy clean.

## Keep the savings

This release is the diagnosis. Want it to keep routing automatically and hold the savings? → **https://frugon.rodiun.io**

---

Built by [Rodiun]. MIT licensed.

[tokencost]: https://github.com/AgentOps-AI/tokencost
[LiteLLM's registry]: https://github.com/BerriAI/litellm
[LiteLLM model_prices_and_context_window.json]: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
[RouteLLM/LMSYS]: https://github.com/lm-sys/RouteLLM
[RouteLLM]: https://github.com/lm-sys/RouteLLM
[LMSYS]: https://lmsys.org/
[Rodiun]: https://rodiun.io
