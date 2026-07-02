# frugon v0.2.1

A much bigger recommendation roster, better quality coverage for reasoning models, a demo that runs on the exact same roster a real analysis uses, and freshly synced data tables.

## Added

- **A 23-model recommendation roster across 11 vendors.** The default candidate pool grew from 10 to 23 current top models spanning OpenAI, Anthropic, Google, DeepSeek, Moonshot, xAI, Mistral, Z.ai, MiniMax, Alibaba, and Meta (open-source Llama) — every one both priced and quality-rated — so recommendations on your real logs draw from a much wider, more current field. The two open-source Llama 4 checkpoints, which have no single first-party price, are priced via Groq as a labeled reference host rather than an invented number.
- **See what else was considered.** A default (no `--candidates`) run now shows a "Candidates considered" block alongside the headline recommendation — the recommended model plus the next four cheapest candidates that also beat your current spend, each one with its own projected monthly cost, saving%, and quality tier. Previously this transparency only appeared when you passed `--candidates` yourself; now it's there by default, on the demo too, with a line telling you how many models were in the full pool and how to compare specific ones with `--candidates`. When two or more candidates save the same amount to the decimal place shown, the recommendation goes to whichever has the higher quality tier — the Quality tier column makes that call visible, not just asserted.

## Improved

- **Better quality coverage for reasoning models.** frugon now recognises that LMArena often rates a reasoning model under an effort-tagged or dated name (`gpt-5-high`, `grok-4-0709`) while your logs carry the bare name (`gpt-5`, `grok-4`) — and honestly attributes the rating to the base model, since reasoning effort changes how many tokens a call spends thinking, not the per-token rate. More of the models in your logs come back rated instead of unrated. Pricing is never folded this way — a thinking variant keeps its own price.
- **The demo is no longer special-cased.** `frugon analyze --demo` used to route against a small, fixed illustrative candidate set — a stand-in that could drift from what the tool actually recommends on your own logs. It now uses the SAME default 23-model roster a real run does, so the demo shows exactly what you'd get. The one narrower exception: `--demo --measure` still samples a single pinned model so the try-out path only needs an `OPENAI_API_KEY` — that pin never touches the recommendation itself.
- **The demo's recommendation moved.** With the un-pinned roster and the refreshed data below, the demo's headline recommendation now routes easy calls to `deepseek-v4-flash` (it previously routed to a fixed illustrative pick) — an honest side effect of no longer special-casing the demo.
- **Fresh pricing and quality data.** The bundled pricing table and LMArena quality tiers are re-synced to their public sources as of this release, the same weekly process `frugon update` runs for you.
- **Honest disclosure copy.** The post-recommendation note no longer claims a "fixed demo candidate set" — it says plainly that `--demo` runs on bundled sample data, full stop.
- **`--measure`/`--judge` now recognise every vendor in the 23-model roster.** DeepSeek, xAI (Grok), Moonshot (Kimi), Z.ai (GLM), MiniMax, and Alibaba (Qwen) models — plus the two reference-host Llama 4 checkpoints — now get the correct provider-key prompt and route correctly to their provider, instead of a generic "bad request" the first time you tried to measure one of them.

## Install / upgrade

```bash
uvx frugon@latest          # or:  uv tool upgrade frugon  /  pipx upgrade frugon  /  pip install -U frugon
```
