# Security Policy

Frugon is a local-first LLM cost analyzer. Its strongest security property is structural:
`frugon analyze` runs entirely on your machine — no network, no telemetry, nothing sent to us.
The smallest attack surface is the one that never collects your data.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report privately:

- **Preferred:** GitHub's **"Report a vulnerability"** button (this repo → **Security** tab → private advisory).
- **Email:** security@rodiun.io

Include the affected version, a description, steps to reproduce, and any suggested fix.
You'll get an acknowledgment within **48 hours** and a triage response within **7 days**. We credit
reporters who follow responsible disclosure, unless you'd prefer to stay anonymous.

## Supported versions

Frugon is pre-1.0 and ships from `main`. Security fixes land on the latest PyPI release —
please `pipx upgrade frugon` before reporting, in case it's already fixed.

| Version | Supported |
|---|---|
| latest (PyPI) | ✅ |
| older | ⚠️ upgrade first |

## In scope (what matters most)

Frugon handles potentially sensitive inputs, so these matter most:

- **Local data handling** — any path that could leak your logs / prompts / completions off-machine, or write them somewhere unexpected.
- **`--measure` + API keys** — `--measure` calls *your own* providers with *your own* keys; any path that could route a key or prompt anywhere other than the configured provider (e.g. a proxy/redirect side-channel) is in scope.
- **`capture` proxy** — any way the local logging proxy could forward credentials or data to an unintended destination.
- **Supply chain** — how frugon pins or fetches its dependencies and pricing data.

Out of scope: vulnerabilities in your own provider accounts, your local OS, or third-party models.

## Our posture (verifiable in source)

- **No data egress in `analyze`** — tokenize + price + arithmetic, fully local; asserted at the socket layer in tests (a regression that adds an outbound call breaks CI).
- **`--measure` goes only to your providers, with your keys** — never to us; keys are never logged or persisted (defense-in-depth socket-patching tests guard this).
- **Open source** — every claim above is checkable.

---

Built by [Rodiun](https://rodiun.io). MIT licensed.
