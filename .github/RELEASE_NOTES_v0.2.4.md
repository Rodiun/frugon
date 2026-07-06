# frugon v0.2.4

A hardening release: sturdier log ingest, a safer capture proxy, and a tighter supply chain. No behaviour change for well-formed logs — every number renders exactly as before.

## Hardened

- **Compressed logs can't blow up your machine.** `.gz` files are now decompressed as a stream with a 512MB ceiling (override with `FRUGON_MAX_GZIP_BYTES`), and a truncated or mislabeled `.gz` gets a clean one-line error instead of a raw traceback.
- **Malformed log lines can't take down a run.** A JSONL line that parses as a bare array, string, or number — valid JSON, but not a record — used to crash `analyze` outright. It's now counted into the same "malformed records skipped" total you already see, and the run carries on.
- **Report writes are atomic and symlink-safe.** Every report file is written to a temp file and swapped into place, so a crash mid-write can't leave a half-written report, and a symlinked output path can no longer overwrite the symlink's target.
- **`capture` treats your log like the credential it is.** The capture file is created private (`0o600`) on macOS/Linux, and the startup panel now carries an explicit caution on every platform: it contains your full prompts and completions.
- **The capture proxy fails loud, not silent.** Streaming requests get a clear 400 explaining capture doesn't support them yet (instead of silently breaking the stream); requests to unsupported paths log a one-time warning naming the path and what is supported (instead of a bare 404); and a `kill`/systemd stop now shuts the proxy down cleanly through its normal cleanup path.
- **Deterministic sampling dedup.** `--measure`'s prompt dedup key now uses sha256 instead of Python's process-salted `hash()`, so the same logs dedup identically across runs and machines.

## Supply chain

- **litellm is pinned below 2.0** in the `measure` extra, so a future major release can't silently break `--measure` installs.
- **Every CI workflow now runs with least-privilege permissions**, enforced by a test that checks every workflow file — including ones added later.

## Install / upgrade

```bash
uvx frugon@latest          # or:  uv tool upgrade frugon  /  pipx upgrade frugon  /  pip install -U frugon
```
