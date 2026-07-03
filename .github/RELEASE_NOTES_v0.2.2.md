# frugon v0.2.2

A dramatically faster analysis pass, a cleaner "Candidates considered" legend, and a fixed demo image on PyPI.

## Fixed

- **Analysis is dramatically faster — two hot-loop costs fixed.** First, pricing a large log was checking the pricing file on disk once per record — tens of thousands of redundant filesystem checks per run. It now checks once per run; this was the largest single cost, and it speeds up **every** analysis (single-model, `--wholesale`, and `--candidates` runs alike). Second, the "Candidates considered" block introduced in v0.2.1 was re-running the easy/hard call classification once per candidate — up to 23 times over the same call set — even though that classification never depends on the candidate; it now runs once and every candidate's projection is derived from it. Together, on the bundled ~56,100-call demo, the analysis pass dropped from roughly 17 seconds to about 1 second, and the pause after "Priced" is gone. The figures are unchanged to the cent.
- **The demo image now renders on PyPI.** The README's demo GIF used a relative path that GitHub resolves but PyPI's project-page renderer does not, so the animated demo never showed up on `pypi.org/project/frugon`. It now points at the raw GitHub URL, so it renders on both.

## Improved

- **A clearer "Candidates considered" legend on a default (no `--candidates`) run.** The pool size and how many rows are shown now sit right in the header — "23 in pool · top 5 shown" — instead of a trailing sentence you'd read after the table. The two paragraphs of prose below the table are now three short bullet points: what each row represents, how the recommendation is picked, and how to compare specific models or measure quality yourself. An explicit `--candidates` run is unchanged.

## Install / upgrade

```bash
uvx frugon@latest          # or:  uv tool upgrade frugon  /  pipx upgrade frugon  /  pip install -U frugon
```
