# frugon roadmap

## Changelog

### v0.2.6

- **A tie now gets checked for a shared failure.** The pairwise judge defaults
  to a tie whenever neither answer is clearly better, but that alone can't
  distinguish "both are equally good" from "both equally failed to address the
  prompt." Every tie now gets a second, single-answer check; when neither side
  addressed the prompt, `--judge` marks the row `[both failed]` and excludes it
  from the judged-success count instead of silently counting it as a win.
- **New `Eff. $/success` column for `--judge` runs.** Each candidate's price
  divided by its judged-success rate, so quality and cost read as one number —
  what you actually pay per answer that held up, not per call made. Reads
  `n/a` when there's no honest figure to show (unpriced candidate, no
  verdicts, or zero judged successes) rather than a misleading number.
- **`--judge` now warns below 20 samples.** A judge verdict built on a
  handful of prompts is noisy; frugon now prints a heads-up recommending
  25-30 samples for a steadier read, rather than staying silent about it.
### v0.2.5

- **`--judge` and `--judge-model` now fail fast when their prerequisite is
  missing.** Passing `--judge` without `--measure`, or `--judge-model`
  without `--judge`, used to run the analysis and silently ignore the flag.
  Both now error immediately with a clear message instead of quietly doing
  nothing.
- **Sampling failures during `--measure` now name the cause.** A drained
  account or bad key used to surface as one generic error. frugon now tells
  you whether it was quota exhaustion, an auth failure, or a rate limit, so
  you know what to fix.
- **Saving percentages now floor, never round up.** Displayed saving
  percentages used to round to one decimal, which could nudge a figure up
  across a boundary (e.g. 49.96% → 50.0%), marginally overstating the saving.
  They now floor (truncate) instead, so the displayed percentage is never
  higher than the computed saving.
- **Local and unpriced candidate pools get an honest verdict.** Racing an
  all-local or otherwise unpriced candidate pool used to render "no cheaper
  candidate found," reading like an evaluated-and-lost verdict when the cost
  race never ran. frugon now says plainly when a candidate pool has no list
  price to race against, while still showing the full quality comparison when
  you ran `--measure`.

### v0.2.4

- **Compressed logs cannot exhaust your machine.** `.gz` input is decompressed
  as a stream under a 512MB ceiling (`FRUGON_MAX_GZIP_BYTES` to override), and a
  truncated or mislabeled archive gets a clean one-line error, not a traceback.
- **A malformed log line no longer ends the run.** JSON that parses but is not a
  record (a bare array, string, or number) counts into the "malformed records
  skipped" total you already see, and the analysis carries on.
- **Report writes are atomic and symlink-safe.** Every report is written to a
  temp file and swapped into place, so a crash cannot leave half a report and a
  symlinked output path cannot overwrite the symlink's target.
- **`capture` treats your log like the credential it is.** The capture file is
  created private (`0o600`) on macOS and Linux, and the startup panel carries an
  explicit caution on every platform: it holds your full prompts and completions.
- **The capture proxy fails loud.** Streaming requests get a clear 400 rather
  than a silently broken stream, unsupported paths log a one-time warning naming
  the path, and a stop signal shuts the proxy down through its normal cleanup.
- **Deterministic sampling dedup.** `--measure`'s dedup key uses sha256 instead
  of Python's process-salted `hash()`, so the same logs dedup identically across
  runs and machines.

### v0.2.3

- **The "Compared" checkpoint shows its own elapsed time**, and the progress bar
  names the comparison stage while it runs, so ranking a couple dozen candidates
  no longer looks like a stall.
- **`--demo --measure` says upfront why it only needs an OpenAI key.** The demo's
  recommendation runs against the full roster like any other run; only its
  measure try-out samples a single pinned model. That is now disclosed where it
  happens rather than in the fine print.
- **The bare `mistral-*` family routes correctly for `--measure`.** LiteLLM does
  not infer a provider from a bare Mistral name; frugon now routes it explicitly
  instead of failing with a generic bad-request.

### v0.2.2

- **Analysis is dramatically faster.** Pricing checked the pricing file on disk
  once per record, and the candidates block re-ran easy/hard classification once
  per candidate though it never depends on the candidate. Both now run once per
  run. On the bundled demo the analysis pass dropped from roughly 17 seconds to
  about 1. Figures are unchanged to the cent.
- **The demo image renders on PyPI.** The README's GIF used a relative path that
  GitHub resolves and PyPI does not; it now points at the raw URL.
- **A clearer "Candidates considered" legend.** Pool size and rows shown sit in
  the header, and the prose below is three short bullets.

### v0.2.1

- **A 23-model recommendation roster across 11 vendors**, every one priced and
  quality-rated, so recommendations draw from a much wider and more current
  field. The two open-source Llama 4 checkpoints, which have no single
  first-party price, are priced via a labeled reference host rather than an
  invented number.
- **See what else was considered.** A default run now shows a "Candidates
  considered" block alongside the headline: the recommended model plus the next
  four cheapest that also beat your current spend, each with its own projected
  cost, saving, and quality tier. Ties on saving go to the higher quality tier,
  and the column makes that call visible rather than asserted.
- **Better quality coverage for reasoning models.** A rating published under an
  effort-tagged or dated name is attributed to the base model, since reasoning
  effort changes how many tokens a call spends thinking, not the per-token rate.
  Pricing is never folded this way.
- **The demo is no longer special-cased.** `--demo` uses the same default roster
  a real run does, so it shows what you would actually get.
- **`--measure` and `--judge` recognise every vendor in the roster**, so each
  model gets the correct provider key prompt and routes to its provider.

### v0.2.0

- **Curated recommendation roster.** The default candidate pool is now a
  hand-picked set of current top models across providers — drawn from OpenRouter
  usage rankings, each one both priced and quality-rated — so recommendations
  reflect what teams run today rather than a stale list.
- **New `frugon update`** — refresh the pricing and quality tables from their
  public sources in one step (equivalent to `frugon pricing update` then
  `frugon quality update`). No account, and nothing about your usage is sent
  anywhere.
- **Modernized `--demo`** — the bundled demo now reflects a current baseline and
  recommendation, and names the data behind it.

### v0.1.3

- Money totals now display at 2 decimal places (e.g. `$389.88` instead of
  `$389.8849`). Sub-cent amounts keep the precision they need: amounts below
  $0.01 display at 4 dp, amounts below $0.0001 display at 6 dp, so a real
  cost never collapses to `$0.00`. All reported savings are derived from the
  rounded components, so Current − New = SAVING is verifiable from the printed
  figures.

---

Planned enhancements, kept public and concrete. Items here are not yet
implemented; they describe where the routing and reporting are headed.

## Multi-tier / per-bucket routing

**Today (the limitation).** frugon's routing recommendation is a *binary,
single-candidate* split. It classifies each logged call as easy or hard, routes
the easy calls of the dominant baseline model to one cheaper candidate, and
keeps the hard calls on the full-price baseline. Two kinds of saving are left on
the table:

- **Hard calls stay on the full-price baseline.** A hard call is never moved,
  even when a cheaper-but-still-capable model could handle it. The only choices
  for a hard call are "baseline" or "the one easy-call candidate" — there is no
  middle tier to step it down to.
- **Already-cheaper calls are never re-optimized.** Calls already running on a
  model cheaper than the baseline are reported as "already optimal" and left
  untouched, even if an equally capable but cheaper model exists for them.

**Proposed enhancement.** Route *each bucket* of traffic to the cheapest viable
model that still matches the bucket's quality needs, instead of a single
easy/hard cut against one candidate:

- **Easy calls** → the cheapest capable model (as today).
- **Hard calls** → a mid-tier model that is cheaper than the baseline yet still
  rated high enough to handle the harder work — rather than always paying the
  full baseline price.
- **Already-cheaper calls** → re-checked against the candidate pool so they, too,
  move to a cheaper quality-matched model when one exists.

The result is a per-bucket routing plan (easy / hard / already-cheaper, each to
its own cheapest viable quality-matched model) that captures savings the current
binary split cannot reach, while keeping the same offline, no-network,
no-LLM-call method and the same honest "quality is an offline estimate until you
run `--measure`" disclosure.

## Validated-models cache

**Today (the limitation).** frugon is stateless between runs. A model that has
no published quality rating is treated as *unrated* and held out of the
recommended route — even after you have personally verified it with
`--measure --judge` on your own logs. Because nothing is remembered, that same
model is reported as "unrated, excluded" again on the very next run, and you
would have to re-measure it to unlock it once more. The verification work you
already paid for (the provider calls behind the scored verdict) is thrown away
the moment the process exits.

**Proposed enhancement.** Persist the models you have confirmed, locally, the
same way the pricing and quality tables are already cached. When a
`--measure --judge` run produces a clear verdict for a model, record it to a
small local file (e.g. `~/.frugon/validated.json`) capturing:

- the **model** name,
- the **date** it was verified,
- the **verdict** (the win/loss/tie outcome that confirmed it), and
- a **dataset fingerprint** — a hash of the logs the verdict was measured
  against, so a confirmation can be tied to the traffic it was actually proven
  on.

On future runs frugon reads this cache and treats a once-confirmed model as
**eligible** for the recommended route without re-measuring — surfaced honestly,
e.g. *"you verified `claude-haiku-4-5` on 2026-06-12 — treating it as eligible"*.
The cache lives entirely on your machine; nothing about it is sent anywhere, in
keeping with the local-first, no-network guarantee.

**Open questions.**

- **Staleness / expiry.** A model's behaviour (and its provider's weights) can
  change over time. Should a confirmation expire after some interval, or carry a
  "last verified N days ago" note and re-prompt past a threshold?
- **Per-dataset vs global validity.** A model proven on one kind of traffic may
  not hold on another. Should a confirmation count only for logs whose
  fingerprint matches (strict, per-dataset), or apply globally once verified
  (convenient, but less precise)? A middle path is to honour it globally while
  flagging when the current logs differ from the dataset it was proven on.

## Quality — a second source

**Today (the limitation).** Quality comes only from LMArena. It is the most
credible public, license-clean, head-to-head ranking, but it does not cover
every model, and a single source is a single point of staleness. (The tier
*scale* is no longer a gap — tiers are now self-recalibrating percentile bands
over the current Arena field, and the bundled seed is the full current overall
registry; see the `quality` module docstring for the design.)

**Proposed enhancement.** Blend an additional public signal (e.g. a benchmark
composite) as a disclosed secondary source to widen coverage and reduce
single-source staleness — never mixed silently, never overriding the "quality
is an offline estimate until you run `--measure`" guarantee.

## Canonicaliser coverage for bare / dotted API names

**Today (the limitation).** Model-ID canonicalisation now folds gateway
prefixes, dashed date snapshots, the Vertex `@YYYYMMDD` form and dotted vendor
prefixes, and the pricing and quality tables resolve by canonical key. But some
plain API names still fall through: bare or dotted forms such as
`gpt-3.5-turbo` or `claude-3.5-sonnet` can resolve to *unrated* (and sometimes
unpriced) even though the underlying model is present under a different key. The
gap is shared by the pricing and quality lookups, since both ride the same
canonicaliser in `model_id.py`.

**Proposed enhancement.** Extend the canonicaliser (and/or its alias map) so
bare and dotted API names resolve to the same key as their canonical
equivalents, and add a coverage check so a logged model that exists in the
registry under *any* form is never reported as unrated/unpriced. Same offline,
local-only method; purely a lookup-robustness improvement.

## Report rendering — split the monolith

**Today (the limitation).** All report rendering lives in a single
`src/frugon/report.py` — roughly 7,800 lines, about 5× the next-largest module.
One file renders every surface (terminal, Markdown, HTML v1, HTML v2) and holds
the shared formatting helpers (daggers, footnotes, share bars, candidate and
swap-plan tables, verdict styling). It is well-tested (~97% line coverage) and
correct, so this is a maintainability concern, not a defect — but the size makes
the file hard to navigate, raises the cost of each new report feature, and
concentrates merge risk in one place.

**Proposed enhancement.** Split into a `report/` package, behaviour-preserving:

- `report/terminal.py` — the terminal renderer,
- `report/markdown.py` — the Markdown renderer,
- `report/html.py` — the HTML v1 + v2 renderers,
- `report/formatting.py` — shared helpers (daggers, footnotes, share bars,
  candidate + swap-plan tables, verdict styling),
- `report/__init__.py` — re-exports the public surface so every caller is
  unchanged.

The existing ~97% coverage makes this safe: keep every current test green and
the split is correct by construction (characterisation-tested). It should run
through the normal review gate like any implementation change.

**Trigger (why it is deferred).** frugon is in its lean phase, where refactoring
a working, well-tested file purely for its size is out of scope. Pull this
trigger when it next *bites* — the next non-trivial report feature that has to
edit `report.py`, or the file crossing ~9–10K lines, or contributor friction
showing up — and not before. A large file that is correct and covered is not an
emergency.

## Test infrastructure — hermetic quality seed

**Today (the limitation).** Most tests read the live quality table from the
per-machine user-data dir, falling back to the bundled seed. Tests that assert a
specific tier scenario pin a synthetic table via the `install_synthetic_quality`
conftest helper, but the remaining live-seed tests are only deterministic when
the user-data seed equals the bundled seed — the state a fresh CI runner always
has. On a developer box whose user-data seed has drifted (an older
`frugon quality update`, or an interrupted run leaving a partial write), those
tests can read stale tiers and report confusing, machine-specific results.

**Proposed enhancement.** Add an autouse conftest fixture that copies the bundled
seed to a temp file and points the quality-table path at it for every test —
making every run hermetic and CI-faithful regardless of the dev box's user-data
state. `install_synthetic_quality` continues to override it where a test needs
controlled tiers, and the seed-validation tests keep reading the real bundled
seed (via that same copy). Care: the registry-sync tests must keep writing to
their own temp output so the autouse copy is never mutated.

## Demo fixture — basis-divergent saving %

**Today (the limitation).** The `--demo` reconciliation tests verify the rendered
saving % matches the total-basis figure, but the current demo fixture's
total-basis (~37.41%) and baseline-basis (~37.47%) savings both round to 37%, so
the rendered string cannot by itself prove the report uses the total basis. The
basis-confusion guard is therefore asserted at the figure-source level instead.

**Proposed enhancement.** Add (or tune) a demo/sample fixture whose two saving
bases diverge by more than the rendered precision, so the rendered string itself
discriminates the total basis from the baseline basis — restoring an end-to-end
basis-confusion regression guard on every render surface.

## Routing candidate pool — keep it current

**Today (the limitation).** The default candidate pool frugon auto-routes against
(`_ROUTING_CANDIDATES`) is a small fixed list of older models. As the leaderboard
re-anchors and newer cheap-but-capable models ship, that list goes stale: it can
miss a newer model that would be the genuinely best quality-matched cheaper
option, and — because every pool member may sit in the same lower tier — the
"next rung up" escalation offered after a NOT-confirmed `--judge` verdict can
dead-end (no pool member out-tiers the failed candidate while staying cheaper than
the baseline).

**Proposed enhancement.** Refresh the default pool to track current
frontier-and-value models — or derive it from the synced pricing + quality tables
rather than hard-coding it — so both the auto-recommendation and the escalation
ladder draw from up-to-date candidates. Same offline, local-only method; users can
always override with `--candidates`.

## Response-time / latency as a judging dimension

**Today (the limitation).** `--judge` scores sampled responses on quality alone.
A cheaper or local candidate that wins on quality but answers far slower is
recommended exactly the same as one that is both cheaper and fast; latency never
factors into the verdict.

**Proposed enhancement.** Capture per-sample wall-time and tokens-per-second
during `--measure` sampling, and surface p50/p95 latency per candidate beside the
judged quality verdict. When a cheaper or local candidate wins on quality but
loses on latency, say so plainly in the verdict, so the tradeoff is visible
before you route to it.

## Nonce-variance stability probe

**Today (the limitation).** `--measure --judge` samples each prompt once per
candidate. A prompt is inherently non-deterministic: the same model can answer
the identical prompt differently across separate calls, so a single sample
cannot tell genuine prompt instability apart from ordinary decoding variance.

**Proposed enhancement.** Add an opt-in probe that re-samples identical prompts
(baseline decoding variance) alongside nonce-perturbed prompts (prompt-
instability variance). Only variance in excess of the baseline is attributed to
prompt instability. Opt-in because it costs extra samples; the added cost is
disclosed up front in the pre-run estimate.

## Trace ingest (under consideration)

frugon is considering support for ingesting OpenInference/OTel traces as an
additional `capture` input format, alongside the log formats it already reads.
No timeline or commitment yet; noting it here because it keeps coming up.
