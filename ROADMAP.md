# frugon roadmap

## Changelog

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
total-basis (~34.83%) and baseline-basis (~34.95%) savings both round to 35%, so
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

## Saving-percent display — floor, never round up

**Today.** Displayed saving percentages are rounded to one decimal. Rounding can
nudge a figure up across a boundary (e.g. 49.96% → 50.0%), marginally overstating
the saving.

**Proposed enhancement.** Floor (truncate) displayed saving percentages so the
shown number is never higher than the computed saving — strictly conservative, in
keeping with the "never inflate" guarantee. Display-only; the underlying math is
unchanged.
