"""Generator script for data/sample_logs.jsonl.gz — the bundled ``--demo`` fixture.

Run from the repo root:
    python scripts/gen_sample_logs.py

This fixture is the SINGLE SOURCE OF TRUTH for the ``frugon analyze --demo``
output (the repo GIF is recorded from it and the landing demo card is a
screenshot of it).  It models a believable **team-scale** workload, not a
personal dev log:

  * A production assistant / support stack on ``gpt-5.5`` (Elite
    quality tier; $5 / $30 per 1M tokens) doing tens of thousands of calls
    over a one-month window.
  * The bulk are routine, short calls — classification, routing, Q&A, short
    summaries, translations — that frugon's per-call difficulty classifier marks
    EASY and proposes routing to ``claude-haiku-4-5`` (Strong tier; $1 / $5
    per 1M tokens) — a genuine quality-tier step-down with a compelling saving.
  * A minority are genuinely demanding calls — incident post-mortems, large
    multi-function reviews, long design-doc critiques, deep debugging — that the
    classifier KEEPS on the premium baseline.
  * A slice of traffic is already running on the cheaper ``claude-haiku-4-5`` (a
    team part-way through a migration) so the demo's accounting line reconciles
    *every* call: routed + kept + already-on-cheaper == analyzed.

Determinism (binding):
  The dataset is built by a fixed, seeded, deterministic procedure — corpus
  entries are cycled in order and prompts are padded to fixed target sizes from
  a fixed lorem corpus.  No randomness, no clock reads.  ``python
  scripts/gen_sample_logs.py`` produces a byte-identical file every run, so
  ``frugon analyze --demo`` is byte-identical every run.

Honesty (binding):
  Every record's ``usage`` block is computed with the tokencost tokenizer, so
  the bundled counts equal exactly what the cost engine recomputes from the
  messages (verified by tests/test_sample_data.py).  The demo headline can never
  be silently inflated: ``_report_stats`` prints the routed / kept / blended /
  saving figures from the REAL frugon engine on every run, so the headline can
  never drift from what the tool actually computes.

Target register (achieved by the counts below, verified by tests):
  * tens of thousands of analyzed calls
  * a believable monthly TOTAL spend (across every model) in the
    hundreds-to-low-thousands of USD — this is the "Current" the demo headlines,
    and it equals the sum of the per-model "Cost by model" rows exactly
  * an honest blended saving (of that TOTAL) in the 30-40% band
"""

from __future__ import annotations

import datetime
import gzip
import io
import itertools
import json
from pathlib import Path

import tokencost  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Deterministic filler corpus.
#
# Prompts and completions are padded to fixed target token sizes by appending
# whole sentences from this fixed corpus, in order.  This keeps the bundled
# token counts realistic and the generation byte-deterministic — no randomness,
# no clock.  The text is plausible product / engineering prose so a reader
# skimming the fixture sees believable content, not "lorem ipsum".
# ---------------------------------------------------------------------------

# tokencost does not recognise newer model names (e.g. gpt-5.5, claude-haiku-4-5)
# in its count_string_tokens() call — Anthropic models raise ValueError, and
# unrecognised names raise KeyError.  Fall back to a known GPT model (gpt-4 uses
# cl100k_base encoding) which gives correct counts for both GPT and Claude
# families.  This constant must be defined BEFORE _count_tokens below.
_TOKENCOST_FALLBACK_MODEL = "gpt-4"  # known-good model for count_string_tokens

# Per-message overhead tokens (role + structural tokens).  The frugon cost engine
# uses the same constant, so the bundled usage counts stay byte-identical to what
# the engine recomputes (verified by tests/test_sample_data.py).
_MSG_OVERHEAD_TOKENS = 4  # per message: <|start|>role\n<|content|><|end|>
_REPLY_OVERHEAD_TOKENS = 3  # per reply priming

_FILLER_SENTENCES = [
    "The customer opened the ticket after the second failed checkout attempt.",
    "Our retention policy keeps raw events for ninety days before aggregation.",
    "The dashboard renders the rolling seven-day cost trend by default.",
    "Latency budgets are enforced per route at the gateway tier.",
    "The migration moved roughly forty percent of routine traffic to the cheaper model.",
    "Each request carries an idempotency key so retries never double-charge.",
    "The support team triages inbound messages into four standard categories.",
    "Token usage is recorded from the provider response, not estimated client-side.",
    "We sample real prompts before switching any model in production.",
    "The pricing table is synced from the public LiteLLM registry weekly.",
    "Most conversations resolve within a single turn of the assistant.",
    "Hard cases are escalated to a human reviewer with full context attached.",
    "The pipeline batches writes to keep the primary database under load.",
    "Observability scrapes metrics every fifteen seconds during incidents.",
    "Routing decisions are computed entirely offline from the call shape.",
]


def _pad_to_tokens(seed_text: str, target_tokens: int, model: str) -> str:
    """Append filler sentences to *seed_text* until it reaches ~*target_tokens*.

    Deterministic: the filler corpus is cycled in a fixed order from a fixed
    start, so the same (seed, target, model) always yields the same text.  The
    result lands within a sentence of the target — close enough for a realistic
    fixture, and exactly reproducible.

    Padding is done in bulk (joining many sentences at once) and the token count
    is checked only after each bulk append, so building a multi-thousand-token
    prompt costs a handful of tokenizer calls rather than thousands — generation
    stays fast while remaining byte-deterministic.
    """
    text = seed_text
    filler = itertools.cycle(_FILLER_SENTENCES)
    avg_sentence_tokens = max(
        1, sum(_count_tokens(s, model) for s in _FILLER_SENTENCES) // len(_FILLER_SENTENCES)
    )
    while True:
        current = _count_tokens(text, model)
        if current >= target_tokens:
            break
        remaining = target_tokens - current
        # Add roughly enough sentences to close the gap, but never overshoot the
        # last sentence — append one short of the estimate, then top up one by one.
        bulk = max(1, remaining // avg_sentence_tokens - 1)
        text = " ".join([text, *(next(filler) for _ in range(bulk))])
    return text


def _count_tokens(text: str, model: str) -> int:
    """Token count of a bare string under *model*.

    tokencost raises for Anthropic models (they don't support count_string_tokens)
    and for unrecognised model names.  Fall back to a known GPT model (cl100k_base
    encoding) which gives correct counts for both GPT and Claude families.
    """
    try:
        return int(tokencost.count_string_tokens(text, model))
    except (ValueError, KeyError):
        return int(tokencost.count_string_tokens(text, _TOKENCOST_FALLBACK_MODEL))


# ---------------------------------------------------------------------------
# EASY corpus — short, routine calls the classifier routes to gpt-4o-mini.
#
# These are real, realistic support / assistant tasks.  Each is padded to a
# fixed, modest target size that keeps the difficulty score below the easy
# threshold so the classifier proposes routing it.  The bundled token counts are
# honest (tokencost-derived); padding only makes the calls believably sized, it
# never inflates the recorded usage beyond what the messages actually contain.
# ---------------------------------------------------------------------------

EASY_SEEDS = [
    {
        "system": "You are a customer support classifier. Classify the message as one of: billing, technical, account, general.",
        "user": "I cannot log in to my account after the latest update.",
        "completion": "technical",
        "prompt_target": 360,
        "completion_target": 120,
    },
    {
        "system": "You are a customer support classifier. Classify the message as one of: billing, technical, account, general.",
        "user": "My invoice shows the wrong amount for last month.",
        "completion": "billing",
        "prompt_target": 360,
        "completion_target": 120,
    },
    {
        "system": "You are a triage assistant. Summarize the customer's request in one sentence and name the next action.",
        "user": "How do I reset my password and enable two-factor authentication?",
        "completion": "The customer wants to reset their password and enable 2FA; send the self-serve security guide.",
        "prompt_target": 370,
        "completion_target": 120,
    },
    {
        "system": "You are a helpful assistant. Answer the user's question concisely.",
        "user": "What does our API return when a request exceeds the rate limit?",
        "completion": "It returns HTTP 429 with a Retry-After header indicating when to retry.",
        "prompt_target": 360,
        "completion_target": 120,
    },
    {
        "system": "You are a document summarizer. Summarize the following note in two or three sentences.",
        "user": "The quarterly review covered revenue, churn, and the upcoming migration plan for routine traffic.",
        "completion": "Revenue and churn were reviewed and a migration plan for routine traffic was agreed.",
        "prompt_target": 380,
        "completion_target": 120,
    },
    {
        "system": "Translate the following short English message to French.",
        "user": "Please submit your expense report by Friday and attach all receipts.",
        "completion": "Veuillez soumettre votre note de frais avant vendredi et joindre tous les recus.",
        "prompt_target": 360,
        "completion_target": 120,
    },
]


# ---------------------------------------------------------------------------
# HARD corpus — long, genuinely demanding calls the classifier KEEPS on the
# premium baseline.  These carry real long content so their bundled token
# counts are honest, and they score above the easy threshold on prompt and
# completion length, so frugon keeps them on gpt-4-turbo.
# ---------------------------------------------------------------------------

HARD_SEEDS = [
    {
        "system": (
            "You are a senior site-reliability engineer. Read the incident timeline, "
            "application logs, and the relevant code, then produce a root-cause analysis "
            "with a remediation plan and concrete preventative actions."
        ),
        "user": (
            "At 02:14 UTC our checkout service began returning HTTP 503 for roughly 40% of "
            "requests. The on-call engineer was paged at 02:19 after the error-rate alert "
            "crossed 5%. The p99 latency climbed from 180ms to 9.2s over four minutes. The "
            "service runs eight replicas behind an L7 load balancer, talks to a Postgres "
            "primary with two read replicas, and uses a Redis cluster for idempotency keys. "
            "The connection-acquire path holds a pooled connection across an external PSP "
            "charge call that takes about 600ms, all inside a single open transaction. Pool "
            "size is 20 per replica. A marketing push raised traffic 30% earlier that day. "
            "Walk through the failure, identify the primary and contributing causes, and give "
            "an ordered remediation and prevention plan."
        ),
        "completion": (
            "Root cause: connections are held across a slow external PSP call inside an open "
            "transaction, so a 20-connection pool saturates at about 33 checkouts per second; "
            "the 30% traffic bump pushed sustained load past that ceiling and the pool "
            "exhausted, so queued requests 503'd. Contributing factors: connection recycling "
            "amplified the queue, and row locks held across the PSP call produced a deadlock. "
            "Remediation, in order: move the PSP charge outside the transaction; make the "
            "charge idempotent on the key; add a bounded acquire timeout and shed load with "
            "429 plus Retry-After; right-size the pool to concurrency over hold-time. "
            "Prevention: load-test holding connections across a slow dependency, alert on the "
            "pool in-use ratio, and lint against network calls inside a DB transaction."
        ),
        "prompt_target": 3000,
        "completion_target": 600,
    },
    {
        "system": (
            "You are a staff engineer doing a thorough code review. Identify correctness bugs, "
            "security issues, race conditions, and resource leaks across the whole module, and "
            "propose concrete fixes."
        ),
        "user": (
            "Review this payment-webhook handler module. It verifies a signature by comparing "
            "sha256(secret + payload) with == , dedupes events in an in-process dict guarded "
            "by a lock that is not held across the read, mutates order state without a "
            "transaction, trusts the event amount without checking it against the order total, "
            "and swallows all exceptions in the receipt sender after three tries. It runs in a "
            "multi-threaded WSGI server across several worker processes. Give a prioritised "
            "list of every defect you find with the fix."
        ),
        "completion": (
            "Critical: the signature check is open to timing and length-extension attacks — use "
            "hmac.compare_digest with HMAC-SHA256, not a raw sha256(secret+payload) with ==. "
            "The in-process dedupe does not work across processes, so each worker reprocesses "
            "the event and receipts double-fire; persist processed IDs with a unique constraint "
            "inside the transaction. already_processed also has a check-then-set race because it "
            "reads the dict without the lock. High: the dict grows unbounded; order mutation is "
            "not transactional and has no concurrency guard; the amount is trusted blindly; and "
            "the receipt sender hides failures. Fixes: HMAC plus compare_digest, DB-level "
            "idempotency inside the transaction, a row lock around state changes, amount "
            "validation, and a durable logged retry for receipts."
        ),
        "prompt_target": 3000,
        "completion_target": 600,
    },
    {
        "system": (
            "You are a principal architect. Summarise the proposed design, then critique it for "
            "scalability, consistency, failure modes, and operational risk, and recommend changes."
        ),
        "user": (
            "Design proposal: real-time notification fan-out to up to 5 million connected "
            "clients with a p95 end-to-end latency under one second. A single fanout service "
            "holds all WebSocket connections in memory, every instance consumes all 12 Kafka "
            "partitions, each instance caches the full 80-million-row team-membership table by "
            "polling Postgres with SELECT * every 30 seconds, there is no backpressure so a "
            "slow socket blocks the consumer thread, delivery is best-effort fire-and-forget, "
            "and scaling is by adding full replicas. Evaluate this against the SLO and "
            "recommend a concrete revised architecture."
        ),
        "completion": (
            "Summary: a fully-replicated fan-out where every instance holds all sockets, "
            "consumes all partitions, and caches all memberships via a 30s full-table poll, "
            "best-effort with no backpressure. Critique: the per-instance 80M-row cache and "
            "poll do not scale and will saturate Postgres; consuming every partition on every "
            "instance defeats Kafka's scaling and multiplies cost; local socket maps mean most "
            "lookups miss; one slow socket blocks delivery for everyone (a head-of-line failure "
            "that blows the 1s p95); and fire-and-forget cannot meet a real delivery "
            "expectation. Revision: shard connections by user_id behind a consistent-hashing "
            "router, key Kafka by user_id, replace the poll with a CDC subscription, add bounded "
            "per-socket queues for backpressure, persist ownership for fast failover, and add "
            "per-message acknowledgement with short-window replay."
        ),
        "prompt_target": 3000,
        "completion_target": 600,
    },
    {
        "system": (
            "You are an expert data engineer. Diagnose why the nightly batch job produces "
            "incorrect aggregates, explain the mechanism, and give a corrected implementation."
        ),
        "user": (
            "Our nightly revenue rollup intermittently reports totals that are 3-8% too low, "
            "but only on days after a daylight-saving change or when a late event arrives. The "
            "job partitions by event_date, a DATE derived from event_ts in each ingestor's "
            "local timezone, then selects WHERE event_date = CURRENT_DATE - 1. Ingestors stamp "
            "event_ts with their own local clock; late events keep their original event_ts but "
            "are assigned a partition at insert time; finance reconciles in UTC calendar days; "
            "and analytics queries now hit a read replica 2-5 minutes behind. Explain every "
            "mechanism that can make the total too low and give a timezone-safe aggregation."
        ),
        "completion": (
            "Several independent mechanisms push the total down. Timezone skew: event_date is "
            "computed in local time but finance reconciles in UTC, so a UTC day's revenue is "
            "split across two local partitions and selecting one local date misses the rest. "
            "DST transitions shift the local-day boundary relative to UTC, matching the "
            "'day after a DST change' symptom. Late events keep their original event_ts but get "
            "a partition chosen at insert time, so some are never counted for the day they "
            "belong to. Replica lag drops the most recent events near the cutoff. Corrected "
            "strategy: stamp and partition by a single canonical UTC timestamp, aggregate on a "
            "UTC day with a half-open range event_ts >= start AND event_ts < start + 1 day, run "
            "against the primary or wait out replica lag, and make the job idempotent with a "
            "watermark so late events trigger a recomputation of the affected UTC day."
        ),
        "prompt_target": 3000,
        "completion_target": 600,
    },
]


# ---------------------------------------------------------------------------
# Dataset composition.
#
# Counts are chosen so the REAL frugon engine lands the demo in the target
# register: a few thousand analyzed calls, a believable monthly baseline in the
# hundreds-to-low-thousands of USD, and an honest blended saving in the 30-40%
# band.  The exact routed/kept/blended/saving figures are printed by
# _report_stats from the engine on every run and asserted by the test suite, so
# they can never silently drift.
#
# The window spans exactly 30 days, so the monthly projection equals the observed
# total (a clean, honest 1:1 — no awkward extrapolation factor).
# ---------------------------------------------------------------------------

# Counts are chosen so the REAL frugon engine lands the demo in the target
# register: a believable monthly baseline in the hundreds-to-low-thousands of
# USD (verified by _report_stats on every run), with an honest 30-40% blended
# saving and a total "Current" that equals the sum of the per-model cost rows.
#
# Baseline: gpt-5.5 at $5/$30 per 1M tokens (Elite quality, tier 0).
# Candidate: claude-haiku-4-5 at $1/$5 per 1M tokens (Strong quality, tier 1).
# This is a genuine tier-0 → tier-1 step-down (tier_drop = 1) with a compelling
# ~5x input-price reduction — exactly the coherent story frugon is built to tell.
#
# Total monthly spend lands in the hundreds-to-low-thousands register with an
# honest 30-40% blended saving.  Per-call costs are uniform per seed-group
# (deterministic padding), so these counts pin the totals precisely.  The exact
# routed/kept/blended/saving figures are printed by _report_stats from the engine
# on every run and asserted by the test suite, so they can never silently drift.
N_TURBO_EASY = 36100  # routine gpt-5.5 calls — routed to claude-haiku-4-5
N_TURBO_HARD = 10000  # genuinely hard gpt-5.5 calls — kept on baseline
N_CANDIDATE = 10000  # already migrated to the cheaper claude-haiku-4-5 (not part of the split)

WINDOW_DAYS = 30

# Number of distinct timestamp instants spread across the window.  Records share
# instants in round-robin so the file stays compact in time while the span is a
# clean 30 days.  The exact span is what frugon discloses; with first and last
# records at day 0 and day 30 the disclosed span is exactly 30.0 days.
_BASE_TS = datetime.datetime(2026, 5, 4, 9, 0, 0, tzinfo=datetime.timezone.utc)


def _count_message_tokens(messages: list[dict], model: str) -> int:
    """Count prompt tokens including per-message overhead, mirroring the frugon engine."""
    total = _REPLY_OVERHEAD_TOKENS
    for msg in messages:
        total += _MSG_OVERHEAD_TOKENS
        total += _count_tokens(msg["content"], model)
    return total


def _make_record(seed: dict, model: str, ts: str) -> dict:
    """Build one log record with honest, tokencost-derived usage counts.

    The prompt and completion are padded deterministically to the seed's target
    sizes, then the usage block is computed with tokencost so it matches exactly
    what the cost engine recomputes (tests/test_sample_data.py).

    tokencost.calculate_all_costs_and_tokens() raises KeyError for models not in
    its registry (e.g. gpt-5.5, claude-haiku-4-5).  We therefore count tokens
    directly via count_string_tokens (which falls back to cl100k_base for unknown
    model names — correct for both GPT and Claude families) and compute the
    message overhead manually to match the frugon engine's counting path.
    """
    user_text = _pad_to_tokens(seed["user"], seed["prompt_target"], model)
    completion = _pad_to_tokens(seed["completion"], seed["completion_target"], model)
    messages = [
        {"role": "system", "content": seed["system"]},
        {"role": "user", "content": user_text},
    ]
    try:
        counts = tokencost.calculate_all_costs_and_tokens(messages, completion, model)
        prompt_tokens = int(counts["prompt_tokens"])
        completion_tokens = int(counts["completion_tokens"])
    except (KeyError, Exception):
        # Fallback for models not in tokencost's registry: count via cl100k_base.
        prompt_tokens = _count_message_tokens(messages, _TOKENCOST_FALLBACK_MODEL)
        completion_tokens = _count_tokens(completion, _TOKENCOST_FALLBACK_MODEL)
    return {
        "model": model,
        "timestamp": ts,
        "request": {"messages": messages},
        "response": {"choices": [{"message": {"role": "assistant", "content": completion}}]},
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _timestamp_for(index: int, total: int) -> str:
    """Deterministic ISO-8601 Z timestamp for record *index* of *total*.

    Records are spread linearly across exactly WINDOW_DAYS so the first record
    sits at day 0 and the last at day WINDOW_DAYS, giving a clean disclosed span.
    """
    if total <= 1:
        offset = datetime.timedelta(0)
    else:
        seconds = WINDOW_DAYS * 86400 * index / (total - 1)
        offset = datetime.timedelta(seconds=int(seconds))
    return (_BASE_TS + offset).isoformat().replace("+00:00", "Z")


def generate(output_path: Path) -> None:
    """Generate the bundled fixture with honest, tokencost-derived token counts.

    Writes a gzip-compressed ``sample_logs.jsonl.gz``.  A real assistant stack
    reuses the same handful of prompt templates thousands of times, so the log is
    extremely repetitive.  We exploit that twice over: each distinct (template,
    model) pair renders to byte-identical message content, and we emit all the
    records sharing that content **contiguously** so gzip's back-references land
    inside its 32 KB window and collapse the repetition almost completely.  A
    team-scale, several-tens-of-thousands-call workload therefore still ships as a
    roughly one-megabyte artifact.

    Deterministic, byte-identical every run: the corpora are cycled in a fixed
    order, each record's timestamp is fixed by its position in a stable
    interleaved sequence spanning the 30-day window, the records are then emitted
    in a fixed content-grouped order, and the gzip stream is written with mtime=0
    (so the header carries no wall clock).
    """
    # Build the (seed_index, seed, model) plan first.  seed_index keys the content
    # group an entry belongs to so records with byte-identical message content can
    # be emitted contiguously for tight gzip back-referencing.
    plan: list[tuple[int, dict, str]] = []

    easy_cycle: itertools.cycle[int] = itertools.cycle(range(len(EASY_SEEDS)))
    for _ in range(N_TURBO_EASY):
        i = next(easy_cycle)
        plan.append((i, EASY_SEEDS[i], "gpt-5.5"))

    hard_cycle: itertools.cycle[int] = itertools.cycle(range(len(HARD_SEEDS)))
    for _ in range(N_TURBO_HARD):
        i = next(hard_cycle)
        plan.append((i, HARD_SEEDS[i], "gpt-5.5"))

    candidate_cycle: itertools.cycle[int] = itertools.cycle(range(len(EASY_SEEDS)))
    for _ in range(N_CANDIDATE):
        i = next(candidate_cycle)
        plan.append((i, EASY_SEEDS[i], "claude-haiku-4-5"))

    total = len(plan)

    # Assign every entry a timestamp by its ORIGINAL position so the disclosed
    # span stays a clean 30 days (first at day 0, last at day 30), independent of
    # the emission order chosen below.
    timestamps = [_timestamp_for(i, total) for i in range(total)]

    # Emit in a CONTENT-GROUPED order: group key is (model, seed_index, target
    # sizes) — every entry in a group renders to byte-identical messages, so
    # placing them adjacently lets gzip dedupe the (otherwise large) repeated
    # bodies and only spend bytes on the differing timestamp.  The sort is stable
    # and key-deterministic, so the output is byte-identical every run.
    order = sorted(
        range(total),
        key=lambda i: (
            plan[i][2],  # model
            plan[i][0],  # seed_index
            plan[i][1]["prompt_target"],
            plan[i][1]["completion_target"],
        ),
    )

    records = [
        _make_record(plan[i][1], plan[i][2], timestamps[i])
        for i in order
    ]

    payload = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Compress to bytes with mtime=0 and no embedded filename so the gzip stream
    # is byte-identical every run (header carries no wall clock, no path).
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(payload.encode("utf-8"))
    output_path.write_bytes(buf.getvalue())

    print(
        f"Wrote {len(records)} records to {output_path} "
        f"({output_path.stat().st_size:,} bytes)"
    )
    _report_stats(records)


def _report_stats(records: list[dict]) -> None:
    """Print the honest split + wholesale figures from the REAL frugon engine.

    Uses frugon.cost.analyze_records so the printed routed/kept/blended/saving
    figures are exactly what `frugon analyze --demo` renders — no hand-maintained
    arithmetic that could drift from the engine.
    """
    from frugon.cost import analyze_records, compute_saving_pct, parse_record

    parsed = [parse_record(r) for r in records]
    log_records = [r for r in parsed if r is not None]
    result = analyze_records(log_records)

    print(
        f"Analyzed {result.priced_calls} priced calls; "
        f"observed total ${float(result.total_cost):.2f}"
    )
    if result.monthly_cost is not None:
        print(f"Monthly baseline (projected): ${float(result.monthly_cost):.2f}")

    split = result.split
    if split is not None and split.saving_pct is not None:
        print()
        print("Per-call split routing (the demo headline):")
        print(f"  baseline:  {split.baseline_model}")
        print(
            f"  routed:    {split.routed_count} calls -> {split.candidate_model} "
            "(within tolerance)"
        )
        print(f"  kept:      {split.kept_count} calls -> {split.baseline_model}")
        if split.monthly_baseline is not None and split.monthly_blended is not None:
            print(
                f"  monthly:   ${float(split.monthly_baseline):.2f} -> "
                f"${float(split.monthly_blended):.2f}"
            )
        print(f"  saving:    {float(split.saving_pct):.1f}%")

    if result.candidate_model:
        wholesale = compute_saving_pct(result.total_cost, result.projected_cost)
        if wholesale is not None:
            print()
            print(
                f"Wholesale upper bound (swap all -> {result.candidate_model}): "
                f"{float(wholesale):.1f}%"
            )


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    out = root / "src" / "frugon" / "data" / "sample_logs.jsonl.gz"
    generate(out)
