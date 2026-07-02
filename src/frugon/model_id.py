"""Model-ID canonicalization for Frugon.

Three pure, deterministic, I/O-free functions:

canonicalize(model)
    Strips known gateway/provider prefixes (OpenRouter, Azure, Bedrock,
    Vertex AI, etc.) iteratively and normalises the Bedrock dotted+versioned
    form (anthropic.claude-3-5-sonnet-20241022-v1:0) and the Vertex AI
    ``@version`` pin form (anthropic.claude-haiku-4-5@20251001).
    Bedrock regional prefixes (us., eu., apac.) are stripped before vendor
    folding.  Output is always lowercase (registry keys are lowercase).
    Idempotent for already-bare names.

base_family(model)
    Folds a dated/versioned snapshot to its base model family by stripping
    trailing :tag suffixes (:beta, :free), -latest, Vertex AI ``@version``
    pins (@YYYYMMDD, @YYYY-MM-DD, @latest, @default, @N), and a range of
    date/version-pin suffixes: ISO date (-YYYY-MM-DD), compact date
    (-YYYYMMDD), three-digit leading-zero version (-0NN), compact month-day
    (-MMDD), month-year (-MM-YYYY), year-month (-YYYY-MM), and two-digit
    month-day (-MM-DD).  None of these forms change a model's per-token
    price, so folding them is safe for both pricing and quality lookups.
    Used **only** as a last-resort fallback in lookups — never rewrite
    user-visible output with this result.

effort_family(model)
    Strips a single trailing reasoning-effort suffix (-high, -medium, -low,
    -minimal, -thinking, -no-thinking) so that an effort-tagged variant name
    resolves against its base model for QUALITY-tier lookups only.  Reasoning
    effort changes how many tokens a model spends thinking, not its
    per-token rate, so folding it for quality is honest; folding it for
    PRICING would not be, because some providers price a thinking variant
    differently from its non-thinking counterpart.  Size/SKU suffixes
    (-mini, -nano, -air, -lite, -flash, -pro, -plus, -chat, -instant,
    -fast) are genuinely different models and are never folded.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Gateway / provider prefixes — applied iteratively.
# All comparisons operate on the already-lowercased current string, so these
# literals are lowercase and no IGNORECASE flag is needed.
# The loop breaks after the first match so only one prefix is consumed per
# iteration, then repeats until none remain.
# ---------------------------------------------------------------------------

_PREFIXES: tuple[str, ...] = (
    "openrouter/",
    "openai/",
    "anthropic/",
    "azure/",
    "bedrock/",
    "vertex_ai/",
    "together_ai/",
    "fireworks_ai/",
    "groq/",
    "mistral/",
    "cohere/",
    "deepseek/",
    "xai/",
    "gemini/",
    "google/",
)

# Bedrock vendor: one letter followed by zero or more letters/digits (no hyphens).
# Matches "anthropic", "amazon", "ai21", "meta", "cohere", etc.
# Does NOT match hyphenated tokens like "gpt-3" (hyphen terminates [a-z0-9]*).
# No IGNORECASE needed — canonicalize() lowercases input at entry.
_BEDROCK_VENDOR_RE = re.compile(r"^[a-z][a-z0-9]*$")

# Bedrock version suffix: -v<major>:<minor>  e.g. -v1:0
# Greedy so it consumes as much of the model segment as possible EXCEPT the suffix.
# No IGNORECASE needed — canonicalize() lowercases input at entry.
_BEDROCK_VERSION_RE = re.compile(r"^(.+)-v\d+:\d+$")

# Bedrock regional prefix (us./eu./apac.) stripped before vendor fold.
# No IGNORECASE needed — canonicalize() lowercases input at entry.
_BEDROCK_REGION_RE = re.compile(r"^(?:us|eu|apac)\.")

# Vertex AI version pin: @<version> where version is an 8-digit date,
# ISO date, "latest", "default", or a bare integer (e.g. @002).
# Applied in both canonicalize() and base_family() (defense-in-depth).
# IGNORECASE because base_family() may receive a non-canonicalized name (e.g. an
# Arena name via quality._arena_name_to_key) whose @version casing varies;
# canonicalize() already lowercases at entry, so the flag is a no-op cost there.
_VERTEX_VERSION_RE = re.compile(
    r"@(?:\d{8}|\d{4}-\d{2}-\d{2}|latest|default|\d+)$",
    re.IGNORECASE,
)

# Anthropic dotted-version normalisation: claude-<major>.<minor> → claude-<major>-<minor>.
# Scoped strictly to the "claude-" prefix so that legitimately-dotted model names from other
# providers (gpt-4.1, gemini-2.5-pro, gemini-2.0-flash, etc.) are never touched.
# The pattern matches a literal "claude-" lead followed by a digit, a literal dot, and a digit,
# anchored at the dot position so that "claude-3.5-sonnet" → "claude-3-5-sonnet" but a model
# like "claude-haiku-4-5" (hyphen-separated, no dot) remains unchanged.
# Applied as the last step in canonicalize(), AFTER gateway/Bedrock/Vertex normalisation,
# so the input is already lowercase and prefix-free.
_CLAUDE_DOTTED_VERSION_RE = re.compile(r"^(claude-\d+)\.(\d+)")

# Dated snapshot suffixes stripped by base_family.
_DATE_ISO_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")
_DATE_COMPACT_RE = re.compile(r"-\d{8}$")

# Additional suffixes stripped by base_family for lookup-only fallback.
_LATEST_RE = re.compile(r"-latest$", re.IGNORECASE)
_TAG_RE = re.compile(r":[a-z][a-z0-9]*$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Additional date/version-pin suffixes stripped by base_family.
# All are strictly narrower than a parameter count (-70b, -405b, -8x7b) or a
# dotted minor-version pair (-4-5), so those forms are never touched.
# ---------------------------------------------------------------------------

# Three-digit leading-zero version, e.g. "gemini-2.0-flash-001" -> "...-flash".
# The leading zero is required so a genuinely numeric-meaningful trailing
# three-digit group (none observed in practice, but the rule is deliberately
# narrow) is never mistaken for a version tag.
_VERSION_3DIGIT_RE = re.compile(r"-0\d{2}$")

# Compact month-day, e.g. "grok-4-0709" -> "grok-4". MM validated 01-12 so a
# YYMM-style ending like "-2507" (month "25" is invalid) is deliberately left
# alone -- that is a year+month compacted form the seed data also carries
# un-folded (e.g. a "-thinking-2507" variant tag), not a date.
_DATE_COMPACT_MMDD_RE = re.compile(r"-(0[1-9]|1[0-2])(\d{2})$")

# Month-year, e.g. "command-r-08-2024" -> "command-r".
_DATE_MM_YYYY_RE = re.compile(r"-(0[1-9]|1[0-2])-\d{4}$")

# Year-month, e.g. "some-model-2025-09" -> "some-model".
_DATE_YYYY_MM_RE = re.compile(r"-\d{4}-(0[1-9]|1[0-2])$")

# Two-digit month-day, e.g. "nova-...-10-09" -> "nova-...". Both sides must be
# exactly two digits so a single-digit minor-version pair like
# "claude-haiku-4-5" (-4-5) never matches.
_DATE_MM_DD_2DIGIT_RE = re.compile(r"-(0[1-9]|1[0-2])-([0-2]\d|3[01])$")

# ---------------------------------------------------------------------------
# Reasoning-effort suffixes stripped by effort_family (quality lookups only).
# Longest-match-first: "-no-thinking" must be tried before "-thinking" so
# "x-no-thinking" folds to "x", not "x-no".  Size/SKU suffixes (-mini, -nano,
# -air, -lite, -flash, -pro, -plus, -chat, -instant, -fast) are deliberately
# absent from this list -- they name genuinely different models, not an
# effort variant of the same model.
# ---------------------------------------------------------------------------
_EFFORT_SUFFIXES: tuple[str, ...] = (
    "-no-thinking",
    "-thinking",
    "-high",
    "-medium",
    "-low",
    "-minimal",
)


def canonicalize(model: str) -> str:
    """Return the canonical bare model name for *model*.

    Strips gateway/provider prefixes iteratively (leftmost first) then
    normalises the Bedrock dotted+versioned form and the Vertex AI
    ``@version`` pin form.  Pure, idempotent, deterministic — no I/O.

    Lowercases the entire input once at entry so that all regex comparisons
    operate on a case-normalised string.  This guarantees idempotency for
    any case variant of the same model ID (e.g.
    US.ANTHROPIC.CLAUDE-3-5-SONNET-20241022-V1:0 and its lowercase
    form both canonicalize to claude-3-5-sonnet-20241022).

    Unknown vendor prefixes are returned unchanged (lowercased).
    """
    # Lowercase once here — every downstream comparison and regex match
    # operates on this normalised string; no further .lower() calls needed.
    current = model.lower()

    # --- Strip prefixes iteratively ---
    while True:
        consumed: str | None = None
        for prefix in _PREFIXES:
            if current.startswith(prefix):
                consumed = current[len(prefix):]
                break
        if consumed is None:
            break
        current = consumed

    # --- Strip Bedrock regional prefix then normalise vendor.model form ---
    # Handles both Bedrock -vN:M and Vertex AI @version pin on the model part.
    region_m = _BEDROCK_REGION_RE.match(current)
    if region_m:
        current = current[region_m.end():]

    dot_pos = current.find(".")
    if dot_pos > 0:
        vendor = current[:dot_pos]
        if _BEDROCK_VENDOR_RE.match(vendor):
            model_part = current[dot_pos + 1:]
            # Strip Bedrock -vN:M suffix.
            version_m = _BEDROCK_VERSION_RE.match(model_part)
            if version_m:
                current = version_m.group(1)
            else:
                # Strip Vertex AI @version pin from the model part, then drop
                # the publisher prefix (same semantics as the Bedrock -vN:M path).
                vertex_m = _VERTEX_VERSION_RE.search(model_part)
                if vertex_m:
                    current = model_part[: vertex_m.start()]

    # --- Strip any remaining Vertex AI @version pin (bare form, no publisher) ---
    # e.g. "claude-haiku-4-5@20251001" after prefix stripping above.
    vertex_bare_m = _VERTEX_VERSION_RE.search(current)
    if vertex_bare_m:
        current = current[: vertex_bare_m.start()]

    # --- Anthropic dotted-version normalisation: claude-<M>.<N> → claude-<M>-<N> ---
    # Handles wire forms like "claude-3.5-sonnet" (used by OpenRouter) and
    # "claude-3.7-sonnet" that arrive with a dot between major and minor version
    # numbers.  Scoped to the "claude-" prefix only: gpt-4.1, gemini-2.5-pro,
    # gemini-2.0-flash, and all other legitimately-dotted provider model names
    # are left unchanged because they do not match _CLAUDE_DOTTED_VERSION_RE.
    # Out of scope by design: a *vendor-prefixed* dotted form ("anthropic.claude-3.5-
    # sonnet") is not normalised — Bedrock IDs use hyphens, not dots, so this wire
    # form does not occur; the anchored ^claude- match intentionally skips it.
    claude_dot_m = _CLAUDE_DOTTED_VERSION_RE.match(current)
    if claude_dot_m:
        current = claude_dot_m.group(1) + "-" + claude_dot_m.group(2) + current[claude_dot_m.end():]

    return current


def base_family(model: str) -> str:
    """Fold a versioned/tagged snapshot to its base model family name.

    Strips in order: trailing :tag (:beta, :free), -latest, Vertex AI
    ``@version`` pin (@YYYYMMDD, @YYYY-MM-DD, @latest, @default, @N), then one
    terminal date/version-pin form: ISO date (-YYYY-MM-DD), compact date
    (-YYYYMMDD), month-year (-MM-YYYY), year-month (-YYYY-MM), three-digit
    leading-zero version (-0NN), compact month-day (-MMDD), or two-digit
    month-day (-MM-DD).
    Used **only** as a lookup fallback (pricing and quality) — never surface
    this result in user-visible output.

    Returns *model* unchanged when none of the above suffixes are present.
    """
    # Two-phase by design, not by accident:
    #   Phase 1 (tag, -latest, @version) strips *stackable* suffixes and
    #   FALLS THROUGH so a name like "gpt-4o-2024-05-13:beta" loses both the
    #   tag and the date; a name like "claude-haiku-4-5@20251001" loses the pin.
    #   Phase 2 (every date/version-pin form below) are mutually exclusive
    #   terminal forms, so the first match RETURNS — a name carries at most one
    #   such trailing form, never two. Ordering within phase 2 matters: the
    #   longer/more-specific patterns (full ISO, compact 8-digit date, MM-YYYY,
    #   YYYY-MM) are tried before the shorter ones (3-digit version, compact
    #   MMDD, 2-digit MM-DD) so a longer match is never left partially stripped
    #   by a shorter pattern matching only a suffix of it.
    tag_m = _TAG_RE.search(model)
    if tag_m:
        model = model[: tag_m.start()]
    latest_m = _LATEST_RE.search(model)
    if latest_m:
        model = model[: latest_m.start()]
    vertex_m = _VERTEX_VERSION_RE.search(model)
    if vertex_m:
        model = model[: vertex_m.start()]

    iso_m = _DATE_ISO_RE.search(model)
    if iso_m:
        return model[: iso_m.start()]
    compact_m = _DATE_COMPACT_RE.search(model)
    if compact_m:
        return model[: compact_m.start()]
    mm_yyyy_m = _DATE_MM_YYYY_RE.search(model)
    if mm_yyyy_m:
        return model[: mm_yyyy_m.start()]
    yyyy_mm_m = _DATE_YYYY_MM_RE.search(model)
    if yyyy_mm_m:
        return model[: yyyy_mm_m.start()]
    version3_m = _VERSION_3DIGIT_RE.search(model)
    if version3_m:
        return model[: version3_m.start()]
    mmdd_m = _DATE_COMPACT_MMDD_RE.search(model)
    if mmdd_m:
        return model[: mmdd_m.start()]
    mm_dd_2digit_m = _DATE_MM_DD_2DIGIT_RE.search(model)
    if mm_dd_2digit_m:
        return model[: mm_dd_2digit_m.start()]
    return model


def effort_family(model: str) -> str:
    """Strip a single trailing reasoning-effort suffix from *model*.

    Checked against ``_EFFORT_SUFFIXES`` in order (longest-match-first, so
    the stacked "-no-thinking" form is recognised before the bare "-thinking"
    suffix would otherwise consume only its tail). Only ONE suffix is ever
    stripped, and only when it is the entire trailing, hyphen-delimited
    token -- "thinking-cap-model" is untouched because "-thinking" is not at
    the end of the string.

    Pure, idempotent for already-bare names, no I/O. Reasoning effort changes
    token volume, not a model's per-token rate, so this fold is valid **for
    quality-tier lookups only** -- it must never be used in a pricing lookup
    (some providers price a thinking variant differently from its
    non-thinking counterpart).
    """
    lowered = model.lower()
    for suffix in _EFFORT_SUFFIXES:
        if lowered.endswith(suffix):
            return model[: -len(suffix)]
    return model
