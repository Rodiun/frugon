"""Model-ID canonicalization for Frugon.

Two pure, deterministic, I/O-free functions:

canonicalize(model)
    Strips known gateway/provider prefixes (OpenRouter, Azure, Bedrock,
    Vertex AI, etc.) iteratively and normalises the Bedrock dotted+versioned
    form (anthropic.claude-3-5-sonnet-20241022-v1:0) and the Vertex AI
    ``@version`` pin form (anthropic.claude-haiku-4-5@20251001).
    Bedrock regional prefixes (us., eu., apac.) are stripped before vendor
    folding.  Output is always lowercase (registry keys are lowercase).
    Idempotent for already-bare names.

base_family(model)
    Folds a dated snapshot to its base model family by stripping trailing
    :tag suffixes (:beta, :free), -latest, ISO/compact date suffixes
    (-YYYY-MM-DD, -YYYYMMDD), and Vertex AI ``@version`` pins
    (@YYYYMMDD, @YYYY-MM-DD, @latest, @default, @N).
    Used **only** as a last-resort fallback in pricing lookups — never
    rewrite user-visible output with this result.
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
    ``@version`` pin (@YYYYMMDD, @YYYY-MM-DD, @latest, @default, @N),
    ISO date (-YYYY-MM-DD), compact date (-YYYYMMDD).
    Used **only** as a pricing-lookup fallback — never surface this result
    in user-visible output.

    Returns *model* unchanged when none of the above suffixes are present.
    """
    # Three-phase by design, not by accident:
    #   Phase 1 (tag, -latest, @version) strips *stackable* suffixes and
    #   FALLS THROUGH so a name like "gpt-4o-2024-05-13:beta" loses both the
    #   tag and the date; a name like "claude-haiku-4-5@20251001" loses the pin.
    #   Phase 2 (ISO date, compact date) are mutually exclusive terminal forms, so
    #   the first match RETURNS — a name carries one date format, never both.
    tag_m = _TAG_RE.search(model)
    if tag_m:
        model = model[: tag_m.start()]
    latest_m = _LATEST_RE.search(model)
    if latest_m:
        model = model[: latest_m.start()]
    vertex_m = _VERTEX_VERSION_RE.search(model)
    if vertex_m:
        model = model[: vertex_m.start()]
    m = _DATE_ISO_RE.search(model)
    if m:
        return model[: m.start()]
    c = _DATE_COMPACT_RE.search(model)
    if c:
        return model[: c.start()]
    return model
