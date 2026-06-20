"""Drift guard: every key in pricing.json and quality.json must already be
in canonical form.

Rationale
---------
``canonicalize()`` defines the normalisation contract for all pricing and
quality lookups.  If a data-sync PR ever introduces a key in an unhandled
form — a raw ``@date`` pin, a publisher-dotted prefix, a gateway prefix, or a
Bedrock versioned form — lookups silently miss and the user sees wrong (or
missing) cost / quality numbers.

This test turns that silent drift into a gated CI failure by asserting, on
every non-``_`` key in both data files, that the key is already canonical::

    canonicalize(key) == key

i.e. no gateway prefix, ``@version`` pin, publisher dot, or Bedrock version
remains to be stripped.  A future data-sync that introduces a key in an
unhandled form fails CI, with the offending key and its canonical form named.

Note: we deliberately do NOT also assert that keys are ``base_family`` fixed
points.  ``pricing.json`` legitimately contains dated snapshot keys (e.g.
``gpt-4o-2024-11-20``, ``claude-3-5-sonnet-20241022``) that LiteLLM prices
distinctly from the bare family.  The lookup reaches those via the exact-match
step that runs *before* any folding, so a dated key is correct and reachable —
not drift.  Forbidding it would reject valid pricing data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_DATA_DIR = Path(__file__).parent.parent / "src" / "frugon" / "data"
_PRICING_JSON = _DATA_DIR / "pricing.json"
_QUALITY_JSON = _DATA_DIR / "quality.json"


def _load_data_keys(path: Path) -> list[str]:
    """Return all non-``_`` top-level keys from *path* (a JSON object file)."""
    data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return [k for k in data if not k.startswith("_")]


# ---------------------------------------------------------------------------
# Parametrize at module load time so each offending key shows as its own
# test failure in the CI output, not a single monolithic assertion.
# ---------------------------------------------------------------------------

_pricing_keys: list[str] = _load_data_keys(_PRICING_JSON)
_quality_keys: list[str] = _load_data_keys(_QUALITY_JSON)

# Deduplicate while preserving insertion order for deterministic parametrize IDs.
_seen: set[str] = set()
_all_keys: list[tuple[str, str]] = []
for _source_label, _keys in (("pricing", _pricing_keys), ("quality", _quality_keys)):
    for _k in _keys:
        if _k not in _seen:
            _seen.add(_k)
            _all_keys.append((_source_label, _k))


@pytest.mark.parametrize(
    ("source", "key"),
    _all_keys,
    ids=[f"{src}:{k}" for src, k in _all_keys],
)
def test_data_key_is_canonical(source: str, key: str) -> None:
    """Arrange: a non-underscore key from pricing.json or quality.json.
    Act: canonicalize(key).
    Assert: key is already in canonical form — it equals its own canonical form.

    Failure here means a future data-sync introduced a key in an unhandled
    form (gateway prefix, ``@version`` pin, publisher dot, Bedrock version) that
    would silently break lookups.
    """
    from frugon.model_id import canonicalize

    canonical = canonicalize(key)
    assert canonical == key, (
        f"[{source}] key {key!r} is not in canonical form: "
        f"canonicalize({key!r}) -> {canonical!r}. "
        f"Either normalise the key in {source}.json to {canonical!r} or "
        f"extend canonicalize() to handle this form."
    )
