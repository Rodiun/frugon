"""Packaging / dependency-pin tests.

Covers:
- FRG-OSS-004: the ``measure`` extra pins litellm with an upper bound so a
  future litellm major release cannot silently break frugon's --measure /
  --judge path on a plain ``pip install frugon[measure]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

ROOT = Path(__file__).resolve().parents[1]


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_measure_extra_pins_litellm_with_upper_bound() -> None:
    """Arrange: parsed pyproject.toml.
    Act: read the ``measure`` optional-dependency entry for litellm.
    Assert: it declares BOTH a lower bound (existing) and an upper bound
    (<2.0.0) — litellm has broken installs across minors before, and an
    unbounded '>=1.0.0' constraint would let a future 2.x land silently.
    """
    pyproject = _load_pyproject()
    optional_deps = pyproject["project"]["optional-dependencies"]  # type: ignore[index]
    measure_deps: list[str] = optional_deps["measure"]  # type: ignore[index]

    litellm_specs = [dep for dep in measure_deps if dep.startswith("litellm")]
    assert litellm_specs, "measure extra must declare a litellm dependency"
    assert len(litellm_specs) == 1, f"expected exactly one litellm spec, got {litellm_specs!r}"

    spec = litellm_specs[0]
    assert ">=1.0.0" in spec, f"litellm spec must keep its lower bound: {spec!r}"
    assert "<2.0.0" in spec, f"litellm spec must declare an upper bound: {spec!r}"


def test_uv_lock_resolves_litellm_within_pinned_range() -> None:
    """Arrange: the committed uv.lock.
    Act: find the resolved litellm package entry.
    Assert: uv.lock actually records the >=1.0.0,<2.0.0 constraint for the
    measure extra (not just pyproject.toml) — a stale, unregenerated lockfile
    would otherwise silently defeat the pin at install time.
    """
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    frugon_pkg = next(p for p in lock["package"] if p["name"] == "frugon")
    litellm_reqs = [
        req
        for req in frugon_pkg["metadata"]["requires-dist"]
        if req["name"] == "litellm"
    ]
    assert litellm_reqs, "uv.lock records no litellm requirement for frugon — run `uv lock`"
    specifier = litellm_reqs[0].get("specifier", "")
    assert ">=1.0.0" in specifier, (
        f"uv.lock lost the litellm lower bound (got {specifier!r}) — run `uv lock`"
    )
    assert "<2.0.0" in specifier, (
        f"uv.lock does not reflect the <2.0.0 litellm pin (got {specifier!r}) — run `uv lock`"
    )
