"""Regression: ``__version__`` must track installed package metadata.

It shipped hardcoded (``"0.1.0"``) once, so the 0.1.1 wheel reported the wrong
version. These tests lock the version to the package metadata so a stale literal
can never silently ship again.
"""

from importlib.metadata import PackageNotFoundError, version

import pytest

import frugon


def test_version_when_installed_matches_metadata() -> None:
    try:
        meta = version("frugon")
    except PackageNotFoundError:
        pytest.skip("frugon not installed; package metadata unavailable")
    assert frugon.__version__ == meta


def test_version_when_installed_is_not_dev_placeholder() -> None:
    try:
        version("frugon")
    except PackageNotFoundError:
        pytest.skip("frugon not installed; package metadata unavailable")
    assert frugon.__version__  # non-empty
    assert frugon.__version__ != "0.0.0+dev"


def test_user_agent_embeds_resolved_version() -> None:
    assert frugon.USER_AGENT == (
        f"frugon/{frugon.__version__} (+https://github.com/Rodiun/frugon)"
    )
