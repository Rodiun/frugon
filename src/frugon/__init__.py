"""frugon — free, local, open-source LLM cost analyzer."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: the installed package metadata (from pyproject).
    # NEVER hardcode the version here — a literal drifts from pyproject on every
    # bump (it shipped 0.1.1 reporting "0.1.0" exactly because it was hardcoded).
    __version__ = _pkg_version("frugon")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+dev"

# Sent on every outbound registry / leaderboard fetch (pricing + quality
# refresh). Some hosts — notably the Hugging Face datasets-server backing the
# LMArena quality table — reject the default ``Python-urllib`` agent with HTTP
# 500, so an explicit, identifying User-Agent is required for the refreshes to
# work at all. Kept here as the single source so both fetchers stay in lockstep.
USER_AGENT = f"frugon/{__version__} (+https://github.com/Rodiun/frugon)"
