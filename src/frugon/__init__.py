"""frugon — free, local, open-source LLM cost analyzer."""

__version__ = "0.1.0"

# Sent on every outbound registry / leaderboard fetch (pricing + quality
# refresh). Some hosts — notably the Hugging Face datasets-server backing the
# LMArena quality table — reject the default ``Python-urllib`` agent with HTTP
# 500, so an explicit, identifying User-Agent is required for the refreshes to
# work at all. Kept here as the single source so both fetchers stay in lockstep.
USER_AGENT = f"frugon/{__version__} (+https://github.com/Rodiun/frugon)"
