"""README structural integrity tests.

Prevents assembly defects (stray fences, artifact strings, missing required
content) from reaching CI or the public repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_contains_privacy_line(readme_text: str) -> None:
    assert (
        "Your data never leaves your machine. Your keys go straight to your own providers. Nothing reaches us."
        in readme_text
    )


def test_readme_contains_install_command(readme_text: str) -> None:
    assert "uvx frugon analyze" in readme_text


def test_readme_contains_routellm_link(readme_text: str) -> None:
    assert "https://github.com/lm-sys/RouteLLM" in readme_text


def test_readme_contains_funnel_link(readme_text: str) -> None:
    assert "https://frugon.rodiun.io" in readme_text


@pytest.mark.parametrize(
    "artifact",
    [
        "wait — let me",
        "let me show the real",
        "🚧 pre-release",
        "back of envelope",
    ],
)
def test_readme_contains_no_known_artifacts(readme_text: str, artifact: str) -> None:
    assert artifact not in readme_text.lower(), f"Artifact found in README: {artifact!r}"


def test_readme_code_fences_are_balanced(readme_text: str) -> None:
    fence_count = readme_text.count("```")
    assert fence_count % 2 == 0, (
        f"Unbalanced code fences: {fence_count} occurrences of ``` (must be even)"
    )


def test_readme_footer_appears_exactly_once(readme_text: str) -> None:
    footer = "Built by [Rodiun](https://rodiun.io). MIT licensed."
    count = readme_text.count(footer)
    assert count == 1, f"Footer appears {count} times (expected exactly 1)"
