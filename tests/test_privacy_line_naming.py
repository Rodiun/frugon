"""Regression test for FRG-OSS-005(b) — the PRIVACY_LINE naming collision.

``frugon.cli`` and ``frugon.report`` each define a privacy-caveat string used
in a different surface (the wholesale CLI panels/help text vs. the split
report's terminal footer).  The two strings are DELIBERATELY different
lengths/wording for their respective surfaces — this is not a duplication bug
to collapse into one value.  The actual defect was the SHARED NAME
``PRIVACY_LINE`` in both modules: importing "the" PRIVACY_LINE without
qualifying the module was one typo away from pulling the wrong string into the
wrong surface.  ``frugon.report`` now names its constant
``SPLIT_FOOTER_PRIVACY_LINE`` so the two are never confusable by name alone.
"""

from __future__ import annotations

from frugon.cli import PRIVACY_LINE as CLI_PRIVACY_LINE
from frugon.report import SPLIT_FOOTER_PRIVACY_LINE


def test_report_module_does_not_export_a_bare_privacy_line_name() -> None:
    """Arrange: frugon.report's public namespace.
    Act: check for a bare `PRIVACY_LINE` attribute.
    Assert: absent — the module-level rename removed the name collision with
    frugon.cli.PRIVACY_LINE at its source.
    """
    import frugon.report as report_mod

    assert not hasattr(report_mod, "PRIVACY_LINE"), (
        "frugon.report must not re-introduce a bare PRIVACY_LINE name that "
        "collides with frugon.cli.PRIVACY_LINE"
    )


def test_cli_and_report_privacy_lines_are_distinct_strings_for_distinct_surfaces() -> None:
    """Arrange: both privacy-caveat constants.
    Act: compare them.
    Assert: they are NOT equal — each surface (wholesale CLI panel/help vs.
    split-report terminal footer) intentionally carries its own length/wording,
    so collapsing them into one shared value would be the wrong fix.
    """
    assert CLI_PRIVACY_LINE != SPLIT_FOOTER_PRIVACY_LINE
    # Both still assert the same substantive privacy guarantee even though the
    # wording differs.
    assert "never leaves your machine" in CLI_PRIVACY_LINE
    assert "never leaves your machine" in SPLIT_FOOTER_PRIVACY_LINE
