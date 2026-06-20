"""Tests for the sampling-error short-circuit in --judge (Tier-1).

When a candidate (or the baseline) returns a SampledOutput with .error set --
the canonical bug-mode is a project that lacks API access to the model, so
every call comes back ``[unavailable -- project lacks access to this model]`` --
the judge MUST NOT see that string as the candidate's output.  Sending an
error message to the judge would have it correctly pick the real answer over
the error and count the comparison as a candidate LOSS -- a quality verdict on
a sampling failure, which is dishonest.

These tests pin the contract: an errored sample short-circuits to the neutral
'error' verdict, the judge call is skipped (no API cost), the tally records
it in the errors column, the per-prompt label reads ``[error]`` (never
``[LOSS]``), and the verdict synthesis tells the user honestly that the
candidate could not be sampled -- pointing at the access check, not at a
quality outcome.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from frugon.cost import LogRecord
from frugon.measure import (
    SampledOutput,
    Tier1Tally,
    run_measure,
)
from frugon.report import (
    _VERDICT_UNMEASURED,
    _classify_verdict,
    _verdict_html_label,
    _verdict_label,
    _verdict_md_label,
)


@pytest.fixture(autouse=True)
def _provider_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")


_RECORD_COUNTER = 0


def _make_record(model: str = "gpt-4o") -> LogRecord:
    """Distinct-prompt records so sample_records does not dedup them away."""
    global _RECORD_COUNTER
    _RECORD_COUNTER += 1
    prompt = f"Say hello {_RECORD_COUNTER}"
    return LogRecord(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        completion_text="ok",
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _make_litellm_mock() -> MagicMock:
    """Minimal litellm whose .completion() returns a parseable judge reply.

    The candidate-error short-circuit tests assert the judge is NEVER reached
    for the errored candidate; any candidate that IS judged sees a TIE so the
    test's assertions stay deterministic regardless of A/B ordering.
    """
    mock = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = "VERDICT: TIE"
    mock.completion.return_value = resp
    return mock


def _errored_output(model: str) -> SampledOutput:
    """Mirror the production 'project lacks access' SampledOutput."""
    return SampledOutput(
        model=model,
        content="",
        error=f"[unavailable -- project lacks access to {model}]",
    )


def _ok_output(model: str, content: str = "real answer") -> SampledOutput:
    return SampledOutput(model=model, content=content)


def test_errored_candidate_sample_is_counted_as_error_not_loss() -> None:
    """Arrange: 3 prompts, 1 candidate ('claude-haiku') whose samples ALL
    error (mirrors the reported bug -- Anthropic project lacked access).  Stub
    _judge_pair to RAISE if called for the errored candidate.
    Act: run_measure with use_judge=True.
    Assert: the tally for claude-haiku is errors=3, wins=losses=ties=0; every
    per-prompt verdict for claude-haiku is the string 'error'; the judge was
    never asked.

    The regression: pre-fix this returned 0 wins / N losses because the judge
    correctly picked the baseline's real answer over claude-haiku's error
    string.  Post-fix it must SKIP the judge call and record 'error'.
    """
    records = [_make_record() for _ in range(3)]
    baseline = "gpt-4o"
    candidates = ["claude-3-haiku-20240307"]

    def _stub_call_model(
        _litellm: object,
        model: str,
        _messages: list[dict[str, str]],
        *,
        is_baseline: bool = False,
    ) -> SampledOutput:
        if model == baseline:
            return _ok_output(model)
        return _errored_output(model)

    def _judge_pair_must_not_be_called(*_a: object, **_k: object) -> str:
        raise AssertionError(
            "_judge_pair must not be invoked when the candidate sample errored"
        )

    with (
        patch("frugon.measure._import_litellm", return_value=_make_litellm_mock()),
        patch("frugon.measure._call_model", side_effect=_stub_call_model),
        patch(
            "frugon.measure._judge_pair", side_effect=_judge_pair_must_not_be_called
        ),
    ):
        result = run_measure(
            records,
            baseline,
            candidates,
            n_samples=3,
            use_judge=True,
            judge_model="gpt-4o",
            seed=0,
        )

    assert result.tier1_tallies is not None
    tally = result.tier1_tallies[0]
    assert tally.candidate == "claude-3-haiku-20240307"
    assert tally.wins == 0
    assert tally.losses == 0, (
        "Errored sample must NOT be counted as a candidate loss -- that was "
        "the bug.  Got: " + repr(tally)
    )
    assert tally.ties == 0
    assert tally.errors == 3, repr(tally)
    for comp in result.comparisons:
        assert comp.verdicts == ["error"], repr(comp.verdicts)


def test_errored_candidate_does_not_block_other_candidates() -> None:
    """Mixed run: gpt-4o-mini samples succeed; claude-haiku samples error.
    Assert: the succeeding candidate is judged normally; only the errored one
    short-circuits.
    """
    records = [_make_record() for _ in range(3)]
    baseline = "gpt-4o"
    candidates = ["gpt-4o-mini", "claude-3-haiku-20240307"]

    def _stub_call_model(
        _litellm: object,
        model: str,
        _messages: list[dict[str, str]],
        *,
        is_baseline: bool = False,
    ) -> SampledOutput:
        if model == "claude-3-haiku-20240307":
            return _errored_output(model)
        return _ok_output(model)

    def _stub_judge_pair(
        _litellm: object,
        _judge_model: str,
        _messages: list[dict[str, str]],
        _current: SampledOutput,
        candidate: SampledOutput,
        **_kwargs: object,
    ) -> str:
        assert candidate.error is None, (
            "Judge received an errored candidate output -- short-circuit failed"
        )
        return "win"

    with (
        patch("frugon.measure._import_litellm", return_value=_make_litellm_mock()),
        patch("frugon.measure._call_model", side_effect=_stub_call_model),
        patch("frugon.measure._judge_pair", side_effect=_stub_judge_pair),
    ):
        result = run_measure(
            records,
            baseline,
            candidates,
            n_samples=3,
            use_judge=True,
            judge_model="gpt-4o",
            seed=0,
        )

    assert result.tier1_tallies is not None
    tallies = {t.candidate: t for t in result.tier1_tallies}
    good = tallies["gpt-4o-mini"]
    assert good.wins == 3
    assert good.errors == 0
    bad = tallies["claude-3-haiku-20240307"]
    assert bad.wins == bad.losses == bad.ties == 0
    assert bad.errors == 3


def test_errored_baseline_skips_all_candidate_judges() -> None:
    """Arrange: baseline errors on the first sampled prompt; baseline OK on
    the others.  Two candidates, both sample successfully.
    Assert: the baseline-errored prompt records 'error' for both candidates
    and the judge is NEVER invoked on it; other prompts are judged normally.
    """
    records = [_make_record() for _ in range(3)]
    baseline = "gpt-4o"
    candidates = ["gpt-4o-mini", "claude-3-haiku-20240307"]

    baseline_calls: list[str] = []

    def _stub_call_model(
        _litellm: object,
        model: str,
        messages: list[dict[str, str]],
        *,
        is_baseline: bool = False,
    ) -> SampledOutput:
        if is_baseline:
            baseline_calls.append(messages[-1]["content"])
            if len(baseline_calls) == 1:
                return _errored_output(model)
            return _ok_output(model)
        return _ok_output(model)

    def _stub_judge_pair(
        _litellm: object,
        _judge_model: str,
        _messages: list[dict[str, str]],
        current: SampledOutput,
        _candidate: SampledOutput,
        **_kwargs: object,
    ) -> str:
        assert current.error is None, (
            "Judge invoked on a prompt whose baseline errored -- short-circuit failed"
        )
        return "tie"

    with (
        patch("frugon.measure._import_litellm", return_value=_make_litellm_mock()),
        patch("frugon.measure._call_model", side_effect=_stub_call_model),
        patch("frugon.measure._judge_pair", side_effect=_stub_judge_pair),
    ):
        result = run_measure(
            records,
            baseline,
            candidates,
            n_samples=3,
            use_judge=True,
            judge_model="gpt-4o",
            seed=0,
        )

    assert result.tier1_tallies is not None
    for tally in result.tier1_tallies:
        assert tally.errors == 1, repr(tally)
        assert tally.wins + tally.losses + tally.ties == 2, repr(tally)
    error_rows = [c for c in result.comparisons if c.verdicts == ["error", "error"]]
    scored_rows = [c for c in result.comparisons if "error" not in c.verdicts]
    assert len(error_rows) == 1
    assert len(scored_rows) == 2


def test_per_prompt_label_for_errored_candidate_is_error_not_loss() -> None:
    """Regression: the per-prompt verdict label on every surface
    (terminal Rich, Markdown, HTML) maps the 'error' verdict to ``[error]``
    -- never to ``[LOSS]`` or any other quality-implying token.

    Before the fix, an errored sample was scored against the baseline and the
    judge typically returned LOSS, producing a red ``[LOSS]`` cue on what was
    in fact a sampling failure.  After the fix the verdict is 'error' and
    every label maps it to a neutral ``[error]``.
    """
    rich_text = _verdict_label("error")
    assert rich_text.plain.strip() == "[error]"
    assert "LOSS" not in rich_text.plain.upper()

    assert _verdict_md_label("error") == "[error]"
    assert "LOSS" not in _verdict_md_label("error").upper()

    html = _verdict_html_label("error")
    assert "[error]" in html
    assert "verdict-error" in html
    assert "LOSS" not in html.upper()


def test_predominantly_errored_candidate_verdict_synthesis_is_honest() -> None:
    """A candidate whose sampling errored on >=50% of prompts must NOT be
    reported as a quality loss.  The verdict line says the candidate could
    not be sampled (access / model-name check), not 'NOT confirmed' (which
    implies measured-and-worse) and not 'worse'.
    """
    fully_errored = Tier1Tally(
        candidate="claude-3-haiku-20240307",
        wins=0,
        losses=0,
        ties=0,
        errors=10,
    )
    state, text = _classify_verdict(fully_errored, current_model="gpt-4o")
    assert "NOT confirmed" not in text, text
    assert " worse" not in text, text
    assert " lost" not in text, text
    assert (
        "access" in text.lower() or "could not be sampled" in text.lower()
    ), text
    assert state in {_VERDICT_UNMEASURED, "not_verified"}, state

    half_errored = Tier1Tally(
        candidate="claude-3-haiku-20240307",
        wins=2,
        losses=1,
        ties=2,
        errors=5,
    )
    state2, text2 = _classify_verdict(half_errored, current_model="gpt-4o")
    assert state2 == _VERDICT_UNMEASURED, (state2, text2)
    assert "could not be sampled" in text2.lower(), text2
    assert "NOT confirmed" not in text2, text2

    lightly_errored = Tier1Tally(
        candidate="claude-3-haiku-20240307",
        wins=5,
        losses=1,
        ties=2,
        errors=2,
    )
    state3, _ = _classify_verdict(lightly_errored, current_model="gpt-4o")
    assert state3 != _VERDICT_UNMEASURED, state3
