"""Tests for the frugon measure engine (--measure / --judge).

Tests run with fully mocked LiteLLM — no real network calls.
All assertions in the privacy invariant tests verify that no data
reaches any Rodiun/Frugon host; calls go only to provider models
controlled by the user's own environment keys.

Coverage target: frugon.measure > 85%.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from frugon.cost import LogRecord
from frugon.measure import (
    Comparison,
    MissingProviderKeyError,
    SampledOutput,
    Tier1Tally,
    _call_model,
    _check_provider_keys,
    _dedup_key,
    _friendly_cell,
    _judge_pair,
    _required_key_for_model,
    run_measure,
    sample_records,
)

# ---------------------------------------------------------------------------
# Module-level key fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _provider_keys_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set provider API keys for every test in this module.

    Tests that specifically verify missing-key behaviour override this via
    their own monkeypatch.delenv call, which takes precedence.  All other
    tests use mocked litellm so no real network calls are made regardless.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RECORD_COUNTER = 0


def _make_record(
    model: str = "gpt-4o",
    content: str = "Hello!",
    prompt_text: str | None = None,
) -> LogRecord:
    """Create a minimal LogRecord with a UNIQUE prompt by default.

    sample_records dedups by prompt content; identical fixture prompts would
    collapse the test universe to a single record.  When ``prompt_text`` is
    None (the historical default callers), an auto-incrementing counter
    builds a distinct prompt string.  Tests that intentionally want
    duplicates pass ``prompt_text=`` with a fixed value.
    """
    global _RECORD_COUNTER
    if prompt_text is None:
        _RECORD_COUNTER += 1
        prompt_text = f"Say hello {_RECORD_COUNTER}"
    return LogRecord(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        completion_text=content,
        prompt_tokens=10,
        completion_tokens=5,
        timestamp=None,
    )


def _make_litellm_mock(content: str = "mocked output") -> MagicMock:
    """Return a mock litellm module whose .completion() returns a realistic response."""
    mock = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = content
    mock.completion.return_value = resp
    return mock


# ---------------------------------------------------------------------------
# sample_records
# ---------------------------------------------------------------------------


def test_sample_records_count_honored() -> None:
    """Arrange: 20 records each with a UNIQUE prompt, n=5.
    Act: sample_records.
    Assert: exactly 5 returned, 20 unique prompts surfaced.
    """
    records = [_make_record(prompt_text=f"msg {i}") for i in range(20)]
    picked, unique_available = sample_records(records, 5, seed=42)
    assert len(picked) == 5
    assert unique_available == 20


def test_sample_records_fewer_records_than_n_returns_all() -> None:
    """Arrange: 3 unique-prompt records, n=10.
    Act: sample_records.
    Assert: all 3 returned, unique_available=3.
    """
    records = [_make_record(prompt_text=f"msg {i}") for i in range(3)]
    picked, unique_available = sample_records(records, 10)
    assert len(picked) == 3
    assert unique_available == 3


def test_sample_records_exact_n_returns_all() -> None:
    """Arrange: exactly n unique-prompt records.
    Act: sample_records.
    Assert: all returned (no sampling needed).
    """
    records = [_make_record(prompt_text=f"msg {i}") for i in range(5)]
    picked, unique_available = sample_records(records, 5)
    assert len(picked) == 5
    assert unique_available == 5


def test_sample_records_empty_returns_empty() -> None:
    """Arrange: empty list.
    Act: sample_records.
    Assert: empty list returned + zero unique prompts.
    """
    picked, unique_available = sample_records([], 5)
    assert picked == []
    assert unique_available == 0


def test_sample_records_seed_is_deterministic() -> None:
    """Arrange: same seed, same records.
    Act: sample twice.
    Assert: same sample both times.
    """
    records = [_make_record(prompt_text=f"msg {i}") for i in range(20)]
    first, _ = sample_records(records, 5, seed=99)
    second, _ = sample_records(records, 5, seed=99)
    assert [r.messages[0]["content"] for r in first] == [
        r.messages[0]["content"] for r in second
    ]


def test_sample_records_dedups_by_unique_prompt_content() -> None:
    """Arrange: 100 records of 3 unique prompts, n=10.
    Act: sample_records.
    Assert: 3 picks returned (all uniques), unique_available=3.
    The deduplication means the candidate model is never compared on the
    same prompt twice — the wasted-call regression frugon ships to prevent.
    """
    prompts = ["alpha question", "beta question", "gamma question"]
    records = [_make_record(prompt_text=prompts[i % 3]) for i in range(100)]
    picked, unique_available = sample_records(records, 10, seed=7)
    assert unique_available == 3
    assert len(picked) == 3
    prompt_texts = {r.messages[0]["content"] for r in picked}
    assert prompt_texts == set(prompts)


def test_sample_records_dedup_picks_first_representative_deterministically() -> None:
    """Arrange: duplicate prompts where the FIRST occurrence carries a
    distinguishing completion_text, so we can verify the FIRST record of each
    group becomes the representative.
    Assert: representatives are the first occurrences, in first-seen order
    when the unique count fits in n.
    """
    records = [
        _make_record(prompt_text="alpha", content="first-alpha"),
        _make_record(prompt_text="beta", content="first-beta"),
        _make_record(prompt_text="alpha", content="second-alpha"),
        _make_record(prompt_text="beta", content="second-beta"),
    ]
    picked, unique_available = sample_records(records, 5, seed=0)
    assert unique_available == 2
    assert len(picked) == 2
    completions = [r.completion_text for r in picked]
    assert completions == ["first-alpha", "first-beta"]


def test_sample_records_dedup_reproducible_across_runs() -> None:
    """Arrange: 100 records of 100 uniques, n=10, fixed seed.
    Assert: two calls with the same (records, n, seed) yield byte-identical
    picks — the deterministic-representative-list-with-seeded-sample contract.
    """
    records = [_make_record(prompt_text=f"unique-{i}") for i in range(100)]
    first, first_avail = sample_records(records, 10, seed=123)
    second, second_avail = sample_records(records, 10, seed=123)
    assert first_avail == second_avail == 100
    assert len(first) == len(second) == 10
    assert [r.messages[0]["content"] for r in first] == [
        r.messages[0]["content"] for r in second
    ]


# ---------------------------------------------------------------------------
# _dedup_key — deterministic cross-process hashing (FRG-OSS-020)
# ---------------------------------------------------------------------------


def test_dedup_key_is_deterministic_across_calls() -> None:
    """Arrange: two records with identical (system, last_user) content.
    Act: hash both independently.
    Assert: same key — the whole point of switching off builtin hash().

    Regression for FRG-OSS-020: builtin ``hash()`` on str is salted by
    PYTHONHASHSEED (randomized per-process by default), so the SAME logical
    key would come out different across two invocations of the CLI. sha256
    has no such per-process salt.
    """
    record_a = LogRecord(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        completion_text="4",
        prompt_tokens=10,
        completion_tokens=1,
        timestamp=None,
    )
    record_b = LogRecord(
        model="gpt-4o-mini",  # different model — key is content-only
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        completion_text="four",
        prompt_tokens=8,
        completion_tokens=1,
        timestamp=None,
    )
    assert _dedup_key(record_a) == _dedup_key(record_b)


def test_dedup_key_returns_stable_sha256_hex_string() -> None:
    """Arrange: a record.
    Act: compute _dedup_key.
    Assert: the return type is a 64-char lowercase hex string (sha256 digest),
    not the builtin hash()'s platform-width signed int — pinning the contract
    so a future refactor cannot silently regress to the non-deterministic path.
    """
    record = LogRecord(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello there"}],
        completion_text="Hi!",
        prompt_tokens=3,
        completion_tokens=2,
        timestamp=None,
    )
    key = _dedup_key(record)
    assert isinstance(key, str)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_dedup_key_distinct_for_distinct_content() -> None:
    """Arrange: two records with different (system, last_user) pairs.
    Act: hash both.
    Assert: distinct keys — the dedup must not collapse genuinely different
    prompts.
    """
    record_a = LogRecord(
        model="gpt-4o",
        messages=[{"role": "user", "content": "prompt one"}],
        completion_text="a",
        prompt_tokens=1,
        completion_tokens=1,
        timestamp=None,
    )
    record_b = LogRecord(
        model="gpt-4o",
        messages=[{"role": "user", "content": "prompt two"}],
        completion_text="b",
        prompt_tokens=1,
        completion_tokens=1,
        timestamp=None,
    )
    assert _dedup_key(record_a) != _dedup_key(record_b)


def test_dedup_key_distinguishes_system_from_user_content() -> None:
    """Arrange: two records whose system/last_user content values are swapped
    across the delimiter boundary (e.g. "a" + "\\0" + "b" vs "a\\0b" formed
    differently would collide under naive '+' concatenation without a
    separator).
    Act: hash both.
    Assert: distinct keys — proves the NUL-delimited join does not let content
    that crosses the system/user boundary collide.
    """
    record_a = LogRecord(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "ab"},
            {"role": "user", "content": "c"},
        ],
        completion_text="x",
        prompt_tokens=1,
        completion_tokens=1,
        timestamp=None,
    )
    record_b = LogRecord(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "a"},
            {"role": "user", "content": "bc"},
        ],
        completion_text="y",
        prompt_tokens=1,
        completion_tokens=1,
        timestamp=None,
    )
    assert _dedup_key(record_a) != _dedup_key(record_b)


# ---------------------------------------------------------------------------
# _call_model (unit, with mocked litellm module)
# ---------------------------------------------------------------------------


def test_call_model_returns_sampled_output() -> None:
    """Arrange: mock litellm returning 'hello world'.
    Act: _call_model.
    Assert: SampledOutput with correct model and content.
    """
    mock_litellm = _make_litellm_mock("hello world")
    messages = [{"role": "user", "content": "Say hello"}]
    out = _call_model(mock_litellm, "gpt-4o-mini", messages)
    assert isinstance(out, SampledOutput)
    assert out.model == "gpt-4o-mini"
    assert out.content == "hello world"
    assert out.error is None


def test_call_model_unknown_error_returns_friendly_cell() -> None:
    """Arrange: litellm raises a generic RuntimeError whose raw message carries
    a URL and an API-key fragment.
    Act: _call_model.
    Assert: SampledOutput with a friendly error cell — the type name is shown but
    NO part of the raw provider message (the URL or key fragment) leaks into the
    user-facing cell.

    Provider exception strings routinely echo back the request URL and can carry
    key fragments. The friendly cell must surface only the exception class name,
    never the raw text — this is the privacy/trust invariant at the error path.
    """
    raw_message = "timeout calling https://api.openai.com/v1 with key sk-secret-abc123"
    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = RuntimeError(raw_message)
    messages = [{"role": "user", "content": "ping"}]
    out = _call_model(mock_litellm, "gpt-4o-mini", messages)
    assert out.error is not None
    # Friendly cell — type name present, raw message absent
    assert "RuntimeError" in out.error
    assert "timeout" not in out.error
    # The raw provider message — including any URL or key fragment — must NOT
    # appear anywhere in the user-facing cell.
    assert raw_message not in out.error
    assert "api.openai.com" not in out.error
    assert "sk-secret-abc123" not in out.error
    assert out.content == ""


def test_call_model_authentication_error_returns_friendly_cell() -> None:
    """Arrange: litellm raises an AuthenticationError-named exception.
    Act: _call_model (no is_baseline).
    Assert: SampledOutput with auth-failed cell — does NOT raise.
    """

    class FakeAuthError(Exception):
        pass

    FakeAuthError.__name__ = "AuthenticationError"

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = FakeAuthError("No key")
    messages = [{"role": "user", "content": "ping"}]
    out = _call_model(mock_litellm, "gpt-4o-mini", messages)
    assert out.error is not None
    assert "auth failed" in out.error
    assert out.content == ""


def test_call_model_baseline_failure_is_explicit() -> None:
    """Arrange: litellm raises on the baseline model call.
    Act: _call_model with is_baseline=True.
    Assert: error cell contains 'baseline unavailable', not blank.
    """

    class FakeNotFoundError(Exception):
        pass

    FakeNotFoundError.__name__ = "NotFoundError"

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = FakeNotFoundError("no access")
    messages = [{"role": "user", "content": "ping"}]
    out = _call_model(mock_litellm, "gpt-4o", messages, is_baseline=True)
    assert out.error is not None
    assert "baseline unavailable" in out.error
    assert out.content == ""


def test_call_model_baseline_not_found_no_double_word() -> None:
    """Arrange: litellm raises NotFoundError for the baseline model.
    Act: _call_model with is_baseline=True.
    Assert: the error cell reads '[baseline unavailable — ...]' without
            a duplicate 'unavailable' word (regression guard for P3-1).
    """

    class FakeNotFoundError(Exception):
        pass

    FakeNotFoundError.__name__ = "NotFoundError"

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = FakeNotFoundError("no access")
    messages = [{"role": "user", "content": "ping"}]
    out = _call_model(mock_litellm, "gpt-4o", messages, is_baseline=True)
    assert out.error is not None
    # Must NOT contain double "unavailable — unavailable".
    assert "unavailable — unavailable" not in out.error, (
        f"Double 'unavailable' in baseline error cell: {out.error!r}"
    )
    # Must read cleanly as "[baseline unavailable — project lacks access to this model]".
    assert out.error == "[baseline unavailable — project lacks access to this model]", (
        f"Unexpected baseline error cell wording: {out.error!r}"
    )


# ---------------------------------------------------------------------------
# _judge_pair (unit)
# ---------------------------------------------------------------------------


def test_judge_pair_candidate_as_b_verdict_b_is_win() -> None:
    """Arrange: candidate shown as OUTPUT B (default), judge prefers B.
    Act: _judge_pair with candidate_is_a=False.
    Assert: 'win' — the candidate (B) was judged better.
    """
    mock_litellm = _make_litellm_mock("Some analysis.\nVERDICT: B")
    current = SampledOutput(model="gpt-4o", content="current")
    candidate = SampledOutput(model="gpt-4o-mini", content="candidate")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        candidate_is_a=False,
    )
    assert result == "win"


def test_judge_pair_candidate_as_b_verdict_a_is_loss() -> None:
    """Arrange: candidate shown as OUTPUT B (default), judge prefers A.
    Act: _judge_pair with candidate_is_a=False.
    Assert: 'loss' — A is the current model, so the candidate lost.
    """
    mock_litellm = _make_litellm_mock("VERDICT: A")
    current = SampledOutput(model="gpt-4o", content="current")
    candidate = SampledOutput(model="gpt-4o-mini", content="candidate")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        candidate_is_a=False,
    )
    assert result == "loss"


def test_judge_pair_candidate_as_a_verdict_a_is_win() -> None:
    """Arrange: candidate shown as OUTPUT A (swapped), judge prefers A.
    Act: _judge_pair with candidate_is_a=True.
    Assert: 'win' — A is the candidate when swapped, so the candidate won.

    This is the inversion the A/B randomisation depends on: the SAME judge
    preference ("A is better") maps to a candidate WIN here but to a candidate
    LOSS in test_judge_pair_candidate_as_b_verdict_a_is_loss above.
    """
    mock_litellm = _make_litellm_mock("VERDICT: A")
    current = SampledOutput(model="gpt-4o", content="current")
    candidate = SampledOutput(model="gpt-4o-mini", content="candidate")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        candidate_is_a=True,
    )
    assert result == "win"


def test_judge_pair_candidate_as_a_verdict_b_is_loss() -> None:
    """Arrange: candidate shown as OUTPUT A (swapped), judge prefers B.
    Act: _judge_pair with candidate_is_a=True.
    Assert: 'loss' — B is the current model when swapped, so the candidate lost.
    """
    mock_litellm = _make_litellm_mock("VERDICT: B")
    current = SampledOutput(model="gpt-4o", content="current")
    candidate = SampledOutput(model="gpt-4o-mini", content="candidate")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        candidate_is_a=True,
    )
    assert result == "loss"


def test_judge_pair_tie_is_order_invariant() -> None:
    """Arrange: judge returns TIE.
    Assert: 'tie' regardless of which slot the candidate occupied — a tie has no
    direction to invert.
    """
    current = SampledOutput(model="gpt-4o", content="current")
    candidate = SampledOutput(model="gpt-4o-mini", content="candidate")
    for candidate_is_a in (False, True):
        mock_litellm = _make_litellm_mock("VERDICT: TIE")
        result = _judge_pair(
            mock_litellm,
            "gpt-4o-mini",
            [{"role": "user", "content": "test"}],
            current,
            candidate,
            candidate_is_a=candidate_is_a,
        )
        assert result == "tie"


def test_judge_pair_prompt_hides_which_output_is_candidate() -> None:
    """Assert: the judge prompt labels outputs ONLY as A/B — it never reveals
    which is the current model and which is the candidate, so the judge cannot be
    primed by the label (the other half of the debiasing, alongside A/B order).
    """
    mock_litellm = _make_litellm_mock("VERDICT: B")
    current = SampledOutput(model="gpt-4o", content="CURRENT_TEXT")
    candidate = SampledOutput(model="gpt-4o-mini", content="CANDIDATE_TEXT")
    _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
    )
    sent = mock_litellm.completion.call_args.kwargs["messages"][0]["content"]
    assert "OUTPUT A" in sent
    assert "OUTPUT B" in sent
    # The judge is never told which model is "current" or "candidate".
    assert "current model" not in sent
    assert "candidate model" not in sent


def test_judge_pair_returns_error_on_unparseable_response() -> None:
    """Arrange: judge model returns gibberish.
    Act: _judge_pair.
    Assert: 'error' returned (no crash).
    """
    mock_litellm = _make_litellm_mock("I cannot decide.")
    current = SampledOutput(model="gpt-4o", content="A")
    candidate = SampledOutput(model="gpt-4o-mini", content="B")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
    )
    assert result == "error"


def test_judge_pair_returns_error_on_exception() -> None:
    """Arrange: judge model raises an exception.
    Act: _judge_pair.
    Assert: 'error' returned (no crash).
    """
    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = RuntimeError("network error")
    current = SampledOutput(model="gpt-4o", content="A")
    candidate = SampledOutput(model="gpt-4o-mini", content="B")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
    )
    assert result == "error"


# ---------------------------------------------------------------------------
# _judge_pair — robust verdict parsing (the 9/50-error fix)
# ---------------------------------------------------------------------------
#
# The old strict "token in verdict_map" match required the reply to be EXACTLY
# "VERDICT: A".  A judge that echoed the old prompt's parenthetical
# ("VERDICT: A (output a is better)"), used markdown emphasis ("**VERDICT: B**"),
# or replied lowercase ("verdict: tie") parsed to no match and collapsed to
# 'error'.  These tests prove the new _parse_verdict tolerates all of those AND
# still maps the A/B token back through the correct candidate-relative table in
# BOTH presentation orderings.


@pytest.mark.parametrize(
    ("reply", "candidate_is_a", "expected"),
    [
        # Trailing parenthetical echoed from the old prompt — the literal
        # regression that produced the 9/50 errors.
        ("VERDICT: A (output a is better)", False, "loss"),  # A=current → loss
        ("VERDICT: A (output a is better)", True, "win"),  # A=candidate → win
        # Lowercase reply.
        ("verdict: tie", False, "tie"),
        ("verdict: tie", True, "tie"),
        # Markdown emphasis around the verdict line.
        ("**VERDICT: B**", False, "win"),  # B=candidate → win
        ("**VERDICT: B**", True, "loss"),  # B=current → loss
        # Leading prose before the verdict on the same/another line.
        ("Here is my call.\nVERDICT: B  (clearly)", False, "win"),
        ("Here is my call.\nVERDICT: B  (clearly)", True, "loss"),
    ],
)
def test_judge_pair_parses_decorated_verdict_in_both_orderings(
    reply: str, candidate_is_a: bool, expected: str
) -> None:
    """Arrange: a judge reply wrapped in punctuation/emphasis/casing.
    Act: _judge_pair in the given A/B ordering.
    Assert: the verdict resolves correctly AND the candidate-relative mapping is
    still applied for whichever slot the candidate occupied.
    """
    mock_litellm = _make_litellm_mock(reply)
    current = SampledOutput(model="gpt-4o", content="current")
    candidate = SampledOutput(model="gpt-4o-mini", content="candidate")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        candidate_is_a=candidate_is_a,
    )
    assert result == expected


@pytest.mark.parametrize("candidate_is_a", [False, True])
def test_judge_pair_no_verdict_keyword_is_error(candidate_is_a: bool) -> None:
    """Arrange: a reply that never contains a VERDICT: A/B/TIE line.
    Act: _judge_pair in both orderings.
    Assert: 'error' — only a genuinely verdict-free reply falls to error, and it
    does so identically regardless of A/B order.
    """
    mock_litellm = _make_litellm_mock("I really cannot decide between these.")
    current = SampledOutput(model="gpt-4o", content="A")
    candidate = SampledOutput(model="gpt-4o-mini", content="B")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        candidate_is_a=candidate_is_a,
    )
    assert result == "error"


def test_judge_pair_tie_not_shortened_to_letter() -> None:
    """Assert: 'VERDICT: TIE' resolves to 'tie', not to a stray 'T'/letter match.

    Guards the alternation order in _VERDICT_RE (TIE listed before A|B) so the
    full word is matched rather than an accidental single letter.
    """
    mock_litellm = _make_litellm_mock("VERDICT: TIE")
    current = SampledOutput(model="gpt-4o", content="A")
    candidate = SampledOutput(model="gpt-4o-mini", content="B")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
    )
    assert result == "tie"


# ---------------------------------------------------------------------------
# JUDGE_PROMPT_TEMPLATE — rubric wording (approved verbatim)
# ---------------------------------------------------------------------------


def test_judge_prompt_template_carries_tie_default_and_no_parentheses() -> None:
    """Assert: the rendered rubric defaults to TIE, forbids parentheses, and
    treats length/style differences as a tie — the approved recalibration
    that makes the judge less harsh on genuine ties.
    """
    from frugon.measure import JUDGE_PROMPT_TEMPLATE

    rendered = JUDGE_PROMPT_TEMPLATE.format(
        prompt="P", output_a="A-text", output_b="B-text"
    )
    assert "Default to TIE." in rendered
    assert "No parentheses." in rendered
    assert (
        "Differences in length, wording, style,\nformatting, or amount of "
        "detail are NOT quality differences" in rendered
    )
    # The placeholders still substitute cleanly.
    assert "A-text" in rendered
    assert "B-text" in rendered


# ---------------------------------------------------------------------------
# _judge_pair — one-retry resilience on a transient failure
# ---------------------------------------------------------------------------


def test_judge_pair_retries_once_then_succeeds() -> None:
    """Arrange: completion raises once (transient rate-limit), then returns a
    clean verdict.
    Act: _judge_pair with the default one retry, backoff suppressed for speed.
    Assert: the REAL verdict is returned — the transient fault did not collapse
    the pair to 'error'.
    """
    resp = MagicMock()
    resp.choices[0].message.content = "VERDICT: TIE"
    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = [RuntimeError("rate limited"), resp]
    current = SampledOutput(model="gpt-4o", content="A")
    candidate = SampledOutput(model="gpt-4o-mini", content="B")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        backoff_s=0.0,
    )
    assert result == "tie"
    assert mock_litellm.completion.call_count == 2


def test_judge_pair_two_failures_collapse_to_error() -> None:
    """Arrange: completion raises on BOTH the first call and its one retry.
    Act: _judge_pair (default max_retries=1), backoff suppressed.
    Assert: 'error' — a genuinely persistent failure stays neutral (never a
    loss) after the retry budget is exhausted.
    """
    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = [
        RuntimeError("rate limited"),
        RuntimeError("still down"),
    ]
    current = SampledOutput(model="gpt-4o", content="A")
    candidate = SampledOutput(model="gpt-4o-mini", content="B")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        backoff_s=0.0,
    )
    assert result == "error"
    assert mock_litellm.completion.call_count == 2


def test_judge_pair_unparseable_reply_is_not_retried() -> None:
    """Assert: a successfully-returned but verdict-free reply is a deterministic
    parse 'error' and is NOT retried — only RAISED exceptions consume the retry
    budget.  The completion is called exactly once.
    """
    mock_litellm = _make_litellm_mock("no verdict here")
    current = SampledOutput(model="gpt-4o", content="A")
    candidate = SampledOutput(model="gpt-4o-mini", content="B")
    result = _judge_pair(
        mock_litellm,
        "gpt-4o-mini",
        [{"role": "user", "content": "test"}],
        current,
        candidate,
        backoff_s=0.0,
    )
    assert result == "error"
    assert mock_litellm.completion.call_count == 1


# ---------------------------------------------------------------------------
# run_measure — Tier-0 (no --judge)
# ---------------------------------------------------------------------------


def test_run_measure_sampling_count_honored() -> None:
    """Arrange: 20 records, n_samples=5, mock litellm.
    Act: run_measure.
    Assert: samples_taken == 5, exactly 5 comparisons.
    """
    records = [_make_record() for _ in range(20)]
    mock_litellm = _make_litellm_mock("mock output")

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=5, seed=0)

    assert result.samples_requested == 5
    assert result.samples_taken == 5
    assert len(result.comparisons) == 5


def test_run_measure_fewer_records_than_samples() -> None:
    """Arrange: 3 records, n_samples=10.
    Act: run_measure.
    Assert: samples_taken == 3 (all records).
    """
    records = [_make_record() for _ in range(3)]
    mock_litellm = _make_litellm_mock("mock output")

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=10)

    assert result.samples_taken == 3
    assert result.samples_requested == 10


def test_run_measure_tier0_comparison_structure() -> None:
    """Arrange: 3 records, 2 candidates, mock litellm.
    Act: run_measure (no --judge).
    Assert: each comparison has current_output and one candidate_output per candidate.
    """
    records = [_make_record() for _ in range(3)]
    mock_litellm = _make_litellm_mock("answer")

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini", "claude-3-haiku-20240307"],
            n_samples=3,
        )

    assert result.tier1_tallies is None
    for comp in result.comparisons:
        assert isinstance(comp, Comparison)
        assert comp.current_output.model == "gpt-4o"
        assert len(comp.candidate_outputs) == 2
        assert comp.candidate_outputs[0].model == "gpt-4o-mini"
        assert comp.candidate_outputs[1].model == "claude-3-haiku-20240307"


def test_run_measure_current_model_and_candidates_stored() -> None:
    """Arrange/Act: run_measure with known current_model + candidates.
    Assert: result stores them correctly.
    """
    records = [_make_record()]
    mock_litellm = _make_litellm_mock("ok")

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=1)

    assert result.current_model == "gpt-4o"
    assert result.candidates == ["gpt-4o-mini"]


# ---------------------------------------------------------------------------
# run_measure — Tier-1 (--judge)
# ---------------------------------------------------------------------------


def test_run_measure_tier1_tally_math() -> None:
    """Arrange: 3 records, 1 candidate; the judge resolves win/loss/tie in order.
    Act: run_measure with use_judge=True.
    Assert: tally shows exactly 1 win, 1 loss, 1 tie, 0 errors.

    _judge_pair (which owns the A/B-order inversion) is stubbed to return fixed
    candidate-relative verdicts so this test isolates the tally arithmetic; the
    A/B mapping is verified directly in the _judge_pair unit tests above.
    """
    records = [_make_record() for _ in range(3)]
    verdicts = iter(["win", "loss", "tie"])

    mock_litellm = _make_litellm_mock("model output")

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=lambda *a, **k: next(verdicts)),
    ):
        result = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=3,
            use_judge=True,
            judge_model="gpt-4o",
            seed=0,
        )

    assert result.tier1_tallies is not None
    assert len(result.tier1_tallies) == 1
    tally = result.tier1_tallies[0]
    assert tally.candidate == "gpt-4o-mini"
    assert tally.wins == 1
    assert tally.losses == 1
    assert tally.ties == 1
    assert tally.errors == 0


def test_run_measure_retains_per_prompt_verdicts_aligned_with_candidates() -> None:
    """Arrange: 3 records, 2 candidates, judge stubbed to fixed per-pair verdicts.
    Act: run_measure with use_judge=True.
    Assert: each Comparison.verdicts holds the per-candidate verdict, aligned
    1:1 with candidate_outputs — so a caller can find WHICH prompt lost.
    """
    records = [_make_record(prompt_text=f"msg {i}") for i in range(3)]
    candidates = ["gpt-4o-mini", "gpt-3.5-turbo"]
    # One verdict per (prompt, candidate) pair, keyed on IDENTITY — NOT call order.
    # The two-stage pipeline judges prompts concurrently, so _judge_pair fires in a
    # nondeterministic order; a call-order-keyed stub would scatter these verdicts
    # to the wrong slots (the historic ~1/3 flake).  Keying on (prompt_text,
    # candidate_model) mirrors exactly what the production code aligns by, so the
    # assertion verifies the real invariant regardless of completion order.
    #   msg 0: gpt-4o-mini=win,  gpt-3.5-turbo=loss
    #   msg 1: gpt-4o-mini=tie,  gpt-3.5-turbo=win
    #   msg 2: gpt-4o-mini=loss, gpt-3.5-turbo=tie
    verdict_by_pair = {
        ("msg 0", "gpt-4o-mini"): "win",
        ("msg 0", "gpt-3.5-turbo"): "loss",
        ("msg 1", "gpt-4o-mini"): "tie",
        ("msg 1", "gpt-3.5-turbo"): "win",
        ("msg 2", "gpt-4o-mini"): "loss",
        ("msg 2", "gpt-3.5-turbo"): "tie",
    }

    def _stub_judge_pair(
        _litellm: object,
        _judge_model: str,
        messages: list[dict[str, str]],
        _current_output: object,
        candidate_output: Any,
        **_kwargs: object,
    ) -> str:
        prompt_text = messages[-1]["content"]
        return verdict_by_pair[(prompt_text, candidate_output.model)]

    mock_litellm = _make_litellm_mock("model output")

    with (
        patch("frugon.measure._import_litellm", return_value=mock_litellm),
        patch("frugon.measure._judge_pair", side_effect=_stub_judge_pair),
    ):
        result = run_measure(
            records,
            "gpt-4o",
            candidates,
            n_samples=3,
            use_judge=True,
            judge_model="gpt-4o-mini",
            seed=0,
        )

    assert len(result.comparisons) == 3
    # Each comparison carries one verdict per candidate, in candidate order.
    assert [c.verdicts for c in result.comparisons] == [
        ["win", "loss"],
        ["tie", "win"],
        ["loss", "tie"],
    ]
    # Alignment invariant: verdicts line up 1:1 with candidate_outputs, and each
    # output's model matches the candidate at the same position.
    for comp in result.comparisons:
        assert len(comp.verdicts) == len(comp.candidate_outputs)
        assert [o.model for o in comp.candidate_outputs] == candidates


def test_run_measure_verdicts_empty_when_judge_off() -> None:
    """Arrange: run_measure WITHOUT use_judge.
    Assert: every Comparison.verdicts is an empty list (backward-compatible
    default) so non-judge callers and their tests are unaffected.
    """
    records = [_make_record() for _ in range(3)]
    mock_litellm = _make_litellm_mock("model output")

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = run_measure(
            records,
            "gpt-4o",
            ["gpt-4o-mini"],
            n_samples=3,
            use_judge=False,
            seed=0,
        )

    assert all(comp.verdicts == [] for comp in result.comparisons)


def test_run_measure_tier1_tally_total_property() -> None:
    """Arrange: Tier1Tally with known values.
    Assert: total property sums all four fields.
    """
    tally = Tier1Tally(candidate="gpt-4o-mini", wins=3, losses=1, ties=1, errors=1)
    assert tally.total == 6


def test_run_measure_tier1_none_when_judge_false() -> None:
    """Arrange: run_measure without use_judge.
    Assert: tier1_tallies is None.
    """
    records = [_make_record()]
    mock_litellm = _make_litellm_mock("ok")

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = run_measure(records, "gpt-4o", ["gpt-4o-mini"], use_judge=False)

    assert result.tier1_tallies is None


# ---------------------------------------------------------------------------
# Privacy invariant — keys-never-to-us
# ---------------------------------------------------------------------------


def test_privacy_model_args_never_contain_rodiun_or_frugon() -> None:
    """Arrange: run_measure with standard provider models.
    Act: record every model arg passed to litellm.completion.
    Assert: no model arg contains 'rodiun' or 'frugon'.

    Privacy invariant §5: calls go ONLY to the user's own providers.
    """
    records = [_make_record() for _ in range(3)]
    called_models: list[str] = []

    def recording_completion(model: str, messages: list[Any], **kw: Any) -> MagicMock:
        called_models.append(model)
        resp = MagicMock()
        resp.choices[0].message.content = "output"
        return resp

    mock_litellm = MagicMock()
    mock_litellm.completion.side_effect = recording_completion

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=3, seed=0)

    assert len(called_models) > 0, "Expected at least one litellm call"
    for model in called_models:
        assert "rodiun" not in model.lower(), (
            f"Model arg '{model}' contains 'rodiun' — privacy violation"
        )
        assert "frugon" not in model.lower(), (
            f"Model arg '{model}' contains 'frugon' — privacy violation"
        )


def test_privacy_no_hardcoded_rodiun_frugon_urls() -> None:
    """Assert: the measure module source has no hardcoded Rodiun/Frugon URLs.

    Static invariant — scanning source catches accidental hardcoded endpoints.
    """
    import inspect

    import frugon.measure as measure_mod

    source = inspect.getsource(measure_mod)
    assert "rodiun.io" not in source, "Hardcoded rodiun.io URL found in measure.py"
    assert "frugon.io" not in source, "Hardcoded frugon.io URL found in measure.py"


# ---------------------------------------------------------------------------
# Privacy invariant — socket-layer guard (load-bearing, not mock-only)
# ---------------------------------------------------------------------------

_PRIVACY_ALLOWLIST = frozenset({
    "api.openai.com",
    "api.anthropic.com",
    "localhost",
    "127.0.0.1",
    "::1",
})


def _host_denied(host: str) -> bool:
    """True when *host* is not in the provider allowlist."""
    h = str(host).lower()
    if h in ("localhost", "127.0.0.1", "::1") or h.startswith("127."):
        return False
    return h not in _PRIVACY_ALLOWLIST


@pytest.fixture
def socket_privacy_guard(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace socket primitives to block any host outside the provider allowlist.

    Monkeypatches socket.socket (class), socket.create_connection, and
    socket.getaddrinfo.  Any access to a denied host appends to the yielded
    list and raises AssertionError('...privacy violation...').  Tests that
    run to completion with an empty list prove no denied hosts were accessed.
    """
    import socket as _socket

    denied: list[str] = []

    def _blocked_socket_class(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "socket.socket() instantiated — all network traffic is blocked in this test"
        )

    def _guarded_create_connection(
        address: Any, *args: object, **kwargs: object
    ) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if _host_denied(host):
            denied.append(host)
            raise AssertionError(
                f"socket.create_connection to '{host}' — privacy violation: "
                "host is outside the provider allowlist"
            )
        raise AssertionError(
            f"socket.create_connection to '{host}' — network blocked in test"
        )

    def _guarded_getaddrinfo(host: Any, *args: object, **kwargs: object) -> list[Any]:
        h = str(host)
        if _host_denied(h):
            denied.append(h)
            raise AssertionError(
                f"socket.getaddrinfo for '{h}' — privacy violation: "
                "host is outside the provider allowlist"
            )
        raise AssertionError(
            f"socket.getaddrinfo for '{h}' — network blocked in test"
        )

    monkeypatch.setattr(_socket, "socket", _blocked_socket_class)
    monkeypatch.setattr(_socket, "create_connection", _guarded_create_connection)
    monkeypatch.setattr(_socket, "getaddrinfo", _guarded_getaddrinfo)

    return denied


def test_privacy_socket_guard_no_denied_connections_run_measure(
    socket_privacy_guard: list[str],
) -> None:
    """Arrange: socket guard active + fully mocked litellm (no real network).
    Act: run_measure.
    Assert: denied list is empty — no host outside the allowlist was accessed.

    Privacy invariant §5: the measure engine must NEVER open a connection to
    any Rodiun/Frugon host.  A future regression adding a side-channel HTTP
    client (e.g. telemetry POST) would cause the socket guard to fire and
    fail this test immediately.
    """
    records = [_make_record() for _ in range(3)]
    mock_litellm = _make_litellm_mock("test output")

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=3, seed=0)

    assert socket_privacy_guard == [], (
        f"Socket guard fired for denied host(s): {socket_privacy_guard}"
    )


def test_privacy_socket_guard_fires_for_rodiun_io(
    socket_privacy_guard: list[str],
) -> None:
    """Negative-control: the guard is load-bearing — it catches denied hosts.

    Assert: calling socket.getaddrinfo for rodiun.io raises AssertionError
    with 'privacy violation', and the denied list records the attempt.

    This proves the fixture is not a no-op: any regression that opens a
    connection to rodiun.io would be caught immediately rather than silently
    passing.
    """
    import socket as _socket

    with pytest.raises(AssertionError, match="privacy violation"):
        _socket.getaddrinfo("rodiun.io", 443)

    assert "rodiun.io" in socket_privacy_guard, (
        "Expected 'rodiun.io' in denied list after the guard fired"
    )


# ---------------------------------------------------------------------------
# Missing API key — clean error, not crash
# ---------------------------------------------------------------------------


def test_missing_key_raises_before_any_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: OPENAI_API_KEY absent from environment.
    Act: run_measure with an OpenAI model.
    Assert: MissingProviderKeyError raised, litellm.completion never called.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    mock_litellm = _make_litellm_mock("should not be called")
    records = [_make_record()]

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        with pytest.raises(MissingProviderKeyError) as exc_info:
            run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=1)

    assert "OPENAI_API_KEY" in str(exc_info.value)
    mock_litellm.completion.assert_not_called()


# ---------------------------------------------------------------------------
# _required_key_for_model
# ---------------------------------------------------------------------------


def test_required_key_for_model_openai_gpt() -> None:
    """Arrange: OpenAI gpt- prefix.
    Assert: returns OPENAI_API_KEY.
    """
    assert _required_key_for_model("gpt-4o") == "OPENAI_API_KEY"
    assert _required_key_for_model("gpt-4o-mini") == "OPENAI_API_KEY"
    assert _required_key_for_model("gpt-3.5-turbo") == "OPENAI_API_KEY"


def test_required_key_for_model_anthropic_claude() -> None:
    """Arrange: Anthropic claude- prefix.
    Assert: returns ANTHROPIC_API_KEY.
    """
    assert _required_key_for_model("claude-3-5-sonnet-20240620") == "ANTHROPIC_API_KEY"
    assert _required_key_for_model("claude-3-haiku-20240307") == "ANTHROPIC_API_KEY"


def test_required_key_for_model_unknown_returns_none() -> None:
    """Arrange: unknown model prefix (e.g. local/llama3).
    Assert: returns None — no pre-flight check applied.
    """
    assert _required_key_for_model("local/llama3") is None
    assert _required_key_for_model("ollama/mistral") is None


# ---------------------------------------------------------------------------
# _check_provider_keys
# ---------------------------------------------------------------------------


def test_check_provider_keys_raises_when_key_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: OPENAI_API_KEY absent.
    Act: _check_provider_keys(['gpt-4o', 'gpt-4o-mini']).
    Assert: MissingProviderKeyError raised listing OPENAI_API_KEY; no duplicates.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingProviderKeyError) as exc_info:
        _check_provider_keys(["gpt-4o", "gpt-4o-mini"])
    msg = str(exc_info.value)
    assert "OPENAI_API_KEY" in msg
    # Should appear exactly once (dedup by seen_vars)
    assert msg.count("OPENAI_API_KEY") == 1


def test_check_provider_keys_raises_all_missing_in_one_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: both OPENAI_API_KEY and ANTHROPIC_API_KEY absent.
    Act: _check_provider_keys(['gpt-4o', 'claude-3-haiku-20240307']).
    Assert: single MissingProviderKeyError listing both keys.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingProviderKeyError) as exc_info:
        _check_provider_keys(["gpt-4o", "claude-3-haiku-20240307"])
    msg = str(exc_info.value)
    assert "OPENAI_API_KEY" in msg
    assert "ANTHROPIC_API_KEY" in msg


def test_check_provider_keys_passes_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: OPENAI_API_KEY set to a non-empty value.
    Act: _check_provider_keys(['gpt-4o']).
    Assert: no exception raised.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    _check_provider_keys(["gpt-4o"])  # must not raise


def test_check_provider_keys_skips_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: model with unknown prefix, no matching env var required.
    Act: _check_provider_keys(['local/llama3']).
    Assert: no exception raised even if no key env vars are set.
    """
    # Remove any keys that might accidentally match
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _check_provider_keys(["local/llama3"])  # must not raise


# ---------------------------------------------------------------------------
# _friendly_cell — error-class to friendly message map
# ---------------------------------------------------------------------------


def test_friendly_cell_not_found_error() -> None:
    """Arrange: exception with NotFoundError in its type name.
    Assert: cell mentions 'unavailable'.
    """

    class NotFoundError(Exception):
        pass

    assert "unavailable" in _friendly_cell(NotFoundError("x"))


def test_friendly_cell_authentication_error() -> None:
    """Arrange: exception with AuthenticationError in its type name.
    Assert: cell mentions 'auth failed'.
    """

    class AuthenticationError(Exception):
        pass

    assert "auth failed" in _friendly_cell(AuthenticationError("x"))


def test_friendly_cell_rate_limit_error() -> None:
    """Arrange: exception with RateLimitError in its type name.
    Assert: cell mentions 'rate limited'.
    """

    class RateLimitError(Exception):
        pass

    assert "rate limited" in _friendly_cell(RateLimitError("x"))


def test_friendly_cell_internal_server_error() -> None:
    """Arrange: exception with InternalServerError in its type name.
    Assert: cell mentions 'provider error'.
    """

    class InternalServerError(Exception):
        pass

    assert "provider error" in _friendly_cell(InternalServerError("x"))


def test_friendly_cell_unknown_error_includes_type_name_not_raw_message() -> None:
    """Arrange: unknown exception type with a verbose raw message.
    Assert: cell contains type name; raw message text is absent.
    """

    class SomeObscureProviderError(Exception):
        pass

    exc = SomeObscureProviderError(
        "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new"
    )
    cell = _friendly_cell(exc)
    assert "SomeObscureProviderError" in cell
    assert "Give Feedback" not in cell
    assert "github.com" not in cell


# ---------------------------------------------------------------------------
# run_measure — pre-flight makes zero calls on missing key
# ---------------------------------------------------------------------------


def test_run_measure_preflight_makes_zero_calls_on_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: OPENAI_API_KEY absent.
    Act: run_measure with OpenAI models.
    Assert: MissingProviderKeyError raised; litellm.completion never invoked.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock_litellm = _make_litellm_mock("should never be reached")
    records = [_make_record()]

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        with pytest.raises(MissingProviderKeyError):
            run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=1)

    mock_litellm.completion.assert_not_called()


def test_run_measure_success_path_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: OPENAI_API_KEY present, mock litellm returns content.
    Act: run_measure.
    Assert: comparisons populated with content, no errors.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_litellm = _make_litellm_mock("good output")
    records = [_make_record()]

    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=1)

    assert result.samples_taken == 1
    comp = result.comparisons[0]
    assert comp.current_output.content == "good output"
    assert comp.current_output.error is None
    assert comp.candidate_outputs[0].content == "good output"
    assert comp.candidate_outputs[0].error is None


def test_required_key_openrouter_prefix_requires_openrouter_api_key() -> None:
    """Arrange: openrouter/... model strings.
    Act: _required_key_for_model.
    Assert: OPENROUTER_API_KEY is required — not the underlying provider's key.

    When routing through OpenRouter the user authenticates to OpenRouter's API
    with a single OPENROUTER_API_KEY, regardless of which model sits behind it.
    """
    assert _required_key_for_model("openrouter/openai/gpt-4o") == "OPENROUTER_API_KEY"
    assert (
        _required_key_for_model("openrouter/anthropic/claude-3-5-sonnet-20241022")
        == "OPENROUTER_API_KEY"
    )
    assert (
        _required_key_for_model("openrouter/meta-llama/llama-3-8b-instruct")
        == "OPENROUTER_API_KEY"
    )


def test_required_key_anthropic_prefix_resolves_to_anthropic() -> None:
    """Arrange: anthropic/claude-3-5-sonnet-20241022 — gateway prefix.
    Act: _required_key_for_model.
    Assert: ANTHROPIC_API_KEY returned after canonicalization strips prefix.
    """
    assert _required_key_for_model("anthropic/claude-3-5-sonnet-20241022") == "ANTHROPIC_API_KEY"


def test_required_key_openrouter_anthropic_claude_resolves() -> None:
    """Arrange: openrouter/anthropic/claude-3-5-sonnet-20241022.
    Act: _required_key_for_model.
    Assert: OPENROUTER_API_KEY — OpenRouter prefix takes precedence over the
            underlying provider, so the user supplies one key for all OpenRouter
            models rather than per-provider credentials.
    """
    assert (
        _required_key_for_model("openrouter/anthropic/claude-3-5-sonnet-20241022")
        == "OPENROUTER_API_KEY"
    )


def test_required_key_bedrock_prefix_still_returns_aws_key() -> None:
    """Arrange: bedrock/anthropic.claude-3-5-sonnet-20241022-v1:0 — bedrock routing.
    Act: _required_key_for_model.
    Assert: AWS_ACCESS_KEY_ID (original map entry wins before canonicalization).

    Backward-compatibility: users routing through Bedrock need AWS credentials,
    not Anthropic credentials — the original-first check preserves this.
    """
    assert (
        _required_key_for_model("bedrock/anthropic.claude-3-5-sonnet-20241022-v1:0")
        == "AWS_ACCESS_KEY_ID"
    )


def test_missing_litellm_import_raises_helpful_error() -> None:
    """Arrange: _import_litellm raises ImportError (litellm not installed).
    Act: run_measure.
    Assert: ImportError propagates with a helpful install hint.
    """
    records = [_make_record()]

    def raise_import() -> Any:
        raise ImportError("Install frugon[measure]")

    with patch("frugon.measure._import_litellm", side_effect=raise_import):
        with pytest.raises(ImportError) as exc_info:
            run_measure(records, "gpt-4o", ["gpt-4o-mini"], n_samples=1)

    assert "measure" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# LiteLLM / botocore import-time noise suppression
# ---------------------------------------------------------------------------


def test_litellm_loggers_are_at_error_level_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_import_litellm() sets LiteLLM, litellm, and botocore loggers to ERROR level.

    Arrange: reset loggers to NOTSET so the change is detectable; inject a mock
             litellm into sys.modules so no real network import occurs.
    Act: call _import_litellm().
    Assert: all three loggers are at logging.ERROR — import-time WARNING chatter
            from LiteLLM's Bedrock/botocore integration is suppressed.

    The levels must be set BEFORE the import statement so that module-level log
    calls triggered at import time are already silenced.
    """
    import logging
    import sys

    from frugon.measure import _import_litellm

    # Reset so we can detect the change.
    logging.getLogger("LiteLLM").setLevel(logging.NOTSET)
    logging.getLogger("litellm").setLevel(logging.NOTSET)
    logging.getLogger("botocore").setLevel(logging.NOTSET)

    mock_litellm = MagicMock()
    monkeypatch.setitem(sys.modules, "litellm", mock_litellm)

    _import_litellm()

    assert logging.getLogger("LiteLLM").level == logging.ERROR, (
        f"LiteLLM logger at {logging.getLevelName(logging.getLogger('LiteLLM').level)}, "
        "expected ERROR"
    )
    assert logging.getLogger("litellm").level == logging.ERROR, (
        f"litellm logger at {logging.getLevelName(logging.getLogger('litellm').level)}, "
        "expected ERROR"
    )
    assert logging.getLogger("botocore").level == logging.ERROR, (
        f"botocore logger at {logging.getLevelName(logging.getLogger('botocore').level)}, "
        "expected ERROR — botocore WARNING noise must be suppressed on --measure"
    )


# ---------------------------------------------------------------------------
# CLI integration — --measure / --samples / --judge flags visible in help
# ---------------------------------------------------------------------------


def test_analyze_help_shows_measure_samples_judge_flags() -> None:
    """Assert: 'analyze --help' output contains --measure, --samples, and --judge."""
    from .conftest import help_text

    # Render-independent canonical help (ANSI stripped, box borders flattened,
    # whitespace collapsed) so phrases that wrap at CI's forced 80-col width — such
    # as "A/B order" — are still discoverable as contiguous substrings.
    out = help_text("analyze")

    assert "measure" in out, f"'--measure' not in help:\n{out}"
    assert "samples" in out, f"'--samples' not in help:\n{out}"
    assert "judge" in out, f"'--judge' not in help:\n{out}"
    # The --judge-model flag, its gpt-4o default, and the A/B-order note must all
    # be discoverable from the help text (the methodology is surfaced honestly).
    assert "judge-model" in out, f"'--judge-model' not in help:\n{out}"
    assert "gpt-4o" in out, f"judge default 'gpt-4o' not in help:\n{out}"
    assert "A/B order" in out, f"A/B-order randomisation not in help:\n{out}"
    # --samples default is now 10, and the mental-model guidance is discoverable.
    assert "quick glance" in out, f"sample mental-model not in help:\n{out}"
    assert "confident before switching" in out, f"sample mental-model not in help:\n{out}"
    assert "default: 10" in out, f"--samples default 10 not in help:\n{out}"


def test_analyze_judge_model_override_is_key_prechecked(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: --judge --judge-model gemini/gemini-1.5-pro, GEMINI_API_KEY absent.
    Act: invoke analyze --measure --judge --judge-model <gemini model>.
    Assert: the run fails fast for the OVERRIDDEN judge's missing key (the
    pre-check verifies the chosen judge, not just the default), names the missing
    GEMINI_API_KEY, exits non-zero, and never reaches a provider call.
    """
    import json

    from typer.testing import CliRunner

    from frugon.cli import app

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # baseline + candidate key OK
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # judge key MISSING

    log_file = tmp_path / "logs.jsonl"
    log_file.write_text(
        json.dumps(
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "hi"}]},
                "response": {"choices": [{"message": {"content": "hello"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mock_litellm = _make_litellm_mock("should not be reached")
    runner = CliRunner()
    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = runner.invoke(
            app,
            [
                "analyze",
                str(log_file),
                "--measure",
                "--judge",
                "--judge-model",
                "gemini/gemini-1.5-pro",
                "--candidates",
                "gpt-4o-mini",
            ],
        )

    assert result.exit_code != 0, (
        f"Expected non-zero exit for missing judge key:\n{result.output}"
    )
    assert "GEMINI_API_KEY" in result.output, (
        f"Overridden judge's missing key not named:\n{result.output}"
    )
    assert "Traceback" not in result.output
    mock_litellm.completion.assert_not_called()


def test_analyze_measure_with_missing_key_exits_nonzero(
    tmp_path: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: a valid log file + --measure + OPENAI_API_KEY absent.
    Act: invoke analyze --measure.
    Assert: exits non-zero, no raw Python traceback shown.
    """
    import json

    from typer.testing import CliRunner

    from frugon.cli import app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    log_file = tmp_path / "logs.jsonl"  # type: ignore[operator]
    log_file.write_text(
        json.dumps(
            {
                "model": "gpt-4o",
                "request": {"messages": [{"role": "user", "content": "hi"}]},
                "response": {"choices": [{"message": {"content": "hello"}}]},
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mock_litellm = _make_litellm_mock("should not be reached")

    runner = CliRunner()
    with patch("frugon.measure._import_litellm", return_value=mock_litellm):
        result = runner.invoke(
            app,
            # Use --candidates so there is always a candidate to measure against,
            # regardless of which model is in the log file.
            ["analyze", str(log_file), "--measure", "--candidates", "gpt-4o-mini"],
        )

    assert "Traceback" not in result.output, (
        f"Raw traceback shown — should be a clean error:\n{result.output}"
    )
    assert result.exit_code != 0, (
        f"Expected non-zero exit for missing key, got {result.exit_code}:\n{result.output}"
    )
    mock_litellm.completion.assert_not_called()


def test_analyze_judge_without_measure_exits_nonzero() -> None:
    """Arrange: bare --judge (no --measure).
    Act: invoke analyze --demo --judge.
    Assert: exits non-zero with a clear prerequisite message; no silent no-op.
    """
    from typer.testing import CliRunner

    from frugon.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "--demo", "--judge", "--no-progress"])

    assert result.exit_code != 0, (
        f"Expected non-zero exit for --judge without --measure:\n{result.output}"
    )
    assert "--judge requires --measure" in result.output, (
        f"Prerequisite message missing:\n{result.output}"
    )
    assert "frugon analyze --measure --judge" in result.output, (
        f"Suggested invocation missing:\n{result.output}"
    )
    assert "Traceback" not in result.output
