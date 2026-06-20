"""Tests for frugon.model_id — canonicalize() and base_family().

Table-driven coverage of every prefix class, Bedrock dotted+versioned form,
dated snapshot folding, idempotency for bare names, and unknown vendor
pass-through.  These tests must FAIL before model_id.py exists and PASS
after a correct implementation is in place.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# canonicalize()
# ---------------------------------------------------------------------------


class TestCanonicalize:
    """Table-driven tests for canonicalize()."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # --- Bare names: idempotent, returned unchanged ---
            ("gpt-4o", "gpt-4o"),
            ("gpt-4o-mini", "gpt-4o-mini"),
            ("gpt-4-turbo", "gpt-4-turbo"),
            ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
            ("claude-3-opus-20240229", "claude-3-opus-20240229"),
            ("gemini-1.5-pro", "gemini-1.5-pro"),
            ("mistral-large-latest", "mistral-large-latest"),
            ("llama3-8b-8192", "llama3-8b-8192"),
            # --- openrouter/ prefix (double-nested, iterative stripping) ---
            ("openrouter/openai/gpt-4o", "gpt-4o"),
            ("openrouter/openai/gpt-4o-mini", "gpt-4o-mini"),
            ("openrouter/anthropic/claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
            ("openrouter/google/gemini-pro", "gemini-pro"),
            ("openrouter/mistral/mistral-large", "mistral-large"),
            # --- openai/ direct ---
            ("openai/gpt-4o", "gpt-4o"),
            ("openai/gpt-4-turbo", "gpt-4-turbo"),
            ("openai/gpt-4o-mini", "gpt-4o-mini"),
            # --- anthropic/ direct ---
            ("anthropic/claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
            ("anthropic/claude-3-opus-20240229", "claude-3-opus-20240229"),
            # --- azure/ ---
            ("azure/gpt-4o", "gpt-4o"),
            ("azure/gpt-4-turbo", "gpt-4-turbo"),
            # --- groq/ ---
            ("groq/llama3-8b-8192", "llama3-8b-8192"),
            ("groq/mixtral-8x7b-32768", "mixtral-8x7b-32768"),
            # --- mistral/ ---
            ("mistral/mistral-large-latest", "mistral-large-latest"),
            ("mistral/mistral-small-latest", "mistral-small-latest"),
            # --- cohere/ ---
            ("cohere/command-r-plus", "command-r-plus"),
            ("cohere/command-r", "command-r"),
            # --- deepseek/ ---
            ("deepseek/deepseek-chat", "deepseek-chat"),
            ("deepseek/deepseek-coder", "deepseek-coder"),
            # --- gemini/ / google/ ---
            ("gemini/gemini-1.5-pro", "gemini-1.5-pro"),
            ("gemini/gemini-1.5-flash", "gemini-1.5-flash"),
            ("google/gemini-1.5-pro", "gemini-1.5-pro"),
            # --- vertex_ai/ ---
            ("vertex_ai/gemini-1.5-pro", "gemini-1.5-pro"),
            # Vertex @version pin: prefix stripped then @date stripped.
            ("vertex_ai/claude-3-5-sonnet@20241022", "claude-3-5-sonnet"),
            # --- together_ai/ (single prefix, org/model path preserved; lowercased) ---
            # --- fireworks_ai/ ---
            ("fireworks_ai/llama-v3p1-405b-instruct", "llama-v3p1-405b-instruct"),
            # --- bedrock/ + dotted+versioned form ---
            ("bedrock/anthropic.claude-3-5-sonnet-20241022-v1:0", "claude-3-5-sonnet-20241022"),
            ("bedrock/anthropic.claude-3-opus-20240229-v1:0", "claude-3-opus-20240229"),
            ("bedrock/amazon.titan-text-express-v1:0", "titan-text-express"),
            ("bedrock/meta.llama3-8b-instruct-v1:0", "llama3-8b-instruct"),
            # --- Raw dotted+versioned Bedrock form (no bedrock/ prefix) ---
            ("anthropic.claude-3-5-sonnet-20241022-v1:0", "claude-3-5-sonnet-20241022"),
            ("amazon.titan-text-express-v1:0", "titan-text-express"),
            # --- Case-insensitive prefix matching ---
            ("OpenRouter/openai/gpt-4o", "gpt-4o"),
            ("AZURE/gpt-4o", "gpt-4o"),
            ("Groq/llama3-8b-8192", "llama3-8b-8192"),
            # --- Unknown vendor prefix: returned unchanged ---
            ("my-custom-provider/gpt-4o", "my-custom-provider/gpt-4o"),
            ("local/llama3", "local/llama3"),
            ("ollama/mistral", "ollama/mistral"),
            # --- Bedrock region prefix (us./eu./apac.) stripped before vendor fold ---
            ("us.anthropic.claude-3-5-sonnet-20241022-v1:0", "claude-3-5-sonnet-20241022"),
            ("eu.anthropic.claude-3-5-sonnet-20241022-v1:0", "claude-3-5-sonnet-20241022"),
            ("apac.meta.llama3-8b-instruct-v1:0", "llama3-8b-instruct"),
            ("bedrock/us.anthropic.claude-3-5-sonnet-20241022-v1:0", "claude-3-5-sonnet-20241022"),
            ("bedrock/eu.amazon.titan-text-express-v1:0", "titan-text-express"),
            # --- Uppercase model names lowercased in output (registry keys are lowercase) ---
            ("GPT-4O", "gpt-4o"),
            ("Claude-3-5-Sonnet-20241022", "claude-3-5-sonnet-20241022"),
            # together_ai mixed-case model lowercased
            ("together_ai/meta-llama/Llama-3-70b-chat-hf", "meta-llama/llama-3-70b-chat-hf"),
            # --- Vertex AI @version pin — bare (no publisher prefix) ---
            ("claude-haiku-4-5@20251001", "claude-haiku-4-5"),
            ("claude-haiku-4-5@latest", "claude-haiku-4-5"),
            ("claude-haiku-4-5@default", "claude-haiku-4-5"),
            ("gemini-1.5-pro@002", "gemini-1.5-pro"),
            # --- Vertex AI @version pin — publisher-dotted form ---
            ("anthropic.claude-haiku-4-5@20251001", "claude-haiku-4-5"),
            ("anthropic.claude-3-5-sonnet@20240620", "claude-3-5-sonnet"),
            # --- Vertex AI @version pin — full path: vertex_ai/ prefix + publisher ---
            ("vertex_ai/anthropic.claude-haiku-4-5@20251001", "claude-haiku-4-5"),
            ("vertex_ai/anthropic.claude-3-5-sonnet@20240620", "claude-3-5-sonnet"),
            # --- Dotted version names that must stay UNCHANGED (no publisher stripping) ---
            # Vendor before dot contains a hyphen → _BEDROCK_VENDOR_RE rejects it safely.
            # gemini-1.5-pro and gemini-1.5-flash already appear above; new entries only.
            ("gpt-4.1", "gpt-4.1"),
            ("gpt-4.1-mini", "gpt-4.1-mini"),
            ("gpt-4.1-nano", "gpt-4.1-nano"),
            ("gemini-2.5-pro", "gemini-2.5-pro"),
            ("gemini-2.5-flash", "gemini-2.5-flash"),
            ("gemini-2.0-flash", "gemini-2.0-flash"),
            ("gemini-1.5-flash", "gemini-1.5-flash"),
            # --- Anthropic dotted-version normalisation: claude-<M>.<N> → claude-<M>-<N> ---
            # Wire forms used by OpenRouter and other gateways that insert a dot
            # between the major and minor version number.  Scoped to the
            # "claude-" prefix only — all gpt-/gemini- dotted names above must
            # remain unchanged (regression guard on the scope).
            ("claude-3.5-sonnet", "claude-3-5-sonnet"),
            ("claude-3.7-sonnet", "claude-3-7-sonnet"),
            ("claude-3.5-haiku", "claude-3-5-haiku"),
            ("claude-3.5-opus", "claude-3-5-opus"),  # hypothetical; same rule
            # Dotted form with suffix preserved after normalisation.
            ("claude-3.5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
            # Gateway-prefixed dotted form: prefix stripped first, then dot normalised.
            ("openrouter/anthropic/claude-3.5-sonnet", "claude-3-5-sonnet"),
            ("anthropic/claude-3.5-sonnet", "claude-3-5-sonnet"),
            # Idempotency: already-hyphenated bare names unchanged.
            ("claude-3-5-sonnet", "claude-3-5-sonnet"),
            ("claude-3-7-sonnet", "claude-3-7-sonnet"),
        ],
    )
    def test_canonicalize_table(self, model: str, expected: str) -> None:
        """Arrange: model string from table.
        Act: canonicalize(model).
        Assert: result matches expected canonical form.
        """
        from frugon.model_id import canonicalize

        assert canonicalize(model) == expected, (
            f"canonicalize({model!r}) → {canonicalize(model)!r}, want {expected!r}"
        )

    def test_canonicalize_idempotent_bare_name(self) -> None:
        """Arrange: already-canonical bare name.
        Act: canonicalize twice.
        Assert: second call returns same result as first (idempotent).
        """
        from frugon.model_id import canonicalize

        assert canonicalize(canonicalize("gpt-4o")) == "gpt-4o"
        assert canonicalize(canonicalize("claude-3-5-sonnet-20241022")) == (
            "claude-3-5-sonnet-20241022"
        )

    def test_canonicalize_idempotent_after_gateway_strip(self) -> None:
        """Arrange: gateway-prefixed model.
        Act: canonicalize result of canonicalize.
        Assert: double-canonicalize is stable.
        """
        from frugon.model_id import canonicalize

        first = canonicalize("openai/gpt-4o")
        second = canonicalize(first)
        assert first == second == "gpt-4o"

    def test_canonicalize_idempotent_region_prefix(self) -> None:
        """Arrange: region-prefixed Bedrock model.
        Act: canonicalize twice.
        Assert: double-canonicalize is stable and lowercase.
        """
        from frugon.model_id import canonicalize

        result = canonicalize("us.anthropic.claude-3-5-sonnet-20241022-v1:0")
        assert result == "claude-3-5-sonnet-20241022"
        assert canonicalize(result) == result

    def test_canonicalize_idempotent_uppercase(self) -> None:
        """Arrange: uppercase model names.
        Act: canonicalize twice.
        Assert: output is lowercase and stable.
        """
        from frugon.model_id import canonicalize

        assert canonicalize(canonicalize("GPT-4O")) == "gpt-4o"
        assert canonicalize(canonicalize("Claude-3-5-Sonnet-20241022")) == "claude-3-5-sonnet-20241022"

    def test_canonicalize_pure_no_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Assert canonicalize() makes no file or network I/O — pure function."""
        import socket

        def _no_network(*args: object, **kwargs: object) -> None:
            raise AssertionError("canonicalize() must not make network calls")

        monkeypatch.setattr(socket, "socket", _no_network)

        from frugon.model_id import canonicalize

        canonicalize("openai/gpt-4o")  # must not raise

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # UPPERCASE Bedrock region + UPPERCASE version suffix (C1 regression guard)
            ("US.ANTHROPIC.CLAUDE-3-5-SONNET-20241022-V1:0", "claude-3-5-sonnet-20241022"),
            # Mixed-case bedrock/ prefix + UPPERCASE region + UPPERCASE version
            ("Bedrock/EU.Amazon.Titan-Text-Express-V1:0", "titan-text-express"),
        ],
    )
    def test_canonicalize_uppercase_bedrock_region_version(
        self, model: str, expected: str
    ) -> None:
        """Arrange: UPPERCASE Bedrock region prefix and version suffix.
        Act: canonicalize(model).
        Assert: result matches expected canonical form (C1 regression guard).

        Root cause guarded: _BEDROCK_VERSION_RE lacked re.IGNORECASE so
        uppercase -V1:0 did not match, leaving the vendor prefix attached.
        Fix: lowercase at entry of canonicalize() so all regex work sees
        a case-normalised string.
        """
        from frugon.model_id import canonicalize

        result = canonicalize(model)
        assert result == expected, (
            f"canonicalize({model!r}) -> {result!r}, want {expected!r}"
        )

    def test_canonicalize_idempotent_uppercase_bedrock_region_version(self) -> None:
        """Arrange: UPPERCASE Bedrock region+version inputs.
        Act: canonicalize twice.
        Assert: double-canonicalize is stable.

        The prior idempotency tests used lowercase inputs only -- they never
        exercised the uppercase code path that produced the C1 defect.
        """
        from frugon.model_id import canonicalize

        inputs = [
            "US.ANTHROPIC.CLAUDE-3-5-SONNET-20241022-V1:0",
            "Bedrock/EU.Amazon.Titan-Text-Express-V1:0",
            "APAC.META.LLAMA3-8B-INSTRUCT-V1:0",
        ]
        for model in inputs:
            first = canonicalize(model)
            second = canonicalize(first)
            assert first == second, (
                f"canonicalize not idempotent for {model!r}: "
                f"first={first!r}, second={second!r}"
            )

    def test_canonicalize_price_resolution_cross_module(self) -> None:
        """Arrange: lowercase Bedrock dotted+versioned model ID.
        Act: canonicalize then get_model_price.
        Assert: price is not None (canonical key resolves in pricing registry).

        Cross-module regression: verifies that canonicalize() output resolves
        in the pricing module.  This gap let C1 go undetected -- model_id
        tests were isolated from pricing outcomes.
        """
        from frugon.model_id import canonicalize
        from frugon.pricing import get_model_price

        canonical = canonicalize("us.anthropic.claude-3-5-sonnet-20241022-v1:0")
        price = get_model_price(canonical)
        assert price is not None, (
            f"get_model_price({canonical!r}) returned None -- "
            "canonicalize output does not resolve in pricing registry"
        )


# ---------------------------------------------------------------------------
# base_family()
# ---------------------------------------------------------------------------


class TestBaseFamily:
    """Table-driven tests for base_family()."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            # --- ISO date suffix (-YYYY-MM-DD) ---
            ("gpt-4o-2024-08-06", "gpt-4o"),
            ("gpt-4o-2024-11-20", "gpt-4o"),
            ("gpt-4-turbo-2024-04-09", "gpt-4-turbo"),
            ("gpt-4-0125-preview-2024-01-25", "gpt-4-0125-preview"),
            # --- Compact date suffix (-YYYYMMDD) ---
            ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet"),
            ("claude-3-opus-20240229", "claude-3-opus"),
            ("claude-3-haiku-20240307", "claude-3-haiku"),
            # --- No date: returned unchanged ---
            ("gpt-4o", "gpt-4o"),
            ("gpt-4-turbo", "gpt-4-turbo"),
            ("gpt-4o-mini", "gpt-4o-mini"),
            ("claude-3-5-sonnet", "claude-3-5-sonnet"),
            ("llama3-8b-8192", "llama3-8b-8192"),
            ("gemini-1.5-pro", "gemini-1.5-pro"),
            # --- -latest suffix stripped for lookup fallback ---
            ("mistral-large-latest", "mistral-large"),
            ("gpt-4o-latest", "gpt-4o"),
            # --- :tag suffix (:beta, :free) stripped; then date if present ---
            ("claude-3-5-sonnet-20241022:beta", "claude-3-5-sonnet"),
            ("gpt-4o:free", "gpt-4o"),
            ("claude-3-5-sonnet-20241022:free", "claude-3-5-sonnet"),
            # --- Vertex AI @version pin stripped (defense-in-depth) ---
            ("claude-haiku-4-5@20251001", "claude-haiku-4-5"),
            ("claude-haiku-4-5@latest", "claude-haiku-4-5"),
            ("claude-haiku-4-5@default", "claude-haiku-4-5"),
            ("gemini-1.5-pro@002", "gemini-1.5-pro"),
            # Dated base model with Vertex pin: pin stripped, date stripped.
            ("anthropic.claude-3-5-sonnet@20240620", "anthropic.claude-3-5-sonnet"),
            # Already-canonical forms with dotted version numbers unchanged.
            # gemini-1.5-pro is already covered above; new entries only.
            ("gpt-4.1", "gpt-4.1"),
            ("gpt-4.1-mini", "gpt-4.1-mini"),
            ("gemini-2.5-pro", "gemini-2.5-pro"),
        ],
    )
    def test_base_family_table(self, model: str, expected: str) -> None:
        """Arrange: model string from table.
        Act: base_family(model).
        Assert: result matches expected base family.
        """
        from frugon.model_id import base_family

        assert base_family(model) == expected, (
            f"base_family({model!r}) → {base_family(model)!r}, want {expected!r}"
        )

    def test_base_family_idempotent(self) -> None:
        """Arrange: already-base model names and a dated snapshot.
        Act: base_family twice.
        Assert: second call is stable.
        """
        from frugon.model_id import base_family

        assert base_family(base_family("gpt-4o-2024-08-06")) == "gpt-4o"
        assert base_family(base_family("gpt-4o")) == "gpt-4o"

    def test_base_family_pure_no_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Assert base_family() makes no file or network I/O — pure function."""
        import socket

        def _no_network(*args: object, **kwargs: object) -> None:
            raise AssertionError("base_family() must not make network calls")

        monkeypatch.setattr(socket, "socket", _no_network)

        from frugon.model_id import base_family

        base_family("gpt-4o-2024-08-06")  # must not raise
