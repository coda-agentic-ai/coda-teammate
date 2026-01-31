"""Test suite for the PIIScrubber privacy module.

Tests cover:
- Credit card detection and Luhn validation
- Email and phone number redaction
- API key and secret detection
- LangChain message scrubbing
- Preservation of non-sensitive content
"""

import pytest
from sentry.privacy import PIIScrubber


class TestPIIScrubberBasic:
    """Basic functionality tests for PIIScrubber."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_empty_string(self):
        """Test that empty strings are handled correctly."""
        assert self.scrubber.scrub_text("") == ""
        assert self.scrubber.scrub_text(None) is None

    def test_none_input(self):
        """Test that None input returns None."""
        assert self.scrubber.scrub_text(None) is None

    def test_no_pii(self):
        """Test that text without PII is unchanged."""
        text = "Hello, this is a normal message without sensitive data."
        result = self.scrubber.scrub_text(text)
        assert result == text


class TestEmailScrubbing:
    """Tests for email address detection and redaction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_basic_email(self):
        """Test basic email redaction."""
        text = "Contact me at john@example.com"
        result = self.scrubber.scrub_text(text)
        assert result == "Contact me at [REDACTED_EMAIL]"
        assert "[REDACTED_EMAIL]" in result

    def test_multiple_emails(self):
        """Test multiple emails are all redacted."""
        text = "Email john@example.com or jane@company.org"
        result = self.scrubber.scrub_text(text)
        assert result.count("[REDACTED_EMAIL]") == 2

    def test_email_in_sentence(self):
        """Test email embedded in a sentence."""
        text = "Please send the report to data-team@analytics.co.uk by Friday."
        result = self.scrubber.scrub_text(text)
        assert "[REDACTED_EMAIL]" in result
        assert "by Friday" in result

    def test_email_with_subdomains(self):
        """Test email with subdomains."""
        text = "Contact support@mail.server.test.com"
        result = self.scrubber.scrub_text(text)
        assert result == "Contact [REDACTED_EMAIL]"


class TestPhoneScrubbing:
    """Tests for phone number detection and redaction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_us_phone(self):
        """Test US phone format."""
        text = "Call me at 555-123-4567"
        result = self.scrubber.scrub_text(text)
        assert result == "Call me at [REDACTED_PHONE]"

    def test_us_phone_parentheses(self):
        """Test US phone with area code in parentheses."""
        text = "Call (800) 555-1234"
        result = self.scrubber.scrub_text(text)
        assert result == "Call [REDACTED_PHONE]"

    def test_international_phone(self):
        """Test international phone format."""
        text = "Call +1-415-555-0123"
        result = self.scrubber.scrub_text(text)
        assert result == "Call [REDACTED_PHONE]"

    def test_phone_with_dots(self):
        """Test phone format with dots."""
        text = "Phone: +1.415.555.0123"
        result = self.scrubber.scrub_text(text)
        assert result == "Phone: [REDACTED_PHONE]"


class TestCreditCardScrubbing:
    """Tests for credit card detection and Luhn validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_mastercard(self):
        """Test Mastercard detection (valid Luhn)."""
        text = "Mastercard: 5200828282828210"
        result = self.scrubber.scrub_text(text)
        assert result == "Mastercard: [REDACTED_FINANCE]"

    def test_discover(self):
        """Test Discover card detection (valid Luhn)."""
        text = "Discover: 6011111111111117"
        result = self.scrubber.scrub_text(text)
        assert result == "Discover: [REDACTED_FINANCE]"

    def test_invalid_luhn_not_redacted(self):
        """Test that invalid Luhn numbers are not redacted."""
        # This is not a valid credit card number (fails Luhn)
        text = "Not a card: 1234567890123456"
        result = self.scrubber.scrub_text(text)
        # Should not be redacted if Luhn fails
        assert "[REDACTED_FINANCE]" not in result


class TestSecretScrubbing:
    """Tests for API key and secret detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_openai_key(self):
        """Test OpenAI API key redaction."""
        text = "sk-abc123def456ghi789jkl012mno345pqr678"
        result = self.scrubber.scrub_text(text)
        assert result == "[REDACTED_SECRET]"

    def test_github_token(self):
        """Test GitHub PAT redaction."""
        text = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = self.scrubber.scrub_text(text)
        assert result == "[REDACTED_SECRET]"

    def test_github_oauth(self):
        """Test GitHub OAuth token redaction."""
        text = "gho_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = self.scrubber.scrub_text(text)
        assert result == "[REDACTED_SECRET]"

    def test_aws_access_key(self):
        """Test AWS access key redaction."""
        text = "AKIAIOSFODNN7EXAMPLE"
        result = self.scrubber.scrub_text(text)
        assert result == "[REDACTED_SECRET]"

    def test_jwt_token(self):
        """Test JWT token detection."""
        text = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
        result = self.scrubber.scrub_text(text)
        assert "[REDACTED_SECRET]" in result


class TestMarkdownPreservation:
    """Tests that the scrubber doesn't mangle markdown formatting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_code_blocks(self):
        """Test that code blocks are preserved."""
        text = """```python
def hello():
    email = "test@example.com"
    print("Hello")
```"""
        result = self.scrubber.scrub_text(text)
        # The email inside code block should still be redacted
        assert "[REDACTED_EMAIL]" in result
        # Code block formatting should be preserved
        assert "```python" in result
        assert "def hello():" in result

    def test_inline_code(self):
        """Test inline code formatting."""
        text = "Use `api_key = 'sk-abcdefghijklmnopqrstuvwxyz123456'` to authenticate"
        result = self.scrubber.scrub_text(text)
        assert "[REDACTED_SECRET]" in result
        # Backticks should be preserved
        assert "`" in result

    def test_markdown_links(self):
        """Test that markdown links are handled correctly."""
        text = "[Click here](https://example.com?email=test@example.com)"
        result = self.scrubber.scrub_text(text)
        # Email in URL should be redacted
        assert "[REDACTED_EMAIL]" in result
        # Link text should be preserved
        assert "Click here" in result

    def test_markdown_headers(self):
        """Test that markdown headers are preserved."""
        text = """# Contact
Email: admin@company.org

## Support
Phone: 800-555-1234"""
        result = self.scrubber.scrub_text(text)
        # Headers should be preserved
        assert "# Contact" in result
        assert "## Support" in result
        # PII should be redacted
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result

    def test_json_format(self):
        """Test that JSON formatting is preserved."""
        text = """{
    "email": "user@example.com",
    "api_key": "sk-abcdefghijklmnopqrstuvwxyz123456"
}"""
        result = self.scrubber.scrub_text(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_SECRET]" in result
        # JSON structure should be preserved
        assert '"email":' in result
        assert '"api_key":' in result


class TestMixedContent:
    """Tests for text with multiple types of PII."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_multiple_pii_types(self):
        """Test text with multiple PII types."""
        text = """Customer Information:
Email: john.doe@example.com
Phone: 555-123-4567
Card: 5200828282828210
API Key: sk-abcdefghijklmnopqrstuvwxyz123456

Please process this request."""
        result = self.scrubber.scrub_text(text)
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_PHONE]" in result
        assert "[REDACTED_FINANCE]" in result
        assert "[REDACTED_SECRET]" in result
        # Non-PII should be preserved
        assert "Customer Information:" in result
        assert "Please process this request." in result

    def test_api_key_with_context(self):
        """Test API key in context."""
        text = "Set the OpenAI key: sk-abcdefghijklmnopqrstuvwxyz123456"
        result = self.scrubber.scrub_text(text)
        assert "[REDACTED_SECRET]" in result
        assert "Set the OpenAI key:" in result


class TestScrubMessage:
    """Tests for LangChain message scrubbing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_scrub_human_message(self):
        """Test scrubbing a HumanMessage-like object."""
        # Simulate a message object with content
        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.type = "human"

            def copy(self):
                return MockMessage(self.content)

        msg = MockMessage("My email is john@example.com")
        result = self.scrubber.scrub_message(msg)

        assert result.content == "My email is [REDACTED_EMAIL]"
        assert result.type == "human"

    def test_scrub_ai_message(self):
        """Test scrubbing an AIMessage-like object."""
        class MockMessage:
            def __init__(self, content):
                self.content = content
                self.type = "ai"

            def copy(self):
                return MockMessage(self.content)

        msg = MockMessage("I called sk-abcdefghijklmnopqrstuvwxyz12345 for authentication")
        result = self.scrubber.scrub_message(msg)

        assert result.content == "I called [REDACTED_SECRET] for authentication"

    def test_scrub_message_with_metadata(self):
        """Test that message metadata is preserved."""
        class MockMessage:
            def __init__(self, content, metadata):
                self.content = content
                self.metadata = metadata
                self.type = "human"

            def copy(self):
                return MockMessage(self.content, self.metadata)

        msg = MockMessage(
            content="Contact: jane@company.org",
            metadata={"tokens": 100, "model": "gpt-4"}
        )
        result = self.scrubber.scrub_message(msg)

        assert result.content == "Contact: [REDACTED_EMAIL]"
        assert result.metadata == {"tokens": 100, "model": "gpt-4"}

    def test_scrub_non_message(self):
        """Test that non-message objects are returned unchanged."""
        result = self.scrubber.scrub_message("not a message")
        assert result == "not a message"


class TestScrubDict:
    """Tests for dictionary scrubbing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_scrub_nested_dict(self):
        """Test scrubbing nested dictionaries."""
        data = {
            "user": {
                "email": "test@example.com",
                "phone": "555-123-4567"
            },
            "config": {
                "api_key": "sk-abcdefghijklmnopqrstuvwxyz"
            }
        }
        result = self.scrubber.scrub_dict(data)

        assert result["user"]["email"] == "[REDACTED_EMAIL]"
        assert result["user"]["phone"] == "[REDACTED_PHONE]"
        assert result["config"]["api_key"] == "[REDACTED_SECRET]"

    def test_scrub_dict_with_list(self):
        """Test scrubbing dictionaries with lists."""
        data = {
            "contacts": [
                {"email": "a@b.com"},
                {"email": "c@d.com"}
            ]
        }
        result = self.scrubber.scrub_dict(data)

        assert result["contacts"][0]["email"] == "[REDACTED_EMAIL]"
        assert result["contacts"][1]["email"] == "[REDACTED_EMAIL]"


class TestStrictMode:
    """Tests for strict mode behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber_strict = PIIScrubber(strict_mode=True)
        self.scrubber_normal = PIIScrubber(strict_mode=False)

    def test_strict_mode_more_aggressive(self):
        """Test that strict mode is more aggressive (placeholder for future use)."""
        # This is a placeholder - strict mode behavior can be extended
        text = "Normal text with no PII"
        assert self.scrubber_strict.scrub_text(text) == text


class TestEdgeCases:
    """Edge case tests."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scrubber = PIIScrubber()

    def test_unicode_in_text(self):
        """Test handling of unicode characters."""
        text = "Contact: 用户@example.com"
        result = self.scrubber.scrub_text(text)
        # Should not crash
        assert isinstance(result, str)

    def test_very_long_string(self):
        """Test handling of very long strings."""
        prefix = "!" * 1000
        suffix = "@" * 1000
        text = prefix + "test@example.com" + suffix
        result = self.scrubber.scrub_text(text)
        assert "[REDACTED_EMAIL]" in result
        # Should not have truncated the rest
        assert prefix in result
        assert suffix in result

    def test_overlapping_patterns(self):
        """Test handling of overlapping patterns."""
        # Phone number that might look like part of an email
        text = "Contact 555-555-5555 or email@test.com"
        result = self.scrubber.scrub_text(text)
        assert "[REDACTED_PHONE]" in result
        assert "[REDACTED_EMAIL]" in result
