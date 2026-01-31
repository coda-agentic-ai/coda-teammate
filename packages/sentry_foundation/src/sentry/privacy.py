"""Privacy module for PII scrubbing and data redaction.

Provides tools to identify and redact sensitive information from text
and LangChain messages.
"""

import re
from typing import Any


class PIIScrubber:
    """Scrub Personally Identifiable Information (PII) from text and messages.

    This class identifies and redacts:
    - Financial data (credit card numbers with Luhn validation)
    - Contact information (emails, phone numbers)
    - Secrets (API keys, tokens with known prefixes)

    Example:
        >>> scrubber = PIIScrubber()
        >>> clean = scrubber.scrub_text("Contact: john@example.com")
        >>> print(clean)
        "Contact: [REDACTED_EMAIL]"
    """

    # Credit card regex (major card formats)
    CREDIT_CARD_PATTERN = re.compile(
        r"\b(?:"
        r"4[0-9]{12}(?:[0-9]{3})?"  # Visa
        r"|5[1-5][0-9]{14}"         # Mastercard
        r"|3[47][0-9]{13}"          # American Express
        r"|6(?:011|5[0-9]{2})[0-9]{12}"  # Discover
        r"|(?:2131|1800|35\d{3})\d{11}"  # JCB
        r")\b"
    )

    # Email regex
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    # Phone number regex - various formats
    PHONE_PATTERN = re.compile(
        r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"
    )

    # API Key and Secret patterns
    SECRET_PATTERNS = [
        # OpenAI, Anthropic, and similar
        re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"),
        # GitHub Personal Access Token
        re.compile(r"\b(ghp_[a-zA-Z0-9]{36})\b"),
        # GitHub OAuth Token
        re.compile(r"\b(gho_[a-zA-Z0-9]{36})\b"),
        # AWS Access Key
        re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
        # Generic high-entropy secrets
        re.compile(r"\b([a-zA-Z0-9_-]{32,})\b"),
    ]

    # Secret prefixes for detection
    SECRET_PREFIXES = [
        "sk-", "sk_live_", "sk_test_",  # OpenAI/Stripe
        "ghp_", "gho_", "ghs_", "ghr_",  # GitHub
        "AKIA",  # AWS
        "eyJ",  # JWT tokens
        "xoxb-", "xoxp-",  # Slack
        "xmc-", "xmb-", "xma-",  # MongoDB
    ]

    # Placeholder templates
    PLACEHOLDERS = {
        "credit_card": "[REDACTED_FINANCE]",
        "email": "[REDACTED_EMAIL]",
        "phone": "[REDACTED_PHONE]",
        "secret": "[REDACTED_SECRET]",
    }

    def __init__(self, strict_mode: bool = False):
        """Initialize the PII Scrubber.

        Args:
            strict_mode: If True, applies additional aggressive scrubbing.
                         If False, uses conservative patterns.
        """
        self.strict_mode = strict_mode

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm.

        Args:
            card_number: The credit card number to validate (digits only).

        Returns:
            True if the card number passes Luhn validation.
        """
        # Remove non-digits
        digits = re.sub(r"\D", "", card_number)

        if len(digits) < 13 or len(digits) > 19:
            return False

        # Luhn algorithm
        total = 0
        is_second = True

        for digit in reversed(digits):
            n = int(digit)
            if is_second:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
            is_second = not is_second

        return total % 10 == 0

    def _is_likely_secret(self, text: str) -> bool:
        """Check if text matches known secret prefixes.

        Args:
            text: The text to check.

        Returns:
            True if the text appears to be a secret.
        """
        for prefix in self.SECRET_PREFIXES:
            if text.startswith(prefix):
                return True
        return False

    def scrub_text(self, text: str) -> str:
        """Scrub PII from a text string.

        Args:
            text: The input text that may contain sensitive information.

        Returns:
            The text with sensitive information replaced by placeholders.

        Example:
            >>> scrubber = PIIScrubber()
            >>> scrubber.scrub_text("Email: john@example.com")
            'Email: [REDACTED_EMAIL]'
        """
        if not text:
            return text

        result = text

        # Scrub credit card numbers
        for match in self.CREDIT_CARD_PATTERN.finditer(text):
            card_number = match.group()
            # Validate with Luhn
            if self._luhn_check(card_number):
                result = result.replace(
                    card_number,
                    self.PLACEHOLDERS["credit_card"]
                )

        # Scrub emails
        for match in self.EMAIL_PATTERN.finditer(text):
            email = match.group()
            result = result.replace(
                email,
                self.PLACEHOLDERS["email"]
            )

        # Scrub phone numbers
        for match in self.PHONE_PATTERN.finditer(text):
            phone = match.group()
            result = result.replace(
                phone,
                self.PLACEHOLDERS["phone"]
            )

        # Scrub secrets with known prefixes
        for pattern in self.SECRET_PATTERNS:
            for match in pattern.finditer(text):
                secret = match.group(1) if match.groups() else match.group()
                if self._is_likely_secret(secret):
                    result = result.replace(
                        secret,
                        self.PLACEHOLDERS["secret"]
                    )

        return result

    def scrub_message(self, message: Any) -> Any:
        """Scrub PII from a LangChain BaseMessage object.

        Args:
            message: A LangChain BaseMessage or similar object with
                     content and optional metadata fields.

        Returns:
            The message with scrubbed content. Returns the original
            object type with modified content.

        Example:
            >>> from langchain_core.messages import HumanMessage
            >>> scrubber = PIIScrubber()
            >>> msg = HumanMessage(content="Email me at john@example.com")
            >>> scrubbed = scrubber.scrub_message(msg)
        """
        if not hasattr(message, "content"):
            return message

        # Scrub the content
        scrubbed_content = self.scrub_text(message.content)

        # Create a new message with scrubbed content
        # Try to preserve the original type
        if hasattr(message, "copy"):
            # For objects with copy method (like LangChain messages)
            new_message = message.copy()
            new_message.content = scrubbed_content
            return new_message
        elif hasattr(message, "model_copy"):
            # For Pydantic models (newer LangChain versions)
            new_message = message.model_copy()
            new_message.content = scrubbed_content
            return new_message
        else:
            # Fallback: create a new object with same type
            try:
                new_message = type(message)(content=scrubbed_content)
                return new_message
            except (TypeError, AttributeError):
                # If all else fails, return with modified attribute
                message.content = scrubbed_content
                return message

    def scrub_dict(self, data: dict) -> dict:
        """Scrub PII from a dictionary recursively.

        Args:
            data: A dictionary that may contain sensitive information.

        Returns:
            A new dictionary with scrubbed string values.
        """
        result = {}

        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.scrub_text(value)
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = self.scrub_list(value)
            else:
                result[key] = value

        return result

    def scrub_list(self, data: list) -> list:
        """Scrub PII from a list recursively.

        Args:
            data: A list that may contain sensitive information.

        Returns:
            A new list with scrubbed string values.
        """
        return [
            self.scrub_dict(item) if isinstance(item, dict)
            else self.scrub_list(item) if isinstance(item, list)
            else self.scrub_text(item) if isinstance(item, str)
            else item
            for item in data
        ]

    def detect_violations(self, text: str) -> list[dict]:
        """Detect all PII violations in text without scrubbing.

        Args:
            text: The input text to check for PII.

        Returns:
            A list of violation dicts with type, value, and position.
        """
        violations = []

        # Detect credit cards
        for match in self.CREDIT_CARD_PATTERN.finditer(text):
            card_number = match.group()
            if self._luhn_check(card_number):
                violations.append({
                    "type": "credit_card",
                    "value": card_number,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Detect emails
        for match in self.EMAIL_PATTERN.finditer(text):
            violations.append({
                "type": "email",
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })

        # Detect phone numbers
        for match in self.PHONE_PATTERN.finditer(text):
            violations.append({
                "type": "phone",
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
            })

        # Detect secrets
        for pattern in self.SECRET_PATTERNS:
            for match in pattern.finditer(text):
                secret = match.group(1) if match.groups() else match.group()
                if self._is_likely_secret(secret):
                    violations.append({
                        "type": "secret",
                        "value": secret,
                        "start": match.start(),
                        "end": match.end(),
                    })

        return violations

    def scrub_with_violation_report(self, text: str) -> tuple[str, bool]:
        """Scrub text and report if PII was detected.

        Args:
            text: The input text that may contain sensitive information.

        Returns:
            A tuple of (scrubbed_text, has_violation) where has_violation
            is True if any PII was detected and scrubbed.
        """
        violations = self.detect_violations(text)
        scrubbed = self.scrub_text(text)
        return scrubbed, len(violations) > 0
