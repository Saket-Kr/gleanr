"""Anthropic SDK-based providers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gleanr.errors import ProviderError
from gleanr.models import Fact
from gleanr.models.consolidation import ConsolidationAction
from gleanr.providers.parsing import (
    CONSOLIDATION_PROMPT,
    REFLECTION_PROMPT,
    format_prior_facts,
    format_turns,
    parse_consolidation_actions,
    parse_reflection_facts,
)
from gleanr.utils.retry import RetryConfig, with_retry

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from gleanr.models import Episode, Turn

logger = logging.getLogger(__name__)

try:
    from anthropic import (
        APIConnectionError as AnthropicConnectionError,
        APITimeoutError as AnthropicTimeoutError,
        InternalServerError as AnthropicInternalError,
        RateLimitError as AnthropicRateLimitError,
    )

    _ANTHROPIC_RETRYABLE: tuple[type[Exception], ...] = (
        AnthropicTimeoutError,
        AnthropicConnectionError,
        AnthropicRateLimitError,
        AnthropicInternalError,
    )
except ImportError:
    _ANTHROPIC_RETRYABLE = (TimeoutError, ConnectionError)


class AnthropicReflector:
    """Reflector using Anthropic Python SDK for fact extraction.

    Uses Claude to extract semantic facts from episodes.
    """

    def __init__(
        self,
        client: "AsyncAnthropic",
        model: str = "claude-3-haiku-20240307",
        *,
        max_facts: int = 5,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ) -> None:
        """Initialize Anthropic reflector.

        Args:
            client: Configured AsyncAnthropic client
            model: Claude model name
            max_facts: Maximum facts to extract per episode
            max_tokens: Maximum tokens in response
            max_retries: Maximum retry attempts for API calls
        """
        self._client = client
        self._model = model
        self._max_facts = max_facts
        self._max_tokens = max_tokens
        self._retry_config = RetryConfig(
            max_attempts=max_retries,
            base_delay=1.0,
            max_delay=30.0,
            retryable_exceptions=_ANTHROPIC_RETRYABLE,
        )

    async def reflect(self, episode: "Episode", turns: list["Turn"]) -> list[Fact]:
        """Extract semantic facts from an episode.

        Args:
            episode: The episode to reflect on
            turns: Turns belonging to the episode

        Returns:
            List of extracted facts

        Raises:
            ProviderError: If reflection fails after retries
        """
        if not turns:
            return []

        try:
            turns_text = "\n".join(
                f"[{t.role.value}]: {t.content}" for t in turns
            )

            prompt = REFLECTION_PROMPT.format(
                turns=turns_text,
                max_facts=self._max_facts,
            )

            response = await with_retry(
                self._message_create,
                self._retry_config,
                on_retry=self._log_retry,
                prompt=prompt,
            )

            content = self._extract_text(response)
            return self._parse_facts(content, episode)

        except Exception as e:
            if "ProviderError" in type(e).__name__:
                raise
            raise ProviderError(
                f"Anthropic reflection failed: {e}",
                provider="AnthropicReflector",
                retryable=False,
                cause=e,
            ) from e

    def _parse_facts(self, content: str, episode: "Episode") -> list[Fact]:
        """Parse facts from LLM response."""
        return parse_reflection_facts(content, episode)

    async def reflect_with_consolidation(
        self,
        episode: "Episode",
        turns: list["Turn"],
        prior_facts: list[Fact],
    ) -> list[ConsolidationAction]:
        """Consolidate prior facts with new episode content."""
        if not turns:
            return []

        try:
            prompt = CONSOLIDATION_PROMPT.format(
                prior_facts=format_prior_facts(prior_facts),
                turns=format_turns(turns),
            )

            response = await with_retry(
                self._message_create,
                self._retry_config,
                on_retry=self._log_retry,
                prompt=prompt,
            )

            content = self._extract_text(response)
            return parse_consolidation_actions(content)

        except Exception as e:
            if "ProviderError" in type(e).__name__:
                raise
            raise ProviderError(
                f"Anthropic consolidation failed: {e}",
                provider="AnthropicReflector",
                retryable=False,
                cause=e,
            ) from e

    async def _message_create(self, prompt: str) -> Any:
        """Make a single Anthropic messages API call."""
        return await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from an Anthropic response."""
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        return content

    @staticmethod
    def _log_retry(attempt: int, error: Exception) -> None:
        logger.warning(
            "Anthropic API call failed (attempt %d), retrying: %s", attempt, error
        )
