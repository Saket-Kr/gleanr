"""Gleanr configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gleanr.cache import CacheConfig
from gleanr.errors import ConfigurationError
from gleanr.models.types import DEFAULT_MARKER_WEIGHTS, MarkerType


@dataclass(frozen=True, slots=True)
class EpisodeBoundaryConfig:
    """Configuration for automatic episode boundaries."""

    max_turns: int = 6
    """Close episode after this many turns."""

    max_time_gap_seconds: int = 1800  # 30 minutes
    """Close episode if gap between turns exceeds this."""

    close_on_tool_result: bool = True
    """Close episode after a tool result."""

    close_on_patterns: tuple[str, ...] = (r"(?i)\b(done|finished|complete|thanks|thank you)\b",)
    """Regex patterns that trigger episode closure."""

    def should_close_on_content(self, content: str) -> bool:
        """Check if content matches any closure pattern."""
        return any(re.search(pattern, content) for pattern in self.close_on_patterns)


@dataclass(frozen=True, slots=True)
class RecallConfig:
    """Configuration for recall behavior."""

    default_token_budget: int = 4000
    """Default token budget when not specified."""

    current_episode_budget_pct: float = 0.2
    """Percentage of budget reserved for current episode (0-1)."""

    max_vector_results: int = 50
    """Maximum results from vector search."""

    min_relevance_threshold: float = 0.3
    """Minimum relevance score to include in results."""

    max_fact_candidates: int = 20
    """Maximum number of fact candidates to include in recall results.
    After relevance filtering, only the top-K facts by score are kept."""

    facts_only_recall: bool = True
    """When active facts exist, skip vector search on past turns and marked turns.
    Facts are the maintained, current-truth representation of past episodes.
    Raw turns from past episodes can carry stale information that contradicts
    updated facts. Set to False to include past turns alongside facts."""

    def validate(self) -> None:
        """Validate configuration values."""
        if self.current_episode_budget_pct < 0 or self.current_episode_budget_pct > 1:
            raise ConfigurationError(
                f"current_episode_budget_pct must be between 0 and 1, "
                f"got {self.current_episode_budget_pct}"
            )
        if self.max_vector_results <= 0:
            raise ConfigurationError(
                f"max_vector_results must be positive, got {self.max_vector_results}"
            )


@dataclass(frozen=True, slots=True)
class ReflectionConfig:
    """Configuration for reflection (L2 fact extraction)."""

    enabled: bool = True
    """Whether reflection is enabled."""

    background: bool = True
    """Run reflection asynchronously after episode closure.
    When True, episode closure returns immediately and reflection runs
    as a background task. Use ``wait_pending()`` to ensure completion.
    Set to False to block until reflection finishes."""

    min_episode_turns: int = 2
    """Minimum turns in episode to trigger reflection."""

    max_facts_per_episode: int = 10
    """Maximum facts to extract per episode."""

    min_confidence: float = 0.7
    """Minimum confidence score for extracted facts."""

    consolidation_similarity_threshold: float = 0.15
    """Minimum embedding similarity for a prior fact to be included in consolidation scope.
    Only applies when the number of active facts exceeds consolidation_max_unscoped_facts."""

    consolidation_max_unscoped_facts: int = 100
    """Skip similarity-based scoping when the active fact count is at or below this limit.
    All facts are sent to the LLM for consolidation, ensuring none are missed.
    Defaults to 100, matching max_active_facts, so scoping is effectively
    disabled unless max_active_facts is raised above this value."""

    max_active_facts: int = 100
    """Maximum number of active (non-superseded) facts per session.
    When exceeded, lowest-confidence facts are archived after reflection."""

    dedup_similarity_threshold: float = 0.90
    """Cosine similarity above which a new fact is considered a duplicate of an existing active fact.
    Set to 1.0 to disable deduplication."""


@dataclass(slots=True)
class GleanrConfig:
    """Main configuration for Gleanr.

    Controls all aspects of context management behavior.
    """

    # Marker settings
    auto_detect_markers: bool = True
    """Whether to auto-detect markers from content patterns."""

    marker_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MARKER_WEIGHTS))
    """Weights for marker types in scoring."""

    # Episode settings
    episode_boundary: EpisodeBoundaryConfig = field(default_factory=EpisodeBoundaryConfig)
    """Episode boundary detection configuration."""

    # Recall settings
    recall: RecallConfig = field(default_factory=RecallConfig)
    """Recall behavior configuration."""

    # Reflection settings
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    """Reflection (L2) configuration."""

    # Cache settings
    cache: CacheConfig = field(default_factory=CacheConfig)
    """Cache layer configuration."""

    # Misc settings
    max_content_length: int = 100_000
    """Maximum content length per turn (characters)."""

    def validate(self) -> None:
        """Validate the entire configuration.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate marker weights
        for marker_type in MarkerType:
            if marker_type.value not in self.marker_weights:
                self.marker_weights[marker_type.value] = DEFAULT_MARKER_WEIGHTS.get(
                    marker_type.value, 0.2
                )

        for marker, weight in self.marker_weights.items():
            if weight < 0:
                raise ConfigurationError(
                    f"Marker weight must be non-negative, got {weight} for {marker}"
                )

        # Validate sub-configs
        self.recall.validate()

        # Validate content length
        if self.max_content_length <= 0:
            raise ConfigurationError(
                f"max_content_length must be positive, got {self.max_content_length}"
            )

    def get_marker_weight(self, marker: str) -> float:
        """Get weight for a marker.

        Args:
            marker: Marker string

        Returns:
            Weight value (uses default for custom markers)
        """
        if marker in self.marker_weights:
            return self.marker_weights[marker]
        # Default weight for custom markers
        return 0.2

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "auto_detect_markers": self.auto_detect_markers,
            "marker_weights": self.marker_weights,
            "episode_boundary": {
                "max_turns": self.episode_boundary.max_turns,
                "max_time_gap_seconds": self.episode_boundary.max_time_gap_seconds,
                "close_on_tool_result": self.episode_boundary.close_on_tool_result,
                "close_on_patterns": list(self.episode_boundary.close_on_patterns),
            },
            "recall": {
                "default_token_budget": self.recall.default_token_budget,
                "current_episode_budget_pct": self.recall.current_episode_budget_pct,
                "max_vector_results": self.recall.max_vector_results,
                "min_relevance_threshold": self.recall.min_relevance_threshold,
                "max_fact_candidates": self.recall.max_fact_candidates,
                "facts_only_recall": self.recall.facts_only_recall,
            },
            "reflection": {
                "enabled": self.reflection.enabled,
                "background": self.reflection.background,
                "min_episode_turns": self.reflection.min_episode_turns,
                "max_facts_per_episode": self.reflection.max_facts_per_episode,
                "min_confidence": self.reflection.min_confidence,
                "consolidation_similarity_threshold": self.reflection.consolidation_similarity_threshold,
                "consolidation_max_unscoped_facts": self.reflection.consolidation_max_unscoped_facts,
                "max_active_facts": self.reflection.max_active_facts,
                "dedup_similarity_threshold": self.reflection.dedup_similarity_threshold,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "max_turns": self.cache.max_turns,
                "max_episodes": self.cache.max_episodes,
                "max_embeddings": self.cache.max_embeddings,
                "max_facts": self.cache.max_facts,
                "ttl_seconds": self.cache.ttl_seconds,
            },
            "max_content_length": self.max_content_length,
        }


def create_config(**kwargs: Any) -> GleanrConfig:
    """Create an GleanrConfig with validation.

    Args:
        **kwargs: Configuration options

    Returns:
        Validated GleanrConfig

    Raises:
        ConfigurationError: If configuration is invalid
    """
    config = GleanrConfig(**kwargs)
    config.validate()
    return config
