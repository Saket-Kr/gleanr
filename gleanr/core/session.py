"""Gleanr main session class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gleanr.core.config import GleanrConfig
from gleanr.memory import EpisodeManager, IngestionPipeline, RecallPipeline, ReflectionRunner
from gleanr.memory.reflection import ReflectionTraceCallback
from gleanr.models import ContextItem, Role, SessionStats
from gleanr.providers import Embedder, NullEmbedder, NullReflector, Reflector
from gleanr.storage import StorageBackend
from gleanr.utils import (
    HeuristicTokenCounter,
    TokenCounter,
    validate_content,
    validate_relevance_threshold,
    validate_session_id,
    validate_token_budget,
)

if TYPE_CHECKING:
    from gleanr.models import MarkerType


class Gleanr:
    """Session-scoped context manager for AI agents.

    Gleanr (Agent Context Management System) provides intelligent,
    token-budgeted context assembly for AI agent conversations.

    One Gleanr instance = one session.

    Example:
        >>> storage = InMemoryBackend()
        >>> embedder = HTTPEmbedder("https://api.openai.com/v1", api_key=key)
        >>> gleanr = Gleanr("session_123", storage, embedder)
        >>> await gleanr.initialize()
        >>>
        >>> await gleanr.ingest("user", "Hello, can you help me?")
        >>> await gleanr.ingest("assistant", "Of course! What do you need?")
        >>>
        >>> context = await gleanr.recall("user's question", token_budget=2000)
        >>> for item in context:
        ...     print(f"[{item.role}] {item.content}")
    """

    def __init__(
        self,
        session_id: str,
        storage: StorageBackend,
        embedder: Embedder | None = None,
        *,
        reflector: Reflector | None = None,
        token_counter: TokenCounter | None = None,
        config: GleanrConfig | None = None,
    ) -> None:
        """Initialize Gleanr for a session.

        Args:
            session_id: Unique session identifier
            storage: Storage backend for persistence
            embedder: Embedding provider (uses NullEmbedder if None)
            reflector: Optional reflector for L2 fact extraction
            token_counter: Token counter (uses heuristic if None)
            config: Configuration options (uses defaults if None)
        """
        self._session_id = validate_session_id(session_id)
        self._storage = storage
        self._embedder = embedder or NullEmbedder()
        self._reflector = reflector or NullReflector()
        self._token_counter = token_counter or HeuristicTokenCounter()
        self._config = config or GleanrConfig()

        # Validate config
        self._config.validate()

        # Internal components (initialized in initialize())
        self._episode_manager: EpisodeManager | None = None
        self._ingestion: IngestionPipeline | None = None
        self._recall: RecallPipeline | None = None
        self._reflection_runner: ReflectionRunner | None = None
        self._initialized = False
        self._closed = False
        self._trace_callback: ReflectionTraceCallback | None = None

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    @property
    def is_initialized(self) -> bool:
        """Check if Gleanr has been initialized."""
        return self._initialized

    @property
    def current_episode_id(self) -> str | None:
        """Get the current open episode ID."""
        if self._episode_manager:
            return self._episode_manager.current_episode_id
        return None

    def set_trace_callback(self, callback: ReflectionTraceCallback | None) -> None:
        """Set a callback to receive reflection traces.

        When set, each reflection call emits a ``ReflectionTrace``
        with full details of inputs, outputs, and timing.
        Can be called before or after ``initialize()``.

        Args:
            callback: Function to call with each trace, or None to disable.
        """
        self._trace_callback = callback
        if self._reflection_runner:
            self._reflection_runner.set_trace_callback(callback)

    async def initialize(self) -> None:
        """Initialize Gleanr and storage.

        Must be called before using other methods.
        Safe to call multiple times (idempotent).
        """
        if self._initialized:
            return

        # Initialize storage
        await self._storage.initialize()

        # Initialize episode manager
        self._episode_manager = EpisodeManager(
            self._session_id,
            self._storage,
            self._config,
        )
        await self._episode_manager.initialize()

        # Initialize reflection runner
        self._reflection_runner = ReflectionRunner(
            self._session_id,
            self._storage,
            self._reflector,
            self._embedder,
            self._token_counter,
            self._config,
        )
        if self._trace_callback:
            self._reflection_runner.set_trace_callback(self._trace_callback)

        # Wire episode close callback to trigger reflection
        self._episode_manager.set_on_episode_closed(self._handle_episode_closed)

        # Initialize pipelines
        self._ingestion = IngestionPipeline(
            self._session_id,
            self._storage,
            self._embedder,
            self._token_counter,
            self._episode_manager,
            self._config,
        )
        await self._ingestion.initialize()

        self._recall = RecallPipeline(
            self._session_id,
            self._storage,
            self._embedder,
            self._token_counter,
            self._episode_manager,
            self._config,
        )

        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure Gleanr is initialized."""
        if not self._initialized:
            raise RuntimeError("Gleanr not initialized. Call await gleanr.initialize() first.")
        if self._closed:
            raise RuntimeError("Gleanr has been closed.")

    async def ingest(
        self,
        role: str | Role,
        content: str,
        *,
        actor_id: str | None = None,
        markers: list[str | MarkerType] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Ingest a turn into memory.

        Args:
            role: Who produced this turn ("user", "assistant", "tool")
            content: The turn content
            actor_id: Optional identifier for the actor
            markers: Optional importance markers (decision, constraint, failure, goal, custom:*)
            metadata: Optional arbitrary metadata

        Returns:
            Turn ID

        Raises:
            ValidationError: If input is invalid
            ProviderError: If embedding fails
            RuntimeError: If Gleanr not initialized
        """
        self._ensure_initialized()
        assert self._ingestion is not None

        # Convert MarkerType enums to strings
        marker_strings: list[str] | None = None
        if markers:
            marker_strings = [m.value if hasattr(m, "value") else str(m) for m in markers]

        return await self._ingestion.ingest(
            role=role,
            content=content,
            actor_id=actor_id,
            markers=marker_strings,
            metadata=metadata,
        )

    async def recall(
        self,
        query: str,
        *,
        token_budget: int | None = None,
        include_current_episode: bool = True,
        min_relevance: float = 0.0,
    ) -> list[ContextItem]:
        """Recall relevant context for a query.

        Args:
            query: Query to find relevant context for
            token_budget: Maximum tokens to return (default from config)
            include_current_episode: Whether to include current episode turns
            min_relevance: Minimum relevance score (0-1) to include

        Returns:
            Ordered list of context items within token budget

        Raises:
            ValidationError: If input is invalid
            RuntimeError: If Gleanr not initialized
        """
        self._ensure_initialized()
        assert self._recall is not None

        # Validate inputs
        query = validate_content(query)
        if token_budget is not None:
            token_budget = validate_token_budget(token_budget)
        min_relevance = validate_relevance_threshold(min_relevance)

        return await self._recall.recall(
            query=query,
            token_budget=token_budget,
            include_current_episode=include_current_episode,
            min_relevance=min_relevance,
        )

    async def close_episode(
        self,
        reason: str = "manual",
    ) -> str | None:
        """Manually close the current episode.

        Triggers reflection via the episode close callback if enabled.

        Args:
            reason: Reason for closing

        Returns:
            Closed episode ID, or None if no open episode
        """
        self._ensure_initialized()
        assert self._episode_manager is not None

        # Reflection is triggered via the on_episode_closed callback
        # that was wired during initialize()
        return await self._episode_manager.close_current_episode(reason)

    async def _handle_episode_closed(self, episode_id: str) -> None:
        """Handle episode close event by running reflection.

        This callback is invoked by EpisodeManager whenever an episode closes,
        whether manually or automatically via boundary rules.

        Args:
            episode_id: The closed episode's ID
        """
        if not self._config.reflection.enabled:
            return

        if self._reflection_runner is None:
            return

        try:
            episode = await self._storage.get_episode(episode_id)
            if not episode:
                return

            turns = await self._storage.get_turns_by_episode(episode_id)

            # ReflectionRunner handles min_episode_turns check internally
            await self._reflection_runner.reflect_episode(
                episode,
                turns,
                background=self._config.reflection.background,
            )

        except Exception:
            # Reflection failures should not crash Gleanr
            # Log and continue
            pass

    async def get_session_stats(self) -> SessionStats:
        """Get statistics about the current session.

        Returns:
            SessionStats with counts and metadata
        """
        self._ensure_initialized()

        stats = await self._storage.get_session_stats(self._session_id)

        return SessionStats(
            session_id=stats["session_id"],
            total_turns=stats["total_turns"],
            total_episodes=stats["total_episodes"],
            total_facts=stats["total_facts"],
            open_episode_id=stats["open_episode_id"],
            open_episode_turn_count=stats["open_episode_turn_count"],
            total_tokens_ingested=stats["total_tokens_ingested"],
            created_at=stats["created_at"],
            last_activity_at=stats["last_activity_at"],
        )

    async def close(self) -> None:
        """Clean up resources.

        Closes current episode and flushes any pending writes.
        Safe to call multiple times.
        """
        if self._closed:
            return

        if self._initialized:
            # Close current episode (triggers reflection via callback)
            closed_ep_id = await self.close_episode(reason="session_close")

            # Flush any carried turns from short episodes
            if self._reflection_runner and closed_ep_id:
                episode = await self._storage.get_episode(closed_ep_id)
                if episode:
                    await self._reflection_runner.flush_carried_turns(episode)

            # Wait for any pending background reflection tasks
            if self._reflection_runner:
                await self._reflection_runner.wait_pending()

            # Close storage
            await self._storage.close()

        self._closed = True

    async def __aenter__(self) -> Gleanr:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
