"""Context recall pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gleanr.models import ContextItem, Role, ScoredCandidate, Turn
from gleanr.utils import TokenCounter, calculate_marker_boost

if TYPE_CHECKING:
    from gleanr.core.config import GleanrConfig
    from gleanr.memory.episode_manager import EpisodeManager
    from gleanr.providers import Embedder
    from gleanr.storage import StorageBackend

logger = logging.getLogger(__name__)


class RecallPipeline:
    """Pipeline for recalling relevant context.

    Implements the recall algorithm:
    1. Embed query
    2. Gather fact candidates (L2 — maintained, current-truth context)
    3. If no facts exist, fall back to marked turns and vector search on past turns
    4. Always include current episode turns (working context)
    5. Score candidates (relevance + marker boost, NO recency)
    6. Budget allocation and assembly
    """

    def __init__(
        self,
        session_id: str,
        storage: StorageBackend,
        embedder: Embedder,
        token_counter: TokenCounter,
        episode_manager: EpisodeManager,
        config: GleanrConfig,
    ) -> None:
        self._session_id = session_id
        self._storage = storage
        self._embedder = embedder
        self._token_counter = token_counter
        self._episode_manager = episode_manager
        self._config = config

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
            token_budget: Maximum tokens in result
            include_current_episode: Whether to include current episode turns
            min_relevance: Minimum relevance score to include

        Returns:
            Ordered list of context items within budget
        """
        if token_budget is None:
            token_budget = self._config.recall.default_token_budget

        # Step 1: Embed query
        query_embedding = await self._embed_query(query)

        # Step 2: Gather candidates
        current_episode_candidates = []
        if include_current_episode:
            current_episode_candidates = await self._get_current_episode_candidates()

        fact_candidates = await self._get_fact_candidates(query_embedding)

        # When active facts exist, they are the maintained current-truth
        # representation of past episodes. Raw turns from past episodes can
        # carry stale information that contradicts updated facts, so we skip
        # vector search and marked-turn retrieval. When no facts exist
        # (reflection disabled, first episode, or legacy mode), fall back to
        # turn-based recall.
        use_facts_only = self._config.recall.facts_only_recall and bool(fact_candidates)

        marked_candidates: list[ScoredCandidate] = []
        vector_candidates: list[ScoredCandidate] = []

        if not use_facts_only:
            marked_candidates = await self._get_marked_candidates(query_embedding)
            vector_candidates = await self._get_vector_candidates(query_embedding, min_relevance)

        # Step 3: Deduplicate (current episode may overlap with vector results)
        current_ids = {c.id for c in current_episode_candidates}
        marked_ids = {c.id for c in marked_candidates}

        unique_vector_candidates = [
            c for c in vector_candidates if c.id not in current_ids and c.id not in marked_ids
        ]

        # Step 4: Budget allocation
        context_items = self._allocate_budget(
            token_budget=token_budget,
            current_episode=current_episode_candidates,
            marked=marked_candidates,
            facts=fact_candidates,
            vectors=unique_vector_candidates,
        )

        return context_items

    async def _embed_query(self, query: str) -> list[float]:
        """Embed the query for vector search."""
        embeddings = await self._embedder.embed([query])
        return embeddings[0] if embeddings else []

    async def _get_current_episode_candidates(self) -> list[ScoredCandidate]:
        """Get candidates from current episode (always included)."""
        turns = await self._episode_manager.get_current_episode_turns()
        return [self._turn_to_candidate(turn, relevance=1.0) for turn in turns]

    async def _get_marked_candidates(
        self,
        query_embedding: list[float],
    ) -> list[ScoredCandidate]:
        """Get candidates from marked turns in past episodes."""
        current_episode_id = self._episode_manager.current_episode_id
        marked_turns = await self._storage.get_marked_turns(
            self._session_id,
            exclude_episode_id=current_episode_id,
        )

        candidates = []
        for turn in marked_turns:
            # Get relevance from embedding if available
            relevance = 0.5  # Default for turns without embeddings
            if turn.embedding_id and query_embedding:
                embedding = await self._storage.get_embedding(turn.embedding_id)
                if embedding:
                    relevance = self._cosine_similarity(query_embedding, embedding)

            candidates.append(self._turn_to_candidate(turn, relevance))

        # Sort by final score
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates

    async def _get_fact_candidates(
        self,
        query_embedding: list[float],
    ) -> list[ScoredCandidate]:
        """Get candidates from L2 facts.

        Facts below ``min_relevance_threshold`` are excluded (except those
        without embeddings, which default to 0.5 relevance).
        """
        facts = await self._storage.get_active_facts_by_session(self._session_id)
        min_relevance = self._config.recall.min_relevance_threshold

        candidates = []
        for fact in facts:
            # Get relevance from embedding if available
            relevance = 0.5
            if fact.embedding_id and query_embedding:
                embedding = await self._storage.get_embedding(fact.embedding_id)
                if embedding:
                    relevance = self._cosine_similarity(query_embedding, embedding)

            if relevance < min_relevance:
                continue

            # Facts get a boost based on their type (maps to markers)
            marker_boost = self._config.get_marker_weight(fact.fact_type)

            candidates.append(
                ScoredCandidate(
                    id=fact.id,
                    content=fact.content,
                    role=Role.ASSISTANT,  # Facts are agent-derived
                    source="fact",
                    relevance_score=relevance,
                    marker_boost=marker_boost,
                    final_score=relevance + marker_boost,
                    token_count=fact.token_count,
                    metadata=fact.metadata,
                    markers=(fact.fact_type,),
                    timestamp=fact.created_at,
                    episode_id=fact.episode_id,
                )
            )

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[: self._config.recall.max_fact_candidates]

    async def _get_vector_candidates(
        self,
        query_embedding: list[float],
        min_relevance: float,
    ) -> list[ScoredCandidate]:
        """Get candidates from vector similarity search."""
        if not query_embedding:
            return []

        results = await self._storage.vector_search(
            query_embedding,
            k=self._config.recall.max_vector_results,
            filter={"session_id": self._session_id, "type": "turn"},
        )

        candidates = []
        for result in results:
            if result.score < min_relevance:
                continue

            # Get the turn
            turn_id = result.metadata.get("turn_id")
            if not turn_id:
                continue

            turn = await self._storage.get_turn(turn_id)
            if not turn:
                continue

            candidates.append(self._turn_to_candidate(turn, result.score))

        return candidates

    def _turn_to_candidate(self, turn: Turn, relevance: float) -> ScoredCandidate:
        """Convert a turn to a scored candidate."""
        marker_boost = calculate_marker_boost(turn.markers, self._config.marker_weights)

        return ScoredCandidate(
            id=turn.id,
            content=turn.content,
            role=turn.role,
            source="turn",
            relevance_score=relevance,
            marker_boost=marker_boost,
            final_score=relevance + marker_boost,
            token_count=turn.token_count,
            metadata=turn.metadata,
            markers=tuple(turn.markers),
            timestamp=turn.created_at,
            episode_id=turn.episode_id,
        )

    def _allocate_budget(
        self,
        token_budget: int,
        current_episode: list[ScoredCandidate],
        marked: list[ScoredCandidate],
        facts: list[ScoredCandidate],
        vectors: list[ScoredCandidate],
    ) -> list[ContextItem]:
        """Allocate token budget across candidate sources.

        Priority:
        1. Current episode (reserved budget percentage)
        2. Marked turns (by score)
        3. Facts (by score, compact and high-signal)
        4. Vector search results (by score, fills remaining budget)
        """
        result: list[ContextItem] = []
        remaining_budget = token_budget

        # Step 1: Reserve budget for current episode
        current_budget = int(token_budget * self._config.recall.current_episode_budget_pct)
        current_used = 0

        # Handle current episode overflow
        total_current_tokens = sum(c.token_count for c in current_episode)
        if total_current_tokens > current_budget:
            # Keep marked turns and most recent turns
            current_episode = self._handle_episode_overflow(current_episode, current_budget)

        # Add current episode turns (chronological order)
        for candidate in current_episode:
            if current_used + candidate.token_count <= current_budget:
                result.append(self._candidate_to_context_item(candidate))
                current_used += candidate.token_count

        remaining_budget -= current_used

        # Step 2: Add marked turns by score
        marked_used = 0
        for candidate in marked:
            if marked_used + candidate.token_count <= remaining_budget:
                result.append(self._candidate_to_context_item(candidate))
                marked_used += candidate.token_count
            else:
                # Log warning about marked turns overflow
                logger.debug(f"Marked turn {candidate.id} excluded due to budget constraints")

        remaining_budget -= marked_used

        # Step 3: Add facts first (compact, high-signal), then vector turns
        for candidate in facts:
            if remaining_budget <= 0:
                break
            if candidate.token_count <= remaining_budget:
                result.append(self._candidate_to_context_item(candidate))
                remaining_budget -= candidate.token_count

        # Step 4: Fill remaining budget with vector search results
        for candidate in vectors:
            if remaining_budget <= 0:
                break
            if candidate.token_count <= remaining_budget:
                result.append(self._candidate_to_context_item(candidate))
                remaining_budget -= candidate.token_count

        return result

    def _handle_episode_overflow(
        self,
        candidates: list[ScoredCandidate],
        budget: int,
    ) -> list[ScoredCandidate]:
        """Handle case where current episode exceeds budget.

        Strategy: Keep marked turns + most recent turns within budget.
        """
        # Separate marked and unmarked
        marked = [c for c in candidates if c.markers]
        unmarked = [c for c in candidates if not c.markers]

        result = []
        used = 0

        # First, include all marked turns (they're important)
        for c in marked:
            if used + c.token_count <= budget:
                result.append(c)
                used += c.token_count

        # Then fill with most recent unmarked turns
        # (unmarked is already in chronological order, take from end)
        for c in reversed(unmarked):
            if used + c.token_count <= budget:
                result.insert(0, c)  # Keep chronological order
                used += c.token_count

        # Re-sort by original position
        result.sort(key=lambda c: candidates.index(c))
        return result

    def _candidate_to_context_item(self, candidate: ScoredCandidate) -> ContextItem:
        """Convert scored candidate to context item."""
        return ContextItem(
            id=candidate.id,
            content=candidate.content,
            role=candidate.role,
            source=candidate.source,
            score=candidate.final_score,
            token_count=candidate.token_count,
            metadata=candidate.metadata,
            markers=candidate.markers,
            timestamp=candidate.timestamp,
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
