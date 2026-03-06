"""Reflection runner for L2 fact extraction with consolidation support."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from gleanr.errors import ReflectionError
from gleanr.memory.coverage import validate_coverage
from gleanr.models.consolidation import ConsolidationAction, ConsolidationActionType
from gleanr.utils import generate_embedding_id, generate_fact_id
from gleanr.utils.vectors import cosine_similarity

if TYPE_CHECKING:
    from gleanr.core.config import GleanrConfig
    from gleanr.models import Episode, Fact, Turn
    from gleanr.providers import Embedder, Reflector
    from gleanr.storage import StorageBackend
    from gleanr.utils import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class ReflectionTrace:
    """Captures a complete trace of a single reflection call.

    Emitted by ReflectionRunner when tracing is enabled, providing
    full visibility into what the reflector received and produced.
    """

    episode_id: str
    mode: str  # "legacy" or "consolidation"
    input_turn_count: int
    input_turns: list[dict[str, str]] = field(default_factory=list)
    prior_facts: list[dict[str, Any]] | None = None
    scoped_fact_count: int | None = None
    raw_actions: list[dict[str, Any]] | None = None
    raw_facts: list[dict[str, Any]] | None = None
    saved_facts: list[dict[str, Any]] = field(default_factory=list)
    superseded_facts: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize trace to a dictionary."""
        return {
            "episode_id": self.episode_id,
            "mode": self.mode,
            "input_turn_count": self.input_turn_count,
            "input_turns": self.input_turns,
            "prior_facts": self.prior_facts,
            "scoped_fact_count": self.scoped_fact_count,
            "raw_actions": self.raw_actions,
            "raw_facts": self.raw_facts,
            "saved_facts": self.saved_facts,
            "superseded_facts": self.superseded_facts,
            "elapsed_ms": self.elapsed_ms,
        }


#: Callback type for receiving reflection traces.
ReflectionTraceCallback = Callable[["ReflectionTrace"], None]


class ReflectionRunner:
    """Runs reflection on closed episodes to extract L2 facts.

    Supports two paths:
    - **Legacy**: Reflector implements only ``Reflector`` protocol.
      Facts are extracted in isolation per episode.
    - **Consolidation**: Reflector also implements ``ConsolidatingReflector``.
      Prior active facts are sent alongside new episode turns; the
      reflector returns actions (keep/update/add/remove) that merge,
      update, or remove facts.

    The runner:
    1. Takes a closed episode and its turns
    2. Fetches prior active facts (if consolidation is supported)
    3. Calls the reflector (consolidation or legacy)
    4. Validates coverage of prior facts
    5. Applies consolidation actions (supersede/add/remove)
    6. Generates embeddings and saves facts to storage
    """

    def __init__(
        self,
        session_id: str,
        storage: "StorageBackend",
        reflector: "Reflector",
        embedder: "Embedder",
        token_counter: "TokenCounter",
        config: "GleanrConfig",
    ) -> None:
        self._session_id = session_id
        self._storage = storage
        self._reflector = reflector
        self._embedder = embedder
        self._token_counter = token_counter
        self._config = config
        self._pending_tasks: list[asyncio.Task[list["Fact"]]] = []
        self._carried_turns: list["Turn"] = []
        self._trace_callback: ReflectionTraceCallback | None = None

    def set_trace_callback(self, callback: ReflectionTraceCallback | None) -> None:
        """Set a callback to receive reflection traces.

        When set, each reflection call emits a ``ReflectionTrace``
        with full details of inputs, outputs, and timing.

        Args:
            callback: Function to call with each trace, or None to disable.
        """
        self._trace_callback = callback

    # ------------------------------------------------------------------
    # Public API (unchanged)
    # ------------------------------------------------------------------

    async def reflect_episode(
        self,
        episode: "Episode",
        turns: list["Turn"],
        *,
        background: bool = False,
    ) -> list["Fact"]:
        """Run reflection on an episode.

        Args:
            episode: The closed episode
            turns: Turns in the episode
            background: If True, run in background and return immediately

        Returns:
            List of extracted/consolidated facts (empty if background=True)

        Raises:
            ReflectionError: If reflection fails (only if background=False)
        """
        if not self._config.reflection.enabled:
            return []

        # Combine any carried-over turns from short prior episodes
        combined_turns = self._carried_turns + turns

        if len(combined_turns) < self._config.reflection.min_episode_turns:
            logger.debug(
                "Episode %s has %d turns (min: %d); carrying forward",
                episode.id,
                len(combined_turns),
                self._config.reflection.min_episode_turns,
            )
            self._carried_turns = combined_turns
            return []

        # Threshold met — reflect on all accumulated turns
        self._carried_turns = []

        if background:
            task = asyncio.create_task(
                self._reflect_and_save(episode, combined_turns)
            )
            self._pending_tasks.append(task)
            self._pending_tasks = [t for t in self._pending_tasks if not t.done()]
            return []

        return await self._reflect_and_save(episode, combined_turns)

    async def wait_pending(self) -> None:
        """Wait for all pending background reflection tasks."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks = []

    def cancel_pending(self) -> None:
        """Cancel all pending background reflection tasks."""
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
        self._pending_tasks = []

    async def flush_carried_turns(self, episode: "Episode") -> list["Fact"]:
        """Force-reflect any buffered turns, regardless of min count.

        Called during session close to ensure no turns are lost.
        Returns an empty list if there are no carried turns.
        """
        if not self._carried_turns or not self._config.reflection.enabled:
            return []

        turns = self._carried_turns
        self._carried_turns = []

        logger.info(
            "Flushing %d carried turns for episode %s",
            len(turns),
            episode.id,
        )
        try:
            return await self._reflect_and_save(episode, turns)
        except Exception as e:
            logger.error("Failed to flush carried turns: %s", e)
            return []

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _reflect_and_save(
        self,
        episode: "Episode",
        turns: list["Turn"],
    ) -> list["Fact"]:
        """Dispatch to consolidation or legacy path."""
        start = time.perf_counter()
        trace = self._build_trace_header(episode, turns) if self._trace_callback else None

        try:
            if self._supports_consolidation():
                prior_facts = await self._storage.get_active_facts_by_session(
                    self._session_id
                )
                if prior_facts:
                    result = await self._consolidate_and_save(
                        episode, turns, prior_facts, trace
                    )
                    self._emit_trace(trace, start)
                    await self._enforce_active_fact_limit()
                    return result

            # First episode, no prior facts, or legacy reflector
            existing_facts = await self._storage.get_active_facts_by_session(
                self._session_id
            )
            result = await self._legacy_reflect_and_save(
                episode, turns, trace, existing_facts=existing_facts or None
            )
            self._emit_trace(trace, start)
            await self._enforce_active_fact_limit()
            return result

        except Exception as e:
            logger.error("Reflection failed for episode %s: %s", episode.id, e)
            raise ReflectionError(
                f"Reflection failed: {e}",
                episode_id=episode.id,
                cause=e,
            ) from e

    def _supports_consolidation(self) -> bool:
        """Check if the reflector supports consolidation."""
        return hasattr(self._reflector, "reflect_with_consolidation")

    # ------------------------------------------------------------------
    # Consolidation path
    # ------------------------------------------------------------------

    async def _consolidate_and_save(
        self,
        episode: "Episode",
        turns: list["Turn"],
        all_active_facts: list["Fact"],
        trace: ReflectionTrace | None = None,
    ) -> list["Fact"]:
        """Run consolidation: scope facts, call reflector, apply actions."""
        relevant_facts = await self._scope_relevant_facts(
            episode, turns, all_active_facts
        )

        if trace:
            trace.mode = "consolidation"
            trace.prior_facts = [
                {"id": f.id, "content": f.content, "fact_type": f.fact_type}
                for f in all_active_facts
            ]
            trace.scoped_fact_count = len(relevant_facts)

        actions = await self._reflector.reflect_with_consolidation(
            episode, turns, relevant_facts
        )

        if trace and actions:
            trace.raw_actions = [
                {
                    "action": a.action.value,
                    "content": a.content,
                    "fact_type": a.fact_type,
                    "confidence": a.confidence,
                    "source_fact_id": a.source_fact_id,
                    "reason": a.reason,
                }
                for a in actions
            ]

        if not actions:
            logger.warning(
                "Consolidation returned no actions for episode %s; "
                "falling back to legacy path",
                episode.id,
            )
            return await self._legacy_reflect_and_save(
                episode, turns, trace, existing_facts=all_active_facts
            )

        validate_coverage(relevant_facts, actions)

        return await self._apply_consolidation_actions(
            episode, actions, all_active_facts, trace
        )

    async def _scope_relevant_facts(
        self,
        episode: "Episode",
        turns: list["Turn"],
        prior_facts: list["Fact"],
    ) -> list["Fact"]:
        """Select prior facts relevant to this episode via embedding similarity.

        For small fact sets (at or below ``consolidation_max_unscoped_facts``),
        all facts are returned to avoid filtering out facts that need updating.

        For larger sets, embedding similarity is used to select a relevant
        subset, with conservative thresholds to minimize false exclusions.
        """
        if not prior_facts:
            return []

        max_unscoped = self._config.reflection.consolidation_max_unscoped_facts
        if len(prior_facts) <= max_unscoped:
            return prior_facts

        # Large fact set — use embedding similarity to scope
        episode_text = " ".join(t.content for t in turns)
        try:
            embeddings = await self._embedder.embed([episode_text])
        except Exception as e:
            logger.warning("Failed to embed episode for scoping: %s", e)
            return prior_facts

        if not embeddings:
            return prior_facts

        query_embedding = embeddings[0]

        # Detect NullEmbedder (zero vector) — include all facts
        if all(v == 0.0 for v in query_embedding):
            return prior_facts

        threshold = self._config.reflection.consolidation_similarity_threshold
        relevant: list["Fact"] = []

        for fact in prior_facts:
            if fact.embedding_id is None:
                relevant.append(fact)
                continue

            fact_embedding = await self._storage.get_embedding(fact.embedding_id)
            if fact_embedding is None:
                relevant.append(fact)
                continue

            similarity = cosine_similarity(query_embedding, fact_embedding)
            if similarity >= threshold:
                relevant.append(fact)

        # If scoping removed everything, include all (conservative)
        if not relevant and prior_facts:
            logger.debug(
                "Scoping removed all %d facts; including all as fallback",
                len(prior_facts),
            )
            return prior_facts

        return relevant

    async def _apply_consolidation_actions(
        self,
        episode: "Episode",
        actions: list[ConsolidationAction],
        prior_facts: list["Fact"],
        trace: ReflectionTrace | None = None,
    ) -> list["Fact"]:
        """Process consolidation actions and persist changes.

        Returns the list of newly saved facts (ADD + UPDATE replacements).
        """
        from datetime import datetime

        from gleanr.models import Fact

        prior_by_id = {f.id: f for f in prior_facts}
        saved_facts: list["Fact"] = []
        superseded_facts: list[dict[str, Any]] = []

        for action in actions:
            if action.action == ConsolidationActionType.KEEP:
                # No changes needed — fact stays active
                continue

            elif action.action == ConsolidationActionType.ADD:
                if action.confidence < self._config.reflection.min_confidence:
                    continue

                new_fact = Fact(
                    id=generate_fact_id(),
                    session_id=self._session_id,
                    episode_id=episode.id,
                    content=action.content,
                    created_at=datetime.utcnow(),
                    fact_type=action.fact_type,
                    confidence=action.confidence,
                )

                if await self._is_duplicate(new_fact, prior_facts):
                    continue

                new_fact = await self._embed_and_save_fact(new_fact, episode)
                saved_facts.append(new_fact)

            elif action.action == ConsolidationActionType.UPDATE:
                old_fact = prior_by_id.get(action.source_fact_id or "")
                if old_fact is None:
                    logger.warning(
                        "UPDATE references unknown fact ID: %s",
                        action.source_fact_id,
                    )
                    continue

                if action.confidence < self._config.reflection.min_confidence:
                    continue

                # Create replacement fact
                new_fact = Fact(
                    id=generate_fact_id(),
                    session_id=self._session_id,
                    episode_id=episode.id,
                    content=action.content,
                    created_at=datetime.utcnow(),
                    fact_type=action.fact_type,
                    confidence=action.confidence,
                    supersedes=[old_fact.id],
                )
                new_fact = await self._embed_and_save_fact(new_fact, episode)
                saved_facts.append(new_fact)

                # Mark old fact as superseded
                superseded_old = replace(old_fact, superseded_by=new_fact.id)
                await self._storage.update_fact(superseded_old)
                superseded_facts.append(
                    {"id": old_fact.id, "content": old_fact.content, "superseded_by": new_fact.id}
                )

            elif action.action == ConsolidationActionType.REMOVE:
                old_fact = prior_by_id.get(action.source_fact_id or "")
                if old_fact is None:
                    logger.warning(
                        "REMOVE references unknown fact ID: %s",
                        action.source_fact_id,
                    )
                    continue

                removed_marker = f"removed_by_{episode.id}"
                superseded_old = replace(old_fact, superseded_by=removed_marker)
                await self._storage.update_fact(superseded_old)
                superseded_facts.append(
                    {"id": old_fact.id, "content": old_fact.content, "superseded_by": removed_marker}
                )

        if trace:
            trace.saved_facts = [
                {"id": f.id, "content": f.content, "fact_type": f.fact_type}
                for f in saved_facts
            ]
            trace.superseded_facts = superseded_facts

        logger.info(
            "Consolidation produced %d new/updated facts for episode %s",
            len(saved_facts),
            episode.id,
        )
        return saved_facts

    # ------------------------------------------------------------------
    # Legacy path (original behavior)
    # ------------------------------------------------------------------

    async def _legacy_reflect_and_save(
        self,
        episode: "Episode",
        turns: list["Turn"],
        trace: ReflectionTrace | None = None,
        *,
        existing_facts: list["Fact"] | None = None,
    ) -> list["Fact"]:
        """Original reflection path — extract facts in isolation.

        Args:
            episode: The closed episode.
            turns: Turns in the episode.
            trace: Optional observability trace.
            existing_facts: Active facts to dedup against. When provided,
                new facts that are semantic duplicates of existing ones
                are silently skipped.
        """
        if trace:
            trace.mode = "legacy"

        facts = await self._reflector.reflect(episode, turns)

        if trace:
            trace.raw_facts = [
                {"content": f.content, "fact_type": f.fact_type, "confidence": f.confidence}
                for f in facts
            ]

        saved_facts: list["Fact"] = []
        for fact in facts[: self._config.reflection.max_facts_per_episode]:
            if fact.confidence < self._config.reflection.min_confidence:
                continue

            if existing_facts and await self._is_duplicate(fact, existing_facts):
                continue

            fact = await self._embed_and_save_fact(fact, episode)
            saved_facts.append(fact)

        if trace:
            trace.saved_facts = [
                {"id": f.id, "content": f.content, "fact_type": f.fact_type}
                for f in saved_facts
            ]

        logger.info(
            "Reflection extracted %d facts from episode %s",
            len(saved_facts),
            episode.id,
        )
        return saved_facts

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_trace_header(
        self, episode: "Episode", turns: list["Turn"]
    ) -> ReflectionTrace:
        """Create a trace with input metadata populated."""
        return ReflectionTrace(
            episode_id=episode.id,
            mode="unknown",
            input_turn_count=len(turns),
            input_turns=[
                {"role": t.role.value, "content": t.content[:200]}
                for t in turns
            ],
        )

    def _emit_trace(
        self, trace: ReflectionTrace | None, start: float
    ) -> None:
        """Finalize and emit a reflection trace."""
        if trace is None or self._trace_callback is None:
            return
        trace.elapsed_ms = int((time.perf_counter() - start) * 1000)
        try:
            self._trace_callback(trace)
        except Exception:
            logger.warning("Trace callback raised an exception", exc_info=True)

    async def _enforce_active_fact_limit(self) -> None:
        """Archive lowest-confidence facts when the active count exceeds the limit.

        Archived facts are marked with ``superseded_by="archived_excess"``
        so they no longer appear in recall or consolidation queries.
        """
        max_facts = self._config.reflection.max_active_facts
        active_facts = await self._storage.get_active_facts_by_session(
            self._session_id
        )
        excess = len(active_facts) - max_facts
        if excess <= 0:
            return

        # Sort by confidence (ascending), then by creation date (oldest first)
        to_archive = sorted(
            active_facts,
            key=lambda f: (f.confidence, f.created_at),
        )[:excess]

        for fact in to_archive:
            archived = replace(fact, superseded_by="archived_excess")
            await self._storage.update_fact(archived)

        logger.info(
            "Archived %d excess facts for session %s (limit: %d)",
            len(to_archive),
            self._session_id,
            max_facts,
        )

    async def _is_duplicate(
        self,
        new_fact: "Fact",
        existing_facts: list["Fact"],
    ) -> bool:
        """Check if a new fact is a semantic duplicate of any existing fact.

        Uses embedding cosine similarity. Returns True if similarity
        exceeds ``dedup_similarity_threshold``.
        """
        threshold = self._config.reflection.dedup_similarity_threshold
        if threshold >= 1.0:
            return False  # Dedup disabled

        try:
            new_embeddings = await self._embedder.embed([new_fact.content])
        except Exception:
            return False  # Can't dedup without embedding

        if not new_embeddings:
            return False

        new_emb = new_embeddings[0]

        # Skip dedup if embedder returns zero vectors (NullEmbedder)
        if all(v == 0.0 for v in new_emb):
            return False

        for fact in existing_facts:
            if fact.embedding_id is None:
                continue
            fact_emb = await self._storage.get_embedding(fact.embedding_id)
            if fact_emb is None:
                continue
            sim = cosine_similarity(new_emb, fact_emb)
            if sim >= threshold:
                logger.info(
                    "Dedup: skipping ADD '%s' (%.3f similarity to fact %s: '%s')",
                    new_fact.content[:60],
                    sim,
                    fact.id[:12],
                    fact.content[:60],
                )
                return True

        return False

    async def _embed_and_save_fact(
        self,
        fact: "Fact",
        episode: "Episode",
    ) -> "Fact":
        """Count tokens, generate embedding, and save a fact to storage.

        Returns the fact (potentially mutated with embedding_id and token_count).
        """
        fact.token_count = self._token_counter.count(fact.content)

        try:
            embeddings = await self._embedder.embed([fact.content])
            if embeddings:
                emb_id = generate_embedding_id()
                await self._storage.save_embedding(
                    id=emb_id,
                    embedding=embeddings[0],
                    metadata={
                        "session_id": self._session_id,
                        "episode_id": episode.id,
                        "fact_id": fact.id,
                        "type": "fact",
                        "fact_type": fact.fact_type,
                    },
                )
                fact.embedding_id = emb_id
        except Exception as e:
            logger.warning("Failed to embed fact %s: %s", fact.id, e)

        await self._storage.save_fact(fact)
        return fact
