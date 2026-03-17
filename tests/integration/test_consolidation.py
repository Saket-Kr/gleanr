"""Integration tests for the full consolidation cycle.

These tests verify the end-to-end flow: episode 1 extracts facts,
episode 2 consolidates them, and recall only returns active facts.

Uses explicit close_episode() calls for predictable episode boundaries,
since automatic boundary detection (max_turns, patterns) has subtle
timing: episodes close BEFORE the triggering turn, not after.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from gleanr import Gleanr, GleanrConfig
from gleanr.core.config import ReflectionConfig
from gleanr.models import Fact, MarkerType
from gleanr.models.consolidation import ConsolidationAction, ConsolidationActionType
from gleanr.providers.base import NullEmbedder
from gleanr.storage.memory import InMemoryBackend
from gleanr.utils import generate_fact_id


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubConsolidatingReflector:
    """Reflector that returns scripted facts/actions for integration tests.

    Call 1 (no prior facts via legacy path) → returns scripted facts.
    Call 2+ (prior facts via consolidation)  → returns scripted actions.
    """

    def __init__(
        self,
        legacy_facts: list[Fact],
        consolidation_actions: list[ConsolidationAction],
    ) -> None:
        self._legacy_facts = legacy_facts
        self._consolidation_actions = consolidation_actions
        self.reflect_count = 0
        self.consolidation_count = 0

    async def reflect(self, episode: Any, turns: list[Any]) -> list[Fact]:
        self.reflect_count += 1
        return list(self._legacy_facts)

    async def reflect_with_consolidation(
        self,
        episode: Any,
        turns: list[Any],
        prior_facts: list[Any],
    ) -> list[ConsolidationAction]:
        self.consolidation_count += 1
        return list(self._consolidation_actions)


class StubLegacyReflector:
    """Reflector without consolidation support."""

    def __init__(self, facts_per_call: list[list[Fact]]) -> None:
        self._facts_per_call = facts_per_call
        self._call_count = 0

    async def reflect(self, episode: Any, turns: list[Any]) -> list[Fact]:
        idx = min(self._call_count, len(self._facts_per_call) - 1)
        self._call_count += 1
        return list(self._facts_per_call[idx])


def _reflection_config() -> ReflectionConfig:
    return ReflectionConfig(
        enabled=True,
        background=False,
        min_episode_turns=2,
        min_confidence=0.7,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullConsolidationCycle:
    """Two-episode consolidation cycle."""

    @pytest.mark.asyncio
    async def test_two_episode_consolidation(self) -> None:
        """Episode 1 extracts facts, episode 2 consolidates (update + add)."""
        storage = InMemoryBackend()

        fact_ep1 = Fact(
            id="fact_ep1_db",
            session_id="test",
            episode_id="ep_placeholder",
            content="Module A uses PostgreSQL",
            created_at=datetime.utcnow(),
            fact_type=MarkerType.DECISION.value,
            confidence=0.9,
        )

        actions_ep2 = [
            ConsolidationAction(
                action=ConsolidationActionType.UPDATE,
                content="Module A uses MySQL (changed from PostgreSQL)",
                source_fact_id="fact_ep1_db",
                fact_type=MarkerType.DECISION.value,
                confidence=0.95,
                reason="database switched",
            ),
            ConsolidationAction(
                action=ConsolidationActionType.ADD,
                content="All API endpoints require authentication",
                fact_type=MarkerType.CONSTRAINT.value,
                confidence=0.9,
            ),
        ]

        reflector = StubConsolidatingReflector(
            legacy_facts=[fact_ep1],
            consolidation_actions=actions_ep2,
        )

        config = GleanrConfig(reflection=_reflection_config())

        gleanr = Gleanr(
            session_id="test",
            storage=storage,
            embedder=NullEmbedder(dimension=4),
            reflector=reflector,
            config=config,
        )
        await gleanr.initialize()

        # --- Episode 1: setup ---
        await gleanr.ingest("user", "Let's set up Module A")
        await gleanr.ingest("assistant", "I'll use PostgreSQL for the database")
        await gleanr.close_episode()

        # Verify episode 1 facts
        assert reflector.reflect_count == 1
        all_facts = await storage.get_facts_by_session("test")
        assert len(all_facts) == 1
        assert all_facts[0].id == "fact_ep1_db"

        # --- Episode 2: consolidation ---
        await gleanr.ingest("user", "Switch Module A to MySQL and add auth")
        await gleanr.ingest("assistant", "Updated database and added auth requirement")
        await gleanr.close_episode()

        # Verify consolidation
        assert reflector.consolidation_count == 1

        all_facts = await storage.get_facts_by_session("test")
        # 1 original (superseded) + 2 new (MySQL + auth)
        assert len(all_facts) == 3

        active_facts = await storage.get_active_facts_by_session("test")
        assert len(active_facts) == 2

        active_contents = {f.content for f in active_facts}
        assert "Module A uses MySQL (changed from PostgreSQL)" in active_contents
        assert "All API endpoints require authentication" in active_contents

        # Old fact should be superseded
        old_fact = next(f for f in all_facts if f.id == "fact_ep1_db")
        assert old_fact.superseded_by is not None

        await gleanr.close()

    @pytest.mark.asyncio
    async def test_remove_in_consolidation(self) -> None:
        """Episode 2 removes a fact via REMOVE action and adds replacement."""
        storage = InMemoryBackend()

        fact_ep1 = Fact(
            id="fact_to_remove",
            session_id="test",
            episode_id="ep_placeholder",
            content="Use dark mode by default",
            created_at=datetime.utcnow(),
            fact_type=MarkerType.DECISION.value,
            confidence=0.9,
        )

        actions_ep2 = [
            ConsolidationAction(
                action=ConsolidationActionType.REMOVE,
                content="Use dark mode by default",
                source_fact_id="fact_to_remove",
                reason="user wants light mode instead",
            ),
            ConsolidationAction(
                action=ConsolidationActionType.ADD,
                content="Use light mode by default",
                fact_type=MarkerType.DECISION.value,
                confidence=0.9,
            ),
        ]

        reflector = StubConsolidatingReflector(
            legacy_facts=[fact_ep1],
            consolidation_actions=actions_ep2,
        )

        config = GleanrConfig(reflection=_reflection_config())

        gleanr = Gleanr(
            session_id="test",
            storage=storage,
            embedder=NullEmbedder(dimension=4),
            reflector=reflector,
            config=config,
        )
        await gleanr.initialize()

        # Episode 1
        await gleanr.ingest("user", "Use dark mode")
        await gleanr.ingest("assistant", "OK, dark mode it is")
        await gleanr.close_episode()

        # Episode 2
        await gleanr.ingest("user", "Actually, switch to light mode")
        await gleanr.ingest("assistant", "Switched to light mode")
        await gleanr.close_episode()

        active = await storage.get_active_facts_by_session("test")
        assert len(active) == 1
        assert "light mode" in active[0].content

        removed = next(
            f
            for f in await storage.get_facts_by_session("test")
            if f.id == "fact_to_remove"
        )
        assert removed.superseded_by is not None
        assert removed.superseded_by.startswith("removed_by_")

        await gleanr.close()

    @pytest.mark.asyncio
    async def test_keep_action_preserves_facts(self) -> None:
        """KEEP actions leave existing facts untouched."""
        storage = InMemoryBackend()

        fact_ep1 = Fact(
            id="fact_keep",
            session_id="test",
            episode_id="ep_placeholder",
            content="API uses REST",
            created_at=datetime.utcnow(),
            confidence=0.9,
        )

        actions_ep2 = [
            ConsolidationAction(
                action=ConsolidationActionType.KEEP,
                content="API uses REST",
                source_fact_id="fact_keep",
            ),
        ]

        reflector = StubConsolidatingReflector(
            legacy_facts=[fact_ep1],
            consolidation_actions=actions_ep2,
        )

        config = GleanrConfig(reflection=_reflection_config())

        gleanr = Gleanr(
            session_id="test",
            storage=storage,
            embedder=NullEmbedder(dimension=4),
            reflector=reflector,
            config=config,
        )
        await gleanr.initialize()

        await gleanr.ingest("user", "Set up REST API")
        await gleanr.ingest("assistant", "REST API configured")
        await gleanr.close_episode()

        await gleanr.ingest("user", "Any other questions?")
        await gleanr.ingest("assistant", "No changes needed")
        await gleanr.close_episode()

        # Fact stays active, no supersession
        active = await storage.get_active_facts_by_session("test")
        assert len(active) == 1
        assert active[0].id == "fact_keep"
        assert active[0].superseded_by is None

        await gleanr.close()


class TestLegacyReflectorCompatibility:
    """Legacy reflectors work without consolidation."""

    @pytest.mark.asyncio
    async def test_legacy_reflector_two_episodes(self) -> None:
        """Legacy reflector appends facts without consolidation."""
        storage = InMemoryBackend()

        fact1 = Fact(
            id=generate_fact_id(),
            session_id="test",
            episode_id="ep_placeholder",
            content="Fact from episode 1",
            created_at=datetime.utcnow(),
            confidence=0.9,
        )
        fact2 = Fact(
            id=generate_fact_id(),
            session_id="test",
            episode_id="ep_placeholder",
            content="Fact from episode 2",
            created_at=datetime.utcnow(),
            confidence=0.9,
        )

        reflector = StubLegacyReflector(facts_per_call=[[fact1], [fact2]])

        config = GleanrConfig(reflection=_reflection_config())

        gleanr = Gleanr(
            session_id="test",
            storage=storage,
            embedder=NullEmbedder(dimension=4),
            reflector=reflector,
            config=config,
        )
        await gleanr.initialize()

        # Episode 1
        await gleanr.ingest("user", "First message")
        await gleanr.ingest("assistant", "First response")
        await gleanr.close_episode()

        # Episode 2
        await gleanr.ingest("user", "Second message")
        await gleanr.ingest("assistant", "Second response")
        await gleanr.close_episode()

        # Both facts exist, both active (no supersession)
        all_facts = await storage.get_facts_by_session("test")
        assert len(all_facts) == 2

        active = await storage.get_active_facts_by_session("test")
        assert len(active) == 2

        await gleanr.close()
