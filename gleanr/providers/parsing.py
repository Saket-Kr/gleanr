"""Shared parsing utilities for reflection and consolidation responses."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from gleanr.models import Fact, MarkerType
from gleanr.models.consolidation import ConsolidationAction, ConsolidationActionType
from gleanr.utils import generate_fact_id

if TYPE_CHECKING:
    from gleanr.models import Episode, Turn

logger = logging.getLogger(__name__)

# Valid marker type values, computed once at module level.
_VALID_FACT_TYPES: frozenset[str] = frozenset(m.value for m in MarkerType)


def _extract_json(content: str) -> dict | None:
    """Extract the first JSON object from a string.

    Handles both clean JSON responses and responses where the model
    wraps JSON in markdown or extra text.

    Returns:
        Parsed dict, or None if no valid JSON found.
    """
    # Fast path: content is already valid JSON
    stripped = content.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Slow path: find first { and last } in the string
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _normalize_fact_type(raw_type: str) -> str:
    """Normalize a fact type string to a valid MarkerType value."""
    if raw_type in _VALID_FACT_TYPES:
        return raw_type
    return MarkerType.DECISION.value


def parse_reflection_facts(content: str, episode: "Episode") -> list[Fact]:
    """Parse standard reflection JSON into Fact objects.

    Expected format:
        {"facts": [{"content": "...", "type": "decision", "confidence": 0.9}, ...]}

    Returns an empty list on malformed input (never raises).
    """
    data = _extract_json(content)
    if data is None:
        return []

    facts_data = data.get("facts", [])
    if not isinstance(facts_data, list):
        return []

    facts: list[Fact] = []
    for item in facts_data:
        if not isinstance(item, dict):
            continue
        facts.append(
            Fact(
                id=generate_fact_id(),
                session_id=episode.session_id,
                episode_id=episode.id,
                content=item.get("content", ""),
                created_at=datetime.utcnow(),
                fact_type=_normalize_fact_type(item.get("type", "decision")),
                confidence=float(item.get("confidence", 0.8)),
            )
        )

    return facts


def parse_consolidation_actions(content: str) -> list[ConsolidationAction]:
    """Parse consolidation JSON into ConsolidationAction objects.

    Expected format:
        {"actions": [
            {"action": "keep|update|add|remove",
             "content": "...",
             "type": "decision",
             "confidence": 0.9,
             "source_fact_id": "fact_...",
             "reason": "..."},
            ...
        ]}

    Returns an empty list on malformed input (never raises).
    """
    data = _extract_json(content)
    if data is None:
        logger.warning("Consolidation response contained no valid JSON")
        return []

    actions_data = data.get("actions", [])
    if not isinstance(actions_data, list):
        logger.warning("Consolidation response 'actions' is not a list")
        return []

    actions: list[ConsolidationAction] = []
    for item in actions_data:
        if not isinstance(item, dict):
            continue

        raw_action = item.get("action", "")
        try:
            action_type = ConsolidationActionType(raw_action)
        except ValueError:
            logger.warning("Unknown consolidation action type: %s", raw_action)
            continue

        actions.append(
            ConsolidationAction(
                action=action_type,
                content=item.get("content", ""),
                fact_type=_normalize_fact_type(item.get("type", "decision")),
                confidence=float(item.get("confidence", 0.9)),
                source_fact_id=item.get("source_fact_id"),
                reason=item.get("reason", ""),
            )
        )

    return actions


REFLECTION_PROMPT = """You are extracting facts from a conversation episode for a memory system.

A "fact" is a single, atomic piece of information: one decision, one requirement, one parameter, one preference, or one constraint. If multiple details are discussed, extract each as a separate fact.

## Fact Types
- "decision": A choice or determination that was made
- "constraint": A limitation or rule that must be followed
- "goal": An objective or desired outcome
- "failure": Something that did not work or was rejected

## Episode Turns
{turns}

## Instructions
1. Extract up to {max_facts} facts from the episode above.
2. Each fact should capture ONE specific piece of information — not a summary of multiple things.
3. Include specific values, names, and parameters (e.g., "Aspect ratio is 16:9" not "An aspect ratio was chosen").
4. Include both user requests AND assistant confirmations/decisions.
5. If something was rejected or removed, record that as a fact too.

Respond ONLY with valid JSON, no other text:
{{"facts": [
  {{"content": "The database engine is PostgreSQL", "type": "decision", "confidence": 0.95}},
  {{"content": "All API endpoints require authentication", "type": "constraint", "confidence": 0.9}},
  {{"content": "The user wants to build a REST API for inventory management", "type": "goal", "confidence": 0.85}}
]}}"""


CONSOLIDATION_PROMPT = """You are maintaining a set of facts about an ongoing session. Your job is to keep these facts accurate and up-to-date based on new conversation turns.

## Existing Facts
{prior_facts}

## New Episode Turns
{turns}

## Instructions
Follow these three steps IN ORDER:

STEP 1 — Handle every existing fact listed above. For each one, output exactly one action:
- "keep": Fact is still accurate and unchanged. Include source_fact_id.
- "update": ANY detail in the fact has changed. Include source_fact_id, the corrected content, and reason.
- "remove": Fact is no longer true or was explicitly revoked. Include source_fact_id and reason.

STEP 2 — Check for CONTRADICTIONS among existing facts. If two facts contradict each other (e.g., one says "use PostgreSQL" and another says "use MySQL"), REMOVE the outdated one with reason "contradicts [other fact]".

STEP 3 — Check for NEW information. Read through the new turns again. For each specific detail that is NOT already covered by an existing fact, output:
- "add": New information. Include content, type, and confidence.

IMPORTANT RULES:
1. You MUST output one action for every existing fact listed above — do not skip any.
2. If a fact says "X" but the conversation now says "Y", that is an UPDATE — not a keep.
3. After handling all existing facts, you MUST add any new details from the turns that are not covered.
4. One fact = one atomic piece of information. Do not merge unrelated facts.
5. Do not silently drop information from existing facts when updating.
6. If two existing facts contradict each other, REMOVE the older/outdated one.

Example: If existing facts are "birds in the sky" and "daytime lighting", and the turns say "replace birds with fireflies, change to evening, add a river":
- UPDATE "birds in the sky" → "fireflies in the sky"
- UPDATE "daytime lighting" → "evening lighting"
- ADD "a river through the middle"

Respond ONLY with valid JSON, no other text:
{{"actions": [
  {{"action": "keep", "source_fact_id": "fact_abc", "content": "API uses REST architecture", "type": "decision", "confidence": 0.95}},
  {{"action": "update", "source_fact_id": "fact_def", "content": "Database engine is MySQL (changed from PostgreSQL)", "type": "decision", "confidence": 0.9, "reason": "user requested switch to MySQL"}},
  {{"action": "add", "content": "All responses must include pagination metadata", "type": "constraint", "confidence": 0.85}},
  {{"action": "remove", "source_fact_id": "fact_ghi", "content": "Use dark mode by default", "type": "decision", "confidence": 0.9, "reason": "contradicts fact_xyz which says light mode"}}
]}}"""


def format_prior_facts(facts: list["Fact"]) -> str:
    """Format prior facts for inclusion in the consolidation prompt."""
    lines: list[str] = []
    for fact in facts:
        lines.append(f"- [{fact.id}] ({fact.fact_type}) {fact.content}")
    return "\n".join(lines)


def format_turns(turns: list["Turn"]) -> str:
    """Format episode turns for inclusion in reflection prompts."""
    return "\n".join(f"[{t.role.value}]: {t.content}" for t in turns)
