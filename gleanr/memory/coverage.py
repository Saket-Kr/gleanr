"""Coverage validation for fact consolidation.

After the LLM returns consolidation actions, this module checks that
every prior fact is accounted for — either by an explicit source_fact_id
reference or by keyword overlap with an action's content.

Validation is advisory (logs warnings, never raises), since imperfect
consolidation is better than no consolidation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gleanr.models import Fact
    from gleanr.models.consolidation import ConsolidationAction

logger = logging.getLogger(__name__)

# Common English stop words to exclude from keyword extraction.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "has", "have", "been",
    "from", "with", "they", "this", "that", "will", "would", "there",
    "their", "what", "about", "which", "when", "make", "like", "been",
    "could", "into", "than", "its", "over", "such", "after", "also",
    "did", "some", "then", "them", "each", "does", "how", "may",
    "much", "should", "these", "just", "use", "used", "using",
})


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text.

    Lowercases, strips stop words, and keeps only words with 3+ chars.
    """
    words = text.lower().split()
    return {
        w.strip(".,;:!?\"'()[]{}") for w in words
        if len(w) >= 3 and w.lower().strip(".,;:!?\"'()[]{}") not in _STOP_WORDS
    }


def validate_coverage(
    prior_facts: list["Fact"],
    actions: list["ConsolidationAction"],
) -> list[str]:
    """Check that every prior fact is covered by at least one action.

    A fact is considered covered if:
      1. Its ID appears as source_fact_id in any action, OR
      2. At least 50% of its keywords appear in some action's content.

    Returns:
        List of warning strings for uncovered facts. Empty if all covered.
    """
    if not prior_facts:
        return []

    # Build set of all referenced source_fact_ids
    referenced_ids: set[str | None] = {a.source_fact_id for a in actions}

    # Build combined keyword set from all action contents
    all_action_keywords: set[str] = set()
    for action in actions:
        all_action_keywords.update(extract_keywords(action.content))

    warnings: list[str] = []
    for fact in prior_facts:
        # Check 1: explicit ID reference
        if fact.id in referenced_ids:
            continue

        # Check 2: keyword overlap
        fact_keywords = extract_keywords(fact.content)
        if not fact_keywords:
            continue

        overlap = fact_keywords & all_action_keywords
        coverage_ratio = len(overlap) / len(fact_keywords)

        if coverage_ratio >= 0.5:
            continue

        # For short facts (<=3 keywords), a single keyword match is sufficient
        if len(fact_keywords) <= 3 and len(overlap) >= 1:
            continue

        # Substring fallback: catches morphological variants
        # (e.g., "postgres" in "postgresql", "python" in "python3")
        if any(
            fk in ak or ak in fk
            for fk in fact_keywords
            for ak in all_action_keywords
        ):
            continue

        warnings.append(
            f"Fact {fact.id} may not be covered by consolidation "
            f"(keyword overlap {coverage_ratio:.0%}): {fact.content[:80]}"
        )

    if warnings:
        for warning in warnings:
            logger.warning(warning)

    return warnings
