# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2026-03-16

### Fixed

- **Stale turn recall**: When active facts exist, vector search on past
  episode turns and marked-turn retrieval are now skipped. Facts are the
  maintained, current-truth representation; raw turns from past episodes
  carried stale information that contradicted updated facts. Configurable
  via `facts_only_recall` (default: `True`). Falls back to turn-based recall
  when no facts exist (reflection disabled or first episode).
- **Consolidation scoping**: `consolidation_max_unscoped_facts` raised from
  50 to 100, matching `max_active_facts`. Similarity-based scoping
  effectively never triggers, preventing exclusion of facts that need
  updating in contradiction scenarios.

### Added

- **Fact candidate cap**: New `max_fact_candidates` config (default: 20)
  limits recall to the top-K facts by score after relevance filtering,
  reducing noise from weakly-related filler facts and improving precision.
- **Facts-only recall**: New `facts_only_recall` config (default: `True`)
  skips vector search on past turns when active facts exist, eliminating the
  primary source of staleness and low precision.
- **Background reflection**: New `ReflectionConfig.background` config
  (default: `True`) runs reflection asynchronously after episode closure.
  Eliminates ingest latency spikes (p95 reduced from ~43s to <1s at episode
  boundaries). Use `wait_pending()` to drain the queue. Set to `False` for
  synchronous mode.

### Changed

- `consolidation_max_unscoped_facts` default raised from 50 to 100.
- `dedup_similarity_threshold` default lowered from 0.95 to 0.90, catching
  paraphrase-level duplicates that previously accumulated as separate facts.
- Reflection now runs in the background by default (`background=True`).

## [0.2.0] - 2026-03-06

### Fixed

- **Consolidation correctness**: Actions (UPDATE, REMOVE, ADD dedup) now run
  against all active facts instead of only the similarity-scoped subset,
  preventing silent skips that left stale facts active.
- **Fact scoping**: Similarity-based scoping is now skipped for sessions with
  ≤50 active facts (configurable via `consolidation_max_unscoped_facts`),
  eliminating false exclusions of facts that need updating. The similarity
  threshold for larger sets was lowered from 0.3 to 0.15.
- **Legacy fallback duplicates**: The legacy reflection path now deduplicates
  new facts against existing active facts, preventing near-duplicate creation
  when consolidation falls back or when using a legacy reflector.
- **Consolidation prompt accuracy**: Prompt instructions now say "every
  existing fact listed above" instead of "EVERY existing fact", matching what
  the LLM actually receives when scoping is active.
- **Recall precision**: Fact candidates below `min_relevance_threshold`
  (default raised from 0.0 to 0.3) are now filtered out, eliminating
  low-signal noise from recall results.
- **Budget allocation**: Facts are now prioritized over raw vector search
  results in budget allocation, leveraging their compact size and high signal
  density.
- **Coverage validation**: Short facts (≤3 keywords) now require only a single
  keyword match instead of 50%, and substring matching catches morphological
  variants (e.g., "postgres" vs "postgresql").

### Added

- **Provider retry**: OpenAI and Anthropic reflectors now retry API calls with
  exponential backoff on transient failures (timeouts, rate limits, server
  errors), matching the existing HTTPReflector pattern. Configurable via
  `max_retries` constructor parameter (default: 3).
- **Active fact cap**: New `max_active_facts` config (default: 100) archives
  lowest-confidence facts when the limit is exceeded, bounding memory growth
  and O(n) similarity computations.
- **Scoping bypass config**: New `consolidation_max_unscoped_facts` config
  (default: 50) controls when similarity-based scoping kicks in.
- **Evaluation harness resilience**: Each scenario now runs in its own
  try/except so a single scenario failure no longer aborts the remaining run.

### Changed

- `current_episode_budget_pct` default reduced from 0.4 to 0.2, freeing
  budget for facts and historical context.
- `max_facts_per_episode` default increased from 5 to 10, allowing dense
  episodes to retain more information (confidence threshold still filters
  noise).
- `consolidation_similarity_threshold` default lowered from 0.3 to 0.15 for
  when scoping is active on large fact sets.
- `min_relevance_threshold` default raised from 0.0 to 0.3 for fact candidate
  filtering in recall.

## [0.1.0] - 2026-02-15

### Added

- Initial release of Gleanr.
- Session-scoped memory management with L0 (turns), L1 (episodes), and
  L2 (semantic facts) memory levels.
- Turn lifecycle: ingest, recall, and reflection.
- Episode management with configurable boundary detection.
- Consolidation-aware reflection with keep/update/add/remove actions.
- Token-budgeted recall pipeline with marker-based scoring.
- Provider support: OpenAI, Anthropic, and HTTP (OpenAI-compatible).
- Storage backends: in-memory and SQLite.
- LRU caching layer for embeddings, turns, and facts.
- Embedding-based deduplication for new facts.
- Observability via ReflectionTrace callbacks.
- Evaluation harness with configurable scenarios and metrics.

[0.3.0]: https://github.com/Saket-Kr/gleanr/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Saket-Kr/gleanr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Saket-Kr/gleanr/releases/tag/v0.1.0
