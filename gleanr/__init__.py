"""Gleanr - Agent Context Management System.

A session-scoped memory layer for AI agents, providing intelligent,
token-budgeted context assembly.

Example:
    >>> from gleanr import Gleanr
    >>> from gleanr.storage import InMemoryBackend
    >>>
    >>> async def main():
    ...     storage = InMemoryBackend()
    ...     async with Gleanr("session_123", storage) as gleanr:
    ...         await gleanr.ingest("user", "Hello!")
    ...         await gleanr.ingest("assistant", "Hi there!")
    ...         context = await gleanr.recall("greeting")
    ...         print(f"Found {len(context)} items")
"""

from gleanr.core import (
    EpisodeBoundaryConfig,
    Gleanr,
    GleanrConfig,
    RecallConfig,
    ReflectionConfig,
    create_config,
)
from gleanr.errors import (
    ConfigurationError,
    EpisodeNotFoundError,
    GleanrError,
    ProviderError,
    ReflectionError,
    RetryExhaustedError,
    SessionNotFoundError,
    StorageError,
    TokenBudgetExceededError,
    TurnNotFoundError,
    ValidationError,
)
from gleanr.memory.reflection import ReflectionTrace, ReflectionTraceCallback
from gleanr.models import (
    ContextItem,
    Episode,
    EpisodeStatus,
    Fact,
    MarkerType,
    Role,
    SessionStats,
    Turn,
)
from gleanr.providers import Embedder, NullEmbedder, NullReflector, Reflector, TokenCounter
from gleanr.storage import InMemoryBackend, StorageBackend

__version__ = "0.3.0"

__all__ = [
    # Main class
    "Gleanr",
    # Configuration
    "GleanrConfig",
    "EpisodeBoundaryConfig",
    "RecallConfig",
    "ReflectionConfig",
    "create_config",
    # Models
    "Turn",
    "Episode",
    "Fact",
    "ContextItem",
    "SessionStats",
    # Enums
    "Role",
    "EpisodeStatus",
    "MarkerType",
    # Protocols
    "Embedder",
    "Reflector",
    "TokenCounter",
    # Null implementations
    "NullEmbedder",
    "NullReflector",
    # Observability
    "ReflectionTrace",
    "ReflectionTraceCallback",
    # Storage
    "StorageBackend",
    "InMemoryBackend",
    # Exceptions
    "GleanrError",
    "ConfigurationError",
    "ValidationError",
    "StorageError",
    "ProviderError",
    "TokenBudgetExceededError",
    "SessionNotFoundError",
    "EpisodeNotFoundError",
    "TurnNotFoundError",
    "ReflectionError",
    "RetryExhaustedError",
]
