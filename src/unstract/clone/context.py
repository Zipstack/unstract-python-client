"""Shared state passed between clone phases.

Three top-level types:

- ``OrgEndpoint`` — base URL + organization_id + Platform API key for one org.
- ``CloneOptions`` — run flags (dry-run, include/exclude, name-conflict).
- ``CloneContext`` — bundles source/target clients, options, and the
  per-run ``RemapTable``.

``RemapTable`` lives here too because every phase touches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unstract.clone.client import PlatformClient


@dataclass(frozen=True)
class OrgEndpoint:
    """One end of a clone: where to talk to and who to talk as.

    ``organization_id`` is the slug embedded in the URL path; the bearer
    Platform API key must belong to this org. ``api_path_prefix`` matches
    the deployment's URL prefix (defaults to ``api/v1``).
    """

    base_url: str
    organization_id: str
    platform_key: str
    api_path_prefix: str = "api/v1"


DEFAULT_MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB; oversize → manual-upload list
DEFAULT_CONCURRENCY = 4


@dataclass
class CloneOptions:
    """Per-run flags for ``clone()``."""

    dry_run: bool = False
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    on_name_conflict: str = "adopt"  # "adopt" | "abort"
    verbose: bool = False
    # "platform_api": download/upload via existing endpoints (default).
    # "skip": metadata only; operator re-uploads via UI on target.
    file_strategy: str = "platform_api"
    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    # Per-phase worker fan-out. 1 = sequential (no executor).
    concurrency: int = DEFAULT_CONCURRENCY

    def includes(self, phase_name: str) -> bool:
        if self.include is not None and phase_name not in self.include:
            return False
        return phase_name not in self.exclude


class RemapTable:
    """Maps source UUID -> target UUID, scoped per entity type.

    Built up in dependency order; consumed by the JSON walker before POST.
    ``resolve_any`` lets the walker look up a UUID without knowing its
    entity type — necessary because embedded references in JSON payloads
    don't always carry an entity hint.
    """

    def __init__(self) -> None:
        self._table: dict[str, dict[str, str]] = {}

    def record(self, entity: str, src_uuid: str, tgt_uuid: str) -> None:
        self._table.setdefault(entity, {})[src_uuid] = tgt_uuid

    def resolve(self, entity: str, src_uuid: str) -> str | None:
        return self._table.get(entity, {}).get(src_uuid)

    def resolve_any(self, src_uuid: str) -> str | None:
        for mapping in self._table.values():
            hit = mapping.get(src_uuid)
            if hit is not None:
                return hit
        return None

    def snapshot(self) -> dict[str, dict[str, str]]:
        """Read-only snapshot for the post-run report."""
        return {entity: dict(m) for entity, m in self._table.items()}


@dataclass
class CloneContext:
    """Shared state for one ``clone()`` invocation.

    Phases hold a reference to this and call ``ctx.source`` / ``ctx.target``
    to drive HTTP, ``ctx.remap`` to record UUID mappings.
    """

    source: PlatformClient
    target: PlatformClient
    options: CloneOptions
    remap: RemapTable = field(default_factory=RemapTable)
    # Source prompt_registry_ids whose CustomTool was skipped; used to
    # cascade-skip dependent workflows downstream.
    skipped_custom_tool_registry_ids: set[str] = field(default_factory=set)
