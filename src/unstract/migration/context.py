"""Shared state passed between migration phases.

Three top-level types:

- ``OrgEndpoint`` — base URL + organization_id + Platform API key for one org.
- ``MigrationOptions`` — run flags (dry-run, include/exclude, name-conflict).
- ``MigrationContext`` — bundles source/target clients, options, and the
  per-run ``RemapTable``.

``RemapTable`` lives here too because every phase touches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unstract.migration.client import PlatformClient


@dataclass(frozen=True)
class OrgEndpoint:
    """One end of a migration: where to talk to and who to talk as.

    ``organization_id`` is the slug embedded in the URL path (see
    ``OrganizationMiddleware`` regex ``/api/(v1|v2)/unstract/<org_id>/...``).
    ``platform_key`` is the bearer UUID issued by an org admin.

    ``api_path_prefix`` defaults to ``api/v1`` to match the OSS docker
    compose ``PATH_PREFIX`` env. Cloud / on-prem envs that mount on a
    different prefix can override (e.g. ``api/v2``).
    """

    base_url: str
    organization_id: str
    platform_key: str
    api_path_prefix: str = "api/v1"


@dataclass
class MigrationOptions:
    """Per-run flags for ``migrate()``."""

    dry_run: bool = False
    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    on_name_conflict: str = "adopt"  # "adopt" | "abort"
    verbose: bool = False

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
class MigrationContext:
    """Shared state for one ``migrate()`` invocation.

    Phases hold a reference to this and call ``ctx.source`` / ``ctx.target``
    to drive HTTP, ``ctx.remap`` to record UUID mappings.
    """

    source: PlatformClient
    target: PlatformClient
    options: MigrationOptions
    remap: RemapTable = field(default_factory=RemapTable)
