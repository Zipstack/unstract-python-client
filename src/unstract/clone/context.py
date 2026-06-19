"""Shared state passed between clone phases.

Three top-level types:

- ``OrgEndpoint`` — base URL + organization_id + Platform API key for one org.
- ``CloneOptions`` — run flags (dry-run, include/exclude, name-conflict).
- ``CloneContext`` — bundles source/target clients, options, and the
  per-run ``RemapTable``.

``RemapTable`` lives here too because every phase touches it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

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
    # Group phase: also add members (matched by email) to cloned groups.
    clone_group_members: bool = False

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
        # Synthetic target ids minted by record_planned() in dry-run. Tracked
        # so the report can mask them — they are not real target ids.
        self._planned: set[str] = set()

    def record(self, entity: str, src_uuid: str, tgt_uuid: str) -> None:
        self._table.setdefault(entity, {})[src_uuid] = tgt_uuid

    def record_planned(self, entity: str, src_uuid: str) -> str:
        """Dry-run only: record a deterministic synthetic target id so
        dependent phases can resolve the FK and plan-count without writing.
        Never reaches the wire, so the fake id stays in-memory scaffolding.
        """
        tgt_uuid = str(uuid5(NAMESPACE_URL, f"planned:{entity}:{src_uuid}"))
        self.record(entity, src_uuid, tgt_uuid)
        self._planned.add(tgt_uuid)
        return tgt_uuid

    def is_planned(self, tgt_uuid: str) -> bool:
        """True if ``tgt_uuid`` is a dry-run synthetic id (no real row on
        target). Callers use this to skip live target lookups that would
        query a non-existent id.
        """
        return tgt_uuid in self._planned

    def resolve(self, entity: str, src_uuid: str) -> str | None:
        return self._table.get(entity, {}).get(src_uuid)

    def resolve_any(self, src_uuid: str) -> str | None:
        # Snapshot to avoid `RuntimeError: dictionary changed size during
        # iteration` when a concurrent record() inserts a new entity bucket.
        for mapping in list(self._table.values()):
            hit = mapping.get(src_uuid)
            if hit is not None:
                return hit
        return None

    def snapshot(self, *, hide_planned: bool = False) -> dict[str, dict[str, str]]:
        """Read-only snapshot for the post-run report. ``hide_planned`` masks
        dry-run synthetic ids (rendered as ``"(planned)"``) while keeping the
        per-entity counts intact.
        """

        def _val(tgt: str) -> str:
            return "(planned)" if hide_planned and tgt in self._planned else tgt

        return {
            entity: {src: _val(tgt) for src, tgt in m.items()}
            for entity, m in self._table.items()
        }


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
    # Source connector ids skipped because they can't be recreated via the API
    # (OAuth-backed or redacted metadata) and no same-name target connector
    # exists to adopt; workflows whose endpoints use one are cascade-skipped so
    # the clone doesn't create pipelines that fail on every run.
    skipped_connector_ids: set[str] = field(default_factory=set)
    # Per-run memo for users/groups directory listings (sharing replication
    # touches them once per endpoint, never per resource).
    share_cache: dict[str, Any] = field(default_factory=dict)
    share_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    # Capability-probe memo: (id(client), feature_path) -> present?. Probed
    # once per (deployment, feature) so cloud-phase gating costs one GET total.
    probe_cache: dict[tuple[int, str], bool] = field(default_factory=dict)

    def feature_present(self, client: "PlatformClient", path: str) -> bool:
        """Is ``path`` (a feature's list endpoint) installed on ``client``?

        Memoised per run. Plain dict, no lock — probing runs in the
        single-threaded orchestrator loop, before any parallel_map fan-out.
        """
        key = (id(client), path)
        cached = self.probe_cache.get(key)
        if cached is not None:
            return cached
        present = client.probe(path)
        self.probe_cache[key] = present
        return present
