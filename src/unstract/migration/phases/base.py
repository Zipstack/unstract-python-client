"""Base class for migration phases."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from unstract.migration.context import MigrationContext
from unstract.migration.report import MigrationReport, PhaseResult

logger = logging.getLogger(__name__)

# DRF OPTIONS reports any ModelSerializer FK/M2M as writable, but the
# backend's perform_create overrides these server-side. Posting them is
# either noise (silently overwritten) or a 400 (when a source-org value
# doesn't validate against the target org). Strip them universally —
# the phase OPTIONS schema covers the entity-specific writable subset.
SERVER_MANAGED: frozenset[str] = frozenset(
    {
        "id",
        "organization",
        "created_by",
        "created_by_email",
        "modified_by",
        "modified_by_email",
        "created_at",
        "modified_at",
        "shared_users",
    }
)


def build_post_payload(
    src: dict[str, Any], writable: frozenset[str]
) -> dict[str, Any]:
    """Project ``src`` onto the writable schema, dropping server-managed
    fields, ``None`` values, and empty strings (which DRF treats as blank
    and rejects on required fields).
    """
    keys = writable - SERVER_MANAGED
    # Equality with `(None, "")` matched False and 0 too (Python: False == 0,
    # 0 in (None, "") is False, but `0 not in (...)` falsely returns True).
    # Explicit identity / equality checks preserve falsy-but-meaningful
    # values like ``BooleanField`` False and numeric defaults.
    return {
        k: src[k]
        for k in keys
        if k in src and src[k] is not None and src[k] != ""
    }


class Phase(ABC):
    """Abstract phase. One subclass per entity type."""

    name: str = ""

    def __init__(self, ctx: MigrationContext):
        self.ctx = ctx

    @abstractmethod
    def run(self, report: MigrationReport) -> PhaseResult:
        """Migrate all entities of this phase's type. Idempotent across runs."""
        raise NotImplementedError
