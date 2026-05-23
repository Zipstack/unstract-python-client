"""Org-to-org data migration over the Platform API.

Migrates configured resources (adapters, connectors, custom tools, workflows,
etc.) from one Unstract org to another using two admin-issued Platform API
keys. The target deployment is the persistent state — re-runs reconcile
against existing target rows by natural key.
"""

from unstract.migration.context import (
    MigrationContext,
    MigrationOptions,
    OrgEndpoint,
    RemapTable,
)
from unstract.migration.orchestrator import migrate
from unstract.migration.report import MigrationReport

__all__ = [
    "MigrationContext",
    "MigrationOptions",
    "MigrationReport",
    "OrgEndpoint",
    "RemapTable",
    "migrate",
]
