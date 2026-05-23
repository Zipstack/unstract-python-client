"""Top-level ``migrate()`` entry point.

Wires source/target ``PlatformClient`` instances, builds a
``MigrationContext``, runs each phase in strict topological order, and
returns a ``MigrationReport``.

Phase order is owned here — phases must not call each other. Adding a new
entity type means: write a new ``Phase`` subclass and append it to
``PHASES`` at the right dependency position.
"""

from __future__ import annotations

import logging

from unstract.migration.client import PlatformClient
from unstract.migration.context import MigrationContext, MigrationOptions, OrgEndpoint
from unstract.migration.exceptions import MigrationError
from unstract.migration.phases import (
    AdapterPhase,
    ConnectorPhase,
    CustomToolPhase,
    TagPhase,
    ToolInstancePhase,
    WorkflowEndpointPhase,
    WorkflowPhase,
)
from unstract.migration.phases.base import Phase
from unstract.migration.report import MigrationReport

logger = logging.getLogger(__name__)

# Strict dependency order. Each entry: (phase_name, phase_class).
# Adapter, connector, tag are independent leaf phases. Downstream phases
# (custom_tool, workflow, tool_instance, workflow_endpoint) land later
# and consume the remap entries these produce.
PHASES: list[tuple[str, type[Phase]]] = [
    ("adapter", AdapterPhase),
    ("connector", ConnectorPhase),
    ("tag", TagPhase),
    ("custom_tool", CustomToolPhase),
    ("workflow", WorkflowPhase),
    ("tool_instance", ToolInstancePhase),
    ("workflow_endpoint", WorkflowEndpointPhase),
]


def migrate(
    source: OrgEndpoint,
    target: OrgEndpoint,
    options: MigrationOptions | None = None,
) -> MigrationReport:
    """Migrate configured resources from one org to another.

    Returns a ``MigrationReport`` even on partial failure; raises only on
    setup errors or ``on_name_conflict='abort'`` collisions.
    """
    opts = options or MigrationOptions()
    ctx = MigrationContext(
        source=PlatformClient(source),
        target=PlatformClient(target),
        options=opts,
    )
    report = MigrationReport()

    for name, phase_cls in PHASES:
        if not opts.includes(name):
            report.skipped_phases.append(name)
            logger.info("Phase '%s' skipped (excluded)", name)
            continue
        logger.info("=== Phase: %s ===", name)
        try:
            phase_cls(ctx).run(report)
        except MigrationError as e:
            report.aborted = True
            report.abort_reason = str(e)
            logger.error("Phase '%s' aborted: %s", name, e)
            break

    report.remap_snapshot = ctx.remap.snapshot()
    return report
