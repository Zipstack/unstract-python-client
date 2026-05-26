"""Top-level ``clone()`` entry point.

Wires source/target ``PlatformClient`` instances, builds a
``CloneContext``, runs each phase in strict topological order, and
returns a ``CloneReport``.

Phase order is owned here — phases must not call each other. Adding a new
entity type means: write a new ``Phase`` subclass and append it to
``PHASES`` at the right dependency position.
"""

from __future__ import annotations

import logging

from unstract.clone.client import PlatformClient
from unstract.clone.context import CloneContext, CloneOptions, OrgEndpoint
from unstract.clone.exceptions import CloneError
from unstract.clone.phases import (
    AdapterPhase,
    APIDeploymentPhase,
    ConnectorPhase,
    CustomToolPhase,
    FilesPhase,
    PipelinePhase,
    TagPhase,
    ToolInstancePhase,
    WorkflowEndpointPhase,
    WorkflowPhase,
)
from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, Endpoint

logger = logging.getLogger(__name__)

# Strict dependency order. Each entry: (phase_name, phase_class).
# Adapter, connector, tag are independent leaf phases. Downstream phases
# (custom_tool, workflow, tool_instance, workflow_endpoint) land later
# and consume the remap entries these produce. Pipeline + api_deployment
# come last: both FK the workflow and api_deployment additionally
# requires endpoints to be configured before the serializer accepts it.
PHASES: list[tuple[str, type[Phase]]] = [
    ("adapter", AdapterPhase),
    ("connector", ConnectorPhase),
    ("tag", TagPhase),
    ("custom_tool", CustomToolPhase),
    ("files", FilesPhase),
    ("workflow", WorkflowPhase),
    ("tool_instance", ToolInstancePhase),
    ("workflow_endpoint", WorkflowEndpointPhase),
    ("pipeline", PipelinePhase),
    ("api_deployment", APIDeploymentPhase),
]


def clone(
    source: OrgEndpoint,
    target: OrgEndpoint,
    options: CloneOptions | None = None,
) -> CloneReport:
    """Migrate configured resources from one org to another.

    Returns a ``CloneReport`` even on partial failure; raises only on
    setup errors or ``on_name_conflict='abort'`` collisions.
    """
    opts = options or CloneOptions()
    src_client = PlatformClient(source)
    tgt_client = PlatformClient(target)
    try:
        ctx = CloneContext(
            source=src_client,
            target=tgt_client,
            options=opts,
        )
        report = CloneReport(
            source=Endpoint(
                base_url=source.base_url, organization_id=source.organization_id
            ),
            target=Endpoint(
                base_url=target.base_url, organization_id=target.organization_id
            ),
        )

        for name, phase_cls in PHASES:
            if not opts.includes(name):
                report.skipped_phases.append(name)
                logger.info("Phase '%s' skipped (excluded)", name)
                continue
            logger.info("=== Phase: %s ===", name)
            try:
                phase_cls(ctx).run(report)
            except CloneError as e:
                report.aborted = True
                report.abort_reason = str(e)
                logger.error("Phase '%s' aborted: %s", name, e)
                break

        report.remap_snapshot = ctx.remap.snapshot()
        return report
    finally:
        src_client.close()
        tgt_client.close()
