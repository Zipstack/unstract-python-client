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
import time

from unstract.clone.client import PlatformClient
from unstract.clone.context import CloneContext, CloneOptions, OrgEndpoint
from unstract.clone.exceptions import CloneError
from unstract.clone.phases import (
    AdapterPhase,
    AgenticStudioPhase,
    APIDeploymentPhase,
    ConnectorPhase,
    CustomToolPhase,
    FilesPhase,
    GroupPhase,
    LookupsPhase,
    ManualReviewPhase,
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
# Group runs first: every shareable phase consumes its remap entries when
# replicating shared_groups. Adapter, connector, tag are independent leaf
# phases. Downstream phases (custom_tool, workflow, tool_instance,
# workflow_endpoint) land later and consume the remap entries these
# produce. Pipeline + api_deployment come last: both FK the workflow and
# api_deployment additionally requires endpoints to be configured before
# the serializer accepts it.
PHASES: list[tuple[str, type[Phase]]] = [
    ("group", GroupPhase),
    ("adapter", AdapterPhase),
    # Cloud-only; standalone (own project + registry) and FKs four adapters.
    # Probe-gated: auto-skips on OSS deployments via ``probe_path``.
    ("agentic_studio", AgenticStudioPhase),
    ("connector", ConnectorPhase),
    ("tag", TagPhase),
    ("custom_tool", CustomToolPhase),
    ("files", FilesPhase),
    # Cloud-only; consumes custom_tool's prompt + adapter remaps. Probe-gated:
    # auto-skips on OSS deployments via ``probe_path``.
    ("lookups", LookupsPhase),
    ("workflow", WorkflowPhase),
    # Cloud-only; FKs the workflow (RuleEngine / HITLSettings bind to it).
    # Probe-gated: auto-skips on OSS deployments via ``probe_path``.
    ("manual_review", ManualReviewPhase),
    ("tool_instance", ToolInstancePhase),
    ("workflow_endpoint", WorkflowEndpointPhase),
    ("pipeline", PipelinePhase),
    ("api_deployment", APIDeploymentPhase),
]


def _cloud_phase_runnable(
    ctx: CloneContext, report: CloneReport, name: str, probe_path: str
) -> bool:
    """Decide whether a cloud-only phase should run on this deployment pair.

    Probe source first; only probe target if source has the feature. A probe
    failure (unexpected status / transport) must not abort an otherwise-fine
    run — treat it like target-absent: warn + skip, never raise.
    """
    try:
        if not ctx.feature_present(ctx.source, probe_path):
            # OSS source: behave exactly as if this phase didn't exist.
            logger.debug("Phase '%s' skipped: feature absent on source", name)
            return False
        target_present = ctx.feature_present(ctx.target, probe_path)
    except Exception as e:
        msg = f"Phase '{name}' skipped: capability probe failed ({e})"
        logger.warning(msg)
        report.warnings.append(msg)
        return False
    if not target_present:
        msg = (
            f"Phase '{name}' skipped: feature present on source but not on "
            "target deployment"
        )
        logger.warning(msg)
        report.warnings.append(msg)
        return False
    return True


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
            dry_run=opts.dry_run,
        )

        run_started = time.perf_counter()
        for name, phase_cls in PHASES:
            if not opts.includes(name):
                report.skipped_phases.append(name)
                logger.info("Phase '%s' skipped (excluded)", name)
                continue
            probe_path = getattr(phase_cls, "probe_path", None)
            if probe_path is not None and not _cloud_phase_runnable(
                ctx, report, name, probe_path
            ):
                continue
            logger.info("=== Phase: %s ===", name)
            phase_started = time.perf_counter()
            try:
                phase_cls(ctx).run(report)
            except CloneError as e:
                report.aborted = True
                report.abort_reason = str(e)
                logger.error("Phase '%s' aborted: %s", name, e)
                # Stamp duration even on abort so the report reflects time spent.
                report.get_phase(name).duration_s = time.perf_counter() - phase_started
                break
            else:
                report.get_phase(name).duration_s = time.perf_counter() - phase_started
                logger.info(
                    "=== Phase '%s' done in %.2fs ===",
                    name,
                    report.get_phase(name).duration_s,
                )

        report.total_duration_s = time.perf_counter() - run_started
        report.remap_snapshot = ctx.remap.snapshot(hide_planned=opts.dry_run)
        return report
    finally:
        src_client.close()
        tgt_client.close()
