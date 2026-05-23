"""Migrate ToolInstance rows from source org to target org.

Each workflow holds at most one ToolInstance, enforced server-side
(``tool_instance_v2/serializers.py`` raises if a workflow already has one).
The row carries:

- ``workflow`` FK — remapped from the WorkflowPhase remap table.
- ``tool_id`` (CharField, not FK) — a ``prompt_registry_id`` UUID. The
  target's registry was rebuilt in CustomToolPhase, so we remap via the
  ``prompt_studio_registry`` table populated there.
- ``metadata`` JSON — backend's ``create()`` discards the POST metadata
  and rebuilds it from tool defaults. So we POST a bare instance, then
  PATCH the metadata afterwards. Source metadata stores adapter values
  as NAMES (via to_representation in source GET); on PATCH the backend's
  ``update_metadata_with_adapter_instances`` resolves those names to
  the target's adapter UUIDs. Names match across orgs because
  AdapterPhase preserved them.
"""

from __future__ import annotations

import logging
from typing import Any

from unstract.migration.phases.base import Phase
from unstract.migration.report import MigrationReport, PhaseResult

logger = logging.getLogger(__name__)


class ToolInstancePhase(Phase):
    name = "tool_instance"

    def run(self, report: MigrationReport) -> PhaseResult:
        result = report.get_phase(self.name)
        workflow_remap = self.ctx.remap.snapshot().get("workflow", {})
        if not workflow_remap:
            logger.info("No workflows in remap; nothing to do for tool_instance phase")
            return result

        for src_wf_id, tgt_wf_id in workflow_remap.items():
            self._migrate_workflow_tools(src_wf_id, tgt_wf_id, result)
        return result

    def _migrate_workflow_tools(
        self, src_wf_id: str, tgt_wf_id: str, result: PhaseResult
    ) -> None:
        try:
            src_instances = self.ctx.source.list_tool_instances(workflow_id=src_wf_id)
        except Exception as e:
            logger.exception("Failed to list source tool_instances for wf %s: %s", src_wf_id, e)
            result.failed += 1
            result.errors.append(f"list src tool_instances {src_wf_id}: {e}")
            return

        if not src_instances:
            return
        if len(src_instances) > 1:
            # Backend enforces ≤1; warn loudly if invariant breaks on source.
            logger.warning(
                "source workflow %s has %d tool_instances (expected ≤1) — migrating first only",
                src_wf_id, len(src_instances),
            )

        src_ti = src_instances[0]
        src_ti_id = src_ti["id"]
        src_tool_id = src_ti["tool_id"]

        tgt_tool_id = self.ctx.remap.resolve("prompt_studio_registry", src_tool_id)
        if not tgt_tool_id:
            logger.warning(
                "skipping tool_instance %s — no registry remap for tool_id %s "
                "(custom tool likely unpublished on source)",
                src_ti_id, src_tool_id,
            )
            result.skipped += 1
            return

        try:
            existing = self.ctx.target.list_tool_instances(workflow_id=tgt_wf_id)
        except Exception as e:
            logger.exception("Failed to list target tool_instances for wf %s: %s", tgt_wf_id, e)
            result.failed += 1
            result.errors.append(f"list tgt tool_instances {tgt_wf_id}: {e}")
            return

        if existing:
            tgt_ti = existing[0]
            result.adopted += 1
            logger.info(
                "adopted tool_instance src=%s -> tgt=%s (workflow %s)",
                src_ti_id, tgt_ti["id"], tgt_wf_id,
            )
        elif self.ctx.options.dry_run:
            result.skipped += 1
            logger.info(
                "[dry-run] would create tool_instance for tgt workflow %s "
                "(src tool_instance %s)",
                tgt_wf_id, src_ti_id,
            )
            return
        else:
            try:
                tgt_ti = self.ctx.target.create_tool_instance(
                    {"workflow_id": tgt_wf_id, "tool_id": tgt_tool_id}
                )
            except Exception as e:
                logger.exception(
                    "Failed to create tool_instance for wf %s: %s", tgt_wf_id, e
                )
                result.failed += 1
                result.errors.append(f"create tool_instance {tgt_wf_id}: {e}")
                return
            result.created += 1
            logger.info(
                "created tool_instance src=%s -> tgt=%s (workflow %s)",
                src_ti_id, tgt_ti["id"], tgt_wf_id,
            )

        # PATCH the metadata regardless of created/adopted — keeps tool config
        # aligned with source on every run.
        src_metadata = src_ti.get("metadata") or {}
        try:
            self.ctx.target.update_tool_instance_metadata(tgt_ti["id"], src_metadata)
        except Exception as e:
            logger.exception(
                "Failed to PATCH tool_instance %s metadata: %s", tgt_ti["id"], e
            )
            result.failed += 1
            result.errors.append(f"patch metadata {tgt_ti['id']}: {e}")
            return

        self.ctx.remap.record("tool_instance", src_ti_id, tgt_ti["id"])
