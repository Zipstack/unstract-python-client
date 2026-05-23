"""Migrate workflows from source org to target org.

Workflow rows themselves are simple — no required FKs to migration
entities, unique per ``(workflow_name, organization)``. The two
non-trivial bits:

1. ``source_settings`` and ``destination_settings`` are JSON blobs that
   embed connector UUIDs. The walker remaps them using the running
   ``RemapTable`` (connectors already landed in the previous phase).

2. Creating a workflow auto-creates empty ``WorkflowEndpoint`` rows
   server-side. We don't touch those here — the dedicated
   WorkflowEndpoint phase reconciles them after ToolInstance lands.
"""

from __future__ import annotations

import logging
from typing import Any

from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.base import Phase, build_post_payload
from unstract.migration.report import MigrationReport, PhaseResult
from unstract.migration.walker import remap_uuids

logger = logging.getLogger(__name__)

WORKFLOW_PATH = "workflow/"


class WorkflowPhase(Phase):
    name = "workflow"

    def run(self, report: MigrationReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._writable = self.ctx.target.get_post_schema(WORKFLOW_PATH)
        except Exception as e:
            logger.exception("Failed to fetch target POST schema for workflow: %s", e)
            result.failed += 1
            result.errors.append(f"OPTIONS workflow: {e}")
            return result

        try:
            src_workflows = self.ctx.source.list_workflows()
        except Exception as e:
            logger.exception("Failed to list source workflows: %s", e)
            result.failed += 1
            result.errors.append(f"list source workflows: {e}")
            return result

        logger.info("Found %d workflow(s) in source org", len(src_workflows))
        for src in src_workflows:
            self._migrate_one(src, result)
        return result

    def _migrate_one(self, src: dict[str, Any], result: PhaseResult) -> None:
        name = src["workflow_name"]
        src_id = src["id"]

        try:
            existing = self.ctx.target.list_workflows(name=name)
        except Exception as e:
            logger.exception("Failed to GET workflow %s on target: %s", name, e)
            result.failed += 1
            result.errors.append(f"GET {name}: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"workflow '{name}' already exists in target as {tgt['id']}"
                )
            result.adopted += 1
            logger.info("adopted workflow '%s' src=%s -> tgt=%s", name, src_id, tgt["id"])
        elif self.ctx.options.dry_run:
            result.skipped += 1
            logger.info("[dry-run] would create workflow '%s' src=%s", name, src_id)
            return
        else:
            remapped = remap_uuids(src, self.ctx.remap)
            payload = build_post_payload(remapped, self._writable)
            try:
                tgt = self.ctx.target.create_workflow(payload)
            except Exception as e:
                logger.exception("Failed to create workflow %s: %s", name, e)
                result.failed += 1
                result.errors.append(f"create {name}: {e}")
                return
            result.created += 1
            logger.info("created workflow '%s' src=%s -> tgt=%s", name, src_id, tgt["id"])

        self.ctx.remap.record("workflow", src_id, tgt["id"])
