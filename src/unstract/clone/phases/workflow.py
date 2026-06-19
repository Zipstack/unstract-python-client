"""Migrate workflows from source org to target org.

Workflow rows themselves are simple — no required FKs to clone
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
import threading
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase, build_post_payload
from unstract.clone.report import CloneReport, PhaseResult
from unstract.clone.walker import remap_uuids

logger = logging.getLogger(__name__)

WORKFLOW_PATH = "workflow/"


class WorkflowPhase(Phase):
    name = "workflow"
    share_path_template = "workflow/{id}/share/"

    def run(self, report: CloneReport) -> PhaseResult:
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

        # Built once so per-workflow cascade-skip checks stay O(1).
        self._wf_to_src_tool_id = self._collect_wf_tool_map(result)

        logger.info("Found %d workflow(s) in source org", len(src_workflows))
        self.parallel_map(
            src_workflows,
            lambda src, lock: self._clone_one(src, result, lock),
        )
        return result

    def _collect_wf_tool_map(self, result: PhaseResult) -> dict[str, str]:
        """Map source workflow_id to its ToolInstance.tool_id; listed once
        to avoid N+1 fetches.
        """
        if not self.ctx.skipped_custom_tool_registry_ids:
            return {}
        try:
            tis = self.ctx.source.list_tool_instances()
        except Exception as e:
            logger.warning(
                "workflow phase: failed to list source tool_instances for "
                "cascade-skip lookup (%s); proceeding without cascade",
                e,
            )
            return {}
        mapping: dict[str, str] = {}
        for ti in tis:
            wf_id = ti.get("workflow")
            tool_id = ti.get("tool_id")
            if wf_id and tool_id:
                mapping[wf_id] = tool_id
        return mapping

    def _clone_one(
        self, src: dict[str, Any], result: PhaseResult, lock: threading.Lock
    ) -> None:
        name = src["workflow_name"]
        src_id = src["id"]

        src_tool_id = self._wf_to_src_tool_id.get(src_id)
        if src_tool_id and src_tool_id in self.ctx.skipped_custom_tool_registry_ids:
            logger.warning(
                "skipping workflow '%s' src=%s — its tool was skipped in "
                "custom_tool phase (frictionless adapter dependence)",
                name,
                src_id,
            )
            with lock:
                result.skipped += 1
            return

        try:
            existing = self.ctx.target.list_workflows(name=name)
        except Exception as e:
            logger.exception("Failed to GET workflow %s on target: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"GET {name}: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"workflow '{name}' already exists in target as {tgt['id']}"
                )
            with lock:
                result.adopted += 1
            logger.info(
                "adopted workflow '%s' src=%s -> tgt=%s", name, src_id, tgt["id"]
            )
        elif self.ctx.options.dry_run:
            with lock:
                result.created += 1
                self.ctx.remap.record_planned("workflow", src_id)
            logger.info("[dry-run] would create workflow '%s' src=%s", name, src_id)
            return
        else:
            # List endpoints serve stripped payloads; the workflow detail
            # carries the JSON blobs source_settings / destination_settings
            # that embed connector UUIDs.
            try:
                src_detail = self.ctx.source.get_workflow(src_id)
            except Exception as e:
                logger.exception("Failed to GET source workflow %s detail: %s", name, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"GET source detail {name}: {e}")
                return
            remapped = remap_uuids(src_detail, self.ctx.remap)
            payload = build_post_payload(remapped, self._writable)
            try:
                tgt = self.ctx.target.create_workflow(payload)
            except Exception as e:
                logger.exception("Failed to create workflow %s: %s", name, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"create {name}: {e}")
                return
            with lock:
                result.created += 1
            logger.info(
                "created workflow '%s' src=%s -> tgt=%s", name, src_id, tgt["id"]
            )

        with lock:
            self.ctx.remap.record("workflow", src_id, tgt["id"])
        # List rows carry the share axes; detail fn is a safety net.
        self.apply_share(
            src=src,
            tgt_id=tgt["id"],
            label=name,
            result=result,
            lock=lock,
            src_detail_fn=lambda: self.ctx.source.get_workflow(src_id),
        )
