"""Migrate ToolInstance rows from source org to target org.

Each workflow holds at most one ToolInstance, enforced server-side.
The row carries:

- ``workflow`` FK — remapped from the WorkflowPhase remap table.
- ``tool_id`` — a ``prompt_registry_id`` UUID. The target's registry was
  rebuilt in CustomToolPhase, so we remap via the ``prompt_studio_registry``
  table populated there.
- ``metadata`` JSON — the create response carries default metadata rebuilt
  from tool defaults, so we POST a bare instance then PATCH the metadata
  afterwards. Source metadata stores adapter values as NAMES; on PATCH the
  backend resolves those names to the target's adapter UUIDs. Names match
  across orgs because AdapterPhase preserved them.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, PhaseResult

logger = logging.getLogger(__name__)

# The source response emits these sentinel strings when an adapter UUID/name
# in the stored metadata can no longer be resolved (deleted or renamed on
# source). Round-tripping them to target produces an AdapterNotFound on PATCH,
# so we detect and skip the metadata PATCH instead — the row exists with safe
# defaults and the operator can re-bind in the UI.
_BROKEN_ADAPTER_SENTINELS: tuple[str, ...] = (
    "NOT FOUND",
    "[DELETED ADAPTER",
    "[NEEDS UPDATE]",
)

# Fields tied to the source row's own ids — never valid on the target.
# Always rewrite these with target values before PATCHing.
_SOURCE_IDENTITY_FIELDS: tuple[str, ...] = (
    "prompt_registry_id",
    "tool_instance_id",
    "tenant_id",
)


def _broken_adapter_keys(metadata: dict[str, Any]) -> list[str]:
    broken: list[str] = []
    for key, value in metadata.items():
        if isinstance(value, str) and any(
            s in value for s in _BROKEN_ADAPTER_SENTINELS
        ):
            broken.append(f"{key}={value!r}")
    return broken


def _strip_source_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metadata.items() if k not in _SOURCE_IDENTITY_FIELDS}


class ToolInstancePhase(Phase):
    name = "tool_instance"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        workflow_remap = self.ctx.remap.snapshot().get("workflow", {})
        if not workflow_remap:
            logger.info("No workflows in remap; nothing to do for tool_instance phase")
            return result

        self.parallel_map(
            list(workflow_remap.items()),
            lambda pair, lock: self._clone_workflow_tools(
                pair[0], pair[1], result, lock
            ),
        )
        return result

    def _clone_workflow_tools(
        self,
        src_wf_id: str,
        tgt_wf_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            src_instances = self.ctx.source.list_tool_instances(workflow_id=src_wf_id)
        except Exception as e:
            logger.exception(
                "Failed to list source tool_instances for wf %s: %s", src_wf_id, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"list src tool_instances {src_wf_id}: {e}")
            return

        if not src_instances:
            return
        if len(src_instances) > 1:
            logger.warning(
                "source workflow %s has %d tool_instances (>1) — migrating first only",
                src_wf_id,
                len(src_instances),
            )

        src_ti = src_instances[0]
        src_ti_id = src_ti["id"]
        src_tool_id = src_ti["tool_id"]

        with lock:
            tgt_tool_id = self.ctx.remap.resolve("prompt_studio_registry", src_tool_id)
        if not tgt_tool_id:
            logger.warning(
                "skipping tool_instance %s — no registry remap for tool_id %s "
                "(custom tool likely unpublished on source)",
                src_ti_id,
                src_tool_id,
            )
            with lock:
                result.skipped += 1
            return

        # A planned (dry-run) workflow id has no row on target; its
        # tool_instance can't exist yet, so predict a create without the
        # live lookup against a non-existent id.
        if self.ctx.options.dry_run and self.ctx.remap.is_planned(tgt_wf_id):
            existing: list[dict[str, Any]] = []
        else:
            try:
                existing = self.ctx.target.list_tool_instances(workflow_id=tgt_wf_id)
            except Exception as e:
                logger.exception(
                    "Failed to list target tool_instances for wf %s: %s", tgt_wf_id, e
                )
                with lock:
                    result.failed += 1
                    result.errors.append(f"list tgt tool_instances {tgt_wf_id}: {e}")
                return

        if existing:
            tgt_ti = existing[0]
            if self.ctx.options.dry_run:
                with lock:
                    result.adopted += 1
                    self.ctx.remap.record("tool_instance", src_ti_id, tgt_ti["id"])
                logger.info(
                    "[dry-run] would adopt tool_instance (metadata PATCH skipped) "
                    "src=%s -> tgt=%s (workflow %s)",
                    src_ti_id,
                    tgt_ti["id"],
                    tgt_wf_id,
                )
                return
            with lock:
                result.adopted += 1
            logger.info(
                "adopted tool_instance src=%s -> tgt=%s (workflow %s)",
                src_ti_id,
                tgt_ti["id"],
                tgt_wf_id,
            )
        elif self.ctx.options.dry_run:
            with lock:
                result.created += 1
                self.ctx.remap.record_planned("tool_instance", src_ti_id)
            logger.info(
                "[dry-run] would create tool_instance for tgt workflow %s "
                "(src tool_instance %s)",
                tgt_wf_id,
                src_ti_id,
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
                with lock:
                    result.failed += 1
                    result.errors.append(f"create tool_instance {tgt_wf_id}: {e}")
                return
            with lock:
                result.created += 1
            logger.info(
                "created tool_instance src=%s -> tgt=%s (workflow %s)",
                src_ti_id,
                tgt_ti["id"],
                tgt_wf_id,
            )

        src_metadata = src_ti.get("metadata") or {}
        broken = _broken_adapter_keys(src_metadata)
        if broken:
            logger.warning(
                "skipping metadata PATCH for tool_instance src=%s tgt=%s — "
                "source metadata carries broken adapter refs %s; "
                "row exists with backend defaults, re-bind in UI",
                src_ti_id,
                tgt_ti["id"],
                broken,
            )
            with lock:
                result.skipped += 1
                result.errors.append(
                    f"stale adapter refs on src tool_instance {src_ti_id}: {broken}"
                )
        else:
            # PATCH overwrites the whole metadata dict — re-stamp target
            # identity fields or the runtime sees them as empty.
            patch_metadata = {
                **_strip_source_identity(src_metadata),
                "prompt_registry_id": tgt_tool_id,
                "tool_instance_id": tgt_ti["id"],
            }
            try:
                self.ctx.target.update_tool_instance_metadata(
                    tgt_ti["id"], patch_metadata
                )
            except Exception as e:
                logger.exception(
                    "Failed to PATCH tool_instance %s metadata: %s", tgt_ti["id"], e
                )
                with lock:
                    result.failed += 1
                    result.errors.append(f"patch metadata {tgt_ti['id']}: {e}")
                return

        with lock:
            self.ctx.remap.record("tool_instance", src_ti_id, tgt_ti["id"])
