"""Migrate WorkflowEndpoint rows from source org to target org.

The backend auto-creates one SOURCE and one DESTINATION endpoint per
workflow on workflow create (``perform_create`` in WorkflowViewSet), so
there's nothing to POST — we only PATCH the target's existing endpoints
with the source's connection_type, connector_instance, and configuration.

Notes:
- ``workflow`` and ``endpoint_type`` are ``editable=False`` server-side
  and aren't writable on PATCH.
- ``connector_instance`` FK is nullable; we remap via the connector
  remap table populated in ConnectorPhase.
- ``configuration`` is a JSON blob that may embed connector UUIDs;
  walker pass remaps them before PATCH.
- Source ``connector_instance`` arrives as a nested dict (per
  ``WorkflowEndpointSerializer.connector_instance``); we extract its
  ``id`` and remap.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, PhaseResult
from unstract.clone.walker import remap_uuids

logger = logging.getLogger(__name__)


def _extract_connector_id(endpoint: dict[str, Any]) -> str | None:
    """``connector_instance`` is a nested dict on GET; pull out the FK uuid."""
    ci = endpoint.get("connector_instance")
    if isinstance(ci, dict):
        return ci.get("id")
    if isinstance(ci, str):
        return ci
    return None


class WorkflowEndpointPhase(Phase):
    name = "workflow_endpoint"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        workflow_remap = self.ctx.remap.snapshot().get("workflow", {})
        if not workflow_remap:
            logger.info(
                "No workflows in remap; nothing to do for workflow_endpoint phase"
            )
            return result

        self.parallel_map(
            list(workflow_remap.items()),
            lambda pair, lock: self._clone_workflow_endpoints(
                pair[0], pair[1], result, lock
            ),
        )
        return result

    def _clone_workflow_endpoints(
        self,
        src_wf_id: str,
        tgt_wf_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            src_endpoints = self.ctx.source.list_workflow_endpoints(
                workflow_id=src_wf_id
            )
        except Exception as e:
            logger.exception(
                "Failed to list source endpoints for wf %s: %s", src_wf_id, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"list src endpoints {src_wf_id}: {e}")
            return

        # A planned (dry-run) workflow would be freshly created, and the
        # backend auto-creates its SOURCE/DEST endpoints on create. They
        # don't exist on target yet, so predict a patch per source endpoint
        # without the live lookup against the synthetic id.
        if self.ctx.options.dry_run and self.ctx.remap.is_planned(tgt_wf_id):
            for src_ep in src_endpoints:
                with lock:
                    result.created += 1
                    self.ctx.remap.record_planned("workflow_endpoint", src_ep["id"])
                logger.info(
                    "[dry-run] would PATCH %s endpoint on new workflow %s (src=%s)",
                    src_ep["endpoint_type"],
                    tgt_wf_id,
                    src_ep["id"],
                )
            return

        try:
            tgt_endpoints = self.ctx.target.list_workflow_endpoints(
                workflow_id=tgt_wf_id
            )
        except Exception as e:
            logger.exception(
                "Failed to list target endpoints for wf %s: %s", tgt_wf_id, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"list tgt endpoints {tgt_wf_id}: {e}")
            return

        tgt_by_type = {ep["endpoint_type"]: ep for ep in tgt_endpoints}

        for src_ep in src_endpoints:
            etype = src_ep["endpoint_type"]
            tgt_ep = tgt_by_type.get(etype)
            if tgt_ep is None:
                logger.warning(
                    "target workflow %s missing %s endpoint — skipping",
                    tgt_wf_id,
                    etype,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"missing tgt {etype} endpoint for wf {tgt_wf_id}"
                    )
                continue

            self._patch_endpoint(src_ep, tgt_ep, result, lock)

    def _patch_endpoint(
        self,
        src_ep: dict[str, Any],
        tgt_ep: dict[str, Any],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        src_ep_id = src_ep["id"]
        tgt_ep_id = tgt_ep["id"]
        etype = src_ep["endpoint_type"]

        src_conn_id = _extract_connector_id(src_ep)
        tgt_conn_id: str | None = None
        connector_unmapped = False
        if src_conn_id:
            with lock:
                tgt_conn_id = self.ctx.remap.resolve("connector", src_conn_id)
            if not tgt_conn_id:
                # Connector wasn't cloned (e.g. OAuth). Still patch the
                # connection_type so the endpoint is valid and the operator
                # only needs to re-bind the connector — skipping the whole
                # patch would leave connection_type empty and fail runs with
                # "Invalid source connection type".
                connector_unmapped = True
                logger.warning(
                    "%s endpoint src=%s tgt=%s: source connector %s has no "
                    "target remap — setting type only, connector needs UI config",
                    etype,
                    src_ep_id,
                    tgt_ep_id,
                    src_conn_id,
                )

        if self.ctx.options.dry_run:
            with lock:
                result.created += 1
                self.ctx.remap.record_planned("workflow_endpoint", src_ep_id)
            logger.info(
                "[dry-run] would PATCH %s endpoint src=%s -> tgt=%s (connector %s)",
                etype,
                src_ep_id,
                tgt_ep_id,
                tgt_conn_id or "<unset>",
            )
            return

        payload: dict[str, Any] = {
            "configuration": remap_uuids(
                src_ep.get("configuration") or {}, self.ctx.remap
            ),
        }
        src_connection_type = src_ep.get("connection_type")
        if src_connection_type is not None:
            payload["connection_type"] = src_connection_type
        # Omit connector_instance_id only when the source connector couldn't be
        # remapped, so the target keeps its connector for re-binding. A source
        # with no connector (e.g. API) still patches null to clear any stale one.
        if not connector_unmapped:
            payload["connector_instance_id"] = tgt_conn_id

        try:
            self.ctx.target.update_workflow_endpoint(tgt_ep_id, payload)
        except Exception as e:
            logger.exception(
                "Failed to PATCH %s endpoint tgt=%s: %s", etype, tgt_ep_id, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"patch {etype} {tgt_ep_id}: {e}")
            return

        with lock:
            result.created += 1
            self.ctx.remap.record("workflow_endpoint", src_ep_id, tgt_ep_id)
            if connector_unmapped:
                result.warnings.append(
                    f"{etype} endpoint {src_ep_id}: connector not cloned "
                    f"(src_connector={src_conn_id}) — connection_type set, "
                    "configure the connector in the UI"
                )
        logger.info(
            "patched %s endpoint src=%s -> tgt=%s (connector %s)",
            etype,
            src_ep_id,
            tgt_ep_id,
            tgt_conn_id or "<unset>",
        )
