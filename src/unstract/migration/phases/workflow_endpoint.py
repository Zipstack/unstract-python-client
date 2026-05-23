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
from typing import Any

from unstract.migration.phases.base import Phase
from unstract.migration.report import MigrationReport, PhaseResult
from unstract.migration.walker import remap_uuids

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

    def run(self, report: MigrationReport) -> PhaseResult:
        result = report.get_phase(self.name)
        workflow_remap = self.ctx.remap.snapshot().get("workflow", {})
        if not workflow_remap:
            logger.info("No workflows in remap; nothing to do for workflow_endpoint phase")
            return result

        for src_wf_id, tgt_wf_id in workflow_remap.items():
            self._migrate_workflow_endpoints(src_wf_id, tgt_wf_id, result)
        return result

    def _migrate_workflow_endpoints(
        self, src_wf_id: str, tgt_wf_id: str, result: PhaseResult
    ) -> None:
        try:
            src_endpoints = self.ctx.source.list_workflow_endpoints(
                workflow_id=src_wf_id
            )
        except Exception as e:
            logger.exception(
                "Failed to list source endpoints for wf %s: %s", src_wf_id, e
            )
            result.failed += 1
            result.errors.append(f"list src endpoints {src_wf_id}: {e}")
            return

        try:
            tgt_endpoints = self.ctx.target.list_workflow_endpoints(
                workflow_id=tgt_wf_id
            )
        except Exception as e:
            logger.exception(
                "Failed to list target endpoints for wf %s: %s", tgt_wf_id, e
            )
            result.failed += 1
            result.errors.append(f"list tgt endpoints {tgt_wf_id}: {e}")
            return

        tgt_by_type = {ep["endpoint_type"]: ep for ep in tgt_endpoints}

        for src_ep in src_endpoints:
            etype = src_ep["endpoint_type"]
            tgt_ep = tgt_by_type.get(etype)
            if tgt_ep is None:
                # Target should have auto-created this; missing means the
                # workflow create flow failed earlier — surface loudly.
                logger.warning(
                    "target workflow %s missing %s endpoint — skipping",
                    tgt_wf_id, etype,
                )
                result.failed += 1
                result.errors.append(
                    f"missing tgt {etype} endpoint for wf {tgt_wf_id}"
                )
                continue

            self._patch_endpoint(src_ep, tgt_ep, result)

    def _patch_endpoint(
        self, src_ep: dict[str, Any], tgt_ep: dict[str, Any], result: PhaseResult
    ) -> None:
        src_ep_id = src_ep["id"]
        tgt_ep_id = tgt_ep["id"]
        etype = src_ep["endpoint_type"]

        if self.ctx.options.dry_run:
            result.skipped += 1
            logger.info(
                "[dry-run] would PATCH %s endpoint src=%s -> tgt=%s",
                etype, src_ep_id, tgt_ep_id,
            )
            return

        src_conn_id = _extract_connector_id(src_ep)
        tgt_conn_id: str | None = None
        if src_conn_id:
            tgt_conn_id = self.ctx.remap.resolve("connector", src_conn_id)
            if not tgt_conn_id:
                logger.warning(
                    "no connector remap for %s on %s endpoint %s — leaving unset",
                    src_conn_id, etype, src_ep_id,
                )

        payload: dict[str, Any] = {
            "connection_type": src_ep.get("connection_type") or "",
            "configuration": remap_uuids(src_ep.get("configuration") or {}, self.ctx.remap),
            "connector_instance_id": tgt_conn_id,
        }

        try:
            self.ctx.target.update_workflow_endpoint(tgt_ep_id, payload)
        except Exception as e:
            logger.exception(
                "Failed to PATCH %s endpoint tgt=%s: %s", etype, tgt_ep_id, e
            )
            result.failed += 1
            result.errors.append(f"patch {etype} {tgt_ep_id}: {e}")
            return

        result.created += 1
        logger.info(
            "patched %s endpoint src=%s -> tgt=%s (connector %s)",
            etype, src_ep_id, tgt_ep_id, tgt_conn_id,
        )
        self.ctx.remap.record("workflow_endpoint", src_ep_id, tgt_ep_id)
