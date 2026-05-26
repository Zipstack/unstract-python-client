"""Migrate API deployments from source org to target org.

APIDeployment FKs ``workflow`` — remap via the WorkflowPhase table.
Backend enforces one active deployment per workflow and one
``api_name`` per organization, so adopt-by-name is the only safe
re-run strategy.

On create the backend auto-provisions a single active API key and
returns it on the response. Extra rotated keys on the source are NOT
mirrored (server-generated UUIDs can't be preserved; rotate
post-clone).
"""

from __future__ import annotations

import logging
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase, build_post_payload
from unstract.clone.report import CloneReport, PhaseResult
from unstract.clone.walker import remap_uuids

logger = logging.getLogger(__name__)

API_DEPLOYMENT_PATH = "api/deployment/"


class APIDeploymentPhase(Phase):
    name = "api_deployment"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._writable = self.ctx.target.get_post_schema(API_DEPLOYMENT_PATH)
        except Exception as e:
            logger.exception(
                "Failed to fetch target POST schema for api_deployment: %s", e
            )
            result.failed += 1
            result.errors.append(f"OPTIONS api_deployment: {e}")
            return result

        try:
            src_deployments = self.ctx.source.list_api_deployments()
        except Exception as e:
            logger.exception("Failed to list source api_deployments: %s", e)
            result.failed += 1
            result.errors.append(f"list source api_deployments: {e}")
            return result

        logger.info("Found %d source API deployment(s)", len(src_deployments))
        for src in src_deployments:
            self._clone_one(src, result)
        return result

    def _clone_one(self, src: dict[str, Any], result: PhaseResult) -> None:
        api_name = src["api_name"]
        src_id = src["id"]
        src_wf_id = src.get("workflow") or src.get("workflow_id")

        if not src_wf_id:
            logger.warning(
                "source api_deployment '%s' has no workflow FK — skipping", api_name
            )
            result.skipped += 1
            return

        tgt_wf_id = self.ctx.remap.resolve("workflow", src_wf_id)
        if not tgt_wf_id:
            logger.warning(
                "no workflow remap for api_deployment '%s' (src workflow %s) — skipping",
                api_name,
                src_wf_id,
            )
            result.skipped += 1
            return

        try:
            existing = self.ctx.target.list_api_deployments(api_name=api_name)
        except Exception as e:
            logger.exception(
                "Failed to GET api_deployment %s on target: %s", api_name, e
            )
            result.failed += 1
            result.errors.append(f"GET {api_name}: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"api_deployment '{api_name}' already exists in target as {tgt['id']}"
                )
            result.adopted += 1
            logger.info(
                "adopted api_deployment '%s' src=%s -> tgt=%s",
                api_name,
                src_id,
                tgt["id"],
            )
        elif self.ctx.options.dry_run:
            result.skipped += 1
            logger.info(
                "[dry-run] would create api_deployment '%s' src=%s", api_name, src_id
            )
            return
        else:
            try:
                # list serializer can strip fields the create serializer expects.
                full_src = self.ctx.source.get_api_deployment(src_id)
            except Exception as e:
                logger.exception(
                    "Failed to GET source api_deployment %s: %s", api_name, e
                )
                result.failed += 1
                result.errors.append(f"GET src api_deployment {api_name}: {e}")
                return
            remapped = remap_uuids(full_src, self.ctx.remap)
            payload = build_post_payload(remapped, self._writable)
            payload["workflow"] = tgt_wf_id
            try:
                tgt = self.ctx.target.create_api_deployment(payload)
            except Exception as e:
                logger.exception("Failed to create api_deployment %s: %s", api_name, e)
                result.failed += 1
                result.errors.append(f"create {api_name}: {e}")
                return
            result.created += 1
            logger.info(
                "created api_deployment '%s' src=%s -> tgt=%s",
                api_name,
                src_id,
                tgt["id"],
            )
            self._warn_if_extra_source_keys(src_id, api_name)

        self.ctx.remap.record("api_deployment", src_id, tgt["id"])

    def _warn_if_extra_source_keys(self, src_deployment_id: str, name: str) -> None:
        try:
            keys = self.ctx.source.list_api_deployment_keys(src_deployment_id)
        except Exception as e:
            # WARNING (not DEBUG) — the operator needs to know we couldn't
            # check whether they have additional keys to recreate manually.
            logger.warning(
                "Could not list source keys for api_deployment %s "
                "(extra-key check skipped; re-verify in source UI): %s",
                name,
                e,
            )
            return
        active = [k for k in keys if k.get("is_active")]
        if len(active) > 1:
            logger.warning(
                "source api_deployment '%s' had %d active API keys; "
                "target has only the auto-provisioned default — "
                "re-create the rest manually if your clients depend on them",
                name,
                len(active),
            )
