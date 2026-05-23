"""Migrate ETL/TASK pipelines from source org to target org.

Pipelines FK ``workflow`` — the only entity remap needed. On create the
backend force-sets ``active=True`` and auto-provisions one active API
key per pipeline; if the source had additional rotated keys, those are
NOT mirrored (their UUIDs are server-generated and can't be preserved,
and operators rotate post-migration anyway).

``DEFAULT`` (legacy) and ``APP`` pipeline types are skipped — DEFAULT is
dead code from the v1 era; APP is a Streamlit-style deployment whose
lifecycle isn't shaped like an ETL/TASK pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.base import Phase, build_post_payload
from unstract.migration.report import MigrationReport, PhaseResult
from unstract.migration.walker import remap_uuids

logger = logging.getLogger(__name__)

PIPELINE_PATH = "pipeline/"
_MIGRATABLE_TYPES: frozenset[str] = frozenset({"ETL", "TASK"})


class PipelinePhase(Phase):
    name = "pipeline"

    def run(self, report: MigrationReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._writable = self.ctx.target.get_post_schema(PIPELINE_PATH)
        except Exception as e:
            logger.exception("Failed to fetch target POST schema for pipeline: %s", e)
            result.failed += 1
            result.errors.append(f"OPTIONS pipeline: {e}")
            return result

        try:
            src_pipelines = self.ctx.source.list_pipelines()
        except Exception as e:
            logger.exception("Failed to list source pipelines: %s", e)
            result.failed += 1
            result.errors.append(f"list source pipelines: {e}")
            return result

        migratable = [
            p for p in src_pipelines if p.get("pipeline_type") in _MIGRATABLE_TYPES
        ]
        skipped_types = len(src_pipelines) - len(migratable)
        if skipped_types:
            logger.info(
                "Found %d source pipeline(s); skipping %d of unsupported type (DEFAULT/APP)",
                len(src_pipelines), skipped_types,
            )
        else:
            logger.info("Found %d source pipeline(s)", len(src_pipelines))

        for src in migratable:
            self._migrate_one(src, result)
        return result

    def _migrate_one(self, src: dict[str, Any], result: PhaseResult) -> None:
        name = src["pipeline_name"]
        src_id = src["id"]
        src_wf_id = src.get("workflow") or src.get("workflow_id")

        if not src_wf_id:
            logger.warning("source pipeline '%s' has no workflow FK — skipping", name)
            result.skipped += 1
            return

        tgt_wf_id = self.ctx.remap.resolve("workflow", src_wf_id)
        if not tgt_wf_id:
            logger.warning(
                "no workflow remap for pipeline '%s' (src workflow %s) — skipping",
                name, src_wf_id,
            )
            result.skipped += 1
            return

        try:
            existing = self.ctx.target.list_pipelines(name=name)
        except Exception as e:
            logger.exception("Failed to GET pipeline %s on target: %s", name, e)
            result.failed += 1
            result.errors.append(f"GET {name}: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"pipeline '{name}' already exists in target as {tgt['id']}"
                )
            result.adopted += 1
            logger.info(
                "adopted pipeline '%s' src=%s -> tgt=%s", name, src_id, tgt["id"]
            )
        elif self.ctx.options.dry_run:
            result.skipped += 1
            logger.info("[dry-run] would create pipeline '%s' src=%s", name, src_id)
            return
        else:
            remapped = remap_uuids(src, self.ctx.remap)
            payload = build_post_payload(remapped, self._writable)
            payload["workflow"] = tgt_wf_id
            try:
                tgt = self.ctx.target.create_pipeline(payload)
            except Exception as e:
                logger.exception("Failed to create pipeline %s: %s", name, e)
                result.failed += 1
                result.errors.append(f"create {name}: {e}")
                return
            result.created += 1
            logger.info(
                "created pipeline '%s' src=%s -> tgt=%s", name, src_id, tgt["id"]
            )
            self._warn_if_extra_source_keys(src_id, name)

        self.ctx.remap.record("pipeline", src_id, tgt["id"])

    def _warn_if_extra_source_keys(self, src_pipeline_id: str, name: str) -> None:
        try:
            keys = self.ctx.source.list_pipeline_keys(src_pipeline_id)
        except Exception as e:
            logger.debug("Could not list source keys for pipeline %s: %s", name, e)
            return
        active = [k for k in keys if k.get("is_active")]
        if len(active) > 1:
            logger.warning(
                "source pipeline '%s' had %d active API keys; "
                "target has only the auto-provisioned default — "
                "re-create the rest manually if your clients depend on them",
                name, len(active),
            )
