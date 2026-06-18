"""Migrate ETL/TASK pipelines from source org to target org.

Pipelines FK ``workflow`` — the only entity remap needed. On create the
backend force-sets ``active=True`` and auto-provisions one active API
key per pipeline; if the source had additional rotated keys, those are
NOT mirrored (their UUIDs are server-generated and can't be preserved,
and operators rotate post-clone anyway).

``DEFAULT`` (legacy) and ``APP`` pipeline types are skipped — DEFAULT is
dead code from the v1 era; APP is a Streamlit-style deployment whose
lifecycle isn't shaped like an ETL/TASK pipeline.
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

PIPELINE_PATH = "pipeline/"
_MIGRATABLE_TYPES: frozenset[str] = frozenset({"ETL", "TASK"})


class PipelinePhase(Phase):
    name = "pipeline"
    share_path_template = "pipeline/{id}/share/"

    def run(self, report: CloneReport) -> PhaseResult:
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
                "Found %d source pipeline(s); skipping %d unsupported (DEFAULT/APP)",
                len(src_pipelines),
                skipped_types,
            )
        else:
            logger.info("Found %d source pipeline(s)", len(src_pipelines))

        self.parallel_map(
            migratable,
            lambda src, lock: self._clone_one(src, result, lock),
        )
        return result

    def _clone_one(
        self, src: dict[str, Any], result: PhaseResult, lock: threading.Lock
    ) -> None:
        name = src["pipeline_name"]
        src_id = src["id"]
        src_wf_id = src.get("workflow") or src.get("workflow_id")

        if not src_wf_id:
            logger.warning("source pipeline '%s' has no workflow FK — skipping", name)
            with lock:
                result.skipped += 1
            return

        with lock:
            tgt_wf_id = self.ctx.remap.resolve("workflow", src_wf_id)
        if not tgt_wf_id:
            logger.warning(
                "no workflow remap for pipeline '%s' (src workflow %s) — skipping",
                name,
                src_wf_id,
            )
            with lock:
                result.skipped += 1
            return

        try:
            existing = self.ctx.target.list_pipelines(name=name)
        except Exception as e:
            logger.exception("Failed to GET pipeline %s on target: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"GET {name}: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"pipeline '{name}' already exists in target as {tgt['id']}"
                )
            with lock:
                result.adopted += 1
            logger.info(
                "adopted pipeline '%s' src=%s -> tgt=%s", name, src_id, tgt["id"]
            )
        elif self.ctx.options.dry_run:
            with lock:
                result.created += 1
                self.ctx.remap.record_planned("pipeline", src_id)
            logger.info("[dry-run] would create pipeline '%s' src=%s", name, src_id)
            return
        else:
            try:
                full_src = self.ctx.source.get_pipeline(src_id)
            except Exception as e:
                logger.exception("Failed to GET source pipeline %s: %s", name, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"GET src pipeline {name}: {e}")
                return
            remapped = remap_uuids(full_src, self.ctx.remap)
            payload = build_post_payload(remapped, self._writable)
            payload["workflow"] = tgt_wf_id
            try:
                tgt = self.ctx.target.create_pipeline(payload)
            except Exception as e:
                logger.exception("Failed to create pipeline %s: %s", name, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"create {name}: {e}")
                return
            with lock:
                result.created += 1
            logger.info(
                "created pipeline '%s' src=%s -> tgt=%s", name, src_id, tgt["id"]
            )
            self._warn_if_extra_source_keys(src_id, name)

        with lock:
            self.ctx.remap.record("pipeline", src_id, tgt["id"])
        # List rows carry the share axes; detail fn is a safety net.
        self.apply_share(
            src=src,
            tgt_id=tgt["id"],
            label=name,
            result=result,
            lock=lock,
            src_detail_fn=lambda: self.ctx.source.get_pipeline(src_id),
        )

    def _warn_if_extra_source_keys(self, src_pipeline_id: str, name: str) -> None:
        try:
            keys = self.ctx.source.list_pipeline_keys(src_pipeline_id)
        except Exception as e:
            # WARNING (not DEBUG) — the operator needs to know we couldn't
            # check whether they have additional keys to recreate manually.
            logger.warning(
                "Could not list source keys for pipeline %s "
                "(extra-key check skipped; re-verify in source UI): %s",
                name,
                e,
            )
            return
        active = [k for k in keys if k.get("is_active")]
        if len(active) > 1:
            logger.warning(
                "source pipeline '%s' had %d active API keys; "
                "target has only the auto-provisioned default — "
                "re-create the rest manually if your clients depend on them",
                name,
                len(active),
            )
