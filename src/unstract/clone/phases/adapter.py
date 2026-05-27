"""Migrate adapters from source org to target org.

Reference implementation for the get-or-create pattern: list-by-name GET
against target, POST create if missing, record source->target UUID in the
remap table for downstream phases.

Frictionless onboarding adapters are excluded — the backend's
service-account queryset already filters them out, so clone never
sees them.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase, build_post_payload
from unstract.clone.report import CloneReport, PhaseResult

logger = logging.getLogger(__name__)

ADAPTER_PATH = "adapter/"


class AdapterPhase(Phase):
    name = "adapter"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._writable = self.ctx.target.get_post_schema(ADAPTER_PATH)
        except Exception as e:
            logger.exception("Failed to fetch target POST schema for adapter: %s", e)
            result.failed += 1
            result.errors.append(f"OPTIONS adapter: {e}")
            return result
        try:
            src_summaries = self.ctx.source.list_adapters()
        except Exception as e:
            logger.exception("Failed to list source adapters: %s", e)
            result.failed += 1
            result.errors.append(f"list source adapters: {e}")
            return result

        logger.info("Found %d adapter(s) in source org", len(src_summaries))
        self.parallel_map(
            src_summaries,
            lambda summary, lock: self._clone_one(summary, result, lock),
        )
        return result

    def _clone_one(
        self, summary: dict[str, Any], result: PhaseResult, lock: threading.Lock
    ) -> None:
        name = summary["adapter_name"]
        atype = summary["adapter_type"]
        src_id = summary["id"]
        try:
            src = self.ctx.source.get_adapter(src_id)
        except Exception as e:
            logger.exception(
                "Failed to GET source adapter %s [%s] detail: %s", name, atype, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"GET source detail {name} [{atype}]: {e}")
            return

        try:
            existing = self.ctx.target.list_adapters(name=name, adapter_type=atype)
        except Exception as e:
            logger.exception(
                "Failed to GET adapter %s [%s] on target: %s", name, atype, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"GET {name} [{atype}]: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"adapter '{name}' [{atype}] already on target as {tgt['id']}"
                )
            with lock:
                result.adopted += 1
            logger.info(
                "adopted adapter '%s' [%s] src=%s -> tgt=%s",
                name,
                atype,
                src_id,
                tgt["id"],
            )
        elif self.ctx.options.dry_run:
            with lock:
                result.skipped += 1
            logger.info(
                "[dry-run] would create adapter '%s' [%s] src=%s", name, atype, src_id
            )
            return
        else:
            payload = build_post_payload(src, self._writable)
            try:
                tgt = self.ctx.target.create_adapter(payload)
            except Exception as e:
                logger.exception("Failed to create adapter %s [%s]: %s", name, atype, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"create {name} [{atype}]: {e}")
                return
            with lock:
                result.created += 1
            logger.info(
                "created adapter '%s' [%s] src=%s -> tgt=%s",
                name,
                atype,
                src_id,
                tgt["id"],
            )

        with lock:
            self.ctx.remap.record("adapter", src_id, tgt["id"])
