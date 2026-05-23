"""Migrate adapters from source org to target org.

Reference implementation for the get-or-create pattern: list-by-name GET
against target, POST create if missing, record source->target UUID in the
remap table for downstream phases.

``AdapterInstanceManager.for_user(service_account)`` returns only
non-frictionless adapters, so frictionless onboarding adapters are
intentionally excluded from migration.
"""

from __future__ import annotations

import logging
from typing import Any

from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.base import Phase
from unstract.migration.report import MigrationReport, PhaseResult

logger = logging.getLogger(__name__)

# Fields copied verbatim from source GET into target POST. Everything else
# (id, created_by, deprecation flags, icon, etc.) is either auto-set by the
# target backend or derived — carrying it would either be ignored or cause
# validation noise.
ADAPTER_POST_FIELDS = (
    "adapter_id",
    "adapter_name",
    "adapter_type",
    "adapter_metadata",
    "description",
)


class AdapterPhase(Phase):
    name = "adapter"

    def run(self, report: MigrationReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            src_summaries = self.ctx.source.list_adapters()
        except Exception as e:
            logger.exception("Failed to list source adapters: %s", e)
            result.failed += 1
            result.errors.append(f"list source adapters: {e}")
            return result

        logger.info("Found %d adapter(s) in source org", len(src_summaries))
        for summary in src_summaries:
            self._migrate_one(summary, result)
        return result

    def _migrate_one(self, summary: dict[str, Any], result: PhaseResult) -> None:
        name = summary["adapter_name"]
        atype = summary["adapter_type"]
        src_id = summary["id"]
        # List response omits adapter_metadata (see AdapterListSerializer);
        # fetch the detail endpoint to pick up the decrypted metadata.
        try:
            src = self.ctx.source.get_adapter(src_id)
        except Exception as e:
            logger.exception("Failed to GET source adapter %s [%s] detail: %s", name, atype, e)
            result.failed += 1
            result.errors.append(f"GET source detail {name} [{atype}]: {e}")
            return

        try:
            existing = self.ctx.target.list_adapters(name=name, adapter_type=atype)
        except Exception as e:
            logger.exception("Failed to GET adapter %s [%s] on target: %s", name, atype, e)
            result.failed += 1
            result.errors.append(f"GET {name} [{atype}]: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"adapter '{name}' [{atype}] already exists in target as {tgt['id']}"
                )
            result.adopted += 1
            logger.info(
                "adopted adapter '%s' [%s] src=%s -> tgt=%s",
                name, atype, src_id, tgt["id"],
            )
        elif self.ctx.options.dry_run:
            result.skipped += 1
            logger.info("[dry-run] would create adapter '%s' [%s] src=%s", name, atype, src_id)
            return
        else:
            payload = {k: src[k] for k in ADAPTER_POST_FIELDS if k in src and src[k] is not None}
            try:
                tgt = self.ctx.target.create_adapter(payload)
            except Exception as e:
                logger.exception("Failed to create adapter %s [%s]: %s", name, atype, e)
                result.failed += 1
                result.errors.append(f"create {name} [{atype}]: {e}")
                return
            result.created += 1
            logger.info(
                "created adapter '%s' [%s] src=%s -> tgt=%s",
                name, atype, src_id, tgt["id"],
            )

        self.ctx.remap.record("adapter", src_id, tgt["id"])
