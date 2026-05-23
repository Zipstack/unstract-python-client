"""Migrate connectors from source org to target org.

Same list -> per-id GET -> POST/adopt pattern as AdapterPhase. Two
connector-specific wrinkles:

1. **Auto-provisioned UCS connectors are skipped.** The Unstract Cloud
   Storage connector has its ``connector_metadata`` redacted to ``{}``
   on the wire, so we cannot reliably reconstruct it on the target.
   The target org is expected to have its own UCS row already; downstream
   phases (workflow endpoints) must remap by ``connector_id`` lookup
   rather than relying on the remap table here.

2. **OAuth ``connector_auth`` is stripped from responses.** Tokens are
   stored in a sibling ``ConnectorAuth`` row that the public API never
   exposes, so OAuth-backed connectors land on the target without
   refresh tokens. Operator must re-authorise on target.
"""

from __future__ import annotations

import logging
from typing import Any

from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.base import Phase
from unstract.migration.report import MigrationReport, PhaseResult

logger = logging.getLogger(__name__)

UCS_CONNECTOR_ID = "pcs|b8cd25cd-4452-4d54-bd5e-e7d71459b702"

CONNECTOR_POST_FIELDS = (
    "connector_id",
    "connector_name",
    "connector_metadata",
    "connector_version",
    "connector_type",
    "shared_to_org",
)


class ConnectorPhase(Phase):
    name = "connector"

    def run(self, report: MigrationReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            src_summaries = self.ctx.source.list_connectors()
        except Exception as e:
            logger.exception("Failed to list source connectors: %s", e)
            result.failed += 1
            result.errors.append(f"list source connectors: {e}")
            return result

        logger.info("Found %d connector(s) in source org", len(src_summaries))
        for summary in src_summaries:
            self._migrate_one(summary, result)
        return result

    def _migrate_one(self, summary: dict[str, Any], result: PhaseResult) -> None:
        name = summary["connector_name"]
        src_id = summary["id"]
        catalog_id = summary.get("connector_id")

        if catalog_id == UCS_CONNECTOR_ID:
            logger.info(
                "skipping UCS connector '%s' (src=%s) — auto-provisioned per-org",
                name, src_id,
            )
            result.skipped += 1
            return

        try:
            src = self.ctx.source.get_connector(src_id)
        except Exception as e:
            logger.exception("Failed to GET source connector %s detail: %s", name, e)
            result.failed += 1
            result.errors.append(f"GET source detail {name}: {e}")
            return

        try:
            existing = self.ctx.target.list_connectors(name=name)
        except Exception as e:
            logger.exception("Failed to GET connector %s on target: %s", name, e)
            result.failed += 1
            result.errors.append(f"GET {name}: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"connector '{name}' already exists in target as {tgt['id']}"
                )
            result.adopted += 1
            logger.info(
                "adopted connector '%s' src=%s -> tgt=%s",
                name, src_id, tgt["id"],
            )
        elif self.ctx.options.dry_run:
            result.skipped += 1
            logger.info("[dry-run] would create connector '%s' src=%s", name, src_id)
            return
        else:
            payload = {k: src[k] for k in CONNECTOR_POST_FIELDS if k in src and src[k] is not None}
            try:
                tgt = self.ctx.target.create_connector(payload)
            except Exception as e:
                logger.exception("Failed to create connector %s: %s", name, e)
                result.failed += 1
                result.errors.append(f"create {name}: {e}")
                return
            result.created += 1
            logger.info(
                "created connector '%s' src=%s -> tgt=%s",
                name, src_id, tgt["id"],
            )

        self.ctx.remap.record("connector", src_id, tgt["id"])
