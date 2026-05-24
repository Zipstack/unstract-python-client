"""Migrate connectors from source org to target org.

Same list -> per-id GET -> POST/adopt pattern as AdapterPhase. Two
connector-specific wrinkles:

1. **Connectors with redacted metadata are skipped.** The backend
   serializer strips ``connector_metadata`` for auto-provisioned rows
   (e.g. Unstract Cloud Storage), so the SDK cannot reconstruct them
   on the target. We detect this by inspecting the source GET response:
   a falsy ``connector_metadata`` means the operator must rely on the
   target's own provisioning (or re-create the row manually) — the
   remap table records no entry for these.

2. **OAuth ``connector_auth`` is stripped from responses.** Tokens are
   stored in a sibling ``ConnectorAuth`` row that the public API never
   exposes, so OAuth-backed connectors land on the target without
   refresh tokens. Operator must re-authorise on target.
"""

from __future__ import annotations

import logging
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase, build_post_payload
from unstract.clone.report import CloneReport, PhaseResult

logger = logging.getLogger(__name__)

CONNECTOR_PATH = "connector/"


class ConnectorPhase(Phase):
    name = "connector"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._writable = self.ctx.target.get_post_schema(CONNECTOR_PATH)
        except Exception as e:
            logger.exception("Failed to fetch target POST schema for connector: %s", e)
            result.failed += 1
            result.errors.append(f"OPTIONS connector: {e}")
            return result
        try:
            src_summaries = self.ctx.source.list_connectors()
        except Exception as e:
            logger.exception("Failed to list source connectors: %s", e)
            result.failed += 1
            result.errors.append(f"list source connectors: {e}")
            return result

        logger.info("Found %d connector(s) in source org", len(src_summaries))
        for summary in src_summaries:
            self._clone_one(summary, result)
        return result

    def _clone_one(self, summary: dict[str, Any], result: PhaseResult) -> None:
        name = summary["connector_name"]
        src_id = summary["id"]

        try:
            src = self.ctx.source.get_connector(src_id)
        except Exception as e:
            logger.exception("Failed to GET source connector %s detail: %s", name, e)
            result.failed += 1
            result.errors.append(f"GET source detail {name}: {e}")
            return

        # Empty metadata means the backend redacted it (auto-provisioned rows
        # like Unstract Cloud Storage). We cannot reconstruct it on target.
        if not src.get("connector_metadata"):
            logger.info(
                "skipping connector '%s' (src=%s, catalog=%s) — source returned no metadata",
                name, src_id, src.get("connector_id"),
            )
            result.skipped += 1
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
            payload = build_post_payload(src, self._writable)
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
