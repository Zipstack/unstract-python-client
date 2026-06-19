"""Migrate connectors from source org to target org.

Same list -> per-id GET -> adopt/POST pattern as AdapterPhase, but a
same-name target connector is adopted *before* any recreate attempt.
Two connector-specific wrinkles make that ordering matter:

1. **Redacted-metadata connectors can't be recreated.** Auto-provisioned
   rows (e.g. Unstract Cloud Storage) come back without
   ``connector_metadata``, so the SDK has nothing to reconstruct.

2. **OAuth credentials can't be cloned.** OAuth refresh tokens are never
   returned by the API, and a token can only be minted by completing the
   OAuth flow as the target user — which only happens in the UI.

Neither can be created from source. Both are skipped, their ids recorded
so dependent workflows cascade-skip (avoiding pipelines that fail every
run). The recovery path is the same for both: provision the connector on
the target with the *same name*, then re-run — the adopt-first lookup
picks it up and wires the dependent workflow endpoints.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase, build_post_payload
from unstract.clone.report import CloneReport, PhaseResult

logger = logging.getLogger(__name__)

CONNECTOR_PATH = "connector/"

# A POST carrying these OAuth token keys triggers a token refresh against
# the source user's credentials — guaranteed to fail on the target. Detect
# here and skip ahead of POST.
_OAUTH_TOKEN_KEYS: frozenset[str] = frozenset({"access_token", "refresh_token"})


def _has_oauth_tokens(metadata: dict[str, Any]) -> bool:
    return any(metadata.get(k) for k in _OAUTH_TOKEN_KEYS)


class ConnectorPhase(Phase):
    name = "connector"
    share_path_template = "connector/{id}/share/"

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
        self.parallel_map(
            src_summaries,
            lambda summary, lock: self._clone_one(summary, result, lock),
        )
        return result

    def _clone_one(
        self, summary: dict[str, Any], result: PhaseResult, lock: threading.Lock
    ) -> None:
        name = summary["connector_name"]
        src_id = summary["id"]

        try:
            src = self.ctx.source.get_connector(src_id)
        except Exception as e:
            logger.exception("Failed to GET source connector %s detail: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"GET source detail {name}: {e}")
            return

        try:
            existing = self.ctx.target.list_connectors(name=name)
        except Exception as e:
            logger.exception("Failed to GET connector %s on target: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"GET {name}: {e}")
            return

        # Adopt a same-name target connector before anything else. This is the
        # only way OAuth / redacted-metadata connectors come across: the
        # operator provisions one on the target (where OAuth can complete), and
        # a re-run adopts it and wires dependent workflow endpoints.
        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"connector '{name}' already exists in target as {tgt['id']}"
                )
            with lock:
                result.adopted += 1
            logger.info(
                "adopted connector '%s' src=%s -> tgt=%s",
                name,
                src_id,
                tgt["id"],
            )
        else:
            tgt = self._recreate_or_skip(src, name, src_id, result, lock)
            if tgt is None:
                return

        with lock:
            self.ctx.remap.record("connector", src_id, tgt["id"])
        # Source detail (fetched above) carries the share axes.
        self.apply_share(
            src=src, tgt_id=tgt["id"], label=name, result=result, lock=lock
        )

    def _recreate_or_skip(
        self,
        src: dict[str, Any],
        name: str,
        src_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> dict[str, Any] | None:
        """Recreate a connector from source, or skip when it can't be.

        Returns the target connector dict on create, or ``None`` when the
        connector is skipped (no metadata to reconstruct, or OAuth credentials
        that the API can't clone). Skipped ids are recorded so dependent
        workflows cascade-skip.
        """
        metadata = src.get("connector_metadata") or {}
        if not metadata:
            logger.info(
                "skipping connector '%s' (src=%s, catalog=%s) — metadata not "
                "exposed by the API (auto-provisioned, e.g. Cloud Storage); "
                "provision it on the target with the same name and re-run to adopt",
                name,
                src_id,
                src.get("connector_id"),
            )
            with lock:
                result.skipped += 1
                self.ctx.skipped_connector_ids.add(src_id)
                result.warnings.append(
                    f"connector '{name}' skipped — metadata not exposed by the API "
                    "(auto-provisioned); provision it on target with the same name "
                    "and re-run to adopt"
                )
            return None

        if _has_oauth_tokens(metadata):
            logger.warning(
                "skipping connector '%s' (src=%s, catalog=%s) — OAuth-backed; "
                "credentials can't be cloned. Recreate + authorise it on the "
                "target with the same name, then re-run to adopt and wire "
                "dependent workflows.",
                name,
                src_id,
                src.get("connector_id"),
            )
            with lock:
                result.skipped += 1
                self.ctx.skipped_connector_ids.add(src_id)
                result.warnings.append(
                    f"connector '{name}' skipped — OAuth-backed, credentials can't "
                    "be cloned; recreate + authorise on target with the same name, "
                    "then re-run to adopt"
                )
            return None

        if self.ctx.options.dry_run:
            with lock:
                result.created += 1
                self.ctx.remap.record_planned("connector", src_id)
            logger.info("[dry-run] would create connector '%s' src=%s", name, src_id)
            return None

        payload = build_post_payload(src, self._writable)
        try:
            tgt = self.ctx.target.create_connector(payload)
        except Exception as e:
            logger.exception("Failed to create connector %s: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"create {name}: {e}")
            return None
        with lock:
            result.created += 1
        logger.info(
            "created connector '%s' src=%s -> tgt=%s",
            name,
            src_id,
            tgt["id"],
        )
        return tgt
