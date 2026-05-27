"""Migrate tags from source org to target org.

Tags are flat (``name`` + ``description``) with a per-org uniqueness
constraint on ``name``. No metadata, no encryption, no list-vs-detail
divergence — the simplest entity in the clone set.

List endpoint paginates; ``PlatformClient.list_tags`` already unwraps
the envelope.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase, build_post_payload
from unstract.clone.report import CloneReport, PhaseResult

logger = logging.getLogger(__name__)

TAG_PATH = "tags/"


class TagPhase(Phase):
    name = "tag"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._writable = self.ctx.target.get_post_schema(TAG_PATH)
        except Exception as e:
            logger.exception("Failed to fetch target POST schema for tag: %s", e)
            result.failed += 1
            result.errors.append(f"OPTIONS tag: {e}")
            return result
        try:
            src_tags = self.ctx.source.list_tags()
        except Exception as e:
            logger.exception("Failed to list source tags: %s", e)
            result.failed += 1
            result.errors.append(f"list source tags: {e}")
            return result

        logger.info("Found %d tag(s) in source org", len(src_tags))
        self.parallel_map(
            src_tags,
            lambda src, lock: self._clone_one(src, result, lock),
        )
        return result

    def _clone_one(
        self, src: dict[str, Any], result: PhaseResult, lock: threading.Lock
    ) -> None:
        name = src["name"]
        src_id = src["id"]

        try:
            existing = self.ctx.target.list_tags(name=name)
        except Exception as e:
            logger.exception("Failed to GET tag %s on target: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"GET {name}: {e}")
            return

        if existing:
            tgt = existing[0]
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"tag '{name}' already exists in target as {tgt['id']}"
                )
            with lock:
                result.adopted += 1
            logger.info("adopted tag '%s' src=%s -> tgt=%s", name, src_id, tgt["id"])
        elif self.ctx.options.dry_run:
            with lock:
                result.skipped += 1
            logger.info("[dry-run] would create tag '%s' src=%s", name, src_id)
            return
        else:
            payload = build_post_payload(src, self._writable)
            try:
                tgt = self.ctx.target.create_tag(payload)
            except Exception as e:
                logger.exception("Failed to create tag %s: %s", name, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"create {name}: {e}")
                return
            with lock:
                result.created += 1
            logger.info("created tag '%s' src=%s -> tgt=%s", name, src_id, tgt["id"])

        with lock:
            self.ctx.remap.record("tag", src_id, tgt["id"])
