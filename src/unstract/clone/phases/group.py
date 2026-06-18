"""Migrate org user groups from source org to target org.

Groups are matched by name and a like-named target group is always reused
(idempotent merge) — ``--on-name-conflict`` does not apply because merging
a sharing container cannot lose configuration. Runs first so downstream
phases can remap group ids when replicating share state.

With ``--clone-group-members`` each group's members are matched to
target-org users by email and bulk-added; misses surface as report
warnings. Service accounts never migrate.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, PhaseResult
from unstract.clone.sharing import is_service_account, target_user_id_by_email

logger = logging.getLogger(__name__)

GROUP_PATH = "groups/"


class GroupPhase(Phase):
    name = "group"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            src_groups = self.ctx.source.list_groups()
        except Exception as e:
            logger.exception("Failed to list source groups: %s", e)
            result.failed += 1
            result.errors.append(f"list source groups: {e}")
            return result
        try:
            tgt_groups = self.ctx.target.list_groups()
        except Exception as e:
            logger.exception("Failed to list target groups: %s", e)
            result.failed += 1
            result.errors.append(f"list target groups: {e}")
            return result

        # Single listing for the whole phase — the endpoint has no name
        # filter. Mutated under lock as creates land.
        target_by_name: dict[str, dict[str, Any]] = {g["name"]: g for g in tgt_groups}

        logger.info("Found %d group(s) in source org", len(src_groups))
        self.parallel_map(
            src_groups,
            lambda src, lock: self._clone_one(src, target_by_name, result, lock),
        )
        return result

    def _clone_one(
        self,
        src: dict[str, Any],
        target_by_name: dict[str, dict[str, Any]],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        name = src["name"]
        src_id = src["id"]

        with lock:
            tgt = target_by_name.get(name)

        if tgt is not None:
            with lock:
                result.adopted += 1
            logger.info("reusing group '%s' src=%s -> tgt=%s", name, src_id, tgt["id"])
        elif self.ctx.options.dry_run:
            with lock:
                result.created += 1
                self.ctx.remap.record_planned("group", str(src_id))
            logger.info("[dry-run] would create group '%s' src=%s", name, src_id)
            if self.ctx.options.clone_group_members:
                # Still computed so would-skip members show up in the report.
                self._clone_members(src, None, result, lock)
            return
        else:
            tgt = self._create_group(src, result, lock)
            if tgt is None:
                return
            with lock:
                result.created += 1
                target_by_name[name] = tgt
            logger.info("created group '%s' src=%s -> tgt=%s", name, src_id, tgt["id"])

        with lock:
            self.ctx.remap.record("group", str(src_id), str(tgt["id"]))

        if self.ctx.options.clone_group_members:
            self._clone_members(src, tgt, result, lock)

    def _create_group(
        self, src: dict[str, Any], result: PhaseResult, lock: threading.Lock
    ) -> dict[str, Any] | None:
        name = src["name"]
        payload = {"name": name, "description": src.get("description") or ""}
        try:
            self.ctx.target.create_group(payload)
            # Create response has no id — re-list and match by name.
            created = next(
                (g for g in self.ctx.target.list_groups() if g["name"] == name),
                None,
            )
        except Exception as e:
            logger.exception("Failed to create group %s: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"create {name}: {e}")
            return None
        if created is None:
            logger.error("group '%s' created but missing from target listing", name)
            with lock:
                result.failed += 1
                result.errors.append(f"create {name}: not found in target after POST")
            return None
        return created

    def _clone_members(
        self,
        src_group: dict[str, Any],
        tgt_group: dict[str, Any] | None,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        """Email-match source members to target users and bulk-add the hits.

        ``tgt_group`` is None only on the dry-run would-create path; the
        matching (and its warnings) still run for report visibility.
        """
        name = src_group["name"]
        try:
            members = self.ctx.source.list_group_members(src_group["id"])
        except Exception as e:
            logger.exception("Failed to list members of group %s: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"list members {name}: {e}")
            return
        if not members:
            return

        tgt_by_email = target_user_id_by_email(self.ctx)
        if tgt_by_email is None:
            with lock:
                result.warnings.append(
                    f"group '{name}': target users listing unavailable — "
                    f"{len(members)} member(s) not migrated"
                )
            return

        to_add: list[int] = []
        skipped: list[str] = []
        for member in members:
            email = (member.get("email") or "").lower()
            if not email or is_service_account(member):
                continue
            tgt_uid = tgt_by_email.get(email)
            if tgt_uid is None:
                skipped.append(email)
            else:
                to_add.append(tgt_uid)

        if skipped:
            with lock:
                result.warnings.extend(
                    f"group '{name}': member {email} not found in target org — skipped"
                    for email in skipped
                )

        if self.ctx.options.dry_run:
            logger.info(
                "[dry-run] would add %d member(s) to group '%s' (%d skipped)",
                len(to_add),
                name,
                len(skipped),
            )
            return
        if not to_add:
            return

        try:
            # Bulk-add is idempotent server-side, so adopt re-runs are safe.
            self.ctx.target.add_group_members(tgt_group["id"], to_add)
        except Exception as e:
            logger.exception("Failed to add members to group %s: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"add members {name}: {e}")
            return
        logger.info(
            "added %d member(s) to group '%s' (%d skipped)",
            len(to_add),
            name,
            len(skipped),
        )
