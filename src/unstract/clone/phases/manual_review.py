"""Migrate cloud-only Manual Review (HITL) configuration.

Cloud-only: gated by ``probe_path`` — the orchestrator probes
``manual_review/auto_approval_settings/`` on source/target and skips the
phase entirely on an OSS deployment. ``auto_approval_settings/`` is a plain
ModelViewSet ``list`` that 200s with no query params, so it is the only MR
GET route safe to probe (the ``rule_engine`` / ``settings`` bare routes map
to workflow-scoped actions that need a URL kwarg).

Runs after ``workflow`` — every RuleEngine and HITLSettings row FKs a
workflow, so the workflow remap must already exist.

Config only. Runtime/queue data (HITLQueue packets, edited_data,
highlights, documents) is deliberately out of scope.

Three passes:

1. **Per-workflow** — for each source workflow that has a target mapping,
   replay its RuleEngine rows (one per ``rule_type``, with nested
   ``confidence_filters``) and its HITLSettings row, rebinding ``workflow``
   to the target id. Adopt-by-presence on the target.
2. **AutoApprovalSettings** — org-level, cloned once. ``auto_approved_users``
   holds source-org user pks; remapped by email (same as share replication).
   ``auto_approved_document_classes`` holds workflow/class-name strings with no
   reliable cross-org remap; carried verbatim with a warning.

MR config rows (RuleEngine/HITLSettings/AutoApprovalSettings) are workflow- or
org-scoped and inherit visibility from there — no per-entity share replication.
3. **ReviewApiKey** — recreated (the secret is server-minted and cannot be
   copied); operator is warned to re-wire external consumers.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, PhaseResult
from unstract.clone.sharing import (
    is_service_account,
    source_user_by_id,
    target_user_id_by_email,
)

logger = logging.getLogger(__name__)

AUTO_APPROVAL_PATH = "manual_review/auto_approval_settings/"

# RuleEngine create fields (RuleEngineSerializer, minus server-managed id /
# created_by / modified_by). ``confidence_filters`` is nested.
_RULE_FIELDS: tuple[str, ...] = (
    "rule_type",
    "percentage",
    "rule_string",
    "rule_json",
    "rule_logic",
)
# ConfidenceFilterSerializer create fields (id is server-managed).
_FILTER_FIELDS: tuple[str, ...] = ("field_key", "confidence_threshold")
# HITLSettingsSerializer is ``__all__``; these are the writable, non-FK-leaking
# fields (workflow is rebound separately; the rest are server-managed).
_SETTINGS_FIELDS: tuple[str, ...] = ("sync_with", "ttl_hours")


class ManualReviewPhase(Phase):
    name = "manual_review"
    probe_path = AUTO_APPROVAL_PATH

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            src_workflows = self.ctx.source.list_workflows()
        except Exception as e:
            logger.exception("Failed to list source workflows for manual_review: %s", e)
            result.failed += 1
            result.errors.append(f"list source workflows: {e}")
            return result

        logger.info(
            "manual_review: scanning %d source workflow(s) for HITL config",
            len(src_workflows),
        )
        self.parallel_map(
            src_workflows,
            lambda wf, lock: self._clone_workflow_config(wf, result, lock),
        )

        # Org-level entities — cloned once, outside the per-workflow fan-out.
        self._clone_auto_approval(result)
        self._clone_api_keys(result)
        return result

    # ----- per-workflow rules + settings -----

    def _clone_workflow_config(
        self, src_wf: dict[str, Any], result: PhaseResult, lock: threading.Lock
    ) -> None:
        src_wf_id = src_wf["id"]
        tgt_wf_id = self.ctx.remap.resolve("workflow", str(src_wf_id))
        if tgt_wf_id is None:
            # Workflow wasn't cloned (e.g. its tool was skipped) — nothing to bind to.
            logger.debug(
                "manual_review: workflow %s has no target mapping — skipping its "
                "HITL config",
                src_wf_id,
            )
            return

        self._clone_rules(src_wf_id, tgt_wf_id, result, lock)
        self._clone_settings(src_wf_id, tgt_wf_id, result, lock)

    def _clone_rules(
        self,
        src_wf_id: str,
        tgt_wf_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        for rule_type in self.ctx.source.MR_RULE_TYPES:
            try:
                src_rule = self.ctx.source.get_review_rule(src_wf_id, rule_type)
            except Exception as e:
                logger.exception(
                    "manual_review: failed to GET source %s rule for workflow %s: %s",
                    rule_type,
                    src_wf_id,
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"GET source {rule_type} rule wf={src_wf_id}: {e}"
                    )
                continue
            if not src_rule:
                continue

            if self.ctx.options.dry_run:
                with lock:
                    result.created += 1
                logger.info(
                    "[dry-run] would create %s rule for workflow %s",
                    rule_type,
                    tgt_wf_id,
                )
                continue

            # Adopt if the target workflow already carries a rule of this type
            # (unique per workflow+rule_type+org).
            if not self.ctx.remap.is_planned(tgt_wf_id):
                try:
                    existing = self.ctx.target.get_review_rule(tgt_wf_id, rule_type)
                except Exception as e:
                    logger.exception(
                        "manual_review: failed to GET target %s rule for wf %s: %s",
                        rule_type,
                        tgt_wf_id,
                        e,
                    )
                    with lock:
                        result.failed += 1
                        result.errors.append(
                            f"GET target {rule_type} rule wf={tgt_wf_id}: {e}"
                        )
                    continue
                if existing:
                    with lock:
                        result.adopted += 1
                    logger.info(
                        "adopted %s rule on workflow %s (already present)",
                        rule_type,
                        tgt_wf_id,
                    )
                    continue

            payload = self._rule_payload(src_rule, tgt_wf_id)
            try:
                self.ctx.target.create_review_rule(payload)
            except Exception as e:
                logger.exception(
                    "manual_review: failed to create %s rule for wf %s: %s",
                    rule_type,
                    tgt_wf_id,
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"create {rule_type} rule wf={tgt_wf_id}: {e}"
                    )
                continue
            with lock:
                result.created += 1
            logger.info("created %s rule for workflow %s", rule_type, tgt_wf_id)

    def _rule_payload(
        self, src_rule: dict[str, Any], tgt_wf_id: str
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"workflow": tgt_wf_id}
        for field in _RULE_FIELDS:
            if field in src_rule and src_rule[field] is not None:
                payload[field] = src_rule[field]
        filters = [
            {f: cf[f] for f in _FILTER_FIELDS if f in cf}
            for cf in src_rule.get("confidence_filters") or []
        ]
        if filters:
            payload["confidence_filters"] = filters
        return payload

    def _clone_settings(
        self,
        src_wf_id: str,
        tgt_wf_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            src_settings = self.ctx.source.get_review_settings(src_wf_id)
        except Exception as e:
            logger.exception(
                "manual_review: failed to GET source settings for wf %s: %s",
                src_wf_id,
                e,
            )
            with lock:
                result.failed += 1
                result.errors.append(f"GET source settings wf={src_wf_id}: {e}")
            return
        if not src_settings:
            return

        if self.ctx.options.dry_run:
            with lock:
                result.created += 1
            logger.info(
                "[dry-run] would create HITL settings for workflow %s", tgt_wf_id
            )
            return

        # HITLSettings is OneToOne on workflow — adopt if one already exists.
        if not self.ctx.remap.is_planned(tgt_wf_id):
            try:
                existing = self.ctx.target.get_review_settings(tgt_wf_id)
            except Exception as e:
                logger.exception(
                    "manual_review: failed to GET target settings for wf %s: %s",
                    tgt_wf_id,
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(f"GET target settings wf={tgt_wf_id}: {e}")
                return
            if existing:
                with lock:
                    result.adopted += 1
                logger.info(
                    "adopted HITL settings on workflow %s (already present)", tgt_wf_id
                )
                return

        payload: dict[str, Any] = {"workflow": tgt_wf_id}
        for field in _SETTINGS_FIELDS:
            if field in src_settings and src_settings[field] is not None:
                payload[field] = src_settings[field]
        try:
            self.ctx.target.create_review_settings(payload)
        except Exception as e:
            logger.exception(
                "manual_review: failed to create settings for wf %s: %s", tgt_wf_id, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"create settings wf={tgt_wf_id}: {e}")
            return
        with lock:
            result.created += 1
        logger.info("created HITL settings for workflow %s", tgt_wf_id)

    # ----- org-level: auto-approval -----

    def _clone_auto_approval(self, result: PhaseResult) -> None:
        try:
            src_rows = self.ctx.source.list_auto_approval_settings()
        except Exception as e:
            logger.exception(
                "manual_review: failed to list source auto-approval: %s", e
            )
            result.failed += 1
            result.errors.append(f"list source auto_approval: {e}")
            return
        if not src_rows:
            return
        src = src_rows[0]  # Unique per org — at most one row.

        doc_classes = src.get("auto_approved_document_classes") or []
        # Mixed workflow-id / class-name strings with no reliable cross-org
        # remap — carried verbatim and flagged for manual verification.
        if doc_classes:
            result.warnings.append(
                "auto-approval cloned with source-org strings in "
                "auto_approved_document_classes — these may need manual "
                "verification on the target"
            )

        users = self._remap_auto_approved_users(
            src.get("auto_approved_users") or [], result
        )

        if self.ctx.options.dry_run:
            result.created += 1
            logger.info("[dry-run] would create org auto-approval settings")
            return

        try:
            existing = self.ctx.target.list_auto_approval_settings()
        except Exception as e:
            logger.exception(
                "manual_review: failed to list target auto-approval: %s", e
            )
            result.failed += 1
            result.errors.append(f"list target auto_approval: {e}")
            return
        if existing:
            result.adopted += 1
            logger.info("adopted org auto-approval settings (already present)")
            return

        payload = {
            "auto_approved_document_classes": doc_classes,
            "auto_approved_users": users,
        }
        try:
            self.ctx.target.create_auto_approval_settings(payload)
        except Exception as e:
            logger.exception("manual_review: failed to create auto-approval: %s", e)
            result.failed += 1
            result.errors.append(f"create auto_approval: {e}")
            return
        result.created += 1
        logger.info("created org auto-approval settings")

    def _remap_auto_approved_users(
        self, src_user_ids: list[Any], result: PhaseResult
    ) -> list[str]:
        """Map source-org user pks to target pks by email (mirrors share
        replication). Unmappable users are skipped with a warning; an
        unavailable listing carries the field empty rather than failing."""
        if not src_user_ids:
            return []
        src_users = source_user_by_id(self.ctx)
        tgt_by_email = target_user_id_by_email(self.ctx)
        if src_users is None or tgt_by_email is None:
            result.warnings.append(
                "auto-approval: users listing unavailable — "
                f"{len(src_user_ids)} auto-approved user(s) not replicated"
            )
            return []
        mapped: list[str] = []
        for uid in src_user_ids:
            row = src_users.get(str(uid))
            if row is None:
                result.warnings.append(
                    f"auto-approval: source user id {uid} not in source users "
                    "listing — skipped"
                )
                continue
            if is_service_account(row):
                continue
            email = row["email"]
            tgt_uid = tgt_by_email.get(email.lower())
            if tgt_uid is None:
                result.warnings.append(
                    f"auto-approval: user {email} not found in target org — skipped"
                )
                continue
            # Stored as CharField (str) ids in the ArrayField.
            mapped.append(str(tgt_uid))
        return mapped

    # ----- org-level: review api keys -----

    def _clone_api_keys(self, result: PhaseResult) -> None:
        try:
            src_keys = self.ctx.source.list_review_api_keys()
        except Exception as e:
            logger.exception(
                "manual_review: failed to list source review api keys: %s", e
            )
            result.failed += 1
            result.errors.append(f"list source review_api_keys: {e}")
            return
        if not src_keys:
            return

        if self.ctx.options.dry_run:
            result.created += len(src_keys)
            logger.info("[dry-run] would recreate %d review api key(s)", len(src_keys))
            return

        # The api_key secret is server-minted and non-copyable; recreating
        # yields a NEW secret, so external consumers must be re-wired.
        result.warnings.append(
            f"{len(src_keys)} review API key(s) recreated with freshly minted "
            "secrets — the original key values cannot be copied; re-wire any "
            "external consumers to the new keys"
        )
        for src in src_keys:
            payload = {
                k: src[k]
                for k in ("class_name", "description", "is_active")
                if k in src and src[k] is not None
            }
            try:
                self.ctx.target.create_review_api_key(payload)
            except Exception as e:
                logger.exception(
                    "manual_review: failed to create review api key: %s", e
                )
                result.failed += 1
                result.errors.append(f"create review_api_key: {e}")
                continue
            result.created += 1
        logger.info("recreated %d review api key(s) on target", len(src_keys))
