"""Migrate cloud-only Agentic ("Agentic Prompt Studio") projects + children.

Cloud-only: gated by ``probe_path`` — the orchestrator probes
``agentic/projects/`` on source/target and skips the phase entirely on an
OSS deployment. Runs after ``adapter`` (a project FKs four adapters).

Standalone: owns its own project + registry, independent of ``custom_tool``.

Per source project:

1. Adopt-by-name if the target already has a project with the same ``name``
   (respecting ``on_name_conflict``), else create fresh from the OPTIONS
   schema, remapping the four adapter FKs
   (``llm_connector_id`` / ``agent_llm_connector_id`` /
   ``lightweight_llm_connector_id`` / ``text_extractor_connector_id``) via the
   ``adapter`` remap. An adapter that doesn't resolve is omitted with a warning
   (the slot stays unset, like a draft lookup adapter). Records an
   ``agentic_project`` remap.
2. In dependency order under the target project:
   - **prompt-versions**: ``parent_version`` is a self-FK, so parents (whose
     ``parent_version`` is ``None``) clone first; each new id is recorded in a
     per-run ``agentic_prompt_version`` table and the child's ``parent_version``
     resolves through it. ``project`` is bound to the target id.
   - **schemas**: bound to the target ``project`` and created.
   - **documents**: source uploads re-uploaded to the target project (skipping
     filenames already present); they live in their own store, not the Prompt
     Studio document table.
   - **verified data**: ground-truth rows re-pointed to the cloned document by
     filename. Extracted/comparison data is regenerable and not cloned.
3. **Registry**: if the project has an active schema + prompt, republish its
   registry entry via the ``export`` action (mirror of custom_tool) and
   record an ``agentic_studio_registry`` remap. Projects with no source
   registry are left unexported.

Org-level agentic-setting rows (global key/value, no project FK) are cloned
once as a flat adopt-by-key / create pass — they are org singletons, not
per-project config.
"""

from __future__ import annotations

import logging
import mimetypes
import threading
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase, build_post_payload
from unstract.clone.report import CloneReport, PhaseResult
from unstract.clone.sharing import apply_share_state

logger = logging.getLogger(__name__)

AGENTIC_PROJECTS_PATH = "agentic/projects/"

# Project adapter FK slots: serializer field name -> ``adapter`` remap source.
_ADAPTER_SLOTS: tuple[str, ...] = (
    "llm_connector_id",
    "agent_llm_connector_id",
    "lightweight_llm_connector_id",
    "text_extractor_connector_id",
)


class AgenticStudioPhase(Phase):
    name = "agentic_studio"
    probe_path = AGENTIC_PROJECTS_PATH

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._writable = self.ctx.target.get_post_schema(AGENTIC_PROJECTS_PATH)
        except Exception as e:
            logger.exception("Failed to fetch target POST schema for agentic: %s", e)
            result.failed += 1
            result.errors.append(f"OPTIONS agentic projects: {e}")
            return result
        try:
            src_projects = self.ctx.source.list_agentic_projects()
        except Exception as e:
            logger.exception("Failed to list source agentic projects: %s", e)
            result.failed += 1
            result.errors.append(f"list source agentic projects: {e}")
            return result
        try:
            tgt_projects = self.ctx.target.list_agentic_projects()
        except Exception as e:
            logger.exception("Failed to list target agentic projects: %s", e)
            result.failed += 1
            result.errors.append(f"list target agentic projects: {e}")
            return result

        logger.info("Found %d agentic project(s) in source org", len(src_projects))
        # Updated under lock on fresh create so same-name source rows adopt.
        target_by_name: dict[str, dict[str, Any]] = {
            p["name"]: p for p in tgt_projects
        }

        self.parallel_map(
            src_projects,
            lambda src, lock: self._clone_project(src, target_by_name, result, lock),
        )

        # Org-global settings: a flat pass, not tied to any project.
        self._clone_settings(result)
        return result

    # ----- projects -----

    def _clone_project(
        self,
        src: dict[str, Any],
        target_by_name: dict[str, dict[str, Any]],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        name = src["name"]
        src_project_id = src["id"]

        with lock:
            match = target_by_name.get(name)

        if match is not None:
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"agentic project '{name}' already exists in target "
                    f"as {match['id']}"
                )
            tgt_project_id = match["id"]
            with lock:
                result.adopted += 1
                self.ctx.remap.record("agentic_project", src_project_id, tgt_project_id)
            logger.info(
                "adopted agentic project '%s' src=%s -> tgt=%s",
                name,
                src_project_id,
                tgt_project_id,
            )
        elif self.ctx.options.dry_run:
            with lock:
                result.created += 1
                self.ctx.remap.record_planned("agentic_project", src_project_id)
            logger.info(
                "[dry-run] would create agentic project '%s' src=%s",
                name,
                src_project_id,
            )
            # Plan child ids so downstream plan-counts stay consistent.
            self._plan_children(src_project_id, result, lock)
            return
        else:
            payload = self._build_project_payload(src, name, result, lock)
            try:
                tgt = self.ctx.target.create_agentic_project(payload)
            except Exception as e:
                logger.exception("Failed to create agentic project '%s': %s", name, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"create agentic project {name}: {e}")
                return
            tgt_project_id = tgt["id"]
            with lock:
                result.created += 1
                target_by_name[name] = {"id": tgt_project_id, "name": name}
                self.ctx.remap.record("agentic_project", src_project_id, tgt_project_id)
            logger.info(
                "created agentic project '%s' src=%s -> tgt=%s",
                name,
                src_project_id,
                tgt_project_id,
            )

        # Children + registry + share write to the real target only.
        if self.ctx.options.dry_run:
            return
        self._replicate_share(src, name, tgt_project_id, result, lock)
        self._clone_prompt_versions(name, src_project_id, tgt_project_id, result, lock)
        self._clone_schemas(name, src_project_id, tgt_project_id, result, lock)
        self._clone_documents(name, src_project_id, tgt_project_id, result, lock)
        self._clone_verified_data(name, src_project_id, tgt_project_id, result, lock)
        self._republish_registry(name, src_project_id, tgt_project_id, result, lock)

    def _build_project_payload(
        self,
        src: dict[str, Any],
        name: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> dict[str, Any]:
        payload = build_post_payload(src, self._writable)
        # Remap each adapter FK; omit any slot that doesn't resolve (leaves it
        # unset — the operator wires it on target later).
        for slot in _ADAPTER_SLOTS:
            src_adapter_id = src.get(slot)
            if not src_adapter_id:
                payload.pop(slot, None)
                continue
            tgt_adapter_id = self.ctx.remap.resolve("adapter", str(src_adapter_id))
            if tgt_adapter_id is None:
                payload.pop(slot, None)
                logger.warning(
                    "agentic project '%s': %s adapter %s has no target mapping — "
                    "leaving it unset",
                    name,
                    slot,
                    src_adapter_id,
                )
                with lock:
                    result.warnings.append(
                        f"agentic project {name}: {slot} adapter not remapped — "
                        "left unset"
                    )
                continue
            payload[slot] = tgt_adapter_id
        return payload

    # ----- share state -----

    def _replicate_share(
        self,
        src: dict[str, Any],
        name: str,
        tgt_project_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        # The list row already carries shared_users / shared_groups /
        # shared_to_org, so no detail fetch needed. Share axes are
        # write-protected on the detail endpoint (a detail PATCH is a silent
        # no-op); they're written via the dedicated share action, which handles
        # the group axis too — so groups replicate like every other shared
        # resource.
        apply_share_state(
            self.ctx,
            share_path=f"agentic/projects/{tgt_project_id}/share/",
            entity_label=f"agentic project '{name}'",
            src=src,
            result=result,
            lock=lock,
        )

    # ----- prompt versions -----

    def _clone_prompt_versions(
        self,
        name: str,
        src_project_id: str,
        tgt_project_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            src_versions = self.ctx.source.list_agentic_prompt_versions(
                project_id=src_project_id
            )
        except Exception as e:
            logger.exception("agentic '%s': prompt-version listing failed: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"agentic {name} list prompt-versions: {e}")
            return

        # Adopt versions already on target (keyed by version number) so a
        # re-run against the same pair doesn't re-create duplicates.
        try:
            tgt_versions = self.ctx.target.list_agentic_prompt_versions(
                project_id=tgt_project_id
            )
        except Exception as e:
            logger.warning(
                "agentic '%s': target prompt-version listing failed "
                "(re-run may duplicate): %s",
                name,
                e,
            )
            tgt_versions = []
        tgt_by_version = {v.get("version"): v for v in tgt_versions}

        # parent_version is a self-FK: clone roots (no parent) first so a child's
        # parent already resolves. Sort by version ascending as a stable order.
        ordered = sorted(
            src_versions,
            key=lambda v: (v.get("parent_version") is not None, v.get("version") or 0),
        )
        for src in ordered:
            self._clone_one_prompt_version(
                name, src, tgt_project_id, tgt_by_version, result, lock
            )

    def _clone_one_prompt_version(
        self,
        name: str,
        src: dict[str, Any],
        tgt_project_id: str,
        tgt_by_version: dict[Any, dict[str, Any]],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        src_vid = src["id"]

        existing = tgt_by_version.get(src.get("version"))
        if existing is not None:
            with lock:
                result.adopted += 1
                self.ctx.remap.record(
                    "agentic_prompt_version", src_vid, existing["id"]
                )
            return
        payload = {
            k: v
            for k, v in src.items()
            if k
            in {
                "version",
                "short_desc",
                "long_desc",
                "prompt_text",
                "accuracy",
                "is_active",
                "created_by_agent",
            }
            and v is not None
        }
        payload["project"] = tgt_project_id

        src_parent = src.get("parent_version")
        if src_parent is not None:
            tgt_parent = self.ctx.remap.resolve(
                "agentic_prompt_version", str(src_parent)
            )
            if tgt_parent is not None:
                payload["parent_version"] = tgt_parent
            else:
                # Root cloned first should always resolve; warn but keep the row.
                logger.warning(
                    "agentic '%s': prompt v%s parent %s unresolved — left unset",
                    name,
                    src.get("version"),
                    src_parent,
                )
                with lock:
                    result.warnings.append(
                        f"agentic {name}: prompt version {src.get('version')} "
                        "parent not remapped — left unset"
                    )

        try:
            tgt = self.ctx.target.create_agentic_prompt_version(payload)
        except Exception as e:
            logger.exception(
                "agentic '%s': prompt v%s create failed: %s",
                name,
                src.get("version"),
                e,
            )
            with lock:
                result.failed += 1
                result.errors.append(
                    f"agentic {name} create prompt v{src.get('version')}: {e}"
                )
            return
        with lock:
            result.created += 1
            self.ctx.remap.record("agentic_prompt_version", src_vid, tgt["id"])

    # ----- schemas -----

    def _clone_schemas(
        self,
        name: str,
        src_project_id: str,
        tgt_project_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            src_schemas = self.ctx.source.list_agentic_schemas(
                project_id=src_project_id
            )
        except Exception as e:
            logger.exception("agentic '%s': schema listing failed: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"agentic {name} list schemas: {e}")
            return

        # Adopt schemas already on target (keyed by version) so a re-run
        # doesn't re-create duplicates.
        try:
            tgt_schemas = self.ctx.target.list_agentic_schemas(
                project_id=tgt_project_id
            )
        except Exception as e:
            logger.warning(
                "agentic '%s': target schema listing failed (re-run may "
                "duplicate): %s",
                name,
                e,
            )
            tgt_schemas = []
        existing_versions = {s.get("version") for s in tgt_schemas}

        for src in src_schemas:
            if src.get("version") in existing_versions:
                with lock:
                    result.adopted += 1
                continue
            payload = {
                k: v
                for k, v in src.items()
                if k in {"json_schema", "version", "is_active", "created_by_agent"}
                and v is not None
            }
            payload["project"] = tgt_project_id
            try:
                self.ctx.target.create_agentic_schema(payload)
            except Exception as e:
                logger.exception(
                    "agentic '%s': schema v%s create failed: %s",
                    name,
                    src.get("version"),
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"agentic {name} create schema v{src.get('version')}: {e}"
                    )
                continue
            with lock:
                result.created += 1

    # ----- documents -----

    def _clone_documents(
        self,
        name: str,
        src_project_id: str,
        tgt_project_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            src_docs = self.ctx.source.list_agentic_documents(src_project_id)
        except Exception as e:
            logger.exception("agentic '%s': document listing failed: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"agentic {name} list documents: {e}")
            return
        if not src_docs:
            return

        # Honour the file strategy: 'skip' means no binary transfer, like the
        # files and lookups phases. Operator re-uploads on target.
        if self.ctx.options.file_strategy == "skip":
            with lock:
                result.skipped += len(src_docs)
                result.warnings.append(
                    f"agentic {name}: {len(src_docs)} document(s) not copied "
                    "(file_strategy=skip) — upload them manually on target"
                )
            return

        try:
            tgt_docs = self.ctx.target.list_agentic_documents(tgt_project_id)
        except Exception as e:
            logger.warning(
                "agentic '%s': target document listing failed "
                "(re-run may duplicate): %s",
                name,
                e,
            )
            tgt_docs = []
        target_names = {d.get("original_filename") for d in tgt_docs}

        for src in src_docs:
            file_name = src.get("original_filename")
            src_doc_id = src.get("id")
            if not file_name or not src_doc_id:
                continue
            if file_name in target_names:
                with lock:
                    result.skipped += 1
                logger.info(
                    "agentic '%s': document '%s' already on target — skipping",
                    name,
                    file_name,
                )
                continue
            self._clone_one_document(
                name, tgt_project_id, src_doc_id, file_name, result, lock
            )

    def _clone_one_document(
        self,
        name: str,
        tgt_project_id: str,
        src_doc_id: str,
        file_name: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            raw = self.ctx.source.download_agentic_document(src_doc_id)
        except Exception as e:
            logger.exception(
                "agentic '%s': document '%s' download failed: %s", name, file_name, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"agentic {name} download {file_name}: {e}")
            return

        if len(raw) > self.ctx.options.max_file_size:
            with lock:
                result.skipped += 1
                result.warnings.append(
                    f"agentic {name}: document {file_name} exceeds size cap — "
                    "upload it manually on target"
                )
            return

        mime = mimetypes.guess_type(file_name)[0] or "application/pdf"
        try:
            self.ctx.target.upload_agentic_document(
                tgt_project_id, file_name, raw, mime
            )
        except Exception as e:
            logger.exception(
                "agentic '%s': document '%s' upload failed: %s", name, file_name, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"agentic {name} upload {file_name}: {e}")
            return
        with lock:
            result.created += 1
        logger.info("agentic '%s': uploaded document '%s'", name, file_name)

    # ----- verified (ground-truth) data -----

    def _clone_verified_data(
        self,
        name: str,
        src_project_id: str,
        tgt_project_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            src_rows = self.ctx.source.list_agentic_verified_data(src_project_id)
        except Exception as e:
            logger.exception("agentic '%s': verified-data listing failed: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"agentic {name} list verified-data: {e}")
            return
        if not src_rows:
            return

        # Verified data FKs a document; under 'skip' no documents are copied,
        # so skip the rows too — matching the plan and _clone_documents.
        if self.ctx.options.file_strategy == "skip":
            with lock:
                result.skipped += len(src_rows)
                result.warnings.append(
                    f"agentic {name}: {len(src_rows)} verified-data row(s) not "
                    "copied (file_strategy=skip)"
                )
            return

        # Verified data FKs a document; map source rows to target docs by
        # filename, the only identity stable across orgs.
        try:
            tgt_docs = self.ctx.target.list_agentic_documents(tgt_project_id)
            tgt_existing = self.ctx.target.list_agentic_verified_data(tgt_project_id)
        except Exception as e:
            logger.warning(
                "agentic '%s': target verified-data lookup failed "
                "(re-run may duplicate): %s",
                name,
                e,
            )
            tgt_docs, tgt_existing = [], []
        doc_id_by_name = {d.get("original_filename"): d.get("id") for d in tgt_docs}
        verified_doc_ids = {r.get("document") for r in tgt_existing}

        for src in src_rows:
            file_name = src.get("document_name")
            tgt_doc_id = doc_id_by_name.get(file_name)
            if not tgt_doc_id:
                with lock:
                    result.skipped += 1
                    result.warnings.append(
                        f"agentic {name}: verified data for '{file_name}' skipped — "
                        "document not on target"
                    )
                continue
            if tgt_doc_id in verified_doc_ids:
                with lock:
                    result.skipped += 1
                continue
            payload = {
                "project": tgt_project_id,
                "document": tgt_doc_id,
                "data": src.get("data"),
            }
            try:
                self.ctx.target.create_agentic_verified_data(payload)
            except Exception as e:
                logger.exception(
                    "agentic '%s': verified data for '%s' create failed: %s",
                    name,
                    file_name,
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"agentic {name} create verified-data {file_name}: {e}"
                    )
                continue
            with lock:
                result.created += 1
            logger.info("agentic '%s': cloned verified data for '%s'", name, file_name)

    # ----- registry -----

    def _republish_registry(
        self,
        name: str,
        src_project_id: str,
        tgt_project_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        # Projects never exported on source (no active schema/prompt) have no
        # registry row; republishing would fail the same backend guard.
        try:
            src_regs = self.ctx.source.list_agentic_registries(
                agentic_project=src_project_id
            )
        except Exception as e:
            logger.warning(
                "agentic '%s': source registry lookup failed — registry not "
                "republished: %s",
                name,
                e,
            )
            with lock:
                result.warnings.append(
                    f"agentic {name}: source registry lookup failed — "
                    "registry not republished"
                )
            return

        if not src_regs:
            # Nothing to republish — project was never exported.
            return

        try:
            self.ctx.target.export_agentic_project(tgt_project_id)
        except Exception as e:
            # Export needs an active schema + prompt on target; a partial clone
            # can leave it unexportable. Non-fatal: warn and move on.
            logger.warning(
                "agentic '%s': registry republish failed (export) tgt=%s: %s",
                name,
                tgt_project_id,
                e,
            )
            with lock:
                result.warnings.append(
                    f"agentic {name}: registry not republished in v1 "
                    f"(export failed: {e})"
                )
            return

        try:
            tgt_regs = self.ctx.target.list_agentic_registries(
                agentic_project=tgt_project_id
            )
        except Exception as e:
            logger.warning(
                "agentic '%s': target registry lookup failed after export: %s",
                name,
                e,
            )
            with lock:
                result.warnings.append(
                    f"agentic {name}: target registry id not recorded after export"
                )
            return

        if src_regs and tgt_regs:
            with lock:
                self.ctx.remap.record(
                    "agentic_studio_registry",
                    src_regs[0]["registry_id"],
                    tgt_regs[0]["registry_id"],
                )
            logger.info("republished agentic registry for project '%s'", name)

    # ----- dry-run planning -----

    def _plan_children(
        self,
        src_project_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        """Dry-run: count source prompt versions + schemas + documents as
        planned and record planned prompt-version ids so downstream resolves
        don't miss.
        """
        try:
            src_versions = self.ctx.source.list_agentic_prompt_versions(
                project_id=src_project_id
            )
        except Exception:
            src_versions = []
        try:
            src_schemas = self.ctx.source.list_agentic_schemas(
                project_id=src_project_id
            )
        except Exception:
            src_schemas = []
        try:
            src_docs = self.ctx.source.list_agentic_documents(src_project_id)
        except Exception:
            src_docs = []
        try:
            src_verified = self.ctx.source.list_agentic_verified_data(src_project_id)
        except Exception:
            src_verified = []
        with lock:
            for v in src_versions:
                self.ctx.remap.record_planned("agentic_prompt_version", v["id"])
                result.created += 1
            result.created += len(src_schemas)
            # Documents move only when the file strategy copies binaries.
            # Verified data FKs a document, so skipping files strands it too.
            if self.ctx.options.file_strategy == "skip":
                result.skipped += len(src_docs)
                result.skipped += len(src_verified)
            else:
                result.created += len(src_docs)
                result.created += len(src_verified)

    # ----- settings -----

    def _clone_settings(self, result: PhaseResult) -> None:
        if self.ctx.options.dry_run:
            try:
                src_settings = self.ctx.source.list_agentic_settings()
            except Exception:
                src_settings = []
            result.created += len(src_settings)
            return

        try:
            src_settings = self.ctx.source.list_agentic_settings()
        except Exception as e:
            logger.exception("Failed to list source agentic settings: %s", e)
            result.failed += 1
            result.errors.append(f"list source agentic settings: {e}")
            return
        if not src_settings:
            return
        try:
            tgt_by_key = {
                s["key"]: s for s in self.ctx.target.list_agentic_settings()
            }
        except Exception as e:
            logger.exception("Failed to list target agentic settings: %s", e)
            result.failed += 1
            result.errors.append(f"list target agentic settings: {e}")
            return

        for src in src_settings:
            key = src.get("key")
            if not key:
                continue
            payload = {
                k: v
                for k, v in src.items()
                if k in {"key", "value", "description"} and v is not None
            }
            existing = tgt_by_key.get(key)
            try:
                if existing is not None:
                    self.ctx.target.update_agentic_setting(existing["id"], payload)
                    result.adopted += 1
                else:
                    self.ctx.target.create_agentic_setting(payload)
                    result.created += 1
            except Exception as e:
                # An agentic-setting key can collide with a row this org's
                # listing doesn't surface — not data loss the clone can
                # resolve. Warn, don't fail.
                logger.warning("agentic setting '%s' not replicated: %s", key, e)
                result.skipped += 1
                result.warnings.append(
                    f"agentic setting {key}: not replicated "
                    f"(org-global key may already exist elsewhere): {e}"
                )
