"""Migrate cloud-only Lookup definitions + their prompt assignments.

Cloud-only: gated by ``probe_path`` — the orchestrator probes
``lookups/definitions/`` on source/target and skips the phase entirely on an
OSS deployment. Runs after ``custom_tool`` (consumes its ``prompt`` and
``adapter`` remaps) and after ``files``.

Two passes:

1. **Definitions** — per source lookup, adopt-by-name or create fresh.
   Creating a definition auto-spawns an empty DRAFT version with default
   adapters; this phase then patches the draft's ``prompt_template`` and
   remaps the draft adapters (LLM / X2Text) to target-org ids, and replays
   the reference files into the draft (reusing the size-cap / file-strategy
   semantics of the files phase). Records a ``lookup_definition`` remap.

2. **Assignments** — after every lookup + prompt remap exists, replay the
   PromptLookupAssignment rows. Each row's source prompt FK remaps via the
   ``custom_tool`` phase's ``prompt`` table; its target version resolves via
   the ``lookup_version`` remap (recorded by the version replay below — draft
   pins fall back to the cloned lookup's ``draft_version_id``);
   ``variable_mappings`` values that are source prompt UUIDs remap too.

Published-version history is reproduced per definition (see ``_replay_versions``)
so published-pinned assignments resolve, before the target draft is restored to
the source's current draft state.
"""

from __future__ import annotations

import logging
import mimetypes
import threading
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase, build_post_payload
from unstract.clone.report import CloneReport, PhaseResult
from unstract.clone.sharing import replicate_share

logger = logging.getLogger(__name__)

LOOKUP_DEFINITIONS_PATH = "lookups/definitions/"

# Draft adapter slots; each maps a detail ``adapters`` key to the PATCH key.
_ADAPTER_SLOTS: tuple[str, ...] = ("llm", "x2text")

_DEFAULT_MIME = "application/octet-stream"


class LookupsPhase(Phase):
    name = "lookups"
    probe_path = LOOKUP_DEFINITIONS_PATH

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._writable = self.ctx.target.get_post_schema(LOOKUP_DEFINITIONS_PATH)
        except Exception as e:
            logger.exception("Failed to fetch target POST schema for lookups: %s", e)
            result.failed += 1
            result.errors.append(f"OPTIONS lookups: {e}")
            return result
        try:
            src_lookups = self.ctx.source.list_lookup_definitions()
        except Exception as e:
            logger.exception("Failed to list source lookup definitions: %s", e)
            result.failed += 1
            result.errors.append(f"list source lookups: {e}")
            return result
        try:
            tgt_lookups = self.ctx.target.list_lookup_definitions()
        except Exception as e:
            logger.exception("Failed to list target lookup definitions: %s", e)
            result.failed += 1
            result.errors.append(f"list target lookups: {e}")
            return result

        logger.info("Found %d lookup definition(s) in source org", len(src_lookups))
        # Updated under lock on fresh create so same-name source rows adopt.
        target_by_name: dict[str, dict[str, Any]] = {
            lk["name"]: lk for lk in tgt_lookups
        }

        # Pass 1: definitions (+ draft content + reference files).
        self.parallel_map(
            src_lookups,
            lambda src, lock: self._clone_definition(src, target_by_name, result, lock),
        )

        # Pass 2: assignments — needs every lookup + prompt remap from above.
        self._clone_assignments(result)
        return result

    # ----- definitions -----

    def _clone_definition(
        self,
        src: dict[str, Any],
        target_by_name: dict[str, dict[str, Any]],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        name = src["name"]
        src_lookup_id = src["lookup_id"]

        with lock:
            match = target_by_name.get(name)

        if match is not None:
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"lookup '{name}' already exists in target as {match['lookup_id']}"
                )
            tgt_lookup_id = match["lookup_id"]
            with lock:
                result.adopted += 1
                self.ctx.remap.record("lookup_definition", src_lookup_id, tgt_lookup_id)
            logger.info(
                "adopted lookup '%s' src=%s -> tgt=%s",
                name,
                src_lookup_id,
                tgt_lookup_id,
            )
        elif self.ctx.options.dry_run:
            with lock:
                result.created += 1
                self.ctx.remap.record_planned("lookup_definition", src_lookup_id)
            logger.info(
                "[dry-run] would create lookup '%s' src=%s", name, src_lookup_id
            )
            self._plan_versions(name, src_lookup_id, lock)
            return
        else:
            payload = build_post_payload(src, self._writable)
            try:
                tgt = self.ctx.target.create_lookup_definition(payload)
            except Exception as e:
                logger.exception("Failed to create lookup '%s': %s", name, e)
                with lock:
                    result.failed += 1
                    result.errors.append(f"create lookup {name}: {e}")
                return
            tgt_lookup_id = tgt["lookup_id"]
            with lock:
                result.created += 1
                target_by_name[name] = {"lookup_id": tgt_lookup_id, "name": name}
                self.ctx.remap.record("lookup_definition", src_lookup_id, tgt_lookup_id)
            logger.info(
                "created lookup '%s' src=%s -> tgt=%s",
                name,
                src_lookup_id,
                tgt_lookup_id,
            )

        # Draft content + reference files write to the real target draft only.
        if self.ctx.options.dry_run:
            self._plan_versions(name, src_lookup_id, lock)
            return

        try:
            detail = self.ctx.source.get_lookup_definition(src_lookup_id)
        except Exception as e:
            logger.warning(
                "lookup '%s': source detail fetch failed — draft content not "
                "replicated: %s",
                name,
                e,
            )
            with lock:
                result.warnings.append(
                    f"lookup {name}: source detail fetch failed — "
                    f"draft template/adapters not replicated: {e}"
                )
            detail = None

        if detail is not None:
            # Share state is server-managed on create; mirror it post-create.
            replicate_share(
                self.ctx,
                apply_fn=lambda p: self.ctx.target.update_lookup_share(
                    tgt_lookup_id, p
                ),
                entity_label=f"lookup '{name}'",
                src=detail,
                result=result,
                lock=lock,
                include_groups=False,  # lookups model has no group sharing
            )

        # Reproduce published-version history first so its draft churn lands
        # before the final draft restore (last publish spawns a fresh draft).
        self._replay_versions(name, src_lookup_id, tgt_lookup_id, result, lock)

        # Restore target draft to the source's CURRENT draft — must run AFTER
        # the replay so the final draft matches source, not the last published.
        if detail is not None:
            self._replicate_draft(name, tgt_lookup_id, detail, result, lock)
        self._replicate_files(name, src_lookup_id, tgt_lookup_id, result, lock)

        # Draft pins resolve uniformly through the version remap too.
        if detail is not None:
            self._record_draft_version_remap(name, tgt_lookup_id, detail, lock)

    def _replicate_draft(
        self,
        name: str,
        tgt_lookup_id: str,
        detail: dict[str, Any],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        template = detail.get("prompt_template") or ""
        if template:
            try:
                self.ctx.target.update_lookup_draft_template(tgt_lookup_id, template)
            except Exception as e:
                logger.exception(
                    "lookup '%s': draft template patch failed: %s", name, e
                )
                with lock:
                    result.failed += 1
                    result.errors.append(f"lookup {name} draft template: {e}")

        tgt_adapters = self._remap_adapters(
            name, detail.get("adapters") or {}, result, lock
        )
        if tgt_adapters:
            try:
                self.ctx.target.update_lookup_draft_adapters(
                    tgt_lookup_id, tgt_adapters
                )
            except Exception as e:
                logger.exception(
                    "lookup '%s': draft adapters patch failed: %s", name, e
                )
                with lock:
                    result.failed += 1
                    result.errors.append(f"lookup {name} draft adapters: {e}")

    def _remap_adapters(
        self,
        name: str,
        src_adapters: dict[str, Any],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> dict[str, str]:
        """Resolve a detail/version ``adapters`` dict to target ids; unresolved
        slots are omitted with a warning so the draft keeps its default.
        """
        tgt_adapters: dict[str, str] = {}
        for slot in _ADAPTER_SLOTS:
            src_adapter_id = src_adapters.get(slot)
            if not src_adapter_id:
                continue
            tgt_adapter_id = self.ctx.remap.resolve("adapter", src_adapter_id)
            if tgt_adapter_id is None:
                logger.warning(
                    "lookup '%s': %s adapter %s has no target mapping — "
                    "leaving draft default",
                    name,
                    slot,
                    src_adapter_id,
                )
                with lock:
                    result.warnings.append(
                        f"lookup {name}: {slot} adapter not remapped — "
                        "draft kept its default adapter"
                    )
                continue
            tgt_adapters[slot] = tgt_adapter_id
        return tgt_adapters

    # ----- version replay -----

    def _plan_versions(
        self, name: str, src_lookup_id: str, lock: threading.Lock
    ) -> None:
        """Dry-run: record a planned ``lookup_version`` remap per source version
        so the assignment pass can plan-count published-pinned ones too.
        """
        try:
            versions = self.ctx.source.list_lookup_versions(src_lookup_id)
        except Exception as e:
            logger.warning(
                "[dry-run] lookup '%s': version listing failed: %s", name, e
            )
            return
        with lock:
            for v in versions:
                vid = v.get("version_id")
                if vid:
                    self.ctx.remap.record_planned("lookup_version", str(vid))

    def _replay_versions(
        self,
        name: str,
        src_lookup_id: str,
        tgt_lookup_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        """Reproduce the source's published versions on the target in
        ``version_number`` order: set the target draft to each version's
        content + files, then publish. Records a ``lookup_version`` remap per
        published version so its pinned assignments resolve. A target version
        with the same ``version_name`` is adopted (no re-publish) so re-runs
        and adopted definitions stay idempotent.
        """
        try:
            versions = self.ctx.source.list_lookup_versions(src_lookup_id)
        except Exception as e:
            logger.warning("lookup '%s': version listing failed: %s", name, e)
            with lock:
                result.warnings.append(
                    f"lookup {name}: version listing failed — published "
                    f"versions not replayed: {e}"
                )
            return

        # Existing target versions — re-publishing a name that already exists
        # would error or duplicate history; adopt those instead.
        try:
            tgt_versions = self.ctx.target.list_lookup_versions(tgt_lookup_id)
        except Exception as e:
            logger.warning(
                "lookup '%s': target version listing failed: %s", name, e
            )
            tgt_versions = []
        tgt_by_name: dict[str, str] = {
            v["version_name"]: str(v["version_id"])
            for v in tgt_versions
            if v.get("version_name") and v.get("version_id")
        }

        published = sorted(
            (v for v in versions if not v.get("is_draft")),
            key=lambda v: v.get("version_number") or 0,
        )
        for v in published:
            existing_id = tgt_by_name.get(v.get("version_name"))
            if existing_id is not None:
                src_version_id = v.get("version_id")
                with lock:
                    result.adopted += 1
                    if src_version_id:
                        self.ctx.remap.record(
                            "lookup_version", str(src_version_id), existing_id
                        )
                logger.info(
                    "lookup '%s': adopted published version '%s' (already present)",
                    name,
                    v.get("version_name"),
                )
                continue
            self._replay_one_version(
                name, src_lookup_id, tgt_lookup_id, v, result, lock
            )

    def _replay_one_version(
        self,
        name: str,
        src_lookup_id: str,
        tgt_lookup_id: str,
        version_row: dict[str, Any],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        src_version_id = version_row.get("version_id")
        version_name = version_row.get("version_name") or ""
        try:
            detail = self.ctx.source.get_lookup_version(
                src_lookup_id, src_version_id
            )
        except Exception as e:
            logger.exception(
                "lookup '%s': version %s detail fetch failed: %s",
                name,
                src_version_id,
                e,
            )
            with lock:
                result.failed += 1
                result.errors.append(
                    f"lookup {name} version {version_name}: detail fetch: {e}"
                )
            return

        # Stage the version's content onto the target draft before publishing.
        template = detail.get("prompt_template") or ""
        if template:
            try:
                self.ctx.target.update_lookup_draft_template(tgt_lookup_id, template)
            except Exception as e:
                logger.exception(
                    "lookup '%s': version %s template patch failed: %s",
                    name,
                    version_name,
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"lookup {name} version {version_name} template: {e}"
                    )
        tgt_adapters = self._remap_adapters(
            name, detail.get("adapters") or {}, result, lock
        )
        if tgt_adapters:
            try:
                self.ctx.target.update_lookup_draft_adapters(
                    tgt_lookup_id, tgt_adapters
                )
            except Exception as e:
                logger.exception(
                    "lookup '%s': version %s adapters patch failed: %s",
                    name,
                    version_name,
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"lookup {name} version {version_name} adapters: {e}"
                    )

        self._replay_version_files(
            name, src_lookup_id, tgt_lookup_id, src_version_id, detail, result, lock
        )

        # ponytail: assignment_values_snapshot is derived by the backend at
        # publish time from then-existing assignments; we publish before
        # assignments are recreated, so historical snapshots may differ from
        # source. Structure + pinning reproduce; frozen values are best-effort.
        try:
            published = self.ctx.target.publish_lookup_version(
                tgt_lookup_id,
                {"version_name": version_name, "rebind_assignments": False},
            )
        except Exception as e:
            logger.exception(
                "lookup '%s': publish of version %s failed: %s",
                name,
                version_name,
                e,
            )
            with lock:
                result.failed += 1
                result.errors.append(
                    f"lookup {name} publish version {version_name}: {e}"
                )
            return

        tgt_version_id = published.get("version_id")
        with lock:
            if src_version_id and tgt_version_id:
                self.ctx.remap.record(
                    "lookup_version", str(src_version_id), str(tgt_version_id)
                )
            result.warnings.append(
                f"lookup {name} version {version_name}: published; frozen "
                "assignment-value snapshots are best-effort (assignments "
                "recreated after publish)"
            )
        logger.info(
            "lookup '%s': replayed published version '%s' src=%s -> tgt=%s",
            name,
            version_name,
            src_version_id,
            tgt_version_id,
        )

    def _replay_version_files(
        self,
        name: str,
        src_lookup_id: str,
        tgt_lookup_id: str,
        src_version_id: str,
        version_detail: dict[str, Any],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        """Upload a published version's reference files onto the target draft,
        deduped by filename against the current draft files. Honors the same
        size-cap / file-strategy semantics as ``_clone_one_file``.
        """
        if self.ctx.options.file_strategy == "skip":
            return
        src_files = version_detail.get("files") or []
        if not src_files:
            return
        try:
            tgt_files = self.ctx.target.list_lookup_files(tgt_lookup_id)
        except Exception as e:
            logger.exception(
                "lookup '%s': target file listing (version replay) failed: %s",
                name,
                e,
            )
            with lock:
                result.failed += 1
                result.errors.append(f"lookup {name} list target files: {e}")
            return
        target_names = {f.get("file_name") for f in tgt_files}

        for f in src_files:
            file_name = f.get("file_name")
            file_id = f.get("file_id")
            if not file_name or not file_id or file_name in target_names:
                continue
            try:
                raw = self.ctx.source.download_lookup_version_file(
                    src_lookup_id, src_version_id, file_id
                )
            except Exception as e:
                logger.exception(
                    "lookup '%s': version file '%s' download failed: %s",
                    name,
                    file_name,
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"lookup {name} download version file {file_name}: {e}"
                    )
                continue
            if len(raw) > self.ctx.options.max_file_size:
                with lock:
                    result.skipped += 1
                    result.warnings.append(
                        f"lookup {name}: version reference file '{file_name}' "
                        f"({len(raw)} bytes) exceeds cap "
                        f"{self.ctx.options.max_file_size} — re-upload via UI"
                    )
                continue
            mime = mimetypes.guess_type(file_name)[0] or _DEFAULT_MIME
            try:
                self.ctx.target.upload_lookup_file(
                    tgt_lookup_id, file_name, raw, mime
                )
            except Exception as e:
                logger.exception(
                    "lookup '%s': version file '%s' upload failed: %s",
                    name,
                    file_name,
                    e,
                )
                with lock:
                    result.failed += 1
                    result.errors.append(
                        f"lookup {name} upload version file {file_name}: {e}"
                    )
                continue
            target_names.add(file_name)

    def _record_draft_version_remap(
        self,
        name: str,
        tgt_lookup_id: str,
        detail: dict[str, Any],
        lock: threading.Lock,
    ) -> None:
        """Map the source DRAFT version id -> target draft_version_id so
        draft-pinned assignments resolve uniformly via the version remap.
        """
        src_draft_id = detail.get("draft_version_id")
        if not src_draft_id:
            return
        try:
            tgt_detail = self.ctx.target.get_lookup_definition(tgt_lookup_id)
            tgt_draft_id = tgt_detail.get("draft_version_id")
        except Exception as e:
            logger.warning(
                "lookup '%s': target draft id fetch failed: %s", name, e
            )
            return
        if tgt_draft_id:
            with lock:
                self.ctx.remap.record(
                    "lookup_version", str(src_draft_id), str(tgt_draft_id)
                )

    def _replicate_files(
        self,
        name: str,
        src_lookup_id: str,
        tgt_lookup_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            src_files = self.ctx.source.list_lookup_files(src_lookup_id)
        except Exception as e:
            logger.exception("lookup '%s': source file listing failed: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"lookup {name} list files: {e}")
            return

        if self.ctx.options.file_strategy == "skip":
            for f in src_files:
                file_name = f.get("file_name")
                if not file_name:
                    continue
                with lock:
                    result.skipped += 1
                    result.warnings.append(
                        f"lookup {name}: reference file '{file_name}' not cloned "
                        "(file_strategy=skip) — re-upload via UI"
                    )
            return

        try:
            tgt_files = self.ctx.target.list_lookup_files(tgt_lookup_id)
        except Exception as e:
            logger.exception("lookup '%s': target file listing failed: %s", name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"lookup {name} list target files: {e}")
            return
        target_names = {f.get("file_name") for f in tgt_files}

        for f in src_files:
            file_name = f.get("file_name")
            file_id = f.get("file_id")
            if not file_name or not file_id:
                continue
            if file_name in target_names:
                with lock:
                    result.skipped += 1
                continue
            self._clone_one_file(
                name, src_lookup_id, tgt_lookup_id, file_name, file_id, result, lock
            )

    def _clone_one_file(
        self,
        name: str,
        src_lookup_id: str,
        tgt_lookup_id: str,
        file_name: str,
        file_id: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        try:
            raw = self.ctx.source.download_lookup_file(src_lookup_id, file_id)
        except Exception as e:
            logger.exception(
                "lookup '%s': download of '%s' failed: %s", name, file_name, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"lookup {name} download {file_name}: {e}")
            return

        if len(raw) > self.ctx.options.max_file_size:
            with lock:
                result.skipped += 1
                result.warnings.append(
                    f"lookup {name}: reference file '{file_name}' "
                    f"({len(raw)} bytes) exceeds cap "
                    f"{self.ctx.options.max_file_size} — re-upload via UI"
                )
            return

        mime = mimetypes.guess_type(file_name)[0] or _DEFAULT_MIME
        try:
            self.ctx.target.upload_lookup_file(tgt_lookup_id, file_name, raw, mime)
        except Exception as e:
            logger.exception(
                "lookup '%s': upload of '%s' failed: %s", name, file_name, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"lookup {name} upload {file_name}: {e}")
            return
        with lock:
            result.created += 1
        logger.info("lookup '%s': uploaded reference file '%s'", name, file_name)

    # ----- assignments -----

    def _clone_assignments(self, result: PhaseResult) -> None:
        try:
            src_assignments = self.ctx.source.list_lookup_assignments()
        except Exception as e:
            logger.exception("Failed to list source lookup assignments: %s", e)
            result.failed += 1
            result.errors.append(f"list source lookup assignments: {e}")
            return

        if not src_assignments:
            return

        # Index existing target assignments by target prompt id (one per prompt)
        # to honor the ``one_lookup_per_prompt`` uniqueness on re-runs.
        existing_by_prompt: dict[str, dict[str, Any]] = {}
        if not self.ctx.options.dry_run:
            try:
                for a in self.ctx.target.list_lookup_assignments():
                    pid = a.get("prompt")
                    if pid:
                        existing_by_prompt[str(pid)] = a
            except Exception as e:
                logger.warning(
                    "target assignment listing failed — re-run idempotency may "
                    "create duplicates: %s",
                    e,
                )

        # Cache target draft_version_id per target lookup id.
        draft_cache: dict[str, str | None] = {}

        self.parallel_map(
            src_assignments,
            lambda a, lock: self._clone_one_assignment(
                a, existing_by_prompt, draft_cache, result, lock
            ),
        )

    def _clone_one_assignment(
        self,
        src: dict[str, Any],
        existing_by_prompt: dict[str, dict[str, Any]],
        draft_cache: dict[str, str | None],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        assignment_id = src.get("assignment_id")
        src_prompt_id = src.get("prompt")
        src_lookup_id = src.get("lookup_definition")
        src_version_id = src.get("version")
        is_draft_version = bool(src.get("is_draft_version"))

        tgt_prompt_id = (
            self.ctx.remap.resolve("prompt", str(src_prompt_id))
            if src_prompt_id is not None
            else None
        )
        if tgt_prompt_id is None:
            with lock:
                result.skipped += 1
                result.warnings.append(
                    f"assignment {assignment_id}: source prompt {src_prompt_id} "
                    "has no target mapping (its tool wasn't cloned) — skipped"
                )
            return

        tgt_lookup_id = (
            self.ctx.remap.resolve("lookup_definition", str(src_lookup_id))
            if src_lookup_id is not None
            else None
        )
        if tgt_lookup_id is None:
            with lock:
                result.skipped += 1
                result.warnings.append(
                    f"assignment {assignment_id}: source lookup {src_lookup_id} "
                    "has no target mapping — skipped"
                )
            return

        mappings = self._remap_mappings(src.get("variable_mappings"))

        if self.ctx.options.dry_run:
            with lock:
                result.created += 1
            logger.info(
                "[dry-run] would create assignment for prompt %s -> lookup %s",
                tgt_prompt_id,
                tgt_lookup_id,
            )
            return

        with lock:
            if str(tgt_prompt_id) in existing_by_prompt:
                result.adopted += 1
                logger.info(
                    "adopted assignment: target prompt %s already has a lookup "
                    "assignment",
                    tgt_prompt_id,
                )
                return

        # Both draft- and published-pinned assignments resolve through the
        # version remap recorded by the replay; draft pins fall back to the
        # target's current draft when the remap is absent.
        tgt_version_id = (
            self.ctx.remap.resolve("lookup_version", str(src_version_id))
            if src_version_id is not None
            else None
        )
        if tgt_version_id is None and is_draft_version:
            tgt_version_id = self._target_draft_version(
                tgt_lookup_id, draft_cache, lock
            )
        if tgt_version_id is None:
            with lock:
                result.skipped += 1
                result.warnings.append(
                    f"assignment {assignment_id}: source version {src_version_id} "
                    "could not be resolved on target — skipped"
                )
            return

        payload = {
            "prompt": tgt_prompt_id,
            "lookup_definition": tgt_lookup_id,
            "version": tgt_version_id,
            "variable_mappings": mappings,
        }
        try:
            self.ctx.target.create_lookup_assignment(payload)
        except Exception as e:
            logger.exception(
                "Failed to create assignment for prompt %s: %s", tgt_prompt_id, e
            )
            with lock:
                result.failed += 1
                result.errors.append(f"create assignment {assignment_id}: {e}")
            return
        with lock:
            result.created += 1
            existing_by_prompt[str(tgt_prompt_id)] = payload
        logger.info(
            "created assignment: prompt %s -> lookup %s", tgt_prompt_id, tgt_lookup_id
        )

    def _remap_mappings(self, mappings: Any) -> Any:
        """Deep-walk ``mappings``; any string value that is a source prompt
        UUID (present in the ``prompt`` remap) rewrites to its target id.
        Non-prompt strings pass through untouched.
        """
        if isinstance(mappings, dict):
            return {k: self._remap_mappings(v) for k, v in mappings.items()}
        if isinstance(mappings, list):
            return [self._remap_mappings(v) for v in mappings]
        if isinstance(mappings, str):
            return self.ctx.remap.resolve("prompt", mappings) or mappings
        return mappings

    def _target_draft_version(
        self,
        tgt_lookup_id: str,
        draft_cache: dict[str, str | None],
        lock: threading.Lock,
    ) -> str | None:
        with lock:
            if tgt_lookup_id in draft_cache:
                return draft_cache[tgt_lookup_id]
        try:
            detail = self.ctx.target.get_lookup_definition(tgt_lookup_id)
            draft_id = detail.get("draft_version_id")
        except Exception as e:
            logger.warning("target lookup %s draft fetch failed: %s", tgt_lookup_id, e)
            draft_id = None
        with lock:
            # A racing peer may have already cached a valid id; the GET is
            # read-only so a real id always wins over this thread's failure.
            if draft_cache.get(tgt_lookup_id) is None:
                draft_cache[tgt_lookup_id] = draft_id
            return draft_cache[tgt_lookup_id]
