"""Migrate prompt-studio projects via the project-transfer endpoints.

For each source tool the phase:

1. ``GET prompt-studio/project-transfer/{src_tool_id}`` — pulls a
   portable JSON blob (tool_metadata, tool_settings,
   default_profile_settings, prompts, export_metadata).
2. Decides fresh vs adopt by looking up the target tool by name.
3. **Fresh path**: reads the source's default adapter profile to learn the
   adapter UUIDs the profile is bound to, remaps each via the running
   ``adapter`` remap table, and POSTs the import as a multipart upload
   with target-org adapter ids on the form. Backend creates the tool,
   the default profile, and all prompts server-side in one call.
4. **Adopt path**: POSTs ``sync-prompts`` on the existing target tool.
   Backend rip-and-replaces prompts + ``tool_settings`` and leaves the
   target's locally-configured profiles + adapters untouched (which is
   what the operator wants — they may have rewired adapters on target).
5. Republishes the tool's registry entry via the export action and
   records the ``custom_tool`` + ``prompt_studio_registry`` remaps so
   downstream ToolInstancePhase can rewrite the tool instance's tool id.
   Skipped for tools with no source registry entry (never exported —
   e.g. empty projects, which the backend refuses to export).

Adapter id discovery for the fresh path resolves each of LLM,
vector_db, embedding, x2text via the adapter remap on a best-effort
basis. Any that can't be resolved are left unconfigured — the backend
imports the tool with a partial/empty profile and flags
``needs_adapter_config`` for the operator to finish wiring on target
and re-run. Frictionless-bound tools (adapters not even visible to the
source org's Platform key) are the exception: cloud-only with no target
equivalent, so they are skipped + cascade.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, PhaseResult

logger = logging.getLogger(__name__)

_PROFILE_ADAPTER_FIELDS: tuple[tuple[str, str], ...] = (
    ("llm", "llm_adapter_id"),
    ("vector_store", "vector_db_adapter_id"),
    ("embedding_model", "embedding_adapter_id"),
    ("x2text", "x2text_adapter_id"),
)


def _extract_adapter_name(value: Any) -> str | None:
    """Adapter FKs serialise as the adapter NAME on the wire; tolerate a
    nested-dict shape too. Never fall back to the UUID — list_adapters
    matches by name and would silently miss.
    """
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        return value.get("adapter_name") or value.get("name")
    return None


class CustomToolPhase(Phase):
    name = "custom_tool"
    share_path_template = "prompt-studio/{id}/share/"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            src_tools = self.ctx.source.list_custom_tools()
        except Exception as e:
            logger.exception("Failed to list source custom tools: %s", e)
            result.failed += 1
            result.errors.append(f"list source custom tools: {e}")
            return result

        logger.info("Found %d custom tool(s) in source org", len(src_tools))
        try:
            target_tools = self.ctx.target.list_custom_tools()
        except Exception as e:
            logger.exception("Failed to list target tools: %s", e)
            result.failed += 1
            result.errors.append(f"list target tools: {e}")
            return result

        # The source's visible adapter set hides frictionless adapters; a
        # profile-referenced name missing here flags a tool we can't migrate.
        try:
            self._src_adapter_names = {
                a["adapter_name"] for a in self.ctx.source.list_adapters()
            }
        except Exception as e:
            logger.exception("Failed to list source adapters: %s", e)
            result.failed += 1
            result.errors.append(f"list source adapters for visibility check: {e}")
            return result

        # Updated under lock when a fresh create lands so duplicate
        # same-name source rows adopt instead of recreating.
        target_by_name: dict[str, dict[str, Any]] = {
            t["tool_name"]: t for t in target_tools
        }

        self.parallel_map(
            src_tools,
            lambda summary, lock: self._clone_one(
                summary, target_by_name, result, lock
            ),
        )
        return result

    def _clone_one(
        self,
        summary: dict[str, Any],
        target_by_name: dict[str, dict[str, Any]],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        tool_name = summary["tool_name"]
        src_tool_id = summary["tool_id"]

        try:
            export_data = self.ctx.source.export_project(src_tool_id)
        except Exception as e:
            logger.exception("Failed to export source tool '%s': %s", tool_name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"export src tool {tool_name}: {e}")
            return

        with lock:
            match = target_by_name.get(tool_name)

        if match is not None:
            tgt_tool_id = self._adopt(
                match, export_data, result, tool_name, src_tool_id, lock
            )
        else:
            tgt_tool_id = self._create_fresh(
                export_data, src_tool_id, tool_name, result, lock
            )
            if tgt_tool_id is not None:
                with lock:
                    target_by_name[tool_name] = {
                        "tool_id": tgt_tool_id,
                        "tool_name": tool_name,
                    }

        # Each sub-path (adopt / fresh / fresh-dry-run) owns its own
        # custom_tool remap, since only it knows whether the target id is
        # real or a planned synthetic.
        if tgt_tool_id is None:
            return

        # Neither the export blob nor list rows carry share axes —
        # share state comes from the source detail.
        self.apply_share(
            src={},
            tgt_id=tgt_tool_id,
            label=tool_name,
            result=result,
            lock=lock,
            src_detail_fn=lambda: self.ctx.source.get_custom_tool(src_tool_id),
        )

        if self.ctx.options.dry_run:
            # Can't republish the registry without writing, but ToolInstance
            # needs a prompt_studio_registry remap to plan-count. Mirror it
            # with a planned id derived from the source registry (read-only).
            self._record_planned_registry(src_tool_id, tool_name, lock)
            self._record_planned_prompts(src_tool_id, lock)
            return

        # Map source prompt ids -> target prompt ids by prompt_key so
        # prompt-scoped phases (e.g. lookup assignments) can rewrite their
        # prompt FKs. Target prompts already exist here (created by
        # import_project on fresh, sync_prompts on adopt).
        self._remap_prompts(src_tool_id, tgt_tool_id, tool_name, lock)

        # Tools never exported on source (e.g. empty projects — backend
        # blocks their export) have no registry entry and no workflow
        # references; republishing would fail the same backend guard.
        try:
            src_regs = self.ctx.source.list_registries(custom_tool=src_tool_id)
        except Exception as e:
            logger.warning(
                "source registry lookup failed for tool '%s' "
                "(downstream ToolInstance clone may skip): %s",
                tool_name,
                e,
            )
            with lock:
                result.failed += 1
                result.errors.append(f"registry remap lookup {tool_name}: {e}")
            return

        if not src_regs:
            logger.info(
                "tool '%s' was never exported on source; skipping registry republish",
                tool_name,
            )
            return

        try:
            self.ctx.target.export_custom_tool(tgt_tool_id)
            logger.info(
                "republished registry for tool '%s' tgt=%s", tool_name, tgt_tool_id
            )
        except Exception as e:
            # Republish can 500 on incomplete/stale source registries (e.g.
            # empty run prompts). The tool itself cloned fine; only its
            # registry entry is missing, so downstream tool_instances
            # cascade-skip. Warn rather than fail the whole tool.
            logger.warning("Registry republish failed for tool '%s': %s", tool_name, e)
            with lock:
                result.warnings.append(
                    f"republish {tool_name}: skipped ({e}) — downstream tool "
                    "instances will cascade-skip until re-published"
                )
            return

        try:
            tgt_regs = self.ctx.target.list_registries(custom_tool=tgt_tool_id)
        except Exception as e:
            logger.warning(
                "target registry lookup failed for tool '%s' "
                "(downstream ToolInstance clone may skip): %s",
                tool_name,
                e,
            )
            with lock:
                result.failed += 1
                result.errors.append(f"registry remap lookup {tool_name}: {e}")
            return

        if src_regs and tgt_regs:
            with lock:
                self.ctx.remap.record(
                    "prompt_studio_registry",
                    src_regs[0]["prompt_registry_id"],
                    tgt_regs[0]["prompt_registry_id"],
                )

    def _record_planned_registry(
        self, src_tool_id: str, tool_name: str, lock: threading.Lock
    ) -> None:
        """Dry-run: record a planned prompt_studio_registry remap from the
        source registry id, so ToolInstancePhase can resolve tool_id and
        plan-count. No-op for tools never exported on source (no registry).
        """
        try:
            src_regs = self.ctx.source.list_registries(custom_tool=src_tool_id)
        except Exception as e:
            logger.warning(
                "[dry-run] source registry lookup failed for tool '%s' "
                "(tool_instance plan may under-count): %s",
                tool_name,
                e,
            )
            return
        if not src_regs:
            return
        with lock:
            self.ctx.remap.record_planned(
                "prompt_studio_registry", src_regs[0]["prompt_registry_id"]
            )

    def _remap_prompts(
        self,
        src_tool_id: str,
        tgt_tool_id: str,
        tool_name: str,
        lock: threading.Lock,
    ) -> None:
        """Record source->target prompt-id remaps, matched by prompt_key.

        Best-effort: a prompt without a matching key on target is skipped
        (the dependent phase counts it as unresolved), and a listing
        failure leaves the remap empty rather than failing the tool.
        """
        try:
            src_prompts = self.ctx.source.list_prompts(src_tool_id)
            tgt_prompts = self.ctx.target.list_prompts(tgt_tool_id)
        except Exception as e:
            logger.warning(
                "prompt-id remap skipped for tool '%s' "
                "(dependent prompt-scoped phases may under-resolve): %s",
                tool_name,
                e,
            )
            return
        # prompt_key is effectively unique per tool; first match wins.
        tgt_by_key = {p["prompt_key"]: p["prompt_id"] for p in tgt_prompts}
        with lock:
            for sp in src_prompts:
                tgt_pid = tgt_by_key.get(sp["prompt_key"])
                if tgt_pid:
                    self.ctx.remap.record("prompt", sp["prompt_id"], tgt_pid)

    def _record_planned_prompts(
        self, src_tool_id: str, lock: threading.Lock
    ) -> None:
        """Dry-run: record a planned prompt remap per source prompt so
        prompt-scoped phases can resolve their FK and plan-count.
        """
        try:
            src_prompts = self.ctx.source.list_prompts(src_tool_id)
        except Exception as e:
            logger.warning(
                "[dry-run] source prompt listing failed for tool %s "
                "(prompt-scoped plan may under-count): %s",
                src_tool_id,
                e,
            )
            return
        with lock:
            for sp in src_prompts:
                self.ctx.remap.record_planned("prompt", sp["prompt_id"])

    def _adopt(
        self,
        match: dict[str, Any],
        export_data: dict[str, Any],
        result: PhaseResult,
        tool_name: str,
        src_tool_id: str,
        lock: threading.Lock,
    ) -> str | None:
        if self.ctx.options.on_name_conflict == "abort":
            raise NameConflictError(
                f"tool '{tool_name}' already exists in target as {match['tool_id']}"
            )

        tgt_tool_id = match["tool_id"]
        if self.ctx.options.dry_run:
            with lock:
                result.adopted += 1
                self.ctx.remap.record("custom_tool", src_tool_id, tgt_tool_id)
            logger.info(
                "[dry-run] would sync prompts into adopted tool '%s' src=%s -> tgt=%s",
                tool_name,
                src_tool_id,
                tgt_tool_id,
            )
            return tgt_tool_id

        try:
            self.ctx.target.sync_prompts(tgt_tool_id, export_data)
        except Exception as e:
            logger.exception("sync_prompts failed for tool %s: %s", tool_name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"sync {tool_name}: {e}")
            return None

        with lock:
            result.adopted += 1
            self.ctx.remap.record("custom_tool", src_tool_id, tgt_tool_id)
        logger.info(
            "adopted tool '%s' src=%s -> tgt=%s (prompts re-synced)",
            tool_name,
            src_tool_id,
            tgt_tool_id,
        )
        return tgt_tool_id

    def _create_fresh(
        self,
        export_data: dict[str, Any],
        src_tool_id: str,
        tool_name: str,
        result: PhaseResult,
        lock: threading.Lock,
    ) -> str | None:
        # Run the source-side checks even in dry-run — they decide whether a
        # real run would create or frictionless-skip, so plan counts must
        # reflect them. Only the target-write steps are stubbed.
        default_profile = self._source_default_profile(src_tool_id, tool_name)

        # Frictionless adapters are cloud-only with no target equivalent —
        # skip + cascade. Only checkable when a default profile exists; a
        # profile-less tool is mirrored unconfigured (below).
        if default_profile is not None:
            invisible = self._invisible_source_adapter_names(default_profile)
            if invisible:
                self._register_frictionless_skip(
                    src_tool_id, tool_name, invisible, result, lock
                )
                return None

        if self.ctx.options.dry_run:
            # Target adapter resolution is skipped: adapters this run would
            # create don't exist on target yet. The frictionless check above
            # already caught the real skip cases.
            with lock:
                result.created += 1
                tgt_tool_id = self.ctx.remap.record_planned("custom_tool", src_tool_id)
            logger.info(
                "[dry-run] would import tool '%s' src=%s", tool_name, src_tool_id
            )
            return tgt_tool_id

        # Best-effort adapter wiring: resolve what maps, leave the rest
        # unconfigured. The backend tolerates a partial/empty set and flags
        # needs_adapter_config — mirror an incomplete source tool rather than
        # fail the clone (operator finishes wiring on target and re-runs).
        adapter_ids = (
            self._resolve_target_adapter_ids(default_profile, tool_name)
            if default_profile is not None
            else {}
        )

        try:
            tgt = self.ctx.target.import_project(export_data, adapter_ids=adapter_ids)
        except Exception as e:
            logger.exception("import_project failed for tool %s: %s", tool_name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"import {tool_name}: {e}")
            return None

        tgt_tool_id = tgt["tool_id"]
        needs_cfg = tgt.get("needs_adapter_config")
        with lock:
            result.created += 1
            self.ctx.remap.record("custom_tool", src_tool_id, tgt_tool_id)
            if needs_cfg:
                result.warnings.append(
                    f"tool {tool_name}: imported without full adapter config — "
                    "wire adapters on target and re-run to complete downstream"
                )
        logger.info(
            "created tool '%s' src=%s -> tgt=%s (needs_adapter_config=%s)",
            tool_name,
            src_tool_id,
            tgt_tool_id,
            needs_cfg,
        )
        return tgt_tool_id

    def _source_default_profile(
        self, src_tool_id: str, tool_name: str
    ) -> dict[str, Any] | None:
        try:
            src_profiles = self.ctx.source.list_profiles(src_tool_id)
        except Exception as e:
            logger.exception(
                "Failed to list source profiles for tool %s: %s", tool_name, e
            )
            return None

        default = next(
            (p for p in src_profiles if p.get("is_default")),
            src_profiles[0] if src_profiles else None,
        )
        if default is None:
            logger.warning(
                "source tool '%s' has no profiles to derive adapter ids from",
                tool_name,
            )
        return default

    def _invisible_source_adapter_names(
        self, default_profile: dict[str, Any]
    ) -> list[str]:
        """Profile adapter names not in the source's visible adapter set
        (typically frictionless) — these can't be migrated.
        """
        missing: list[str] = []
        for src_field, _ in _PROFILE_ADAPTER_FIELDS:
            adapter_name = _extract_adapter_name(default_profile.get(src_field))
            if adapter_name and adapter_name not in self._src_adapter_names:
                missing.append(adapter_name)
        return missing

    def _register_frictionless_skip(
        self,
        src_tool_id: str,
        tool_name: str,
        missing_adapters: list[str],
        result: PhaseResult,
        lock: threading.Lock,
    ) -> None:
        """Record the skip + source registry id so dependent workflows
        cascade-skip downstream.
        """
        logger.warning(
            "skipping tool '%s' src=%s — default profile references adapters "
            "not visible to this org's adapter listing (frictionless?): %s. "
            "Wire equivalents on target and re-run.",
            tool_name,
            src_tool_id,
            missing_adapters,
        )
        try:
            src_regs = self.ctx.source.list_registries(custom_tool=src_tool_id)
        except Exception as e:
            logger.warning(
                "registry lookup failed for skipped tool '%s' — "
                "downstream cascade-skip may not fire: %s",
                tool_name,
                e,
            )
            src_regs = []
        with lock:
            result.skipped += 1
            for reg in src_regs:
                reg_id = reg.get("prompt_registry_id")
                if reg_id:
                    self.ctx.skipped_custom_tool_registry_ids.add(reg_id)

    def _resolve_target_adapter_ids(
        self, default_profile: dict[str, Any], tool_name: str
    ) -> dict[str, str]:
        """Source profile carries adapter NAMES (per serializer); resolve
        each to a target adapter UUID via ``list_adapters(name=...)``.

        Best-effort: adapters that can't be resolved are omitted (not fatal).
        The backend tolerates a partial/empty set and flags
        ``needs_adapter_config``. AdapterPhase preserves names across orgs,
        so a miss means the adapter wasn't cloned (frictionless, or a failed
        adapter clone) — the operator wires it on target and re-runs.
        """
        resolved: dict[str, str] = {}
        for src_field, form_field in _PROFILE_ADAPTER_FIELDS:
            adapter_name = _extract_adapter_name(default_profile.get(src_field))
            if not adapter_name:
                logger.warning(
                    "source default profile for tool '%s' missing adapter '%s'",
                    tool_name,
                    src_field,
                )
                continue
            try:
                matches = self.ctx.target.list_adapters(name=adapter_name)
            except Exception as e:
                logger.exception(
                    "list_adapters lookup failed for %s on tool '%s': %s",
                    adapter_name,
                    tool_name,
                    e,
                )
                continue
            if not matches:
                logger.warning(
                    "no target adapter named '%s' for field %s on tool '%s' — "
                    "left unconfigured",
                    adapter_name,
                    src_field,
                    tool_name,
                )
                continue
            resolved[form_field] = matches[0]["id"]
        return resolved
