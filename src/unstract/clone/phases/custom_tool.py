"""Migrate prompt-studio projects via the project-transfer endpoints.

For each source tool the phase:

1. ``GET prompt-studio/project-transfer/{src_tool_id}`` — pulls a
   portable JSON blob (tool_metadata, tool_settings,
   default_profile_settings, prompts, export_metadata).
2. Decides fresh vs adopt by looking up the target tool by name.
3. **Fresh path**: reads source's default ProfileManager to learn the
   adapter UUIDs the profile is bound to, remaps each via the running
   ``adapter`` remap table, and POSTs the import as a multipart upload
   with target-org adapter ids on the form. Backend creates the tool,
   the default profile, and all prompts server-side in one call.
4. **Adopt path**: POSTs ``sync-prompts`` on the existing target tool.
   Backend rip-and-replaces prompts + ``tool_settings`` and leaves the
   target's locally-configured profiles + adapters untouched (which is
   what the operator wants — they may have rewired adapters on target).
5. Republishes ``PromptStudioRegistry`` via the export action and
   records the ``custom_tool`` + ``prompt_studio_registry`` remaps so
   downstream ToolInstancePhase can rewrite ``ToolInstance.tool_id``.

Adapter id discovery for the fresh path needs all four of LLM,
vector_db, embedding, x2text. If any source adapter can't be resolved
via the adapter remap, the tool is failed cleanly — we never want to
land a half-wired profile.
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

        # Source's service-account view hides frictionless adapters; a
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

        if tgt_tool_id is None:
            return

        with lock:
            self.ctx.remap.record("custom_tool", src_tool_id, tgt_tool_id)

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
            return

        try:
            self.ctx.target.export_custom_tool(tgt_tool_id)
            logger.info(
                "republished registry for tool '%s' tgt=%s", tool_name, tgt_tool_id
            )
        except Exception as e:
            logger.exception("Registry republish failed for tool %s: %s", tool_name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"export {tool_name}: {e}")
            return

        try:
            src_regs = self.ctx.source.list_registries(custom_tool=src_tool_id)
            tgt_regs = self.ctx.target.list_registries(custom_tool=tgt_tool_id)
        except Exception as e:
            logger.warning(
                "registry remap lookup failed for tool '%s' "
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
                result.skipped += 1
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
        if self.ctx.options.dry_run:
            with lock:
                result.skipped += 1
            logger.info(
                "[dry-run] would import tool '%s' src=%s", tool_name, src_tool_id
            )
            return None

        default_profile = self._source_default_profile(src_tool_id, tool_name)
        if default_profile is None:
            with lock:
                result.failed += 1
                result.errors.append(
                    f"import {tool_name}: no default profile on source"
                )
            return None

        invisible = self._invisible_source_adapter_names(default_profile)
        if invisible:
            self._register_frictionless_skip(
                src_tool_id, tool_name, invisible, result, lock
            )
            return None

        adapter_ids = self._resolve_target_adapter_ids(default_profile, tool_name)
        if adapter_ids is None:
            with lock:
                result.failed += 1
                result.errors.append(
                    f"import {tool_name}: missing target adapter remap for default"
                )
            return None

        try:
            tgt = self.ctx.target.import_project(export_data, adapter_ids=adapter_ids)
        except Exception as e:
            logger.exception("import_project failed for tool %s: %s", tool_name, e)
            with lock:
                result.failed += 1
                result.errors.append(f"import {tool_name}: {e}")
            return None

        tgt_tool_id = tgt["tool_id"]
        with lock:
            result.created += 1
        logger.info(
            "created tool '%s' src=%s -> tgt=%s (needs_adapter_config=%s)",
            tool_name,
            src_tool_id,
            tgt_tool_id,
            tgt.get("needs_adapter_config"),
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
            "not visible to the source service account (frictionless?): %s. "
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
    ) -> dict[str, str] | None:
        """Source profile carries adapter NAMES (per serializer); resolve
        each name to a target adapter UUID via ``list_adapters(name=...)``.

        Returns ``None`` if any of the four required adapters can't be
        found on target — caller fails the tool. AdapterPhase preserves
        names across orgs so this lookup should always hit when the
        adapter clone ran cleanly.
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
                return None
            try:
                matches = self.ctx.target.list_adapters(name=adapter_name)
            except Exception as e:
                logger.exception(
                    "list_adapters lookup failed for %s on tool '%s': %s",
                    adapter_name,
                    tool_name,
                    e,
                )
                return None
            if not matches:
                logger.warning(
                    "no target adapter named '%s' for field %s on tool '%s'",
                    adapter_name,
                    src_field,
                    tool_name,
                )
                return None
            resolved[form_field] = matches[0]["id"]
        return resolved
