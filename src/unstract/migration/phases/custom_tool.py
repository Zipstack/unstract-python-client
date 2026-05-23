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
from typing import Any

from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.base import Phase
from unstract.migration.report import MigrationReport, PhaseResult

logger = logging.getLogger(__name__)

_PROFILE_ADAPTER_FIELDS: tuple[tuple[str, str], ...] = (
    ("llm", "llm_adapter_id"),
    ("vector_store", "vector_db_adapter_id"),
    ("embedding_model", "embedding_adapter_id"),
    ("x2text", "x2text_adapter_id"),
)


def _extract_adapter_id(value: Any) -> str | None:
    """Profile FKs come back as nested dicts via serializer expansion;
    pull the UUID back out for either flat-string or nested-dict shapes.
    """
    if isinstance(value, dict):
        return value.get("id")
    if isinstance(value, str):
        return value
    return None


class CustomToolPhase(Phase):
    name = "custom_tool"

    def run(self, report: MigrationReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            src_tools = self.ctx.source.list_custom_tools()
        except Exception as e:
            logger.exception("Failed to list source custom tools: %s", e)
            result.failed += 1
            result.errors.append(f"list source custom tools: {e}")
            return result

        logger.info("Found %d custom tool(s) in source org", len(src_tools))
        for summary in src_tools:
            self._migrate_one(summary, result)
        return result

    def _migrate_one(self, summary: dict[str, Any], result: PhaseResult) -> None:
        tool_name = summary["tool_name"]
        src_tool_id = summary["tool_id"]

        try:
            export_data = self.ctx.source.export_project(src_tool_id)
        except Exception as e:
            logger.exception("Failed to export source tool '%s': %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"export src tool {tool_name}: {e}")
            return

        try:
            target_tools = self.ctx.target.list_custom_tools()
        except Exception as e:
            logger.exception("Failed to list target tools: %s", e)
            result.failed += 1
            result.errors.append(f"list target tools: {e}")
            return
        match = next(
            (t for t in target_tools if t["tool_name"] == tool_name), None
        )

        if match is not None:
            tgt_tool_id = self._adopt(match, export_data, result, tool_name, src_tool_id)
        else:
            tgt_tool_id = self._create_fresh(
                export_data, src_tool_id, tool_name, result
            )

        if tgt_tool_id is None:
            return

        self.ctx.remap.record("custom_tool", src_tool_id, tgt_tool_id)

        try:
            self.ctx.target.export_custom_tool(tgt_tool_id)
            logger.info("republished registry for tool '%s' tgt=%s", tool_name, tgt_tool_id)
        except Exception as e:
            logger.exception("Registry republish failed for tool %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"export {tool_name}: {e}")
            return

        # Record registry remap so ToolInstancePhase can rewrite
        # ToolInstance.tool_id (which stores a registry UUID as CharField).
        # Source registry exists only if the operator already published
        # the tool there; unpublished source tools simply produce no
        # ToolInstance rows for downstream to remap.
        try:
            src_regs = self.ctx.source.list_registries(custom_tool=src_tool_id)
            tgt_regs = self.ctx.target.list_registries(custom_tool=tgt_tool_id)
        except Exception as e:
            logger.warning(
                "registry remap lookup failed for tool '%s' "
                "(downstream ToolInstance migration may skip): %s",
                tool_name, e,
            )
            return

        if src_regs and tgt_regs:
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
    ) -> str | None:
        if self.ctx.options.on_name_conflict == "abort":
            raise NameConflictError(
                f"tool '{tool_name}' already exists in target as {match['tool_id']}"
            )

        tgt_tool_id = match["tool_id"]
        if self.ctx.options.dry_run:
            result.skipped += 1
            logger.info(
                "[dry-run] would sync prompts into adopted tool '%s' src=%s -> tgt=%s",
                tool_name, src_tool_id, tgt_tool_id,
            )
            return tgt_tool_id

        try:
            self.ctx.target.sync_prompts(tgt_tool_id, export_data)
        except Exception as e:
            logger.exception("sync_prompts failed for tool %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"sync {tool_name}: {e}")
            return None

        result.adopted += 1
        logger.info(
            "adopted tool '%s' src=%s -> tgt=%s (prompts re-synced)",
            tool_name, src_tool_id, tgt_tool_id,
        )
        return tgt_tool_id

    def _create_fresh(
        self,
        export_data: dict[str, Any],
        src_tool_id: str,
        tool_name: str,
        result: PhaseResult,
    ) -> str | None:
        if self.ctx.options.dry_run:
            result.skipped += 1
            logger.info(
                "[dry-run] would import tool '%s' src=%s", tool_name, src_tool_id
            )
            return None

        adapter_ids = self._resolve_target_adapter_ids(src_tool_id, tool_name)
        if adapter_ids is None:
            result.failed += 1
            result.errors.append(
                f"import {tool_name}: missing target adapter remap for default profile"
            )
            return None

        try:
            tgt = self.ctx.target.import_project(export_data, adapter_ids=adapter_ids)
        except Exception as e:
            logger.exception("import_project failed for tool %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"import {tool_name}: {e}")
            return None

        tgt_tool_id = tgt["tool_id"]
        result.created += 1
        logger.info(
            "created tool '%s' src=%s -> tgt=%s (needs_adapter_config=%s)",
            tool_name, src_tool_id, tgt_tool_id, tgt.get("needs_adapter_config"),
        )
        return tgt_tool_id

    def _resolve_target_adapter_ids(
        self, src_tool_id: str, tool_name: str
    ) -> dict[str, str] | None:
        """Read source default profile → remap each adapter UUID to target.

        Returns ``None`` if any of the four required adapters can't be
        resolved via the ``adapter`` remap — caller fails the tool.
        """
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
            return None

        resolved: dict[str, str] = {}
        for src_field, form_field in _PROFILE_ADAPTER_FIELDS:
            src_adapter_id = _extract_adapter_id(default.get(src_field))
            if not src_adapter_id:
                logger.warning(
                    "source default profile for tool '%s' missing adapter '%s'",
                    tool_name, src_field,
                )
                return None
            tgt_adapter_id = self.ctx.remap.resolve("adapter", src_adapter_id)
            if not tgt_adapter_id:
                logger.warning(
                    "no adapter remap for %s (field %s) on tool '%s'",
                    src_adapter_id, src_field, tool_name,
                )
                return None
            resolved[form_field] = tgt_adapter_id
        return resolved
