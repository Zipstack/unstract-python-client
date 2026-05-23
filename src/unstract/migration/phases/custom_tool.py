"""Migrate prompt-studio projects (CustomTool) and their children.

Composite phase: a single project carries ``ProfileManager`` rows (LLM
triad config) and ``ToolStudioPrompt`` rows (the actual prompts). All
three must land together for the project to be functional on target, so
they live in one phase rather than three sibling phases.

Within a project, the create order is:

  1. CustomTool — POST creates the project and auto-creates one default
     ProfileManager on target.
  2. ProfileManagers — on a freshly-created tool we delete the auto-default
     first so the source's profiles land cleanly. On an adopted tool we
     reconcile by ``profile_name`` (per-tool unique).
  3. ToolStudioPrompts — reconcile by ``prompt_key`` (per-tool unique).
  4. Republish PromptStudioRegistry via the ``export-tool`` action so the
     registry row is rebuilt server-side from the now-correct child state.
     Avoids the SDK carrying ``tool_metadata`` JSON across orgs.

Walker remapping: adapter UUIDs embedded in the tool's adapter FKs
(``monitor_llm``, ``challenge_llm``, ``summarize_llm_adapter``), in the
profile's adapter FKs (``llm``, ``embedding_model``, ``vector_store``,
``x2text``), and in the prompt's ``profile_manager`` + ``tool_id`` FKs
are remapped before POST using the running ``RemapTable``.

The ProfileManager GET response expands adapter FKs into nested adapter
dicts (per the backend serializer's ``to_representation``); we flatten
them back to UUIDs before walker pass.
"""

from __future__ import annotations

import logging
from typing import Any

from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.base import SERVER_MANAGED, Phase, build_post_payload
from unstract.migration.report import MigrationReport, PhaseResult
from unstract.migration.walker import remap_uuids

logger = logging.getLogger(__name__)

TOOL_PATH = "prompt-studio/"

# Per-action endpoints on PromptStudioCoreView don't surface their own
# DRF metadata (OPTIONS returns the parent CustomToolSerializer schema).
# Hardcode the model-derived writable subset for the children and let the
# integration test catch backend drift.
PROFILE_WRITABLE: frozenset[str] = frozenset(
    {
        "profile_name",
        "vector_store",
        "embedding_model",
        "llm",
        "x2text",
        "chunk_size",
        "chunk_overlap",
        "reindex",
        "retrieval_strategy",
        "similarity_top_k",
        "section",
        "prompt_studio_tool",
        "is_default",
        "is_summarize_llm",
    }
)

PROMPT_WRITABLE: frozenset[str] = frozenset(
    {
        "prompt_key",
        "enforce_type",
        "prompt",
        "tool_id",
        "sequence_number",
        "prompt_type",
        "profile_manager",
        "output",
        "assert_prompt",
        "assertion_failure_prompt",
        "required",
        "is_assert",
        "active",
        "output_metadata",
        "postprocessing_webhook_url",
        "evaluate",
        "eval_quality_faithfulness",
        "eval_quality_correctness",
        "eval_quality_relevance",
        "eval_security_pii",
        "eval_guidance_toxicity",
        "eval_guidance_completeness",
    }
)

_PROFILE_ADAPTER_KEYS = ("llm", "embedding_model", "vector_store", "x2text")


def _flatten_profile_adapters(profile: dict[str, Any]) -> dict[str, Any]:
    """ProfileManagerSerializer.to_representation expands FK adapters into
    nested dicts; for write paths we need flat UUIDs back.
    """
    out = dict(profile)
    for key in _PROFILE_ADAPTER_KEYS:
        val = out.get(key)
        if isinstance(val, dict) and "id" in val:
            out[key] = val["id"]
    return out


class CustomToolPhase(Phase):
    name = "custom_tool"

    def run(self, report: MigrationReport) -> PhaseResult:
        result = report.get_phase(self.name)
        try:
            self._tool_writable = self.ctx.target.get_post_schema(TOOL_PATH)
        except Exception as e:
            logger.exception("Failed to fetch target POST schema for prompt-studio: %s", e)
            result.failed += 1
            result.errors.append(f"OPTIONS prompt-studio: {e}")
            return result

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
            src_tool = self.ctx.source.get_custom_tool(src_tool_id)
        except Exception as e:
            logger.exception("Failed to GET source tool %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"GET source tool {tool_name}: {e}")
            return

        tgt_tool, fresh = self._get_or_create_tool(src_tool, result)
        if tgt_tool is None:
            return

        tgt_tool_id = tgt_tool["tool_id"]
        self.ctx.remap.record("custom_tool", src_tool_id, tgt_tool_id)

        if self.ctx.options.dry_run:
            logger.info(
                "[dry-run] would reconcile profiles+prompts for tool '%s' src=%s",
                tool_name, src_tool_id,
            )
            return

        try:
            src_profiles = self.ctx.source.list_profiles(src_tool_id)
        except Exception as e:
            logger.exception("Failed to list source profiles for %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"list src profiles {tool_name}: {e}")
            return

        try:
            self._reconcile_profiles(src_profiles, tgt_tool_id, fresh)
        except Exception as e:
            logger.exception("Profile reconcile failed for tool %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"profiles {tool_name}: {e}")
            return

        try:
            src_prompts = src_tool.get("prompts") or []
            self._reconcile_prompts(src_prompts, tgt_tool_id)
        except Exception as e:
            logger.exception("Prompt reconcile failed for tool %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"prompts {tool_name}: {e}")
            return

        try:
            self.ctx.target.export_custom_tool(tgt_tool_id)
            logger.info("republished registry for tool '%s' tgt=%s", tool_name, tgt_tool_id)
        except Exception as e:
            logger.exception("Registry republish failed for tool %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"export {tool_name}: {e}")

    def _get_or_create_tool(
        self, src_tool: dict[str, Any], result: PhaseResult
    ) -> tuple[dict[str, Any] | None, bool]:
        tool_name = src_tool["tool_name"]
        src_tool_id = src_tool["tool_id"]

        try:
            target_tools = self.ctx.target.list_custom_tools()
        except Exception as e:
            logger.exception("Failed to list target tools: %s", e)
            result.failed += 1
            result.errors.append(f"list target tools: {e}")
            return None, False

        match = next((t for t in target_tools if t["tool_name"] == tool_name), None)
        if match is not None:
            if self.ctx.options.on_name_conflict == "abort":
                raise NameConflictError(
                    f"tool '{tool_name}' already exists in target as {match['tool_id']}"
                )
            result.adopted += 1
            logger.info(
                "adopted tool '%s' src=%s -> tgt=%s",
                tool_name, src_tool_id, match["tool_id"],
            )
            return match, False

        if self.ctx.options.dry_run:
            result.skipped += 1
            logger.info("[dry-run] would create tool '%s' src=%s", tool_name, src_tool_id)
            return None, True

        remapped = remap_uuids(src_tool, self.ctx.remap)
        payload = build_post_payload(remapped, self._tool_writable)
        try:
            tgt = self.ctx.target.create_custom_tool(payload)
        except Exception as e:
            logger.exception("Failed to create tool %s: %s", tool_name, e)
            result.failed += 1
            result.errors.append(f"create tool {tool_name}: {e}")
            return None, True
        result.created += 1
        logger.info(
            "created tool '%s' src=%s -> tgt=%s",
            tool_name, src_tool_id, tgt["tool_id"],
        )
        return tgt, True

    def _reconcile_profiles(
        self,
        src_profiles: list[dict[str, Any]],
        tgt_tool_id: str,
        fresh: bool,
    ) -> None:
        if fresh:
            for p in self.ctx.target.list_profiles(tgt_tool_id):
                self.ctx.target.delete_profile(p["profile_id"])
                logger.debug("deleted auto-default profile %s", p["profile_id"])

        src_default_id: str | None = None
        for src_profile in src_profiles:
            src_pid = src_profile["profile_id"]
            if src_profile.get("is_default"):
                src_default_id = src_pid

            target_profiles_by_name = {
                p["profile_name"]: p
                for p in self.ctx.target.list_profiles(tgt_tool_id)
            }
            existing = target_profiles_by_name.get(src_profile["profile_name"])

            if existing is not None:
                tgt_pid = existing["profile_id"]
                logger.info(
                    "adopted profile '%s' src=%s -> tgt=%s",
                    src_profile["profile_name"], src_pid, tgt_pid,
                )
            else:
                flat = _flatten_profile_adapters(src_profile)
                remapped = remap_uuids(flat, self.ctx.remap)
                remapped["prompt_studio_tool"] = tgt_tool_id
                payload = build_post_payload(remapped, PROFILE_WRITABLE)
                tgt = self.ctx.target.create_profile(tgt_tool_id, payload)
                tgt_pid = tgt["profile_id"]
                logger.info(
                    "created profile '%s' src=%s -> tgt=%s",
                    src_profile["profile_name"], src_pid, tgt_pid,
                )
            self.ctx.remap.record("profile_manager", src_pid, tgt_pid)

        if src_default_id:
            tgt_default = self.ctx.remap.resolve("profile_manager", src_default_id)
            if tgt_default:
                self.ctx.target.set_default_profile(tgt_tool_id, tgt_default)

    def _reconcile_prompts(
        self, src_prompts: list[dict[str, Any]], tgt_tool_id: str
    ) -> None:
        existing_prompts = self.ctx.target.list_prompts(tool_id=tgt_tool_id)
        by_key = {p["prompt_key"]: p for p in existing_prompts}

        for src_prompt in src_prompts:
            src_prompt_id = src_prompt["prompt_id"]
            key = src_prompt["prompt_key"]
            existing = by_key.get(key)
            if existing is not None:
                tgt_pid = existing["prompt_id"]
                logger.info(
                    "adopted prompt '%s' src=%s -> tgt=%s",
                    key, src_prompt_id, tgt_pid,
                )
            else:
                remapped = remap_uuids(src_prompt, self.ctx.remap)
                remapped["tool_id"] = tgt_tool_id
                payload = build_post_payload(remapped, PROMPT_WRITABLE - SERVER_MANAGED)
                tgt = self.ctx.target.create_prompt(tgt_tool_id, payload)
                tgt_pid = tgt["prompt_id"]
                logger.info(
                    "created prompt '%s' src=%s -> tgt=%s",
                    key, src_prompt_id, tgt_pid,
                )
            self.ctx.remap.record("prompt", src_prompt_id, tgt_pid)
