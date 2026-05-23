"""Tests for ``CustomToolPhase`` — the composite tool + profile + prompt phase.

Coverage:
- happy path: fresh tool → tool created, auto-default deleted, source
  profiles + prompts created, registry republished.
- idempotency: re-run is a no-op when tool + profile names + prompt keys
  already match on target.
- adopt path: existing tool on target, partial overlap of profiles —
  matching ones adopted, missing ones created.
- dry-run: nothing posted.
- adapter UUID remap into profile FKs.
"""

from __future__ import annotations

from unstract.migration.context import (
    MigrationContext,
    MigrationOptions,
    RemapTable,
)
from unstract.migration.phases.custom_tool import CustomToolPhase
from unstract.migration.report import MigrationReport


TOOL_POST_SCHEMA = frozenset(
    {
        "tool_name",
        "description",
        "author",
        "icon",
        "preamble",
        "postamble",
        "prompt_grammer",
        "monitor_llm",
        "challenge_llm",
        "summarize_llm_adapter",
        "custom_data",
        "single_pass_extraction_mode",
        "shared_users",
        "shared_to_org",
    }
)


class FakeClient:
    """In-memory stand-in for ``PlatformClient`` covering the prompt-studio surface."""

    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}
        self.profiles_by_tool: dict[str, list[dict]] = {}
        self.prompts_by_tool: dict[str, list[dict]] = {}
        self.registries_by_tool: dict[str, dict] = {}
        self.export_calls: list[str] = []
        self._next = 1

    # --- ID helper ---
    def _mint(self, prefix: str) -> str:
        s = f"tgt-{prefix}-{self._next:04d}"
        self._next += 1
        return s

    # --- schema ---
    def get_post_schema(self, entity_path: str) -> frozenset[str]:
        if entity_path == "prompt-studio/":
            return TOOL_POST_SCHEMA
        raise AssertionError(f"unexpected OPTIONS path: {entity_path}")

    # --- tools ---
    def list_custom_tools(self) -> list[dict]:
        return list(self.tools.values())

    def get_custom_tool(self, tool_id: str) -> dict:
        return self.tools[tool_id]

    def create_custom_tool(self, payload: dict) -> dict:
        tool_id = self._mint("tool")
        tool = {**payload, "tool_id": tool_id, "prompts": []}
        self.tools[tool_id] = tool
        # Backend auto-creates a default profile on create.
        auto = {
            "profile_id": self._mint("autoprofile"),
            "profile_name": "Default",
            "is_default": True,
            "prompt_studio_tool": tool_id,
        }
        self.profiles_by_tool[tool_id] = [auto]
        self.prompts_by_tool[tool_id] = []
        return tool

    def export_custom_tool(self, tool_id: str, *, force: bool = True) -> None:
        self.export_calls.append(tool_id)
        # Mimic the backend: export creates/updates a registry row for the tool.
        self.registries_by_tool.setdefault(
            tool_id,
            {"prompt_registry_id": self._mint("registry"), "custom_tool": tool_id},
        )

    def list_registries(self, *, custom_tool: str | None = None) -> list[dict]:
        if custom_tool is None:
            return list(self.registries_by_tool.values())
        reg = self.registries_by_tool.get(custom_tool)
        return [reg] if reg else []

    # --- profiles ---
    def list_profiles(self, tool_id: str) -> list[dict]:
        return list(self.profiles_by_tool.get(tool_id, []))

    def create_profile(self, tool_id: str, payload: dict) -> dict:
        new = {**payload, "profile_id": self._mint("profile")}
        self.profiles_by_tool.setdefault(tool_id, []).append(new)
        return new

    def delete_profile(self, profile_id: str) -> None:
        for tid, profiles in self.profiles_by_tool.items():
            self.profiles_by_tool[tid] = [
                p for p in profiles if p["profile_id"] != profile_id
            ]

    def set_default_profile(self, tool_id: str, profile_id: str) -> None:
        for p in self.profiles_by_tool.get(tool_id, []):
            p["is_default"] = p["profile_id"] == profile_id

    # --- prompts ---
    def list_prompts(self, *, tool_id: str) -> list[dict]:
        return list(self.prompts_by_tool.get(tool_id, []))

    def create_prompt(self, tool_id: str, payload: dict) -> dict:
        new = {**payload, "prompt_id": self._mint("prompt")}
        self.prompts_by_tool.setdefault(tool_id, []).append(new)
        return new


def _ctx(source, target, *, remap=None, **opt_overrides) -> MigrationContext:
    return MigrationContext(
        source=source,
        target=target,
        options=MigrationOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


def _src_tool(tool_id: str, name: str, prompts: list[dict] | None = None) -> dict:
    return {
        "tool_id": tool_id,
        "tool_name": name,
        "description": f"{name} desc",
        "author": "src",
        "icon": "",
        "preamble": "",
        "postamble": "",
        "prompt_grammer": {},
        "monitor_llm": None,
        "challenge_llm": None,
        "summarize_llm_adapter": None,
        "custom_data": {},
        "single_pass_extraction_mode": False,
        "shared_users": [],
        "shared_to_org": False,
        "prompts": prompts or [],
    }


def _src_profile(pid: str, name: str, *, is_default: bool = False) -> dict:
    return {
        "profile_id": pid,
        "profile_name": name,
        "is_default": is_default,
        "is_summarize_llm": False,
        # Mimic to_representation expansion: nested adapter dicts.
        "llm": {"id": "11111111-1111-1111-1111-111111111111", "adapter_name": "L"},
        "embedding_model": {"id": "22222222-2222-2222-2222-222222222222", "adapter_name": "E"},
        "vector_store": {"id": "33333333-3333-3333-3333-333333333333", "adapter_name": "V"},
        "x2text": {"id": "44444444-4444-4444-4444-444444444444", "adapter_name": "X"},
        "chunk_size": 1024,
        "chunk_overlap": 128,
        "reindex": False,
        "retrieval_strategy": "simple",
        "similarity_top_k": 3,
        "section": "default",
        "prompt_studio_tool": None,
    }


def _src_prompt(prompt_id: str, key: str, profile_id: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt_key": key,
        "prompt": f"What is {key}?",
        "enforce_type": "string",
        "prompt_type": "prompt",
        "sequence_number": 1,
        "tool_id": "src-tool-x",
        "profile_manager": profile_id,
        "output": "",
        "active": True,
        "required": False,
    }


def _preload_source(client: FakeClient, tool_id: str) -> dict:
    """Helper to set up a source FakeClient with one tool, one profile, one prompt."""
    profile = _src_profile("src-profile-1", "Default", is_default=True)
    prompt = _src_prompt("src-prompt-1", "field_a", "src-profile-1")
    tool = _src_tool(tool_id, "Invoice Extractor", prompts=[prompt])
    client.tools[tool_id] = tool
    client.profiles_by_tool[tool_id] = [profile]
    client.prompts_by_tool[tool_id] = [prompt]
    client.registries_by_tool[tool_id] = {
        "prompt_registry_id": "55555555-5555-5555-5555-555555555555",
        "custom_tool": tool_id,
    }
    return tool


def _preload_remap_with_adapters(remap: RemapTable) -> None:
    remap.record("adapter", "11111111-1111-1111-1111-111111111111", "a1111111-1111-1111-1111-111111111111")
    remap.record("adapter", "22222222-2222-2222-2222-222222222222", "a2222222-2222-2222-2222-222222222222")
    remap.record("adapter", "33333333-3333-3333-3333-333333333333", "a3333333-3333-3333-3333-333333333333")
    remap.record("adapter", "44444444-4444-4444-4444-444444444444", "a4444444-4444-4444-4444-444444444444")


def test_fresh_tool_creates_tool_profiles_prompts_and_republishes():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source(src, "src-tool-x")
    remap = RemapTable()
    _preload_remap_with_adapters(remap)
    ctx = _ctx(src, tgt, remap=remap)
    report = MigrationReport()

    result = CustomToolPhase(ctx).run(report)

    assert result.created == 1
    assert result.failed == 0
    assert len(tgt.tools) == 1
    tgt_tool_id = next(iter(tgt.tools))

    # Auto-default profile deleted; exactly one profile (the source's) remains.
    profiles = tgt.profiles_by_tool[tgt_tool_id]
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile["profile_name"] == "Default"
    assert profile["is_default"] is True
    # Adapter FKs remapped via walker.
    assert profile["llm"] == "a1111111-1111-1111-1111-111111111111"
    assert profile["embedding_model"] == "a2222222-2222-2222-2222-222222222222"
    assert profile["vector_store"] == "a3333333-3333-3333-3333-333333333333"
    assert profile["x2text"] == "a4444444-4444-4444-4444-444444444444"

    # One prompt landed, pointing at the new tool.
    prompts = tgt.prompts_by_tool[tgt_tool_id]
    assert len(prompts) == 1
    assert prompts[0]["prompt_key"] == "field_a"
    assert prompts[0]["tool_id"] == tgt_tool_id

    # Registry republished exactly once.
    assert tgt.export_calls == [tgt_tool_id]

    # Remap records populated for downstream phases.
    assert ctx.remap.resolve("custom_tool", "src-tool-x") == tgt_tool_id
    assert ctx.remap.resolve("profile_manager", "src-profile-1") == profile["profile_id"]
    assert ctx.remap.resolve("prompt", "src-prompt-1") == prompts[0]["prompt_id"]
    # Registry remap recorded for ToolInstancePhase consumption.
    assert ctx.remap.resolve(
        "prompt_studio_registry", "55555555-5555-5555-5555-555555555555"
    ) == tgt.registries_by_tool[tgt_tool_id]["prompt_registry_id"]


def test_idempotent_rerun_does_not_create_duplicates():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source(src, "src-tool-x")
    remap = RemapTable()
    _preload_remap_with_adapters(remap)
    ctx = _ctx(src, tgt, remap=remap)

    CustomToolPhase(ctx).run(MigrationReport())
    tgt_tool_id = next(iter(tgt.tools))
    profile_count = len(tgt.profiles_by_tool[tgt_tool_id])
    prompt_count = len(tgt.prompts_by_tool[tgt_tool_id])
    export_count = len(tgt.export_calls)

    report2 = MigrationReport()
    result2 = CustomToolPhase(ctx).run(report2)

    assert result2.adopted == 1
    assert result2.created == 0
    assert len(tgt.profiles_by_tool[tgt_tool_id]) == profile_count
    assert len(tgt.prompts_by_tool[tgt_tool_id]) == prompt_count
    # Republish still fires (rebuild registry idempotently).
    assert len(tgt.export_calls) == export_count + 1


def test_adopt_path_fills_missing_profile_only():
    src = FakeClient()
    tgt = FakeClient()
    # Source has TWO profiles.
    extra = _src_profile("src-profile-2", "HighRecall", is_default=False)
    default = _src_profile("src-profile-1", "Default", is_default=True)
    prompt = _src_prompt("src-prompt-1", "field_a", "src-profile-1")
    tool = _src_tool("src-tool-x", "Invoice Extractor", prompts=[prompt])
    src.tools["src-tool-x"] = tool
    src.profiles_by_tool["src-tool-x"] = [default, extra]
    src.prompts_by_tool["src-tool-x"] = [prompt]

    # Target already has the tool + the "Default" profile + the prompt.
    tgt_tool_id = "tgt-pre-tool"
    tgt.tools[tgt_tool_id] = {
        "tool_id": tgt_tool_id,
        "tool_name": "Invoice Extractor",
        "prompts": [],
    }
    tgt.profiles_by_tool[tgt_tool_id] = [
        {
            "profile_id": "tgt-pre-profile",
            "profile_name": "Default",
            "is_default": True,
            "prompt_studio_tool": tgt_tool_id,
        }
    ]
    tgt.prompts_by_tool[tgt_tool_id] = [
        {
            "prompt_id": "tgt-pre-prompt",
            "prompt_key": "field_a",
            "tool_id": tgt_tool_id,
        }
    ]

    remap = RemapTable()
    _preload_remap_with_adapters(remap)
    ctx = _ctx(src, tgt, remap=remap)

    result = CustomToolPhase(ctx).run(MigrationReport())

    assert result.adopted == 1
    # Adopted tool path: only the missing "HighRecall" profile got created.
    profiles = tgt.profiles_by_tool[tgt_tool_id]
    assert len(profiles) == 2
    names = {p["profile_name"] for p in profiles}
    assert names == {"Default", "HighRecall"}

    # Prompt was already there → adopted, not duplicated.
    assert len(tgt.prompts_by_tool[tgt_tool_id]) == 1

    assert ctx.remap.resolve("profile_manager", "src-profile-1") == "tgt-pre-profile"
    assert ctx.remap.resolve("prompt", "src-prompt-1") == "tgt-pre-prompt"


def test_dry_run_creates_nothing():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source(src, "src-tool-x")
    remap = RemapTable()
    _preload_remap_with_adapters(remap)
    ctx = _ctx(src, tgt, remap=remap, dry_run=True)

    result = CustomToolPhase(ctx).run(MigrationReport())

    assert result.skipped == 1
    assert tgt.tools == {}
    assert tgt.profiles_by_tool == {}
    assert tgt.export_calls == []
