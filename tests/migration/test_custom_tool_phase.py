"""Tests for ``CustomToolPhase`` — project-transfer + sync-prompts based.

Coverage:
- fresh path: ``export_project`` on source → ``import_project`` on
  target with adapter ids resolved from source's default profile and
  remapped via the adapter table.
- adopt path: existing target tool with matching name →
  ``sync_prompts`` overwrites prompts; no profile/adapter writes.
- registry remap recorded after ``export_custom_tool``.
- dry-run: no writes on either side.
- abort on name conflict when option is set.
- missing adapter remap fails the tool cleanly.
"""

from __future__ import annotations

import pytest

from unstract.migration.context import (
    MigrationContext,
    MigrationOptions,
    RemapTable,
)
from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.custom_tool import CustomToolPhase
from unstract.migration.report import MigrationReport


SRC_LLM = "11111111-1111-1111-1111-111111111111"
SRC_EMB = "22222222-2222-2222-2222-222222222222"
SRC_VEC = "33333333-3333-3333-3333-333333333333"
SRC_X2T = "44444444-4444-4444-4444-444444444444"
TGT_LLM = "a1111111-1111-1111-1111-111111111111"
TGT_EMB = "a2222222-2222-2222-2222-222222222222"
TGT_VEC = "a3333333-3333-3333-3333-333333333333"
TGT_X2T = "a4444444-4444-4444-4444-444444444444"
SRC_REG = "55555555-5555-5555-5555-555555555555"


class FakeClient:
    """In-memory stand-in for ``PlatformClient`` covering project-transfer."""

    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}
        self.profiles_by_tool: dict[str, list[dict]] = {}
        self.export_blobs: dict[str, dict] = {}
        self.registries_by_tool: dict[str, dict] = {}
        # Call recorders.
        self.import_calls: list[tuple[dict, dict | None]] = []
        self.sync_calls: list[tuple[str, dict, bool]] = []
        self.export_tool_calls: list[str] = []
        self._next = 1

    def _mint(self, prefix: str) -> str:
        s = f"tgt-{prefix}-{self._next:04d}"
        self._next += 1
        return s

    # --- reads ---
    def list_custom_tools(self) -> list[dict]:
        return [
            {"tool_id": tid, "tool_name": t["tool_name"]}
            for tid, t in self.tools.items()
        ]

    def list_profiles(self, tool_id: str) -> list[dict]:
        return list(self.profiles_by_tool.get(tool_id, []))

    def export_project(self, tool_id: str) -> dict:
        return self.export_blobs[tool_id]

    def list_registries(self, *, custom_tool: str | None = None) -> list[dict]:
        if custom_tool is None:
            return list(self.registries_by_tool.values())
        reg = self.registries_by_tool.get(custom_tool)
        return [reg] if reg else []

    # --- writes ---
    def import_project(
        self, export_data: dict, adapter_ids: dict | None = None
    ) -> dict:
        self.import_calls.append((export_data, adapter_ids))
        tool_id = self._mint("tool")
        tool_name = export_data["tool_metadata"]["tool_name"]
        self.tools[tool_id] = {"tool_name": tool_name}
        return {
            "tool_id": tool_id,
            "message": f"Project imported successfully as '{tool_name}'",
            "needs_adapter_config": adapter_ids is None,
        }

    def sync_prompts(
        self, tool_id: str, export_data: dict, *, create_copy: bool = False
    ) -> dict:
        self.sync_calls.append((tool_id, export_data, create_copy))
        return {
            "prompts_created": len(export_data.get("prompts", [])),
            "prompts_deleted": 0,
            "tool_settings_updated": True,
        }

    def export_custom_tool(self, tool_id: str, *, force: bool = True) -> None:
        self.export_tool_calls.append(tool_id)
        self.registries_by_tool.setdefault(
            tool_id,
            {"prompt_registry_id": self._mint("registry"), "custom_tool": tool_id},
        )


def _ctx(source, target, *, remap=None, **opt_overrides) -> MigrationContext:
    return MigrationContext(
        source=source,
        target=target,
        options=MigrationOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


def _seed_adapter_remap(remap: RemapTable) -> None:
    remap.record("adapter", SRC_LLM, TGT_LLM)
    remap.record("adapter", SRC_EMB, TGT_EMB)
    remap.record("adapter", SRC_VEC, TGT_VEC)
    remap.record("adapter", SRC_X2T, TGT_X2T)


def _src_default_profile(*, nested: bool = True) -> dict:
    """Mimic ProfileManager serializer output.

    ``nested=True`` matches ``to_representation`` expanding FK adapters
    into nested dicts; ``nested=False`` covers the raw-UUID fallback.
    """
    if nested:
        return {
            "profile_id": "src-profile-1",
            "profile_name": "Default",
            "is_default": True,
            "llm": {"id": SRC_LLM, "adapter_name": "L"},
            "embedding_model": {"id": SRC_EMB, "adapter_name": "E"},
            "vector_store": {"id": SRC_VEC, "adapter_name": "V"},
            "x2text": {"id": SRC_X2T, "adapter_name": "X"},
        }
    return {
        "profile_id": "src-profile-1",
        "profile_name": "Default",
        "is_default": True,
        "llm": SRC_LLM,
        "embedding_model": SRC_EMB,
        "vector_store": SRC_VEC,
        "x2text": SRC_X2T,
    }


def _src_export_blob(tool_name: str) -> dict:
    return {
        "tool_metadata": {"tool_name": tool_name, "description": "x", "author": "a", "icon": None},
        "tool_settings": {"preamble": "p", "postamble": "q"},
        "default_profile_settings": {
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "retrieval_strategy": "simple",
            "similarity_top_k": 3,
            "section": "default",
            "profile_name": "Default",
        },
        "prompts": [
            {"prompt_key": "field_a", "prompt": "What is field_a?", "sequence_number": 1}
        ],
        "export_metadata": {"exported_at": "2026-05-24T00:00:00Z"},
    }


def _preload_source_tool(
    client: FakeClient, tool_id: str, tool_name: str, *, nested_profile: bool = True
) -> None:
    client.tools[tool_id] = {"tool_name": tool_name}
    client.profiles_by_tool[tool_id] = [_src_default_profile(nested=nested_profile)]
    client.export_blobs[tool_id] = _src_export_blob(tool_name)
    client.registries_by_tool[tool_id] = {
        "prompt_registry_id": SRC_REG,
        "custom_tool": tool_id,
    }


def test_fresh_imports_with_remapped_adapter_ids_and_records_registry():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Invoice Extractor")
    remap = RemapTable()
    _seed_adapter_remap(remap)
    ctx = _ctx(src, tgt, remap=remap)

    result = CustomToolPhase(ctx).run(MigrationReport())

    assert result.created == 1
    assert result.failed == 0
    # Exactly one import_project call with the right export blob + remapped adapter ids.
    assert len(tgt.import_calls) == 1
    blob, adapter_ids = tgt.import_calls[0]
    assert blob["tool_metadata"]["tool_name"] == "Invoice Extractor"
    assert adapter_ids == {
        "llm_adapter_id": TGT_LLM,
        "vector_db_adapter_id": TGT_VEC,
        "embedding_adapter_id": TGT_EMB,
        "x2text_adapter_id": TGT_X2T,
    }
    # No sync_prompts on fresh path.
    assert tgt.sync_calls == []
    # Registry republish fired exactly once.
    assert len(tgt.export_tool_calls) == 1
    tgt_tool_id = tgt.export_tool_calls[0]

    # Remap records populated for downstream phases.
    assert ctx.remap.resolve("custom_tool", "src-tool-x") == tgt_tool_id
    tgt_reg_id = tgt.registries_by_tool[tgt_tool_id]["prompt_registry_id"]
    assert ctx.remap.resolve("prompt_studio_registry", SRC_REG) == tgt_reg_id


def test_flat_uuid_profile_also_resolves_adapter_ids():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "T", nested_profile=False)
    remap = RemapTable()
    _seed_adapter_remap(remap)
    ctx = _ctx(src, tgt, remap=remap)

    CustomToolPhase(ctx).run(MigrationReport())

    _, adapter_ids = tgt.import_calls[0]
    assert adapter_ids["llm_adapter_id"] == TGT_LLM


def test_adopt_path_calls_sync_prompts_and_skips_import():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Invoice Extractor")
    # Target already has the tool with the same name.
    tgt.tools["tgt-existing"] = {"tool_name": "Invoice Extractor"}

    remap = RemapTable()
    _seed_adapter_remap(remap)
    ctx = _ctx(src, tgt, remap=remap)

    result = CustomToolPhase(ctx).run(MigrationReport())

    assert result.adopted == 1
    assert result.created == 0
    # sync_prompts ran against the pre-existing target tool, not a new one.
    assert len(tgt.sync_calls) == 1
    tool_id, blob, create_copy = tgt.sync_calls[0]
    assert tool_id == "tgt-existing"
    assert blob["tool_metadata"]["tool_name"] == "Invoice Extractor"
    assert create_copy is False
    # Import path never fired on adopt.
    assert tgt.import_calls == []
    # Registry still republished against the adopted tool.
    assert tgt.export_tool_calls == ["tgt-existing"]
    assert ctx.remap.resolve("custom_tool", "src-tool-x") == "tgt-existing"


def test_abort_on_name_conflict_raises():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Conflict")
    tgt.tools["tgt-existing"] = {"tool_name": "Conflict"}

    remap = RemapTable()
    _seed_adapter_remap(remap)
    ctx = _ctx(src, tgt, remap=remap, on_name_conflict="abort")

    with pytest.raises(NameConflictError):
        CustomToolPhase(ctx).run(MigrationReport())

    assert tgt.sync_calls == []
    assert tgt.import_calls == []


def test_dry_run_makes_no_writes():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "T")
    remap = RemapTable()
    _seed_adapter_remap(remap)
    ctx = _ctx(src, tgt, remap=remap, dry_run=True)

    result = CustomToolPhase(ctx).run(MigrationReport())

    assert result.skipped == 1
    assert tgt.import_calls == []
    assert tgt.sync_calls == []
    assert tgt.export_tool_calls == []


def test_missing_adapter_remap_fails_tool_cleanly():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "T")
    # Only seed 3 of 4 adapters → x2text remap missing.
    remap = RemapTable()
    remap.record("adapter", SRC_LLM, TGT_LLM)
    remap.record("adapter", SRC_EMB, TGT_EMB)
    remap.record("adapter", SRC_VEC, TGT_VEC)
    ctx = _ctx(src, tgt, remap=remap)

    result = CustomToolPhase(ctx).run(MigrationReport())

    assert result.failed == 1
    assert tgt.import_calls == []
    # Registry republish should NOT fire when the tool fails.
    assert tgt.export_tool_calls == []
    # No custom_tool remap recorded.
    assert ctx.remap.resolve("custom_tool", "src-tool-x") is None
