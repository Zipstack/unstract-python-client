"""Tests for ``CustomToolPhase`` — project-transfer + sync-prompts based.

Coverage:
- fresh path: ``export_project`` on source → ``import_project`` on
  target with adapter ids resolved by looking up each source-profile
  adapter NAME against the target via ``list_adapters(name=...)``.
- adopt path: existing target tool with matching name →
  ``sync_prompts`` overwrites prompts; no profile/adapter writes.
- registry remap recorded after ``export_custom_tool``.
- dry-run: no writes on either side.
- abort on name conflict when option is set.
- incomplete source tools (missing target adapter / no profile) mirror
  unconfigured instead of failing; frictionless adapters still skip.
- registry republish 500 warns, doesn't fail the tool.
"""

from __future__ import annotations

import pytest

from unstract.clone.context import (
    CloneContext,
    CloneOptions,
    RemapTable,
)
from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.custom_tool import CustomToolPhase
from unstract.clone.report import CloneReport

ADAPTER_NAMES = {
    "llm": "gpt4",
    "embedding_model": "ada-embed",
    "vector_store": "pgvector",
    "x2text": "llmw",
}
TGT_ADAPTER_IDS = {
    "gpt4": "a1111111-1111-1111-1111-111111111111",
    "ada-embed": "a2222222-2222-2222-2222-222222222222",
    "pgvector": "a3333333-3333-3333-3333-333333333333",
    "llmw": "a4444444-4444-4444-4444-444444444444",
}
SRC_REG = "55555555-5555-5555-5555-555555555555"


class FakeClient:
    """In-memory stand-in for ``PlatformClient`` covering project-transfer."""

    def __init__(self) -> None:
        self.tools: dict[str, dict] = {}
        self.profiles_by_tool: dict[str, list[dict]] = {}
        self.export_blobs: dict[str, dict] = {}
        self.registries_by_tool: dict[str, dict] = {}
        self.adapters_by_name: dict[str, dict] = {}
        self.prompts_by_tool: dict[str, list[dict]] = {}
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

    def list_adapters(
        self,
        *,
        name: str | None = None,
        adapter_type: str | None = None,
    ) -> list[dict]:
        if name is None:
            return list(self.adapters_by_name.values())
        ad = self.adapters_by_name.get(name)
        return [ad] if ad else []

    def list_registries(self, *, custom_tool: str | None = None) -> list[dict]:
        if custom_tool is None:
            return list(self.registries_by_tool.values())
        reg = self.registries_by_tool.get(custom_tool)
        return [reg] if reg else []

    def list_prompts(self, tool_id: str) -> list[dict]:
        return list(self.prompts_by_tool.get(tool_id, []))

    # --- writes ---
    _REQUIRED_ADAPTER_FIELDS = (
        "llm_adapter_id",
        "vector_db_adapter_id",
        "embedding_adapter_id",
        "x2text_adapter_id",
    )

    def import_project(
        self, export_data: dict, adapter_ids: dict | None = None
    ) -> dict:
        self.import_calls.append((export_data, adapter_ids))
        tool_id = self._mint("tool")
        tool_name = export_data["tool_metadata"]["tool_name"]
        self.tools[tool_id] = {"tool_name": tool_name}
        # Backend flags needs_adapter_config unless all four are wired.
        fully_wired = bool(adapter_ids) and all(
            adapter_ids.get(k) for k in self._REQUIRED_ADAPTER_FIELDS
        )
        return {
            "tool_id": tool_id,
            "message": f"Project imported successfully as '{tool_name}'",
            "needs_adapter_config": not fully_wired,
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


def _ctx(source, target, *, remap=None, **opt_overrides) -> CloneContext:
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


def _seed_target_adapters(target: FakeClient) -> None:
    """ProfileManagerSerializer surfaces adapter NAMES — target must
    expose name → id lookups for the phase to resolve them.
    """
    for name, adapter_id in TGT_ADAPTER_IDS.items():
        target.adapters_by_name[name] = {"id": adapter_id, "adapter_name": name}


def _seed_source_adapters(source: FakeClient) -> None:
    """Source-visible adapter set; phase uses it for frictionless detection."""
    for name in ADAPTER_NAMES.values():
        source.adapters_by_name[name] = {"id": f"src-{name}", "adapter_name": name}


def _src_default_profile(*, nested: bool = False) -> dict:
    """Mirror the live ProfileManager serializer: adapter FKs render as
    flat NAME strings. ``nested=True`` covers the alternate dict shape
    in case backend behavior changes.
    """
    if nested:
        return {
            "profile_id": "src-profile-1",
            "profile_name": "Default",
            "is_default": True,
            "llm": {"adapter_name": ADAPTER_NAMES["llm"]},
            "embedding_model": {"adapter_name": ADAPTER_NAMES["embedding_model"]},
            "vector_store": {"adapter_name": ADAPTER_NAMES["vector_store"]},
            "x2text": {"adapter_name": ADAPTER_NAMES["x2text"]},
        }
    return {
        "profile_id": "src-profile-1",
        "profile_name": "Default",
        "is_default": True,
        "llm": ADAPTER_NAMES["llm"],
        "embedding_model": ADAPTER_NAMES["embedding_model"],
        "vector_store": ADAPTER_NAMES["vector_store"],
        "x2text": ADAPTER_NAMES["x2text"],
    }


def _src_export_blob(tool_name: str) -> dict:
    return {
        "tool_metadata": {
            "tool_name": tool_name,
            "description": "x",
            "author": "a",
            "icon": None,
        },
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
            {
                "prompt_key": "field_a",
                "prompt": "What is field_a?",
                "sequence_number": 1,
            }
        ],
        "export_metadata": {"exported_at": "2026-05-24T00:00:00Z"},
    }


def _preload_source_tool(
    client: FakeClient, tool_id: str, tool_name: str, *, nested_profile: bool = False
) -> None:
    client.tools[tool_id] = {"tool_name": tool_name}
    client.profiles_by_tool[tool_id] = [_src_default_profile(nested=nested_profile)]
    client.export_blobs[tool_id] = _src_export_blob(tool_name)
    client.registries_by_tool[tool_id] = {
        "prompt_registry_id": SRC_REG,
        "custom_tool": tool_id,
    }


def test_fresh_imports_with_name_resolved_adapter_ids_and_records_registry():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Invoice Extractor")
    _seed_source_adapters(src)
    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt)

    result = CustomToolPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    # Exactly one import_project call with the right export blob + name-resolved
    # adapter ids.
    assert len(tgt.import_calls) == 1
    blob, adapter_ids = tgt.import_calls[0]
    assert blob["tool_metadata"]["tool_name"] == "Invoice Extractor"
    assert adapter_ids == {
        "llm_adapter_id": TGT_ADAPTER_IDS["gpt4"],
        "vector_db_adapter_id": TGT_ADAPTER_IDS["pgvector"],
        "embedding_adapter_id": TGT_ADAPTER_IDS["ada-embed"],
        "x2text_adapter_id": TGT_ADAPTER_IDS["llmw"],
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


def test_nested_adapter_dict_also_resolves():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "T", nested_profile=True)
    _seed_source_adapters(src)
    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt)

    CustomToolPhase(ctx).run(CloneReport())

    _, adapter_ids = tgt.import_calls[0]
    assert adapter_ids["llm_adapter_id"] == TGT_ADAPTER_IDS["gpt4"]


def test_adopt_path_calls_sync_prompts_and_skips_import():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Invoice Extractor")
    # Target already has the tool with the same name.
    tgt.tools["tgt-existing"] = {"tool_name": "Invoice Extractor"}

    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt)

    result = CustomToolPhase(ctx).run(CloneReport())

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

    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt, on_name_conflict="abort")

    with pytest.raises(NameConflictError):
        CustomToolPhase(ctx).run(CloneReport())

    assert tgt.sync_calls == []
    assert tgt.import_calls == []


def test_dry_run_makes_no_writes():
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "T")
    _seed_source_adapters(src)
    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt, dry_run=True)

    result = CustomToolPhase(ctx).run(CloneReport())

    # Predicts the import (count matches a real run) without writing, and
    # records a planned custom_tool remap so the files phase can plan.
    assert result.created == 1
    assert result.skipped == 0
    assert tgt.import_calls == []
    assert tgt.sync_calls == []
    assert tgt.export_tool_calls == []
    planned = ctx.remap.resolve("custom_tool", "src-tool-x")
    assert planned is not None and ctx.remap.is_planned(planned)


def test_dry_run_on_adopt_path_does_not_republish_registry():
    # Adopt path used to return tgt_tool_id even on dry-run, falling
    # through to export_custom_tool (a real POST to the target).
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Invoice Extractor")
    tgt.tools["tgt-existing"] = {"tool_name": "Invoice Extractor"}
    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt, dry_run=True)

    result = CustomToolPhase(ctx).run(CloneReport())

    # Adopt path counts as adopted (matching a real run).
    assert result.adopted == 1
    assert result.skipped == 0
    assert tgt.sync_calls == []
    assert tgt.import_calls == []
    # Critical regression: registry republish must NOT fire on dry-run.
    assert tgt.export_tool_calls == []
    # Remap still recorded so downstream dry-run output stays coherent.
    assert ctx.remap.resolve("custom_tool", "src-tool-x") == "tgt-existing"


def test_frictionless_adapter_dependence_skips_tool_and_records_for_cascade():
    """Source profile references an adapter NAME the source's
    service-account view can't list (frictionless). Tool is skipped
    cleanly and source registry id is recorded for WorkflowPhase to
    cascade-skip dependent workflows.
    """
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Frictionless-Bound Tool")
    # Source-visible adapters cover 3 of 4 — llm "gpt4" hidden.
    for name in ("ada-embed", "pgvector", "llmw"):
        src.adapters_by_name[name] = {"id": f"src-{name}", "adapter_name": name}
    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt)

    result = CustomToolPhase(ctx).run(CloneReport())

    assert result.skipped == 1
    assert result.failed == 0
    assert tgt.import_calls == []
    assert tgt.export_tool_calls == []
    assert ctx.remap.resolve("custom_tool", "src-tool-x") is None
    assert SRC_REG in ctx.skipped_custom_tool_registry_ids


def test_never_exported_source_tool_skips_registry_republish():
    """A source tool with no registry entry (e.g. an empty project — the
    backend blocks exporting those) clones cleanly without republishing.
    """
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Empty Project")
    del src.registries_by_tool["src-tool-x"]
    src.export_blobs["src-tool-x"]["prompts"] = []
    _seed_source_adapters(src)
    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt)

    result = CustomToolPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    # No registry on source → republish must not fire (it would hit the
    # backend's empty-project export guard).
    assert tgt.export_tool_calls == []
    # Tool remap still recorded; registry remap absent.
    assert ctx.remap.resolve("custom_tool", "src-tool-x") is not None
    assert ctx.remap.resolve("prompt_studio_registry", SRC_REG) is None


def test_missing_target_adapter_imports_unconfigured():
    """A source adapter with no target match isn't fatal: the tool is
    mirrored with a partial adapter set, flagged needs_adapter_config, and
    a warning tells the operator to wire it + re-run.
    """
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "T")
    _seed_source_adapters(src)
    # Only seed 3 of 4 adapters → x2text lookup misses on target.
    for name in ("gpt4", "ada-embed", "pgvector"):
        tgt.adapters_by_name[name] = {"id": TGT_ADAPTER_IDS[name], "adapter_name": name}
    ctx = _ctx(src, tgt)

    result = CustomToolPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    # Import fired with the 3 resolvable adapters; x2text omitted.
    assert len(tgt.import_calls) == 1
    _, adapter_ids = tgt.import_calls[0]
    assert adapter_ids == {
        "llm_adapter_id": TGT_ADAPTER_IDS["gpt4"],
        "vector_db_adapter_id": TGT_ADAPTER_IDS["pgvector"],
        "embedding_adapter_id": TGT_ADAPTER_IDS["ada-embed"],
    }
    assert "x2text_adapter_id" not in adapter_ids
    assert any("full adapter config" in w for w in result.warnings)
    # Source had a registry → still republished + remap recorded.
    assert tgt.export_tool_calls == [ctx.remap.resolve("custom_tool", "src-tool-x")]


def test_no_default_profile_imports_unconfigured():
    """A source tool with no profiles can't derive adapter ids; mirror it
    anyway (backend auto-creates an unconfigured default profile) rather
    than failing the whole clone.
    """
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Profileless")
    src.profiles_by_tool["src-tool-x"] = []  # no default profile on source
    _seed_source_adapters(src)
    _seed_target_adapters(tgt)
    ctx = _ctx(src, tgt)

    result = CustomToolPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    # Imported with an empty adapter set (no profile to derive from).
    assert tgt.import_calls == [(src.export_blobs["src-tool-x"], {})]
    assert any("full adapter config" in w for w in result.warnings)
    assert ctx.remap.resolve("custom_tool", "src-tool-x") is not None


def test_republish_failure_warns_not_fails():
    """A registry republish 500 (e.g. stale/empty source registry) must not
    fail the whole tool — the tool itself cloned; downstream tool_instances
    just cascade-skip.
    """
    src = FakeClient()
    tgt = FakeClient()
    _preload_source_tool(src, "src-tool-x", "Stale Registry Tool")
    _seed_source_adapters(src)
    _seed_target_adapters(tgt)

    def boom(tool_id, *, force=True):
        raise RuntimeError("500 export failed: no run prompts")

    tgt.export_custom_tool = boom
    ctx = _ctx(src, tgt)

    result = CustomToolPhase(ctx).run(CloneReport())

    # Tool cloned; republish failure is a warning, not a failure.
    assert result.created == 1
    assert result.failed == 0
    assert any("republish" in w for w in result.warnings)
    # Tool remap recorded; registry remap absent (republish never landed).
    assert ctx.remap.resolve("custom_tool", "src-tool-x") is not None
    assert ctx.remap.resolve("prompt_studio_registry", SRC_REG) is None


def test_remap_prompts_maps_src_to_tgt_by_prompt_key():
    import threading

    src = FakeClient()
    tgt = FakeClient()
    src.prompts_by_tool["src-tool"] = [
        {"prompt_id": "sp1", "prompt_key": "k1"},
        {"prompt_id": "sp2", "prompt_key": "k2"},
    ]
    tgt.prompts_by_tool["tgt-tool"] = [
        {"prompt_id": "tp1", "prompt_key": "k1"},
        {"prompt_id": "tp2", "prompt_key": "k2"},
    ]
    ctx = _ctx(src, tgt)

    CustomToolPhase(ctx)._remap_prompts(
        "src-tool", "tgt-tool", "T", threading.Lock()
    )

    assert ctx.remap.resolve("prompt", "sp1") == "tp1"
    assert ctx.remap.resolve("prompt", "sp2") == "tp2"


def test_record_planned_prompts_records_synthetic_remaps():
    import threading

    src = FakeClient()
    src.prompts_by_tool["src-tool"] = [
        {"prompt_id": "sp1", "prompt_key": "k1"},
    ]
    ctx = _ctx(src, FakeClient(), dry_run=True)

    CustomToolPhase(ctx)._record_planned_prompts("src-tool", threading.Lock())

    planned = ctx.remap.resolve("prompt", "sp1")
    assert planned is not None and ctx.remap.is_planned(planned)
