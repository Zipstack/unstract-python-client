"""Tests for ``ToolInstancePhase``.

ToolInstance is unique among phases:
- The source list of "things to clone" comes from the workflow remap
  table, not a top-level entity list.
- Create is a two-step dance (POST bare, PATCH metadata) because the
  backend rebuilds metadata from defaults on POST.
"""

from __future__ import annotations

from unstract.clone.context import (
    CloneContext,
    CloneOptions,
    RemapTable,
)
from unstract.clone.phases.tool_instance import ToolInstancePhase
from unstract.clone.report import CloneReport


class FakeClient:
    def __init__(self) -> None:
        # Keyed by workflow_id -> list of tool_instances.
        self.instances: dict[str, list[dict]] = {}
        self.create_calls: list[dict] = []
        self.patch_calls: list[tuple[str, dict]] = []
        self._next = 1

    def _mint(self) -> str:
        s = f"tgt-ti-{self._next:04d}"
        self._next += 1
        return s

    def list_tool_instances(self, *, workflow_id: str | None = None) -> list[dict]:
        if workflow_id is None:
            return [ti for instances in self.instances.values() for ti in instances]
        return list(self.instances.get(workflow_id, []))

    def create_tool_instance(self, payload: dict) -> dict:
        wf = payload["workflow_id"]
        new = {**payload, "id": self._mint(), "metadata": {"defaults": True}}
        self.instances.setdefault(wf, []).append(new)
        self.create_calls.append(new)
        return new

    def update_tool_instance_metadata(
        self, instance_id: str, metadata: dict
    ) -> dict:
        self.patch_calls.append((instance_id, metadata))
        for wf_instances in self.instances.values():
            for ti in wf_instances:
                if ti["id"] == instance_id:
                    ti["metadata"] = metadata
                    return ti
        raise KeyError(instance_id)


def _ctx(source, target, *, remap=None, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


def _src_ti(ti_id: str, wf_id: str, tool_id: str, metadata: dict) -> dict:
    return {
        "id": ti_id,
        "workflow": wf_id,
        "tool_id": tool_id,
        "metadata": metadata,
        "step": 1,
    }


SRC_WF = "10000000-0000-0000-0000-000000000001"
TGT_WF = "20000000-0000-0000-0000-000000000001"
SRC_REG = "30000000-0000-0000-0000-000000000001"
TGT_REG = "40000000-0000-0000-0000-000000000001"


def _seed_remap() -> RemapTable:
    remap = RemapTable()
    remap.record("workflow", SRC_WF, TGT_WF)
    remap.record("prompt_studio_registry", SRC_REG, TGT_REG)
    return remap


def test_happy_path_creates_instance_then_patches_metadata():
    src = FakeClient()
    src.instances[SRC_WF] = [
        _src_ti(
            "src-ti-1", SRC_WF, SRC_REG,
            {
                "llm": "My OpenAI",
                "embedding": "MyEmb",
                # Identity fields that the backend populated server-side
                # at source create time — must NOT cross the org boundary.
                "tenant_id": "src-org",
                "prompt_registry_id": "src-registry-uuid",
                "tool_instance_id": "src-ti-1-pk",
            },
        )
    ]
    tgt = FakeClient()
    ctx = _ctx(src, tgt, remap=_seed_remap())

    result = ToolInstancePhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    assert len(tgt.create_calls) == 1
    posted = tgt.create_calls[0]
    assert posted["workflow_id"] == TGT_WF
    assert posted["tool_id"] == TGT_REG
    # PATCH carries the source settings but never the source-internal
    # identity fields — the target row already has its own.
    assert len(tgt.patch_calls) == 1
    patched_id, patched_metadata = tgt.patch_calls[0]
    assert patched_id == posted["id"]
    assert patched_metadata == {"llm": "My OpenAI", "embedding": "MyEmb"}
    assert ctx.remap.resolve("tool_instance", "src-ti-1") == posted["id"]


def test_skip_when_registry_remap_missing():
    src = FakeClient()
    src.instances[SRC_WF] = [_src_ti("src-ti-1", SRC_WF, "unknown-reg", {})]
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", SRC_WF, TGT_WF)
    # No prompt_studio_registry remap entry → SDK must skip.
    ctx = _ctx(src, tgt, remap=remap)

    result = ToolInstancePhase(ctx).run(CloneReport())

    assert result.skipped == 1
    assert result.created == 0
    assert tgt.create_calls == []


def test_adopt_existing_target_instance_and_repatch_metadata():
    src = FakeClient()
    src_meta = {"llm": "My OpenAI"}
    src.instances[SRC_WF] = [_src_ti("src-ti-1", SRC_WF, SRC_REG, src_meta)]
    tgt = FakeClient()
    tgt.instances[TGT_WF] = [
        {"id": "tgt-pre-ti", "workflow": TGT_WF, "tool_id": TGT_REG, "metadata": {}}
    ]
    ctx = _ctx(src, tgt, remap=_seed_remap())

    result = ToolInstancePhase(ctx).run(CloneReport())

    assert result.adopted == 1
    assert result.created == 0
    assert tgt.create_calls == []
    # PATCH still fires for the adopted instance to align metadata.
    assert tgt.patch_calls == [("tgt-pre-ti", src_meta)]
    assert ctx.remap.resolve("tool_instance", "src-ti-1") == "tgt-pre-ti"


def test_no_op_when_no_workflows_in_remap():
    src = FakeClient()
    tgt = FakeClient()
    ctx = _ctx(src, tgt, remap=RemapTable())

    result = ToolInstancePhase(ctx).run(CloneReport())

    assert result.created == 0
    assert result.skipped == 0
    assert tgt.create_calls == []


def test_dry_run_does_not_create_or_patch():
    src = FakeClient()
    src.instances[SRC_WF] = [_src_ti("src-ti-1", SRC_WF, SRC_REG, {"x": 1})]
    tgt = FakeClient()
    ctx = _ctx(src, tgt, remap=_seed_remap(), dry_run=True)

    result = ToolInstancePhase(ctx).run(CloneReport())

    assert result.skipped == 1
    assert tgt.create_calls == []
    assert tgt.patch_calls == []
