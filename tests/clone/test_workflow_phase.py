"""Tests for ``WorkflowPhase``.

Coverage:
- happy path: source workflow created on target, connector UUIDs in
  ``source_settings`` / ``destination_settings`` rewritten via walker.
- idempotency: re-run on existing target adopts and doesn't duplicate.
- dry-run: no POST.
- abort on name conflict.
"""

from __future__ import annotations

import pytest

from unstract.clone.context import (
    CloneContext,
    CloneOptions,
    RemapTable,
)
from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.workflow import WorkflowPhase
from unstract.clone.report import CloneReport


WORKFLOW_POST_SCHEMA = frozenset(
    {
        "workflow_name",
        "description",
        "is_active",
        "deployment_type",
        "source_settings",
        "destination_settings",
        "max_file_execution_count",
        "shared_users",
        "shared_to_org",
    }
)


class FakeClient:
    def __init__(self, workflows: list[dict] | None = None):
        self.workflows: list[dict] = list(workflows or [])
        self.posts: list[dict] = []
        self.tool_instances: list[dict] = []
        self._next_id = 1

    def get_post_schema(self, entity_path: str) -> frozenset[str]:
        return WORKFLOW_POST_SCHEMA

    def list_workflows(self, *, name: str | None = None):
        result = self.workflows
        if name is not None:
            result = [w for w in result if w["workflow_name"] == name]
        return list(result)

    def list_tool_instances(self, *, workflow_id: str | None = None) -> list[dict]:
        if workflow_id is None:
            return list(self.tool_instances)
        return [ti for ti in self.tool_instances if ti.get("workflow") == workflow_id]

    def create_workflow(self, payload: dict) -> dict:
        new = dict(payload)
        new["id"] = f"tgt-{self._next_id:08d}-0000-0000-0000-000000000000"
        self._next_id += 1
        self.workflows.append(new)
        self.posts.append(new)
        return new


def _src(id_, name, *, source_settings=None, destination_settings=None):
    return {
        "id": id_,
        "workflow_name": name,
        "description": f"{name} desc",
        "is_active": True,
        "deployment_type": "DEFAULT",
        "source_settings": source_settings or {},
        "destination_settings": destination_settings or {},
        "max_file_execution_count": None,
        "shared_users": [],
        "shared_to_org": False,
    }


def _ctx(source, target, *, remap=None, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


def test_happy_path_creates_workflow_and_remaps_connector_uuids():
    src_conn = "11111111-1111-1111-1111-111111111111"
    tgt_conn = "a1111111-1111-1111-1111-111111111111"
    src = FakeClient(
        [
            _src(
                "wf-src-1",
                "Invoice ETL",
                source_settings={"connector_id": src_conn, "extras": {"a": 1}},
                destination_settings={"connector_id": src_conn},
            )
        ]
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("connector", src_conn, tgt_conn)
    ctx = _ctx(src, tgt, remap=remap)

    result = WorkflowPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    assert len(tgt.posts) == 1
    posted = tgt.posts[0]
    # Walker rewrote both occurrences of the source connector UUID.
    assert posted["source_settings"]["connector_id"] == tgt_conn
    assert posted["destination_settings"]["connector_id"] == tgt_conn
    # Unrelated nested data passes through untouched.
    assert posted["source_settings"]["extras"] == {"a": 1}

    assert ctx.remap.resolve("workflow", "wf-src-1") == posted["id"]


def test_idempotent_rerun_adopts_existing_workflow():
    src = FakeClient([_src("wf-src-1", "Invoice ETL")])
    tgt = FakeClient(
        [{"id": "wf-tgt-pre", "workflow_name": "Invoice ETL"}]
    )
    ctx = _ctx(src, tgt, on_name_conflict="adopt")

    result = WorkflowPhase(ctx).run(CloneReport())

    assert result.adopted == 1
    assert result.created == 0
    assert tgt.posts == []
    assert ctx.remap.resolve("workflow", "wf-src-1") == "wf-tgt-pre"


def test_dry_run_creates_nothing():
    src = FakeClient([_src("wf-src-1", "Invoice ETL")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt, dry_run=True)

    result = WorkflowPhase(ctx).run(CloneReport())

    assert result.skipped == 1
    assert tgt.posts == []


def test_abort_on_name_conflict_raises():
    src = FakeClient([_src("wf-src-1", "Invoice ETL")])
    tgt = FakeClient(
        [{"id": "wf-tgt-pre", "workflow_name": "Invoice ETL"}]
    )
    ctx = _ctx(src, tgt, on_name_conflict="abort")

    with pytest.raises(NameConflictError):
        WorkflowPhase(ctx).run(CloneReport())


def test_cascade_skip_when_workflow_tool_was_skipped():
    """Workflow whose ToolInstance references a registry id in the
    cascade-skip set must not land on target. Re-runs after the operator
    wires the missing adapter pick it up naturally.
    """
    skipped_reg = "skipped-registry-id"
    src = FakeClient([_src("wf-skipped", "Frictionless WF"), _src("wf-ok", "OK WF")])
    src.tool_instances = [
        {"workflow": "wf-skipped", "tool_id": skipped_reg},
        {"workflow": "wf-ok", "tool_id": "other-registry-id"},
    ]
    tgt = FakeClient()
    ctx = _ctx(src, tgt)
    ctx.skipped_custom_tool_registry_ids.add(skipped_reg)

    result = WorkflowPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.skipped == 1
    assert [p["workflow_name"] for p in tgt.posts] == ["OK WF"]
    assert ctx.remap.resolve("workflow", "wf-skipped") is None
    assert ctx.remap.resolve("workflow", "wf-ok") is not None
