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
        self.endpoints: list[dict] = []
        self._next_id = 1

    def get_post_schema(self, entity_path: str) -> frozenset[str]:
        return WORKFLOW_POST_SCHEMA

    def list_workflows(self, *, name: str | None = None):
        result = self.workflows
        if name is not None:
            result = [w for w in result if w["workflow_name"] == name]
        return list(result)

    def get_workflow(self, workflow_id: str) -> dict:
        for w in self.workflows:
            if w["id"] == workflow_id:
                return dict(w)
        raise KeyError(workflow_id)

    def list_tool_instances(self, *, workflow_id: str | None = None) -> list[dict]:
        if workflow_id is None:
            return list(self.tool_instances)
        return [ti for ti in self.tool_instances if ti.get("workflow") == workflow_id]

    def list_workflow_endpoints(self, *, workflow_id: str | None = None) -> list[dict]:
        if workflow_id is None:
            return list(self.endpoints)
        return [e for e in self.endpoints if e.get("workflow") == workflow_id]

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
    tgt = FakeClient([{"id": "wf-tgt-pre", "workflow_name": "Invoice ETL"}])
    ctx = _ctx(src, tgt, on_name_conflict="adopt")

    result = WorkflowPhase(ctx).run(CloneReport())

    assert result.adopted == 1
    assert result.created == 0
    assert tgt.posts == []
    assert ctx.remap.resolve("workflow", "wf-src-1") == "wf-tgt-pre"


def test_dry_run_predicts_create_without_writing():
    src = FakeClient([_src("wf-src-1", "Invoice ETL")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt, dry_run=True)

    result = WorkflowPhase(ctx).run(CloneReport())

    # Count matches a real run; nothing posted; planned remap recorded so
    # downstream phases (tool_instance, endpoint, pipeline) can plan.
    assert result.created == 1
    assert result.skipped == 0
    assert tgt.posts == []
    planned = ctx.remap.resolve("workflow", "wf-src-1")
    assert planned is not None and ctx.remap.is_planned(planned)


def test_abort_on_name_conflict_raises():
    src = FakeClient([_src("wf-src-1", "Invoice ETL")])
    tgt = FakeClient([{"id": "wf-tgt-pre", "workflow_name": "Invoice ETL"}])
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
    assert any("its tool was skipped" in w for w in result.warnings)


def test_dual_skip_reports_both_tool_and_connector_reasons():
    """A workflow blocked by both a skipped tool and a skipped connector
    surfaces both reasons in one pass, not one per re-run.
    """
    skipped_reg = "skipped-registry-id"
    skipped_conn = "oauth-conn-id"
    src = FakeClient([_src("wf-both", "Blocked WF")])
    src.tool_instances = [{"workflow": "wf-both", "tool_id": skipped_reg}]
    src.endpoints = [
        {
            "workflow": "wf-both",
            "endpoint_type": "SOURCE",
            "connector_instance": {"id": skipped_conn},
        }
    ]
    tgt = FakeClient()
    ctx = _ctx(src, tgt)
    ctx.skipped_custom_tool_registry_ids.add(skipped_reg)
    ctx.skipped_connector_ids.add(skipped_conn)

    result = WorkflowPhase(ctx).run(CloneReport())

    assert result.skipped == 1
    assert tgt.posts == []
    assert ctx.remap.resolve("workflow", "wf-both") is None
    assert any("its tool was skipped" in w for w in result.warnings)
    assert any("un-clonable connector" in w for w in result.warnings)


def test_cascade_skip_when_endpoint_connector_skipped():
    """Workflow whose endpoint uses an un-clonable (OAuth/redacted) connector
    must not land on target — else its pipelines fail every run. Downstream
    pipeline / api_deployment cascade off the missing workflow remap.
    """
    skipped_conn = "oauth-conn-id"
    src = FakeClient([_src("wf-oauth", "Gdrive ETL"), _src("wf-ok", "OK WF")])
    src.endpoints = [
        {
            "workflow": "wf-oauth",
            "endpoint_type": "SOURCE",
            "connector_instance": {"id": skipped_conn},
        },
        {
            "workflow": "wf-ok",
            "endpoint_type": "SOURCE",
            "connector_instance": {"id": "good-conn"},
        },
    ]
    tgt = FakeClient()
    ctx = _ctx(src, tgt)
    ctx.skipped_connector_ids.add(skipped_conn)

    result = WorkflowPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.skipped == 1
    assert [p["workflow_name"] for p in tgt.posts] == ["OK WF"]
    assert ctx.remap.resolve("workflow", "wf-oauth") is None
    assert ctx.remap.resolve("workflow", "wf-ok") is not None
    assert any("un-clonable" in w for w in result.warnings)
