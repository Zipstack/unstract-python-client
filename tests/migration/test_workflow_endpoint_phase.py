"""Tests for ``WorkflowEndpointPhase``.

WorkflowEndpoints are PATCH-only — backend auto-creates them on workflow
POST. Tests verify that the SDK:
- pairs source/target endpoints by ``endpoint_type``;
- remaps the embedded ``connector_instance`` UUID;
- walker-rewrites UUIDs nested in ``configuration``;
- silently leaves connector_instance_id null when no remap exists.
"""

from __future__ import annotations

from unstract.migration.context import (
    MigrationContext,
    MigrationOptions,
    RemapTable,
)
from unstract.migration.phases.workflow_endpoint import WorkflowEndpointPhase
from unstract.migration.report import MigrationReport


class FakeClient:
    def __init__(self) -> None:
        self.endpoints: dict[str, list[dict]] = {}
        self.patch_calls: list[tuple[str, dict]] = []

    def list_workflow_endpoints(
        self, *, workflow_id: str | None = None
    ) -> list[dict]:
        if workflow_id is None:
            return [ep for eps in self.endpoints.values() for ep in eps]
        return list(self.endpoints.get(workflow_id, []))

    def update_workflow_endpoint(
        self, endpoint_id: str, payload: dict
    ) -> dict:
        self.patch_calls.append((endpoint_id, payload))
        for eps in self.endpoints.values():
            for ep in eps:
                if ep["id"] == endpoint_id:
                    ep.update(payload)
                    return ep
        raise KeyError(endpoint_id)


def _ctx(source, target, *, remap=None, **opt_overrides):
    return MigrationContext(
        source=source,
        target=target,
        options=MigrationOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


SRC_WF = "10000000-0000-0000-0000-000000000001"
TGT_WF = "20000000-0000-0000-0000-000000000001"
SRC_CONN = "30000000-0000-0000-0000-000000000001"
TGT_CONN = "40000000-0000-0000-0000-000000000001"


def _src_endpoint(ep_id, etype, connector_id, configuration):
    return {
        "id": ep_id,
        "workflow": SRC_WF,
        "endpoint_type": etype,
        "connection_type": "FILESYSTEM",
        "configuration": configuration,
        "connector_instance": {"id": connector_id, "connector_name": "src-conn"},
    }


def _tgt_endpoint(ep_id, etype):
    return {
        "id": ep_id,
        "workflow": TGT_WF,
        "endpoint_type": etype,
        "connection_type": "",
        "configuration": {},
        "connector_instance": None,
    }


def _seed_remap() -> RemapTable:
    remap = RemapTable()
    remap.record("workflow", SRC_WF, TGT_WF)
    remap.record("connector", SRC_CONN, TGT_CONN)
    return remap


def test_pairs_endpoints_by_type_and_remaps_connector():
    src = FakeClient()
    src.endpoints[SRC_WF] = [
        _src_endpoint(
            "src-ep-source",
            "SOURCE",
            SRC_CONN,
            {"connector_id": SRC_CONN, "path": "/in"},
        ),
        _src_endpoint(
            "src-ep-dest",
            "DESTINATION",
            SRC_CONN,
            {"connector_id": SRC_CONN, "path": "/out"},
        ),
    ]
    tgt = FakeClient()
    tgt.endpoints[TGT_WF] = [
        _tgt_endpoint("tgt-ep-source", "SOURCE"),
        _tgt_endpoint("tgt-ep-dest", "DESTINATION"),
    ]
    ctx = _ctx(src, tgt, remap=_seed_remap())

    result = WorkflowEndpointPhase(ctx).run(MigrationReport())

    assert result.created == 2
    assert result.failed == 0
    assert len(tgt.patch_calls) == 2

    patches_by_id = dict(tgt.patch_calls)
    src_patch = patches_by_id["tgt-ep-source"]
    assert src_patch["connection_type"] == "FILESYSTEM"
    assert src_patch["connector_instance_id"] == TGT_CONN
    assert src_patch["configuration"]["connector_id"] == TGT_CONN
    assert src_patch["configuration"]["path"] == "/in"

    dst_patch = patches_by_id["tgt-ep-dest"]
    assert dst_patch["configuration"]["path"] == "/out"
    assert dst_patch["connector_instance_id"] == TGT_CONN

    assert ctx.remap.resolve("workflow_endpoint", "src-ep-source") == "tgt-ep-source"
    assert ctx.remap.resolve("workflow_endpoint", "src-ep-dest") == "tgt-ep-dest"


def test_endpoint_without_source_connector_patches_with_null():
    src = FakeClient()
    src.endpoints[SRC_WF] = [
        {
            "id": "src-ep-source",
            "endpoint_type": "SOURCE",
            "connection_type": "API",
            "configuration": {"foo": "bar"},
            "connector_instance": None,
        }
    ]
    tgt = FakeClient()
    tgt.endpoints[TGT_WF] = [_tgt_endpoint("tgt-ep-source", "SOURCE")]
    ctx = _ctx(src, tgt, remap=_seed_remap())

    result = WorkflowEndpointPhase(ctx).run(MigrationReport())

    assert result.created == 1
    assert len(tgt.patch_calls) == 1
    _, payload = tgt.patch_calls[0]
    assert payload["connector_instance_id"] is None
    assert payload["configuration"] == {"foo": "bar"}


def test_unknown_connector_uuid_skips_endpoint_and_flags_error():
    """Source had a connector but its remap is missing — patching with
    connector=None would silently detach the endpoint on target. Skip
    the PATCH and record an operator-visible error entry instead.
    """
    src = FakeClient()
    src.endpoints[SRC_WF] = [
        _src_endpoint(
            "src-ep-source",
            "SOURCE",
            "unmapped-but-uuid-99999999-9999-9999-9999-999999999999"[:36],
            {},
        )
    ]
    tgt = FakeClient()
    tgt.endpoints[TGT_WF] = [_tgt_endpoint("tgt-ep-source", "SOURCE")]
    ctx = _ctx(src, tgt, remap=_seed_remap())

    result = WorkflowEndpointPhase(ctx).run(MigrationReport())

    assert result.created == 0
    assert result.skipped == 1
    assert tgt.patch_calls == []
    assert any("unmapped connector" in e for e in result.errors)


def test_missing_target_endpoint_fails_loudly():
    src = FakeClient()
    src.endpoints[SRC_WF] = [
        _src_endpoint("src-ep-source", "SOURCE", SRC_CONN, {})
    ]
    tgt = FakeClient()
    tgt.endpoints[TGT_WF] = []  # No endpoints — anomaly.
    ctx = _ctx(src, tgt, remap=_seed_remap())

    result = WorkflowEndpointPhase(ctx).run(MigrationReport())

    assert result.failed == 1
    assert tgt.patch_calls == []


def test_dry_run_makes_no_patches():
    src = FakeClient()
    src.endpoints[SRC_WF] = [
        _src_endpoint("src-ep-source", "SOURCE", SRC_CONN, {})
    ]
    tgt = FakeClient()
    tgt.endpoints[TGT_WF] = [_tgt_endpoint("tgt-ep-source", "SOURCE")]
    ctx = _ctx(src, tgt, remap=_seed_remap(), dry_run=True)

    result = WorkflowEndpointPhase(ctx).run(MigrationReport())

    assert result.skipped == 1
    assert tgt.patch_calls == []


def test_no_workflows_in_remap_is_noop():
    src = FakeClient()
    tgt = FakeClient()
    ctx = _ctx(src, tgt, remap=RemapTable())

    result = WorkflowEndpointPhase(ctx).run(MigrationReport())

    assert result.created == 0
    assert tgt.patch_calls == []
