"""Tests for ``APIDeploymentPhase``.

Coverage:
- happy path: source api_deployments created with workflow FK remapped.
- adopt by api_name on existing target deployment.
- skipped when workflow remap missing.
- dry-run is a no-op.
- abort raises ``NameConflictError``.
- extra source keys produce a warning, never a failure.
"""

from __future__ import annotations

import logging

import pytest

from unstract.clone.context import (
    CloneContext,
    CloneOptions,
    RemapTable,
)
from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.api_deployment import APIDeploymentPhase
from unstract.clone.report import CloneReport

API_DEPLOYMENT_POST_SCHEMA = frozenset(
    {
        "display_name",
        "description",
        "workflow",
        "is_active",
        "api_name",
        "shared_users",
        "shared_to_org",
    }
)


class FakeClient:
    def __init__(self, deployments: list[dict] | None = None):
        self.deployments: list[dict] = list(deployments or [])
        self.posts: list[dict] = []
        self.keys_by_deployment: dict[str, list[dict]] = {}
        self._next = 1

    def get_post_schema(self, entity_path: str) -> frozenset[str]:
        return API_DEPLOYMENT_POST_SCHEMA

    def list_api_deployments(self, *, api_name: str | None = None):
        result = self.deployments
        if api_name is not None:
            result = [d for d in result if d["api_name"] == api_name]
        return list(result)

    def get_api_deployment(self, deployment_id: str) -> dict:
        for d in self.deployments:
            if d["id"] == deployment_id:
                return dict(d)
        raise KeyError(deployment_id)

    def create_api_deployment(self, payload: dict) -> dict:
        new = dict(payload)
        new["id"] = f"tgt-dep-{self._next:04d}"
        new["api_key"] = f"key-{self._next:04d}"
        self._next += 1
        self.deployments.append(new)
        self.posts.append(new)
        return new

    def list_api_deployment_keys(self, deployment_id: str) -> list[dict]:
        return list(self.keys_by_deployment.get(deployment_id, []))


def _src_deployment(
    id_: str, api_name: str, workflow_id: str, *, display_name: str | None = None
) -> dict:
    return {
        "id": id_,
        "api_name": api_name,
        "display_name": display_name or api_name,
        "description": f"{api_name} desc",
        "workflow": workflow_id,
        "workflow_id": workflow_id,
        "is_active": True,
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


def test_happy_path_creates_deployment_with_remapped_workflow():
    src = FakeClient([_src_deployment("src-dep-1", "invoices_api", "wf-src-1")])
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    result = APIDeploymentPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    posted = tgt.posts[0]
    assert posted["api_name"] == "invoices_api"
    assert posted["workflow"] == "wf-tgt-1"
    assert ctx.remap.resolve("api_deployment", "src-dep-1") == posted["id"]


def test_adopts_existing_deployment_by_api_name():
    src = FakeClient([_src_deployment("src-dep-1", "invoices_api", "wf-src-1")])
    tgt = FakeClient([{"id": "tgt-existing", "api_name": "invoices_api"}])
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    result = APIDeploymentPhase(ctx).run(CloneReport())

    assert result.adopted == 1
    assert result.created == 0
    assert tgt.posts == []
    assert ctx.remap.resolve("api_deployment", "src-dep-1") == "tgt-existing"


def test_skipped_when_workflow_remap_missing():
    src = FakeClient([_src_deployment("src-dep-1", "orphan", "wf-src-1")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)  # No workflow remap.

    result = APIDeploymentPhase(ctx).run(CloneReport())

    assert result.skipped == 1
    assert tgt.posts == []


def test_dry_run_makes_no_writes():
    src = FakeClient([_src_deployment("src-dep-1", "invoices_api", "wf-src-1")])
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap, dry_run=True)

    result = APIDeploymentPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.skipped == 0
    assert tgt.posts == []
    planned = ctx.remap.resolve("api_deployment", "src-dep-1")
    assert planned is not None and ctx.remap.is_planned(planned)


def test_abort_on_name_conflict_raises():
    src = FakeClient([_src_deployment("src-dep-1", "invoices_api", "wf-src-1")])
    tgt = FakeClient([{"id": "tgt-existing", "api_name": "invoices_api"}])
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap, on_name_conflict="abort")

    with pytest.raises(NameConflictError):
        APIDeploymentPhase(ctx).run(CloneReport())


def test_extra_source_keys_log_warning_not_failure(caplog):
    src = FakeClient([_src_deployment("src-dep-1", "invoices_api", "wf-src-1")])
    src.keys_by_deployment["src-dep-1"] = [
        {"id": "k1", "is_active": True},
        {"id": "k2", "is_active": True},
    ]
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    with caplog.at_level(
        logging.WARNING, logger="unstract.clone.phases.api_deployment"
    ):
        result = APIDeploymentPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    assert any("2 active API keys" in r.message for r in caplog.records)
