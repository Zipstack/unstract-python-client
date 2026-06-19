"""Tests for ``PipelinePhase``.

Coverage:
- happy path: source ETL/TASK pipelines created with workflow FK remapped.
- DEFAULT and APP types are skipped (out of clone scope).
- adopt path on name conflict.
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
from unstract.clone.phases.pipeline import PipelinePhase
from unstract.clone.report import CloneReport

PIPELINE_POST_SCHEMA = frozenset(
    {
        "pipeline_name",
        "workflow",
        "pipeline_type",
        "cron_string",
        "app_id",
        "app_icon",
        "app_url",
        "access_control_bundle_id",
        "shared_users",
        "shared_to_org",
    }
)


class FakeClient:
    def __init__(self, pipelines: list[dict] | None = None):
        self.pipelines: list[dict] = list(pipelines or [])
        self.posts: list[dict] = []
        self.patches: list[tuple[str, dict]] = []
        self.keys_by_pipeline: dict[str, list[dict]] = {}
        self._next = 1

    def get_post_schema(self, entity_path: str) -> frozenset[str]:
        return PIPELINE_POST_SCHEMA

    def list_pipelines(
        self, *, name: str | None = None, pipeline_type: str | None = None
    ):
        result = self.pipelines
        if name is not None:
            result = [p for p in result if p["pipeline_name"] == name]
        if pipeline_type is not None:
            result = [p for p in result if p.get("pipeline_type") == pipeline_type]
        return list(result)

    def get_pipeline(self, pipeline_id: str) -> dict:
        for p in self.pipelines:
            if p["id"] == pipeline_id:
                return dict(p)
        raise KeyError(pipeline_id)

    def create_pipeline(self, payload: dict) -> dict:
        new = dict(payload)
        new["id"] = f"tgt-pipeline-{self._next:04d}"
        self._next += 1
        self.pipelines.append(new)
        self.posts.append(new)
        return new

    def update_pipeline(self, pipeline_id: str, payload: dict) -> dict:
        self.patches.append((pipeline_id, payload))
        for p in self.pipelines:
            if p["id"] == pipeline_id:
                p.update(payload)
                return dict(p)
        raise KeyError(pipeline_id)

    def list_pipeline_keys(self, pipeline_id: str) -> list[dict]:
        return list(self.keys_by_pipeline.get(pipeline_id, []))


def _src_pipeline(
    id_: str,
    name: str,
    workflow_id: str,
    *,
    pipeline_type: str = "ETL",
    cron_string: str | None = None,
) -> dict:
    return {
        "id": id_,
        "pipeline_name": name,
        "workflow": workflow_id,
        "workflow_id": workflow_id,
        "workflow_name": "wf",
        "pipeline_type": pipeline_type,
        "active": True,
        "scheduled": cron_string is not None,
        "cron_string": cron_string,
        "app_id": None,
        "app_icon": None,
        "app_url": None,
        "access_control_bundle_id": None,
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


def test_happy_path_creates_pipeline_with_remapped_workflow():
    src = FakeClient([_src_pipeline("src-pl-1", "Daily Invoices", "wf-src-1")])
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    result = PipelinePhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    posted = tgt.posts[0]
    assert posted["pipeline_name"] == "Daily Invoices"
    assert posted["workflow"] == "wf-tgt-1"
    assert ctx.remap.resolve("pipeline", "src-pl-1") == posted["id"]


def test_create_uses_per_id_get_not_stripped_list_payload():
    # list_pipelines can omit fields the create serializer expects. Phase
    # must re-fetch the full record via get_pipeline before POSTing.
    full = _src_pipeline("src-pl-1", "Daily Invoices", "wf-src-1")
    full["cron_string"] = "0 5 * * *"  # only present on detail serializer.
    stripped = {k: v for k, v in full.items() if k not in ("cron_string",)}

    class StripListFakeClient(FakeClient):
        def list_pipelines(self, *, name=None, pipeline_type=None):
            base = (
                [stripped]
                if (
                    (name is None or stripped["pipeline_name"] == name)
                    and (
                        pipeline_type is None
                        or stripped["pipeline_type"] == pipeline_type
                    )
                )
                else []
            )
            return list(base)

        def get_pipeline(self, pipeline_id):
            assert pipeline_id == full["id"]
            return dict(full)

    src = StripListFakeClient([full])
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    PipelinePhase(ctx).run(CloneReport())

    posted = tgt.posts[0]
    # cron_string only existed on the detail GET — proves we did NOT
    # POST the stripped list-item payload.
    assert posted["cron_string"] == "0 5 * * *"


def test_inactive_source_pipeline_deactivated_on_target():
    # Backend force-activates on create; an inactive source pipeline must be
    # patched back to inactive so its schedule doesn't run on the target.
    pl = _src_pipeline("src-pl-1", "Disabled ETL", "wf-src-1", cron_string="0 5 * * *")
    pl["active"] = False
    src = FakeClient([pl])
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    result = PipelinePhase(ctx).run(CloneReport())

    assert result.created == 1
    posted = tgt.posts[0]
    assert tgt.patches == [(posted["id"], {"active": False})]


def test_active_source_pipeline_not_patched():
    src = FakeClient(
        [_src_pipeline("src-pl-1", "Live ETL", "wf-src-1", cron_string="0 5 * * *")]
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    PipelinePhase(ctx).run(CloneReport())

    assert tgt.patches == []


def test_default_and_app_pipeline_types_are_skipped():
    src = FakeClient(
        [
            _src_pipeline(
                "src-1", "default-legacy", "wf-src-1", pipeline_type="DEFAULT"
            ),
            _src_pipeline("src-2", "streamlit-app", "wf-src-1", pipeline_type="APP"),
            _src_pipeline("src-3", "real-etl", "wf-src-1", pipeline_type="ETL"),
        ]
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    result = PipelinePhase(ctx).run(CloneReport())

    assert result.created == 1
    assert len(tgt.posts) == 1
    assert tgt.posts[0]["pipeline_name"] == "real-etl"


def test_adopts_existing_pipeline_by_name():
    src = FakeClient([_src_pipeline("src-pl-1", "Daily Invoices", "wf-src-1")])
    tgt = FakeClient([{"id": "tgt-existing", "pipeline_name": "Daily Invoices"}])
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    result = PipelinePhase(ctx).run(CloneReport())

    assert result.adopted == 1
    assert result.created == 0
    assert tgt.posts == []
    assert ctx.remap.resolve("pipeline", "src-pl-1") == "tgt-existing"


def test_skipped_when_workflow_remap_missing():
    src = FakeClient([_src_pipeline("src-pl-1", "Orphan", "wf-src-1")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)  # No workflow remap.

    result = PipelinePhase(ctx).run(CloneReport())

    assert result.skipped == 1
    assert result.failed == 0
    assert tgt.posts == []


def test_dry_run_makes_no_writes():
    src = FakeClient([_src_pipeline("src-pl-1", "Daily Invoices", "wf-src-1")])
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap, dry_run=True)

    result = PipelinePhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.skipped == 0
    assert tgt.posts == []
    planned = ctx.remap.resolve("pipeline", "src-pl-1")
    assert planned is not None and ctx.remap.is_planned(planned)


def test_abort_on_name_conflict_raises():
    src = FakeClient([_src_pipeline("src-pl-1", "Daily Invoices", "wf-src-1")])
    tgt = FakeClient([{"id": "tgt-existing", "pipeline_name": "Daily Invoices"}])
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap, on_name_conflict="abort")

    with pytest.raises(NameConflictError):
        PipelinePhase(ctx).run(CloneReport())


def test_extra_source_keys_log_warning_not_failure(caplog):
    src = FakeClient([_src_pipeline("src-pl-1", "Daily Invoices", "wf-src-1")])
    src.keys_by_pipeline["src-pl-1"] = [
        {"id": "k1", "is_active": True},
        {"id": "k2", "is_active": True},
        {"id": "k3", "is_active": False},
    ]
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "wf-src-1", "wf-tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    with caplog.at_level(logging.WARNING, logger="unstract.clone.phases.pipeline"):
        result = PipelinePhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.failed == 0
    assert any("2 active API keys" in r.message for r in caplog.records)
