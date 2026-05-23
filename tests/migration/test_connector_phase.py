"""Tests for ``ConnectorPhase``.

Mirrors the adapter phase suite — happy path, idempotency, dry-run,
abort — plus connector-specific behavior: UCS auto-provisioned rows are
skipped without consulting the target.
"""

from __future__ import annotations

import pytest

from unstract.migration.context import (
    MigrationContext,
    MigrationOptions,
    RemapTable,
)
from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.connector import ConnectorPhase
from unstract.migration.report import MigrationReport


class FakeClient:
    POST_SCHEMA = frozenset(
        {
            "connector_id",
            "connector_name",
            "connector_metadata",
            "connector_version",
            "connector_mode",
            "connector_type",
            "shared_to_org",
        }
    )

    def __init__(self, connectors: list[dict] | None = None):
        self.connectors: list[dict] = list(connectors or [])
        self.posts: list[dict] = []
        self._next_id = 1

    def get_post_schema(self, entity_path):
        return self.POST_SCHEMA

    def list_connectors(self, *, name=None, connector_type=None):
        result = self.connectors
        if name is not None:
            result = [c for c in result if c["connector_name"] == name]
        if connector_type is not None:
            result = [c for c in result if c.get("connector_type") == connector_type]
        return list(result)

    def get_connector(self, connector_pk):
        for c in self.connectors:
            if c["id"] == connector_pk:
                return c
        raise KeyError(connector_pk)

    def create_connector(self, payload):
        new = dict(payload)
        new["id"] = f"tgt-{self._next_id:08d}-0000-0000-0000-000000000000"
        self._next_id += 1
        self.connectors.append(new)
        self.posts.append(new)
        return new


def _src(id_, name, catalog_id="postgres|abc", ctype="INPUT"):
    return {
        "id": id_,
        "connector_id": catalog_id,
        "connector_name": name,
        "connector_type": ctype,
        "connector_version": "1.0",
        "connector_metadata": {"host": "db.example.com", "password": "secret"},
        "shared_to_org": False,
    }


def _ctx(source, target, **opt_overrides):
    return MigrationContext(
        source=source,
        target=target,
        options=MigrationOptions(**opt_overrides),
        remap=RemapTable(),
    )


def test_happy_path_creates_all_and_records_remap():
    src = FakeClient([_src("src-a", "Prod PG"), _src("src-b", "Stg S3", "s3|xyz")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)
    report = MigrationReport()

    result = ConnectorPhase(ctx).run(report)

    assert result.created == 2
    assert result.adopted == 0
    assert result.skipped == 0
    assert len(tgt.posts) == 2
    assert ctx.remap.resolve("connector", "src-a") == tgt.posts[0]["id"]
    assert ctx.remap.resolve("connector", "src-b") == tgt.posts[1]["id"]


def test_redacted_metadata_connector_skipped():
    """Source returning empty metadata (redacted by backend) is unmigratable —
    skipped with no POST and no remap entry."""
    redacted = _src("src-ucs", "User Storage")
    redacted["connector_metadata"] = {}  # backend redaction signal
    src = FakeClient([redacted])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)
    report = MigrationReport()

    result = ConnectorPhase(ctx).run(report)

    assert result.skipped == 1
    assert result.created == 0
    assert tgt.posts == []
    assert ctx.remap.resolve("connector", "src-ucs") is None


def test_idempotency_zero_creates_on_rerun():
    src = FakeClient([_src("src-a", "Prod PG")])
    tgt = FakeClient(
        [
            {
                "id": "preexisting",
                "connector_id": "postgres|abc",
                "connector_name": "Prod PG",
                "connector_type": "INPUT",
                "connector_metadata": {},
            }
        ]
    )
    ctx = _ctx(src, tgt, on_name_conflict="adopt")
    report = MigrationReport()

    result = ConnectorPhase(ctx).run(report)

    assert result.created == 0
    assert result.adopted == 1
    assert tgt.posts == []
    assert ctx.remap.resolve("connector", "src-a") == "preexisting"


def test_dry_run_makes_no_posts():
    src = FakeClient([_src("src-a", "Prod PG")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt, dry_run=True)
    report = MigrationReport()

    result = ConnectorPhase(ctx).run(report)

    assert result.skipped == 1
    assert result.created == 0
    assert tgt.posts == []


def test_abort_on_name_conflict_raises():
    src = FakeClient([_src("src-a", "Prod PG")])
    tgt = FakeClient(
        [
            {
                "id": "preexisting",
                "connector_id": "postgres|abc",
                "connector_name": "Prod PG",
                "connector_type": "INPUT",
                "connector_metadata": {},
            }
        ]
    )
    ctx = _ctx(src, tgt, on_name_conflict="abort")
    report = MigrationReport()

    with pytest.raises(NameConflictError):
        ConnectorPhase(ctx).run(report)
