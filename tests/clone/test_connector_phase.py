"""Tests for ``ConnectorPhase``.

Mirrors the adapter phase suite — happy path, idempotency, dry-run,
abort — plus connector-specific behavior: UCS auto-provisioned rows are
skipped without consulting the target.
"""

from __future__ import annotations

import pytest

from unstract.clone.context import (
    CloneContext,
    CloneOptions,
    RemapTable,
)
from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.connector import ConnectorPhase
from unstract.clone.report import CloneReport


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
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=RemapTable(),
    )


def test_happy_path_creates_all_and_records_remap():
    src = FakeClient([_src("src-a", "Prod PG"), _src("src-b", "Stg S3", "s3|xyz")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)
    report = CloneReport()

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
    report = CloneReport()

    result = ConnectorPhase(ctx).run(report)

    assert result.skipped == 1
    assert result.created == 0
    assert tgt.posts == []
    assert ctx.remap.resolve("connector", "src-ucs") is None
    # Recorded for downstream cascade-skip + surfaced in the report.
    assert ctx.skipped_connector_ids == {"src-ucs"}
    assert any("User Storage" in w for w in result.warnings)


def test_oauth_connector_skipped_when_no_target_match():
    """OAuth-backed connectors can't be recreated (credentials can't be cloned)
    and aren't on the target yet — skip, record for cascade, warn.
    """
    oauth = _src("src-gdrive", "Unstract's google drive")
    oauth["connector_metadata"] = {
        "provider": "google-oauth2",
        "uid": "src-user",
        "access_token": "ya29.src-access",
        "refresh_token": "1//src-refresh",
    }
    src = FakeClient([oauth])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)

    result = ConnectorPhase(ctx).run(CloneReport())

    assert result.skipped == 1
    assert result.created == 0
    assert result.failed == 0
    assert tgt.posts == []
    assert ctx.remap.resolve("connector", "src-gdrive") is None
    assert ctx.skipped_connector_ids == {"src-gdrive"}
    assert any("OAuth" in w for w in result.warnings)


def test_oauth_connector_adopted_when_target_exists():
    """The recovery path: operator provisioned a same-name connector on the
    target (where OAuth completes), so a re-run adopts it instead of skipping —
    populating the remap so dependent endpoints wire.
    """
    oauth = _src("src-gdrive", "Unstract's google drive")
    oauth["connector_metadata"] = {
        "provider": "google-oauth2",
        "access_token": "ya29.src-access",
        "refresh_token": "1//src-refresh",
    }
    src = FakeClient([oauth])
    tgt = FakeClient(
        [
            {
                "id": "tgt-gdrive",
                "connector_id": "gdrive|abc",
                "connector_name": "Unstract's google drive",
                "connector_type": "INPUT",
                "connector_metadata": {"refresh_token": "1//tgt-refresh"},
            }
        ]
    )
    ctx = _ctx(src, tgt, on_name_conflict="adopt")

    result = ConnectorPhase(ctx).run(CloneReport())

    assert result.adopted == 1
    assert result.skipped == 0
    assert ctx.skipped_connector_ids == set()
    assert ctx.remap.resolve("connector", "src-gdrive") == "tgt-gdrive"


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
    report = CloneReport()

    result = ConnectorPhase(ctx).run(report)

    assert result.created == 0
    assert result.adopted == 1
    assert tgt.posts == []
    assert ctx.remap.resolve("connector", "src-a") == "preexisting"


def test_dry_run_makes_no_posts():
    src = FakeClient([_src("src-a", "Prod PG")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt, dry_run=True)
    report = CloneReport()

    result = ConnectorPhase(ctx).run(report)

    assert result.created == 1
    assert result.skipped == 0
    assert tgt.posts == []
    planned = ctx.remap.resolve("connector", "src-a")
    assert planned is not None and ctx.remap.is_planned(planned)


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
    report = CloneReport()

    with pytest.raises(NameConflictError):
        ConnectorPhase(ctx).run(report)
