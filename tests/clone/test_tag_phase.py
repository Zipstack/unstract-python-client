"""Tests for ``TagPhase``.

Tag is the simplest entity — no encryption, no list-vs-detail divergence.
Suite covers happy / idempotency / dry-run / abort.
"""

from __future__ import annotations

import pytest

from unstract.clone.context import (
    CloneContext,
    CloneOptions,
    RemapTable,
)
from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.tag import TagPhase
from unstract.clone.report import CloneReport


class FakeClient:
    POST_SCHEMA = frozenset({"name", "description"})

    def __init__(self, tags: list[dict] | None = None):
        self.tags: list[dict] = list(tags or [])
        self.posts: list[dict] = []
        self._next_id = 1

    def get_post_schema(self, entity_path):
        return self.POST_SCHEMA

    def list_tags(self, *, name=None):
        result = self.tags
        if name is not None:
            result = [t for t in result if t["name"] == name]
        return list(result)

    def create_tag(self, payload):
        new = dict(payload)
        new["id"] = f"tgt-{self._next_id:08d}-0000-0000-0000-000000000000"
        self._next_id += 1
        self.tags.append(new)
        self.posts.append(new)
        return new


def _src(id_, name):
    return {"id": id_, "name": name, "description": f"{name} desc"}


def _ctx(source, target, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=RemapTable(),
    )


def test_happy_path_creates_all_and_records_remap():
    src = FakeClient([_src("src-a", "billing"), _src("src-b", "finance")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)
    report = CloneReport()

    result = TagPhase(ctx).run(report)

    assert result.created == 2
    assert result.adopted == 0
    assert len(tgt.posts) == 2
    assert ctx.remap.resolve("tag", "src-a") == tgt.posts[0]["id"]
    assert ctx.remap.resolve("tag", "src-b") == tgt.posts[1]["id"]


def test_idempotency_zero_creates_on_rerun():
    src = FakeClient([_src("src-a", "billing")])
    tgt = FakeClient([{"id": "preexisting", "name": "billing", "description": "x"}])
    ctx = _ctx(src, tgt, on_name_conflict="adopt")
    report = CloneReport()

    result = TagPhase(ctx).run(report)

    assert result.created == 0
    assert result.adopted == 1
    assert tgt.posts == []
    assert ctx.remap.resolve("tag", "src-a") == "preexisting"


def test_dry_run_makes_no_posts():
    src = FakeClient([_src("src-a", "billing")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt, dry_run=True)
    report = CloneReport()

    result = TagPhase(ctx).run(report)

    assert result.created == 1
    assert result.skipped == 0
    assert tgt.posts == []
    planned = ctx.remap.resolve("tag", "src-a")
    assert planned is not None and ctx.remap.is_planned(planned)


def test_abort_on_name_conflict_raises():
    src = FakeClient([_src("src-a", "billing")])
    tgt = FakeClient([{"id": "preexisting", "name": "billing", "description": "x"}])
    ctx = _ctx(src, tgt, on_name_conflict="abort")
    report = CloneReport()

    with pytest.raises(NameConflictError):
        TagPhase(ctx).run(report)
