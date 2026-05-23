"""Tests for ``AdapterPhase``.

Uses an in-process fake ``PlatformClient`` to avoid real HTTP. Verifies:
- happy path: source has N adapters, target gets N POSTs, all remapped
- idempotency: re-run with target already populated → zero POSTs, all adopted
- dry-run: zero POSTs, all skipped
- on_name_conflict='abort' raises on existing
"""

from __future__ import annotations

import pytest

from unstract.migration.context import (
    MigrationContext,
    MigrationOptions,
    RemapTable,
)
from unstract.migration.exceptions import NameConflictError
from unstract.migration.phases.adapter import AdapterPhase
from unstract.migration.report import MigrationReport


class FakeClient:
    """Minimal in-memory stand-in for ``PlatformClient``."""

    def __init__(self, adapters: list[dict] | None = None):
        # Stored as a list of dicts; mutated by create_adapter.
        self.adapters: list[dict] = list(adapters or [])
        self.posts: list[dict] = []
        self._next_id = 1

    def list_adapters(self, *, name=None, adapter_type=None):
        result = self.adapters
        if name is not None:
            result = [a for a in result if a["adapter_name"] == name]
        if adapter_type is not None:
            result = [a for a in result if a["adapter_type"] == adapter_type]
        # Mimic AdapterListSerializer — strip adapter_metadata from list output.
        return [{k: v for k, v in a.items() if k != "adapter_metadata"} for a in result]

    def get_adapter(self, adapter_pk):
        for a in self.adapters:
            if a["id"] == adapter_pk:
                return a
        raise KeyError(adapter_pk)

    def create_adapter(self, payload):
        new = dict(payload)
        new["id"] = f"tgt-{self._next_id:08d}-0000-0000-0000-000000000000"
        self._next_id += 1
        self.adapters.append(new)
        self.posts.append(new)
        return new


def _src_adapter(id_, name, atype="LLM"):
    return {
        "id": id_,
        "adapter_id": "openai-llm-v2",
        "adapter_name": name,
        "adapter_type": atype,
        "adapter_metadata": {"api_key": "sk-secret", "model": "gpt-4"},
        "description": f"{name} desc",
    }


def _ctx(source: FakeClient, target: FakeClient, **opt_overrides):
    ctx = MigrationContext(
        source=source,
        target=target,
        options=MigrationOptions(**opt_overrides),
        remap=RemapTable(),
    )
    return ctx


def test_happy_path_creates_all_and_records_remap():
    src = FakeClient(
        [
            _src_adapter("src-a", "OpenAI Prod"),
            _src_adapter("src-b", "Mistral Stg", atype="EMBEDDING"),
        ]
    )
    tgt = FakeClient()
    ctx = _ctx(src, tgt)
    report = MigrationReport()

    result = AdapterPhase(ctx).run(report)

    assert result.created == 2
    assert result.adopted == 0
    assert result.failed == 0
    assert len(tgt.posts) == 2
    assert ctx.remap.resolve("adapter", "src-a") == tgt.posts[0]["id"]
    assert ctx.remap.resolve("adapter", "src-b") == tgt.posts[1]["id"]


def test_idempotency_zero_creates_on_rerun():
    src_adapters = [_src_adapter("src-a", "OpenAI Prod")]
    src = FakeClient(src_adapters)
    # Target pre-populated with the same name+type — simulates a prior run.
    tgt = FakeClient(
        [
            {
                "id": "preexisting",
                "adapter_id": "openai-llm-v2",
                "adapter_name": "OpenAI Prod",
                "adapter_type": "LLM",
                "adapter_metadata": {},
            }
        ]
    )
    ctx = _ctx(src, tgt, on_name_conflict="adopt")
    report = MigrationReport()

    result = AdapterPhase(ctx).run(report)

    assert result.created == 0
    assert result.adopted == 1
    assert tgt.posts == []  # no new POSTs
    assert ctx.remap.resolve("adapter", "src-a") == "preexisting"


def test_dry_run_makes_no_posts():
    src = FakeClient([_src_adapter("src-a", "OpenAI Prod")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt, dry_run=True)
    report = MigrationReport()

    result = AdapterPhase(ctx).run(report)

    assert result.skipped == 1
    assert result.created == 0
    assert tgt.posts == []


def test_abort_on_name_conflict_raises():
    src = FakeClient([_src_adapter("src-a", "OpenAI Prod")])
    tgt = FakeClient(
        [
            {
                "id": "preexisting",
                "adapter_id": "openai-llm-v2",
                "adapter_name": "OpenAI Prod",
                "adapter_type": "LLM",
                "adapter_metadata": {},
            }
        ]
    )
    ctx = _ctx(src, tgt, on_name_conflict="abort")
    report = MigrationReport()

    with pytest.raises(NameConflictError):
        AdapterPhase(ctx).run(report)
