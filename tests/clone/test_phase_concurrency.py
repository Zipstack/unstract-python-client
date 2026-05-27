"""Thread-safety checks for ``Phase.parallel_map``.

Coverage:
- Many-item fan-out produces exact counts + remap entries with no loss.
- Sequential path (``concurrency=1``) skips the thread pool entirely
  while preserving identical behaviour.
- ``CloneError`` raised inside a worker propagates out of ``parallel_map``
  so the orchestrator's abort handling engages.
- A non-``CloneError`` exception inside a worker still propagates.

We use a fake client that holds a lock around its own mutable state and
injects a small sleep per HTTP call to force real interleaving between
workers, then assert the phase's lock-guarded code keeps counters and
the remap table consistent.
"""

from __future__ import annotations

import threading
import time

import pytest

from unstract.clone.context import CloneContext, CloneOptions, RemapTable
from unstract.clone.exceptions import NameConflictError
from unstract.clone.phases.adapter import AdapterPhase
from unstract.clone.phases.tag import TagPhase
from unstract.clone.report import CloneReport


class _ThreadSafeAdapterClient:
    """Adapter FakeClient with a lock around mutable state + per-call sleep
    so workers actually interleave under ThreadPoolExecutor.
    """

    POST_SCHEMA = frozenset(
        {
            "adapter_id",
            "adapter_name",
            "adapter_type",
            "adapter_metadata",
            "description",
        }
    )

    def __init__(self, adapters=None, sleep_seconds: float = 0.005):
        self._adapters: list[dict] = list(adapters or [])
        self.posts: list[dict] = []
        self._next_id = 1
        self._lock = threading.Lock()
        self._sleep = sleep_seconds

    def get_post_schema(self, entity_path):
        return self.POST_SCHEMA

    def list_adapters(self, *, name=None, adapter_type=None):
        time.sleep(self._sleep)
        with self._lock:
            snap = list(self._adapters)
        result = snap
        if name is not None:
            result = [a for a in result if a["adapter_name"] == name]
        if adapter_type is not None:
            result = [a for a in result if a["adapter_type"] == adapter_type]
        return [{k: v for k, v in a.items() if k != "adapter_metadata"} for a in result]

    def get_adapter(self, adapter_pk):
        time.sleep(self._sleep)
        with self._lock:
            for a in self._adapters:
                if a["id"] == adapter_pk:
                    return dict(a)
        raise KeyError(adapter_pk)

    def create_adapter(self, payload):
        time.sleep(self._sleep)
        with self._lock:
            new = dict(payload)
            new["id"] = f"tgt-{self._next_id:08d}-0000-0000-0000-000000000000"
            self._next_id += 1
            self._adapters.append(new)
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


def _ctx(source, target, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=RemapTable(),
    )


def test_parallel_map_preserves_counts_with_many_items():
    items = 50
    src = _ThreadSafeAdapterClient(
        [_src_adapter(f"src-{i:03d}", f"adapter-{i:03d}") for i in range(items)]
    )
    tgt = _ThreadSafeAdapterClient()
    ctx = _ctx(src, tgt, concurrency=8)
    report = CloneReport()

    result = AdapterPhase(ctx).run(report)

    assert result.created == items
    assert result.adopted == 0
    assert result.skipped == 0
    assert result.failed == 0
    assert len(tgt.posts) == items
    remap = ctx.remap.snapshot().get("adapter", {})
    assert len(remap) == items
    # Every source id should be mapped to a fresh target id.
    assert set(remap.keys()) == {f"src-{i:03d}" for i in range(items)}
    assert len(set(remap.values())) == items


def test_concurrency_one_runs_sequentially_with_no_executor(monkeypatch):
    """With concurrency=1 we should never hit ThreadPoolExecutor."""
    sentinel = {"executor_used": False}

    import unstract.clone.phases.base as base_mod

    original = base_mod.ThreadPoolExecutor

    class _Forbidden:
        def __init__(self, *a, **kw):
            sentinel["executor_used"] = True
            raise AssertionError("ThreadPoolExecutor must not be used at concurrency=1")

    monkeypatch.setattr(base_mod, "ThreadPoolExecutor", _Forbidden)
    src = _ThreadSafeAdapterClient(
        [_src_adapter(f"src-{i}", f"a-{i}") for i in range(5)]
    )
    tgt = _ThreadSafeAdapterClient()
    ctx = _ctx(src, tgt, concurrency=1)
    report = CloneReport()

    result = AdapterPhase(ctx).run(report)

    assert result.created == 5
    assert sentinel["executor_used"] is False
    # restore for any other tests in same module (monkeypatch undoes on teardown).
    base_mod.ThreadPoolExecutor = original  # noqa: F841


class _AbortingAdapterClient(_ThreadSafeAdapterClient):
    """As parent, but ``list_adapters`` claims the named adapter already
    exists on target — used to trigger NameConflictError when the phase
    is run with ``on_name_conflict='abort'``."""

    def list_adapters(self, *, name=None, adapter_type=None):
        time.sleep(self._sleep)
        return [
            {
                "id": "tgt-existing-0001",
                "adapter_name": name or "x",
                "adapter_type": adapter_type or "LLM",
            }
        ]


def test_clone_error_in_worker_propagates_under_concurrency():
    src = _ThreadSafeAdapterClient(
        [_src_adapter(f"src-{i}", f"clash-{i}") for i in range(10)]
    )
    tgt = _AbortingAdapterClient()
    ctx = _ctx(src, tgt, concurrency=4, on_name_conflict="abort")
    report = CloneReport()

    with pytest.raises(NameConflictError):
        AdapterPhase(ctx).run(report)


class _UnexpectedAdapterClient(_ThreadSafeAdapterClient):
    """One of the GETs blows up with a non-Clone RuntimeError."""

    def __init__(self, *a, fail_on_name: str, **kw):
        super().__init__(*a, **kw)
        self._fail_on_name = fail_on_name

    def get_adapter(self, adapter_pk):
        snap = super().get_adapter(adapter_pk)
        if snap["adapter_name"] == self._fail_on_name:
            raise RuntimeError("transport boom")
        return snap


def test_non_clone_exception_recorded_as_failed_not_raised():
    """Workers convert non-Clone errors into ``result.failed`` counts;
    they don't escape the phase. (CloneError is the abort signal —
    arbitrary exceptions are per-item failures.)"""
    src = _UnexpectedAdapterClient(
        adapters=[_src_adapter(f"src-{i}", f"adapter-{i}") for i in range(10)],
        fail_on_name="adapter-3",
    )
    tgt = _ThreadSafeAdapterClient()
    ctx = _ctx(src, tgt, concurrency=4)
    report = CloneReport()

    result = AdapterPhase(ctx).run(report)

    assert result.failed == 1
    # The other 9 still created successfully.
    assert result.created == 9
    assert len(tgt.posts) == 9


class _TagClient:
    """Minimal tag fake with thread-safe state + per-call sleep."""

    POST_SCHEMA = frozenset({"name", "description"})

    def __init__(self, tags=None, sleep_seconds: float = 0.005):
        self._tags: list[dict] = list(tags or [])
        self.posts: list[dict] = []
        self._next_id = 1
        self._lock = threading.Lock()
        self._sleep = sleep_seconds

    def get_post_schema(self, entity_path):
        return self.POST_SCHEMA

    def list_tags(self, *, name=None):
        time.sleep(self._sleep)
        with self._lock:
            snap = list(self._tags)
        if name is not None:
            snap = [t for t in snap if t["name"] == name]
        return snap

    def create_tag(self, payload):
        time.sleep(self._sleep)
        with self._lock:
            new = dict(payload)
            new["id"] = f"tag-tgt-{self._next_id:04d}"
            self._next_id += 1
            self._tags.append(new)
            self.posts.append(new)
            return new


def test_tag_phase_parallel_remap_table_consistent():
    """Distinct phase exercising the same parallel_map path — ensures the
    helper isn't accidentally adapter-specific.
    """
    src = _TagClient(
        [{"id": f"tag-src-{i}", "name": f"tag-{i:03d}"} for i in range(30)]
    )
    tgt = _TagClient()
    ctx = _ctx(src, tgt, concurrency=8)
    report = CloneReport()

    result = TagPhase(ctx).run(report)

    assert result.created == 30
    assert result.failed == 0
    remap = ctx.remap.snapshot().get("tag", {})
    assert len(remap) == 30
    # remap value uniqueness — no two source tags mapped to the same target id.
    assert len(set(remap.values())) == 30


def test_parallel_map_empty_input_no_executor(monkeypatch):
    """No items → no thread pool, no work."""
    import unstract.clone.phases.base as base_mod

    class _Forbidden:
        def __init__(self, *a, **kw):
            raise AssertionError("Should not create pool for empty input")

    monkeypatch.setattr(base_mod, "ThreadPoolExecutor", _Forbidden)
    src = _ThreadSafeAdapterClient([])
    tgt = _ThreadSafeAdapterClient()
    ctx = _ctx(src, tgt, concurrency=8)
    report = CloneReport()

    result = AdapterPhase(ctx).run(report)
    assert result.created == 0
    assert result.adopted == 0
