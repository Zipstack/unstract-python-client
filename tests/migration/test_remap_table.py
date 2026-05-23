"""Tests for ``RemapTable``."""

from unstract.migration.context import RemapTable


def test_record_and_resolve_per_entity():
    t = RemapTable()
    t.record("adapter", "src-1", "tgt-1")
    t.record("adapter", "src-2", "tgt-2")
    t.record("connector", "src-1", "tgt-99")

    assert t.resolve("adapter", "src-1") == "tgt-1"
    assert t.resolve("adapter", "src-2") == "tgt-2"
    assert t.resolve("connector", "src-1") == "tgt-99"


def test_resolve_missing_returns_none():
    t = RemapTable()
    assert t.resolve("adapter", "nope") is None
    assert t.resolve_any("nope") is None


def test_resolve_any_searches_across_entities():
    t = RemapTable()
    t.record("adapter", "src-a", "tgt-a")
    t.record("workflow", "src-w", "tgt-w")
    assert t.resolve_any("src-a") == "tgt-a"
    assert t.resolve_any("src-w") == "tgt-w"


def test_snapshot_is_independent_copy():
    t = RemapTable()
    t.record("adapter", "src-1", "tgt-1")
    snap = t.snapshot()
    t.record("adapter", "src-2", "tgt-2")
    assert "src-2" not in snap["adapter"]
