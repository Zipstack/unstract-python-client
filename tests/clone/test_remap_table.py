"""Tests for ``RemapTable``."""

from unstract.clone.context import RemapTable


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


def test_record_planned_is_deterministic_and_resolvable():
    t1, t2 = RemapTable(), RemapTable()
    a = t1.record_planned("workflow", "src-w")
    b = t2.record_planned("workflow", "src-w")
    # Same (entity, src) → same synthetic id across runs (resume-safe).
    assert a == b
    assert t1.resolve("workflow", "src-w") == a
    assert t1.is_planned(a)
    # Different src or entity → different id.
    assert t1.record_planned("workflow", "src-other") != a
    assert t1.record_planned("pipeline", "src-w") != a


def test_is_planned_false_for_real_ids():
    t = RemapTable()
    t.record("adapter", "src-1", "tgt-1")
    assert not t.is_planned("tgt-1")


def test_snapshot_hide_planned_masks_only_synthetic_ids():
    t = RemapTable()
    t.record("adapter", "src-real", "tgt-real")
    t.record_planned("workflow", "src-planned")

    visible = t.snapshot()
    assert visible["workflow"]["src-planned"] != "(planned)"

    masked = t.snapshot(hide_planned=True)
    assert masked["adapter"]["src-real"] == "tgt-real"
    assert masked["workflow"]["src-planned"] == "(planned)"
