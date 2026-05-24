"""Tests for ``remap_uuids``."""

from unstract.clone.context import RemapTable
from unstract.clone.walker import remap_uuids

SRC_A = "11111111-1111-1111-1111-111111111111"
TGT_A = "22222222-2222-2222-2222-222222222222"
SRC_B = "33333333-3333-3333-3333-333333333333"
TGT_B = "44444444-4444-4444-4444-444444444444"
UNRELATED = "55555555-5555-5555-5555-555555555555"


def _populated_remap():
    t = RemapTable()
    t.record("adapter", SRC_A, TGT_A)
    t.record("workflow", SRC_B, TGT_B)
    return t


def test_remaps_mapped_uuid_string():
    assert remap_uuids(SRC_A, _populated_remap()) == TGT_A


def test_leaves_unmapped_uuid_untouched():
    assert remap_uuids(UNRELATED, _populated_remap()) == UNRELATED


def test_leaves_non_uuid_string_alone():
    assert remap_uuids("hello-world", _populated_remap()) == "hello-world"


def test_remaps_inside_nested_dict_and_list():
    payload = {
        "id": SRC_A,
        "config": {
            "refs": [SRC_B, "not-a-uuid", UNRELATED],
            "nested": {"adapter_id": SRC_A},
        },
        "count": 42,
    }
    result = remap_uuids(payload, _populated_remap())
    assert result == {
        "id": TGT_A,
        "config": {
            "refs": [TGT_B, "not-a-uuid", UNRELATED],
            "nested": {"adapter_id": TGT_A},
        },
        "count": 42,
    }


def test_handles_non_string_scalars():
    payload = {"a": 1, "b": True, "c": None, "d": 3.14}
    assert remap_uuids(payload, _populated_remap()) == payload
