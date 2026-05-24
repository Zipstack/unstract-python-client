"""Tests for ``unstract.migration.phases.base`` helpers."""

from __future__ import annotations

from unstract.migration.phases.base import SERVER_MANAGED, build_post_payload


def test_preserves_false_and_zero_values():
    """Booleans set to False and numeric 0 are legitimate field values.

    Earlier ``value not in (None, "")`` worked for None/"" but dropped
    False and 0 too because of Python's ``False == 0 == in (None, "")``
    edge case. Regression guard.
    """
    src = {
        "is_active": False,
        "retry_count": 0,
        "rate_limit": 0.0,
        "name": "demo",
    }
    writable = frozenset({"is_active", "retry_count", "rate_limit", "name"})

    payload = build_post_payload(src, writable)

    assert payload == {
        "is_active": False,
        "retry_count": 0,
        "rate_limit": 0.0,
        "name": "demo",
    }


def test_strips_none_and_empty_string_but_keeps_zero():
    src = {"a": None, "b": "", "c": 0, "d": False, "e": "kept"}
    writable = frozenset({"a", "b", "c", "d", "e"})

    payload = build_post_payload(src, writable)

    assert payload == {"c": 0, "d": False, "e": "kept"}


def test_drops_server_managed_keys_even_if_writable():
    src = {"id": "X", "name": "demo", "organization": "org", "created_by": "u"}
    # All four are nominally writable but SERVER_MANAGED should win.
    writable = frozenset(src.keys())

    payload = build_post_payload(src, writable)

    assert payload == {"name": "demo"}
    for key in SERVER_MANAGED & set(src.keys()):
        assert key not in payload


def test_ignores_writable_keys_missing_from_src():
    src = {"present": 1}
    writable = frozenset({"present", "absent"})

    assert build_post_payload(src, writable) == {"present": 1}
