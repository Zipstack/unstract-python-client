"""Tests for share-state replication (``unstract.clone.sharing``).

Covers payload building (user email mapping, group remap, org flag),
the skip-when-group-phase-excluded axis, the empty-state short circuit,
dry-run behaviour and the SERVER_MANAGED guarantee that ``shared_users``
never rides the create POST.
"""

from __future__ import annotations

import threading

from unstract.clone.context import CloneContext, CloneOptions, RemapTable
from unstract.clone.phases.base import build_post_payload
from unstract.clone.report import PhaseResult
from unstract.clone.sharing import apply_share_state


class FakeClient:
    def __init__(self, users: list[dict] | None = None):
        self.users: list[dict] = list(users or [])
        self.share_posts: list[tuple[str, dict]] = []

    def list_users(self):
        return list(self.users)

    def share_resource(self, share_path, payload):
        self.share_posts.append((share_path, payload))


def _ctx(source, target, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=RemapTable(),
    )


def _apply(ctx, src, result=None, src_detail_fn=None):
    result = result if result is not None else PhaseResult(name="adapter")
    apply_share_state(
        ctx,
        share_path="adapter/tgt-1/share/",
        entity_label="adapter 'demo'",
        src=src,
        result=result,
        lock=threading.Lock(),
        src_detail_fn=src_detail_fn,
    )
    return result


def test_share_payload_maps_users_groups_and_org_flag():
    src_client = FakeClient(
        users=[
            {"id": "7", "email": "alice@x.com"},
            # service account via email-suffix fallback (no flag in row)
            {"id": "8", "email": "svc@platform.internal"},
            # service account via backend flag (email alone wouldn't tell)
            {"id": "9", "email": "bot@x.com", "is_service_account": True},
        ]
    )
    tgt_client = FakeClient(
        users=[
            {"id": "70", "email": "ALICE@x.com"},
            {"id": "71", "email": "bot@x.com"},
        ]
    )
    ctx = _ctx(src_client, tgt_client)
    ctx.remap.record("group", "1", "10")

    result = _apply(
        ctx,
        {
            "created_by": 99,
            "shared_users": [7, 8, 9, 99],  # svc accounts + owner must be dropped
            "shared_groups": [1],
            "shared_to_org": True,
        },
    )

    assert tgt_client.share_posts == [
        (
            "adapter/tgt-1/share/",
            {"shared_to_org": True, "shared_groups": [10], "shared_users": [70]},
        )
    ]
    assert result.failed == 0
    assert result.warnings == []


def test_share_skips_users_missing_on_target_with_warning():
    src_client = FakeClient(users=[{"id": "7", "email": "ghost@x.com"}])
    tgt_client = FakeClient(users=[])
    ctx = _ctx(src_client, tgt_client)

    result = _apply(
        ctx,
        {"shared_users": [7], "shared_groups": [], "shared_to_org": True},
    )

    # shared_to_org still forces a POST; the unmapped user is just dropped.
    assert tgt_client.share_posts == [
        (
            "adapter/tgt-1/share/",
            {"shared_to_org": True, "shared_groups": [], "shared_users": []},
        )
    ]
    assert any("ghost@x.com" in w for w in result.warnings)


def test_share_unmapped_group_id_is_skipped_with_warning():
    ctx = _ctx(FakeClient(), FakeClient())
    ctx.remap.record("group", "1", "10")

    result = _apply(
        ctx,
        {"shared_users": [], "shared_groups": [1, 2], "shared_to_org": False},
    )

    assert ctx.target.share_posts == [
        (
            "adapter/tgt-1/share/",
            {"shared_to_org": False, "shared_groups": [10], "shared_users": []},
        )
    ]
    assert any("group id 2" in w for w in result.warnings)


def test_share_group_axis_omitted_when_group_phase_excluded():
    ctx = _ctx(FakeClient(), FakeClient(), exclude=("group",))

    result = _apply(
        ctx,
        {"shared_users": [], "shared_groups": [1], "shared_to_org": True},
    )

    (path, payload) = ctx.target.share_posts[0]
    assert "shared_groups" not in payload  # axis untouched on target
    assert payload["shared_to_org"] is True
    assert any("group phase excluded" in w for w in result.warnings)


def test_share_empty_state_skips_the_post():
    ctx = _ctx(FakeClient(), FakeClient())

    result = _apply(
        ctx,
        # Owner-only shared_users counts as empty.
        {
            "created_by": 99,
            "shared_users": [99],
            "shared_groups": [],
            "shared_to_org": False,
        },
    )

    assert ctx.target.share_posts == []
    assert result.failed == 0


def test_share_dry_run_never_posts():
    src_client = FakeClient(users=[{"id": "7", "email": "alice@x.com"}])
    tgt_client = FakeClient(users=[{"id": "70", "email": "alice@x.com"}])
    ctx = _ctx(src_client, tgt_client, dry_run=True)

    _apply(
        ctx,
        {"shared_users": [7], "shared_groups": [], "shared_to_org": True},
    )

    assert tgt_client.share_posts == []


def test_share_fetches_source_detail_when_axes_missing_from_list_row():
    ctx = _ctx(FakeClient(), FakeClient())
    detail = {
        "shared_users": [],
        "shared_groups": [],
        "shared_to_org": True,
    }
    calls = []

    def fetch_detail():
        calls.append(1)
        return detail

    _apply(ctx, {"id": "src-1", "name": "demo"}, src_detail_fn=fetch_detail)

    assert calls == [1]
    assert ctx.target.share_posts[0][1]["shared_to_org"] is True


def test_share_users_listing_caches_across_resources():
    src_client = FakeClient(users=[{"id": "7", "email": "alice@x.com"}])
    tgt_client = FakeClient(users=[{"id": "70", "email": "alice@x.com"}])
    src_calls = []
    orig = src_client.list_users
    src_client.list_users = lambda: (src_calls.append(1), orig())[1]
    ctx = _ctx(src_client, tgt_client)

    share = {"shared_users": [7], "shared_groups": [], "shared_to_org": False}
    _apply(ctx, dict(share))
    _apply(ctx, dict(share))

    assert len(src_calls) == 1  # memoised per run, not per resource
    assert len(tgt_client.share_posts) == 2


def test_share_post_failure_lands_in_errors():
    src_client = FakeClient(users=[{"id": "7", "email": "alice@x.com"}])
    tgt_client = FakeClient(users=[{"id": "70", "email": "alice@x.com"}])

    def boom(share_path, payload):
        raise RuntimeError("503")

    tgt_client.share_resource = boom
    ctx = _ctx(src_client, tgt_client)

    result = _apply(
        ctx, {"shared_users": [7], "shared_groups": [], "shared_to_org": False}
    )

    assert result.failed == 1
    assert any("share adapter 'demo'" in e for e in result.errors)


def test_server_managed_still_strips_shared_users_on_create():
    src = {
        "name": "demo",
        "shared_users": [1, 2],
        "shared_to_org": True,
    }
    # Even a (hypothetically) writable shared_users never rides the POST.
    payload = build_post_payload(src, frozenset({"name", "shared_users"}))
    assert payload == {"name": "demo"}
