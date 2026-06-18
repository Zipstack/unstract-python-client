"""Tests for ``GroupPhase``.

Groups merge idempotently by name (no rename, no abort) and optionally
clone members matched by email — missing members surface as warnings.
"""

from __future__ import annotations

from unstract.clone.context import CloneContext, CloneOptions, RemapTable
from unstract.clone.phases.group import GroupPhase
from unstract.clone.report import CloneReport


class FakeClient:
    def __init__(
        self,
        groups: list[dict] | None = None,
        members: dict[int, list[dict]] | None = None,
        users: list[dict] | None = None,
    ):
        self.groups: list[dict] = list(groups or [])
        self.members: dict[int, list[dict]] = dict(members or {})
        self.users: list[dict] = list(users or [])
        self.group_posts: list[dict] = []
        self.member_posts: list[tuple[int, list[int]]] = []
        self._next_id = 100

    def list_groups(self):
        return list(self.groups)

    def create_group(self, payload):
        new = {
            "id": self._next_id,
            "name": payload["name"],
            "description": payload.get("description", ""),
        }
        self._next_id += 1
        self.groups.append(new)
        self.group_posts.append(payload)
        # Backend echoes only name/description (no id) — callers re-list.
        return {"name": payload["name"], "description": payload.get("description", "")}

    def list_group_members(self, group_id):
        return list(self.members.get(group_id, []))

    def add_group_members(self, group_id, user_ids):
        self.member_posts.append((group_id, list(user_ids)))
        return {"added_user_ids": list(user_ids)}

    def list_users(self):
        return list(self.users)


def _ctx(source, target, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=RemapTable(),
    )


def _grp(id_, name, description="d"):
    return {"id": id_, "name": name, "description": description}


def test_creates_missing_groups_and_records_remap():
    src = FakeClient([_grp(1, "devs"), _grp(2, "qa")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)

    result = GroupPhase(ctx).run(CloneReport())

    assert result.created == 2
    assert result.failed == 0
    created_names = {g["name"] for g in tgt.groups}
    assert created_names == {"devs", "qa"}
    # remap is keyed by stringified int pks
    tgt_devs = next(g for g in tgt.groups if g["name"] == "devs")
    assert ctx.remap.resolve("group", "1") == str(tgt_devs["id"])


def test_reuses_like_named_group_even_with_abort_conflict_mode():
    """Idempotent merge: never error or rename on name collision."""
    src = FakeClient([_grp(1, "devs")])
    tgt = FakeClient([_grp(50, "devs")])
    ctx = _ctx(src, tgt, on_name_conflict="abort")

    result = GroupPhase(ctx).run(CloneReport())

    assert result.adopted == 1
    assert result.created == 0
    assert result.failed == 0
    assert tgt.group_posts == []
    assert ctx.remap.resolve("group", "1") == "50"


def test_dry_run_makes_no_posts():
    src = FakeClient([_grp(1, "devs")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt, dry_run=True)

    result = GroupPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert result.skipped == 0
    assert tgt.group_posts == []
    planned = ctx.remap.resolve("group", "1")
    assert planned is not None and ctx.remap.is_planned(planned)


def test_members_not_cloned_by_default():
    src = FakeClient(
        [_grp(1, "devs")],
        members={1: [{"user_id": 7, "email": "a@x.com"}]},
    )
    tgt = FakeClient(users=[{"id": "70", "email": "a@x.com"}])
    ctx = _ctx(src, tgt)

    GroupPhase(ctx).run(CloneReport())

    assert tgt.member_posts == []


def test_member_cloning_matches_by_email_and_skips_missing():
    src = FakeClient(
        [_grp(1, "devs")],
        members={
            1: [
                {"user_id": 7, "email": "alice@x.com"},
                {"user_id": 8, "email": "ghost@x.com"},  # not in target org
                # service acct via email-suffix fallback (no flag in row)
                {"user_id": 9, "email": "svc@platform.internal"},
                # service acct via backend flag (email alone wouldn't tell)
                {"user_id": 10, "email": "bot@x.com", "is_service_account": True},
            ]
        },
    )
    tgt = FakeClient(
        users=[
            {"id": "70", "email": "Alice@X.com"},  # case-insensitive match
            {"id": "71", "email": "bob@x.com"},
            {"id": "72", "email": "bot@x.com"},
        ]
    )
    ctx = _ctx(src, tgt, clone_group_members=True)

    result = GroupPhase(ctx).run(CloneReport())

    tgt_group_id = next(g for g in tgt.groups if g["name"] == "devs")["id"]
    assert tgt.member_posts == [(tgt_group_id, [70])]
    assert any("ghost@x.com" in w for w in result.warnings)
    # service accounts are skipped silently, not warned about
    assert not any("platform.internal" in w for w in result.warnings)
    assert not any("bot@x.com" in w for w in result.warnings)


def test_member_cloning_dry_run_warns_but_never_posts():
    src = FakeClient(
        [_grp(1, "devs")],
        members={1: [{"user_id": 8, "email": "ghost@x.com"}]},
    )
    tgt = FakeClient(users=[{"id": "70", "email": "alice@x.com"}])
    ctx = _ctx(src, tgt, dry_run=True, clone_group_members=True)

    result = GroupPhase(ctx).run(CloneReport())

    assert tgt.member_posts == []
    assert any("ghost@x.com" in w for w in result.warnings)
