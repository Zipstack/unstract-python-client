"""Tests for ``LookupsPhase`` (cloud-only Lookups feature).

Covers: create-fresh definition + draft template patch + adapter remap;
adopt-by-name; a draft-pinned assignment remapped via the ``prompt`` +
``lookup_definition`` tables; share replication (PATCH with mapped users +
``shared_to_org``); published-version replay (publish in ``version_number``
order + ``lookup_version`` remap recorded + draft restored after replay); a
published-pinned assignment resolved via the version remap; dry-run records a
planned remap and writes nothing.

A single scripted fake plays both source and target; the target side records
every write so assertions read off ``posts`` / ``patches``.
"""

from __future__ import annotations

from unstract.clone.context import CloneContext, CloneOptions, RemapTable
from unstract.clone.phases.lookups import LookupsPhase
from unstract.clone.report import CloneReport


class FakeClient:
    POST_SCHEMA = frozenset({"name", "description", "shared_to_org", "shared_users"})

    def __init__(
        self,
        *,
        lookups=None,
        details=None,
        files=None,
        file_blobs=None,
        assignments=None,
        versions=None,
        version_details=None,
        version_file_blobs=None,
        users=None,
    ):
        self._users = list(users or [])
        self.lookups = list(lookups or [])
        # lookup_id -> detail dict (draft template + adapters + draft_version_id)
        self.details = dict(details or {})
        # lookup_id -> list of file rows
        self.files = {k: list(v) for k, v in (files or {}).items()}
        # file_id -> bytes
        self.file_blobs = dict(file_blobs or {})
        self.assignments = list(assignments or [])
        # lookup_id -> list of version rows (draft + published)
        self.versions = {k: list(v) for k, v in (versions or {}).items()}
        # version_id -> version detail dict
        self.version_details = dict(version_details or {})
        # (version_id, file_id) -> bytes
        self.version_file_blobs = dict(version_file_blobs or {})

        self.created_lookups: list[dict] = []
        self.draft_template_patches: list[tuple[str, str]] = []
        self.draft_adapter_patches: list[tuple[str, dict]] = []
        self.uploaded_files: list[tuple[str, str]] = []
        self.created_assignments: list[dict] = []
        self.share_patches: list[tuple[str, dict]] = []
        self.published_versions: list[tuple[str, dict]] = []
        self._next_id = 1

    # ----- schema / definitions -----

    def get_post_schema(self, entity_path):
        return self.POST_SCHEMA

    def list_lookup_definitions(self):
        return list(self.lookups)

    def get_lookup_definition(self, lookup_id):
        return self.details[lookup_id]

    def create_lookup_definition(self, payload):
        new = dict(payload)
        lid = f"tgt-lookup-{self._next_id:04d}"
        self._next_id += 1
        new["lookup_id"] = lid
        self.lookups.append(new)
        self.created_lookups.append(new)
        # Fresh definition auto-spawns an empty draft with a default version id.
        self.details[lid] = {
            "prompt_template": "",
            "draft_version_id": f"tgt-draft-{lid}",
            "adapters": {"llm": None, "x2text": None},
        }
        return new

    def update_lookup_draft_template(self, lookup_id, prompt_template):
        self.draft_template_patches.append((lookup_id, prompt_template))
        self.details[lookup_id]["prompt_template"] = prompt_template
        return self.details[lookup_id]

    def update_lookup_draft_adapters(self, lookup_id, adapters):
        self.draft_adapter_patches.append((lookup_id, adapters))
        self.details[lookup_id]["adapters"].update(adapters)
        return self.details[lookup_id]["adapters"]

    # ----- files -----

    def list_lookup_files(self, lookup_id):
        return list(self.files.get(lookup_id, []))

    def download_lookup_file(self, lookup_id, file_id):
        return self.file_blobs[file_id]

    def upload_lookup_file(self, lookup_id, file_name, data, mime_type):
        self.uploaded_files.append((lookup_id, file_name))
        self.files.setdefault(lookup_id, []).append(
            {"file_id": f"tgt-file-{self._next_id}", "file_name": file_name}
        )
        self._next_id += 1
        return {"file_id": f"tgt-file-{file_name}"}

    # ----- share -----

    def list_users(self):
        return list(self._users)

    def update_lookup_share(self, lookup_id, payload):
        self.share_patches.append((lookup_id, payload))
        return {"lookup_id": lookup_id, **payload}

    # ----- versions -----

    def list_lookup_versions(self, lookup_id):
        return list(self.versions.get(lookup_id, []))

    def get_lookup_version(self, lookup_id, version_id):
        return self.version_details[version_id]

    def download_lookup_version_file(self, lookup_id, version_id, file_id):
        return self.version_file_blobs[(version_id, file_id)]

    def publish_lookup_version(self, lookup_id, payload):
        """Freeze the current draft into a published version + spawn a fresh
        draft (mirrors the backend's ``_publish_draft``).
        """
        detail = self.details[lookup_id]
        max_num = max(
            (v.get("version_number") or 0 for v in self.versions.get(lookup_id, [])),
            default=0,
        )
        vid = f"tgt-ver-{self._next_id:04d}"
        self._next_id += 1
        published = {
            "version_id": vid,
            "is_draft": False,
            "version_name": payload.get("version_name") or f"v{max_num + 1}",
            "version_number": max_num + 1,
        }
        self.versions.setdefault(lookup_id, []).append(published)
        # New empty-ish draft: backend clones the published content into it,
        # but the phase re-stages content per version anyway.
        new_draft_id = f"tgt-draft-{vid}"
        detail["draft_version_id"] = new_draft_id
        self.published_versions.append((lookup_id, published))
        return published

    # ----- assignments -----

    def list_lookup_assignments(self):
        return list(self.assignments)

    def create_lookup_assignment(self, payload):
        new = dict(payload)
        new["assignment_id"] = f"tgt-asg-{self._next_id:04d}"
        self._next_id += 1
        self.created_assignments.append(new)
        self.assignments.append(new)
        return new


def _ctx(source, target, *, remap=None, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


def _src_lookup(lookup_id, name):
    return {"lookup_id": lookup_id, "name": name, "description": f"{name} desc"}


def _src_detail(
    template,
    *,
    llm=None,
    x2text=None,
    shared_to_org=False,
    shared_users=None,
    created_by=None,
):
    return {
        "prompt_template": template,
        "draft_version_id": "src-draft",
        "adapters": {"llm": llm, "x2text": x2text},
        "shared_to_org": shared_to_org,
        "shared_users": list(shared_users or []),
        "created_by": created_by,
    }


def test_create_fresh_with_draft_template_and_adapter_remap():
    src = FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={
            "src-lk": _src_detail("Find {{vendor}}", llm="src-llm", x2text="src-x2t")
        },
    )
    tgt = FakeClient()
    remap = RemapTable()
    # AdapterPhase recorded these earlier in the run.
    remap.record("adapter", "src-llm", "tgt-llm")
    remap.record("adapter", "src-x2t", "tgt-x2t")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = LookupsPhase(ctx).run(report)

    assert result.created == 1
    assert len(tgt.created_lookups) == 1
    new_id = tgt.created_lookups[0]["lookup_id"]
    assert remap.resolve("lookup_definition", "src-lk") == new_id
    # Draft template replicated.
    assert tgt.draft_template_patches == [(new_id, "Find {{vendor}}")]
    # Both adapters remapped to target ids in one PATCH.
    assert tgt.draft_adapter_patches == [
        (new_id, {"llm": "tgt-llm", "x2text": "tgt-x2t"})
    ]


def test_unresolved_adapter_is_skipped_with_warning():
    src = FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={"src-lk": _src_detail("T", llm="src-llm", x2text="src-x2t")},
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("adapter", "src-llm", "tgt-llm")  # x2text intentionally absent
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = LookupsPhase(ctx).run(report)

    new_id = tgt.created_lookups[0]["lookup_id"]
    assert tgt.draft_adapter_patches == [(new_id, {"llm": "tgt-llm"})]
    assert any("x2text adapter not remapped" in w for w in result.warnings)


def test_adopt_by_name_records_remap_no_create():
    src = FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={"src-lk": _src_detail("T")},
    )
    tgt = FakeClient(
        lookups=[{"lookup_id": "tgt-existing", "name": "Vendors"}],
        details={
            "tgt-existing": {
                "prompt_template": "",
                "draft_version_id": "tgt-existing-draft",
                "adapters": {"llm": None, "x2text": None},
            }
        },
    )
    ctx = _ctx(src, tgt, on_name_conflict="adopt")
    report = CloneReport()

    result = LookupsPhase(ctx).run(report)

    assert result.adopted == 1
    assert result.created == 0
    assert tgt.created_lookups == []
    assert ctx.remap.resolve("lookup_definition", "src-lk") == "tgt-existing"


def test_share_replication_patches_mapped_users_and_org_flag():
    src = FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={
            "src-lk": _src_detail(
                "T",
                shared_to_org=True,
                shared_users=[10, 20],
                created_by=99,  # owner — skipped from share payload
            )
        },
        users=[
            {"id": 10, "email": "a@x.com"},
            {"id": 20, "email": "b@x.com"},
            {"id": 99, "email": "owner@x.com"},
        ],
    )
    tgt = FakeClient(
        users=[
            {"id": 110, "email": "a@x.com"},
            {"id": 120, "email": "b@x.com"},
        ],
    )
    ctx = _ctx(src, tgt)
    report = CloneReport()

    LookupsPhase(ctx).run(report)

    new_id = tgt.created_lookups[0]["lookup_id"]
    assert len(tgt.share_patches) == 1
    lid, payload = tgt.share_patches[0]
    assert lid == new_id
    assert payload["shared_to_org"] is True
    assert sorted(payload["shared_users"]) == [110, 120]
    # Lookups have no group sharing — axis omitted entirely.
    assert "shared_groups" not in payload


def test_draft_pinned_assignment_remapped():
    src = FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={"src-lk": _src_detail("T")},
        assignments=[
            {
                "assignment_id": "src-asg",
                "prompt": "src-prompt",
                "version": "src-draft",
                "lookup_definition": "src-lk",
                "is_draft_version": True,
                "variable_mappings": {"vendor": "src-prompt"},
            }
        ],
    )
    tgt = FakeClient()
    remap = RemapTable()
    # custom_tool phase recorded the prompt remap.
    remap.record("prompt", "src-prompt", "tgt-prompt")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    LookupsPhase(ctx).run(report)

    assert len(tgt.created_assignments) == 1
    asg = tgt.created_assignments[0]
    assert asg["prompt"] == "tgt-prompt"
    new_lookup = tgt.created_lookups[0]["lookup_id"]
    assert asg["lookup_definition"] == new_lookup
    assert asg["version"] == f"tgt-draft-{new_lookup}"
    # Mapping value that is a source prompt uuid is remapped too.
    assert asg["variable_mappings"] == {"vendor": "tgt-prompt"}


def _src_published_lookup():
    """A source lookup carrying one published version + a draft, with a
    published-pinned assignment. Shared by the replay/resolution tests.
    """
    return FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={
            "src-lk": _src_detail("Current draft", llm="src-llm")
        },
        versions={
            "src-lk": [
                {
                    "version_id": "src-draft",
                    "is_draft": True,
                    "version_name": "",
                    "version_number": 0,
                },
                {
                    "version_id": "src-v1",
                    "is_draft": False,
                    "version_name": "v1",
                    "version_number": 1,
                },
            ]
        },
        version_details={
            "src-v1": {
                "version_id": "src-v1",
                "is_draft": False,
                "version_name": "v1",
                "version_number": 1,
                "prompt_template": "Frozen v1",
                "adapters": {"llm": "src-llm", "x2text": None},
                "files": [],
            }
        },
        assignments=[
            {
                "assignment_id": "src-asg",
                "prompt": "src-prompt",
                "version": "src-v1",
                "lookup_definition": "src-lk",
                "is_draft_version": False,
                "variable_mappings": {},
            }
        ],
    )


def test_published_version_replayed_publishes_in_order_and_records_remap():
    src = FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={"src-lk": _src_detail("Current draft", llm="src-llm")},
        versions={
            "src-lk": [
                # Out-of-order on purpose: replay must sort by version_number.
                {
                    "version_id": "src-v2",
                    "is_draft": False,
                    "version_name": "v2",
                    "version_number": 2,
                },
                {
                    "version_id": "src-draft",
                    "is_draft": True,
                    "version_name": "",
                    "version_number": 0,
                },
                {
                    "version_id": "src-v1",
                    "is_draft": False,
                    "version_name": "v1",
                    "version_number": 1,
                },
            ]
        },
        version_details={
            "src-v1": {
                "version_id": "src-v1",
                "is_draft": False,
                "version_name": "v1",
                "version_number": 1,
                "prompt_template": "Frozen v1",
                "adapters": {"llm": "src-llm", "x2text": None},
                "files": [],
            },
            "src-v2": {
                "version_id": "src-v2",
                "is_draft": False,
                "version_name": "v2",
                "version_number": 2,
                "prompt_template": "Frozen v2",
                "adapters": {"llm": "src-llm", "x2text": None},
                "files": [],
            },
        },
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("adapter", "src-llm", "tgt-llm")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    LookupsPhase(ctx).run(report)

    # Published in version_number order.
    assert [p[1]["version_name"] for p in tgt.published_versions] == ["v1", "v2"]
    # A version remap recorded for each published version.
    assert remap.resolve("lookup_version", "src-v1") is not None
    assert remap.resolve("lookup_version", "src-v2") is not None


def test_published_pinned_assignment_resolves_via_version_remap():
    src = _src_published_lookup()
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("prompt", "src-prompt", "tgt-prompt")
    remap.record("adapter", "src-llm", "tgt-llm")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    LookupsPhase(ctx).run(report)

    # No longer skipped — the published pin resolves via the version remap.
    assert len(tgt.created_assignments) == 1
    asg = tgt.created_assignments[0]
    assert asg["prompt"] == "tgt-prompt"
    tgt_v1 = remap.resolve("lookup_version", "src-v1")
    assert tgt_v1 is not None
    assert asg["version"] == tgt_v1


def test_draft_restored_after_replay():
    src = _src_published_lookup()
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("prompt", "src-prompt", "tgt-prompt")
    remap.record("adapter", "src-llm", "tgt-llm")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    LookupsPhase(ctx).run(report)

    new_id = tgt.created_lookups[0]["lookup_id"]
    # The LAST template patch on the target draft is the source's CURRENT
    # draft, not the frozen v1 content staged during replay.
    last_template = [
        t for (lid, t) in tgt.draft_template_patches if lid == new_id
    ][-1]
    assert last_template == "Current draft"
    # Source draft version id maps to the target's final draft id.
    src_draft = src.details["src-lk"]["draft_version_id"]
    tgt_draft = tgt.details[new_id]["draft_version_id"]
    assert remap.resolve("lookup_version", src_draft) == tgt_draft


def test_dry_run_records_planned_and_writes_nothing():
    src = FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={"src-lk": _src_detail("T", llm="src-llm")},
        assignments=[
            {
                "assignment_id": "src-asg",
                "prompt": "src-prompt",
                "version": "src-draft",
                "lookup_definition": "src-lk",
                "is_draft_version": True,
                "variable_mappings": {},
            }
        ],
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("prompt", "src-prompt", "tgt-prompt")
    ctx = _ctx(src, tgt, remap=remap, dry_run=True)
    report = CloneReport()

    result = LookupsPhase(ctx).run(report)

    # One planned definition + one planned assignment.
    assert result.created == 2
    assert tgt.created_lookups == []
    assert tgt.created_assignments == []
    assert tgt.draft_template_patches == []
    assert tgt.draft_adapter_patches == []
    planned = ctx.remap.resolve("lookup_definition", "src-lk")
    assert planned is not None and ctx.remap.is_planned(planned)
