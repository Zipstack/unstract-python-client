"""Tests for ``LookupsPhase`` (cloud-only Lookups feature).

Covers: create-fresh definition + draft template patch + adapter remap;
adopt-by-name; a draft-pinned assignment remapped via the ``prompt`` +
``lookup_definition`` tables; a published-pinned assignment skipped with a
warning; dry-run records a planned remap and writes nothing.

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
    ):
        self.lookups = list(lookups or [])
        # lookup_id -> detail dict (draft template + adapters + draft_version_id)
        self.details = dict(details or {})
        # lookup_id -> list of file rows
        self.files = {k: list(v) for k, v in (files or {}).items()}
        # file_id -> bytes
        self.file_blobs = dict(file_blobs or {})
        self.assignments = list(assignments or [])

        self.created_lookups: list[dict] = []
        self.draft_template_patches: list[tuple[str, str]] = []
        self.draft_adapter_patches: list[tuple[str, dict]] = []
        self.uploaded_files: list[tuple[str, str]] = []
        self.created_assignments: list[dict] = []
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


def _src_detail(template, *, llm=None, x2text=None):
    return {
        "prompt_template": template,
        "draft_version_id": "src-draft",
        "adapters": {"llm": llm, "x2text": x2text},
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


def test_published_pinned_assignment_skipped_with_warning():
    src = FakeClient(
        lookups=[_src_lookup("src-lk", "Vendors")],
        details={"src-lk": _src_detail("T")},
        assignments=[
            {
                "assignment_id": "src-asg",
                "prompt": "src-prompt",
                "version": "src-published",
                "lookup_definition": "src-lk",
                "is_draft_version": False,
                "variable_mappings": {},
            }
        ],
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("prompt", "src-prompt", "tgt-prompt")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = LookupsPhase(ctx).run(report)

    assert tgt.created_assignments == []
    assert result.skipped == 1
    assert any("published-version" in w for w in result.warnings)


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
