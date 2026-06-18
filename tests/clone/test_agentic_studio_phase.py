"""Tests for ``AgenticStudioPhase`` (cloud-only Agentic Prompt Studio).

Covers: create-fresh project + four-adapter remap; an unresolved adapter
omitted with a warning; adopt-by-name; prompt-version parent-before-child
ordering with the self-FK remapped; schema clone bound to the target project;
registry republish via the export action recording a remap; dry-run records a
planned project (+ planned children) and writes nothing.

A single scripted fake plays both source and target; the target side records
every write so assertions read off ``created_*`` lists.
"""

from __future__ import annotations

from unstract.clone.context import CloneContext, CloneOptions, RemapTable
from unstract.clone.phases.agentic_studio import AgenticStudioPhase
from unstract.clone.report import CloneReport


class FakeClient:
    POST_SCHEMA = frozenset(
        {
            "name",
            "description",
            "canary_fields",
            "llm_connector_id",
            "agent_llm_connector_id",
            "lightweight_llm_connector_id",
            "text_extractor_connector_id",
        }
    )

    def __init__(
        self,
        *,
        projects=None,
        versions=None,
        schemas=None,
        settings=None,
        registries=None,
    ):
        self.projects = list(projects or [])
        # project_id -> list of prompt-version rows
        self.versions = {k: list(v) for k, v in (versions or {}).items()}
        # project_id -> list of schema rows
        self.schemas = {k: list(v) for k, v in (schemas or {}).items()}
        self.settings = list(settings or [])
        # project_id -> list of registry rows
        self.registries = {k: list(v) for k, v in (registries or {}).items()}

        self.created_projects: list[dict] = []
        self.created_versions: list[dict] = []
        self.created_schemas: list[dict] = []
        self.created_settings: list[dict] = []
        self.updated_settings: list[tuple[str, dict]] = []
        self.exported_projects: list[str] = []
        self._next_id = 1

    def _mint(self, prefix: str) -> str:
        out = f"{prefix}-{self._next_id:04d}"
        self._next_id += 1
        return out

    def get_post_schema(self, entity_path):
        return self.POST_SCHEMA

    # ----- projects -----

    def list_agentic_projects(self):
        return list(self.projects)

    def create_agentic_project(self, payload):
        new = dict(payload)
        new["id"] = self._mint("tgt-proj")
        self.projects.append(new)
        self.created_projects.append(new)
        return new

    # ----- prompt versions -----

    def list_agentic_prompt_versions(self, *, project_id=None):
        return list(self.versions.get(project_id, []))

    def create_agentic_prompt_version(self, payload):
        new = dict(payload)
        new["id"] = self._mint("tgt-ver")
        self.created_versions.append(new)
        self.versions.setdefault(new["project"], []).append(new)
        return new

    # ----- schemas -----

    def list_agentic_schemas(self, *, project_id=None):
        return list(self.schemas.get(project_id, []))

    def create_agentic_schema(self, payload):
        new = dict(payload)
        new["id"] = self._mint("tgt-schema")
        self.created_schemas.append(new)
        self.schemas.setdefault(new["project"], []).append(new)
        return new

    # ----- settings -----

    def list_agentic_settings(self):
        return list(self.settings)

    def create_agentic_setting(self, payload):
        new = dict(payload)
        new["id"] = self._mint("tgt-setting")
        self.created_settings.append(new)
        self.settings.append(new)
        return new

    def update_agentic_setting(self, setting_id, payload):
        self.updated_settings.append((setting_id, payload))
        return {"id": setting_id, **payload}

    # ----- registry -----

    def export_agentic_project(self, project_id, *, force=True):
        self.exported_projects.append(project_id)
        # Export auto-creates the registry row on the target.
        self.registries.setdefault(project_id, []).append(
            {"registry_id": f"tgt-reg-{project_id}"}
        )
        return {"message": "ok"}

    def list_agentic_registries(self, *, agentic_project=None):
        return list(self.registries.get(agentic_project, []))


def _ctx(source, target, *, remap=None, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


def _src_project(pid, name, **adapters):
    base = {"id": pid, "name": name, "description": f"{name} desc"}
    base.update(adapters)
    return base


def test_create_fresh_with_four_adapter_remap():
    src = FakeClient(
        projects=[
            _src_project(
                "src-p",
                "Receipts",
                llm_connector_id="src-llm",
                agent_llm_connector_id="src-agent",
                lightweight_llm_connector_id="src-light",
                text_extractor_connector_id="src-x2t",
            )
        ]
    )
    tgt = FakeClient()
    remap = RemapTable()
    for s, t in [
        ("src-llm", "tgt-llm"),
        ("src-agent", "tgt-agent"),
        ("src-light", "tgt-light"),
        ("src-x2t", "tgt-x2t"),
    ]:
        remap.record("adapter", s, t)
    ctx = _ctx(src, tgt, remap=remap)

    result = AgenticStudioPhase(ctx).run(CloneReport())

    assert result.created == 1
    assert len(tgt.created_projects) == 1
    payload = tgt.created_projects[0]
    assert payload["llm_connector_id"] == "tgt-llm"
    assert payload["agent_llm_connector_id"] == "tgt-agent"
    assert payload["lightweight_llm_connector_id"] == "tgt-light"
    assert payload["text_extractor_connector_id"] == "tgt-x2t"
    new_id = payload["id"]
    assert remap.resolve("agentic_project", "src-p") == new_id


def test_unresolved_adapter_omitted_with_warning():
    src = FakeClient(
        projects=[
            _src_project(
                "src-p",
                "Receipts",
                llm_connector_id="src-llm",
                agent_llm_connector_id="src-agent",
            )
        ]
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("adapter", "src-llm", "tgt-llm")  # agent intentionally absent
    ctx = _ctx(src, tgt, remap=remap)

    result = AgenticStudioPhase(ctx).run(CloneReport())

    payload = tgt.created_projects[0]
    assert payload["llm_connector_id"] == "tgt-llm"
    assert "agent_llm_connector_id" not in payload
    assert any(
        "agent_llm_connector_id adapter not remapped" in w for w in result.warnings
    )


def test_adopt_by_name_records_remap_no_create():
    src = FakeClient(projects=[_src_project("src-p", "Receipts")])
    tgt = FakeClient(projects=[{"id": "tgt-existing", "name": "Receipts"}])
    ctx = _ctx(src, tgt, on_name_conflict="adopt")

    result = AgenticStudioPhase(ctx).run(CloneReport())

    assert result.adopted == 1
    assert result.created == 0
    assert tgt.created_projects == []
    assert ctx.remap.resolve("agentic_project", "src-p") == "tgt-existing"


def test_prompt_version_parent_before_child_remap():
    # Child (v2) listed first to prove ordering sorts roots ahead of children.
    src = FakeClient(
        projects=[_src_project("src-p", "Receipts")],
        versions={
            "src-p": [
                {
                    "id": "src-v2",
                    "project": "src-p",
                    "version": 2,
                    "prompt_text": "v2",
                    "parent_version": "src-v1",
                },
                {
                    "id": "src-v1",
                    "project": "src-p",
                    "version": 1,
                    "prompt_text": "v1",
                    "parent_version": None,
                },
            ]
        },
    )
    tgt = FakeClient()
    ctx = _ctx(src, tgt)

    AgenticStudioPhase(ctx).run(CloneReport())

    assert len(tgt.created_versions) == 2
    # Root v1 cloned first, no parent.
    first, second = tgt.created_versions
    assert first["version"] == 1
    assert "parent_version" not in first
    # Child v2 second, parent remapped to the freshly created root id.
    assert second["version"] == 2
    assert second["parent_version"] == first["id"]
    tgt_pid = tgt.created_projects[0]["id"]
    assert first["project"] == tgt_pid and second["project"] == tgt_pid


def test_schema_clone_bound_to_target_project():
    src = FakeClient(
        projects=[_src_project("src-p", "Receipts")],
        schemas={
            "src-p": [
                {
                    "id": "src-s1",
                    "project": "src-p",
                    "json_schema": '{"type":"object"}',
                    "version": 1,
                    "is_active": True,
                }
            ]
        },
    )
    tgt = FakeClient()
    ctx = _ctx(src, tgt)

    AgenticStudioPhase(ctx).run(CloneReport())

    assert len(tgt.created_schemas) == 1
    schema = tgt.created_schemas[0]
    assert schema["project"] == tgt.created_projects[0]["id"]
    assert schema["json_schema"] == '{"type":"object"}'


def test_registry_republished_and_remapped():
    src = FakeClient(
        projects=[_src_project("src-p", "Receipts")],
        registries={"src-p": [{"registry_id": "src-reg"}]},
    )
    tgt = FakeClient()
    ctx = _ctx(src, tgt)

    AgenticStudioPhase(ctx).run(CloneReport())

    tgt_pid = tgt.created_projects[0]["id"]
    assert tgt.exported_projects == [tgt_pid]
    assert (
        ctx.remap.resolve("agentic_studio_registry", "src-reg")
        == f"tgt-reg-{tgt_pid}"
    )


def test_no_source_registry_skips_export():
    src = FakeClient(projects=[_src_project("src-p", "Receipts")])
    tgt = FakeClient()
    ctx = _ctx(src, tgt)

    AgenticStudioPhase(ctx).run(CloneReport())

    assert tgt.exported_projects == []


def test_settings_create_and_adopt():
    src = FakeClient(
        projects=[],
        settings=[
            {"id": "src-s1", "key": "model", "value": "gpt-4"},
            {"id": "src-s2", "key": "temp", "value": "0.2"},
        ],
    )
    tgt = FakeClient(settings=[{"id": "tgt-s2", "key": "temp", "value": "0.9"}])
    ctx = _ctx(src, tgt)

    result = AgenticStudioPhase(ctx).run(CloneReport())

    # 'model' is new (create), 'temp' already exists (update/adopt).
    assert [s["key"] for s in tgt.created_settings] == ["model"]
    assert tgt.updated_settings and tgt.updated_settings[0][0] == "tgt-s2"
    assert result.adopted == 1


def test_dry_run_plans_without_writing():
    src = FakeClient(
        projects=[_src_project("src-p", "Receipts")],
        versions={
            "src-p": [
                {"id": "src-v1", "project": "src-p", "version": 1, "prompt_text": "v1"}
            ]
        },
        schemas={
            "src-p": [
                {"id": "src-s1", "project": "src-p", "json_schema": "{}", "version": 1}
            ]
        },
        settings=[{"id": "src-set", "key": "model", "value": "x"}],
    )
    tgt = FakeClient()
    ctx = _ctx(src, tgt, dry_run=True)

    result = AgenticStudioPhase(ctx).run(CloneReport())

    assert tgt.created_projects == []
    assert tgt.created_versions == []
    assert tgt.created_schemas == []
    assert tgt.created_settings == []
    assert tgt.exported_projects == []
    # 1 project + 1 version + 1 schema + 1 setting planned.
    assert result.created == 4
    planned = ctx.remap.resolve("agentic_project", "src-p")
    assert planned is not None and ctx.remap.is_planned(planned)
    planned_v = ctx.remap.resolve("agentic_prompt_version", "src-v1")
    assert planned_v is not None and ctx.remap.is_planned(planned_v)
