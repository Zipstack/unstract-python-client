"""Tests for ``ManualReviewPhase`` (cloud-only HITL feature).

Covers: per-workflow RuleEngine + HITLSettings cloned with the workflow
remap (incl. nested confidence_filters); a workflow with no target mapping is
silently skipped; org-level AutoApprovalSettings cloned once with an id-remap
warning; ReviewApiKey recreation emits the re-wire warning; dry-run plans the
creates without writing.

A single scripted fake plays both source and target; the target side records
every write so assertions read off the ``created_*`` lists.
"""

from __future__ import annotations

from unstract.clone.context import CloneContext, CloneOptions, RemapTable
from unstract.clone.phases.manual_review import ManualReviewPhase
from unstract.clone.report import CloneReport


class FakeClient:
    MR_RULE_TYPES = ("DB", "API")

    def __init__(
        self,
        *,
        workflows=None,
        rules=None,
        settings=None,
        auto_approval=None,
        api_keys=None,
    ):
        self.workflows = list(workflows or [])
        # (workflow_id, rule_type) -> rule dict
        self.rules = dict(rules or {})
        # workflow_id -> settings dict
        self.settings = dict(settings or {})
        self.auto_approval = list(auto_approval or [])
        self.api_keys = list(api_keys or [])

        self.created_rules: list[dict] = []
        self.created_settings: list[dict] = []
        self.created_auto_approval: list[dict] = []
        self.created_api_keys: list[dict] = []

    def list_workflows(self):
        return list(self.workflows)

    def get_review_rule(self, workflow_id, rule_type):
        return self.rules.get((str(workflow_id), rule_type))

    def create_review_rule(self, payload):
        self.created_rules.append(payload)
        self.rules[(str(payload["workflow"]), payload.get("rule_type", "DB"))] = payload
        return payload

    def get_review_settings(self, workflow_id):
        return self.settings.get(str(workflow_id))

    def create_review_settings(self, payload):
        self.created_settings.append(payload)
        self.settings[str(payload["workflow"])] = payload
        return payload

    def list_auto_approval_settings(self):
        return list(self.auto_approval)

    def create_auto_approval_settings(self, payload):
        self.created_auto_approval.append(payload)
        self.auto_approval.append(payload)
        return payload

    def list_review_api_keys(self):
        return list(self.api_keys)

    def create_review_api_key(self, payload):
        self.created_api_keys.append(payload)
        self.api_keys.append(payload)
        return payload


def _ctx(source, target, *, remap=None, **opt_overrides):
    return CloneContext(
        source=source,
        target=target,
        options=CloneOptions(**opt_overrides),
        remap=remap or RemapTable(),
    )


def _src_settings(workflow="src-wf"):
    return {"workflow": workflow, "sync_with": "DB", "ttl_hours": 100}


def _src_rule(rule_type, **over):
    rule = {
        "id": f"src-rule-{rule_type}",
        "workflow": "src-wf",
        "rule_type": rule_type,
        "percentage": 25,
        "rule_string": "x > 1",
        "rule_json": {"x": 1},
        "rule_logic": "OR",
        "confidence_filters": [
            {"id": "cf1", "field_key": "amount", "confidence_threshold": 80}
        ],
    }
    rule.update(over)
    return rule


def test_rule_and_settings_cloned_with_workflow_remap():
    src = FakeClient(
        workflows=[{"id": "src-wf", "workflow_name": "WF"}],
        rules={("src-wf", "DB"): _src_rule("DB")},
        settings={"src-wf": _src_settings()},
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "src-wf", "tgt-wf")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = ManualReviewPhase(ctx).run(report)

    # One DB rule + one settings row created (no API rule in source).
    assert len(tgt.created_rules) == 1
    rule = tgt.created_rules[0]
    assert rule["workflow"] == "tgt-wf"
    assert rule["rule_type"] == "DB"
    assert rule["percentage"] == 25
    # Nested filters carried, server-managed id stripped.
    assert rule["confidence_filters"] == [
        {"field_key": "amount", "confidence_threshold": 80}
    ]
    assert "id" not in rule

    assert len(tgt.created_settings) == 1
    settings = tgt.created_settings[0]
    assert settings == {"workflow": "tgt-wf", "sync_with": "DB", "ttl_hours": 100}
    assert result.created == 2


def test_workflow_without_target_mapping_skipped():
    src = FakeClient(
        workflows=[{"id": "src-wf", "workflow_name": "WF"}],
        rules={("src-wf", "DB"): _src_rule("DB")},
        settings={"src-wf": _src_settings()},
    )
    tgt = FakeClient()
    # No workflow remap recorded — its tool/workflow wasn't cloned.
    ctx = _ctx(src, tgt, remap=RemapTable())
    report = CloneReport()

    result = ManualReviewPhase(ctx).run(report)

    assert tgt.created_rules == []
    assert tgt.created_settings == []
    assert result.created == 0
    assert result.failed == 0


def test_auto_approval_cloned_once_with_warning():
    src = FakeClient(
        workflows=[],
        auto_approval=[
            {
                "id": "aa1",
                "auto_approved_document_classes": ["cls-1"],
                "auto_approved_users": ["7"],
            }
        ],
    )
    tgt = FakeClient()
    ctx = _ctx(src, tgt, remap=RemapTable())
    report = CloneReport()

    result = ManualReviewPhase(ctx).run(report)

    assert len(tgt.created_auto_approval) == 1
    payload = tgt.created_auto_approval[0]
    assert payload["auto_approved_document_classes"] == ["cls-1"]
    assert payload["auto_approved_users"] == ["7"]
    assert "organization" not in payload
    assert any("do not remap across orgs" in w for w in result.warnings)
    assert result.created == 1


def test_review_api_key_recreated_with_warning():
    src = FakeClient(
        workflows=[],
        api_keys=[
            {
                "id": "k1",
                "api_key": "secret-uuid",
                "class_name": "invoices",
                "description": "d",
                "is_active": True,
            }
        ],
    )
    tgt = FakeClient()
    ctx = _ctx(src, tgt, remap=RemapTable())
    report = CloneReport()

    result = ManualReviewPhase(ctx).run(report)

    assert len(tgt.created_api_keys) == 1
    payload = tgt.created_api_keys[0]
    # Secret + server-managed id NOT carried over.
    assert "api_key" not in payload
    assert "id" not in payload
    assert payload == {"class_name": "invoices", "description": "d", "is_active": True}
    assert any("re-wire any external consumers" in w for w in result.warnings)


def test_dry_run_plans_without_writing():
    src = FakeClient(
        workflows=[{"id": "src-wf", "workflow_name": "WF"}],
        rules={
            ("src-wf", "DB"): _src_rule("DB"),
            ("src-wf", "API"): _src_rule("API"),
        },
        settings={"src-wf": _src_settings()},
        auto_approval=[
            {
                "id": "aa",
                "auto_approved_document_classes": [],
                "auto_approved_users": [],
            }
        ],
        api_keys=[{"id": "k1", "class_name": "c", "is_active": True}],
    )
    tgt = FakeClient()
    remap = RemapTable()
    remap.record("workflow", "src-wf", "tgt-wf")
    ctx = _ctx(src, tgt, remap=remap, dry_run=True)
    report = CloneReport()

    result = ManualReviewPhase(ctx).run(report)

    # 2 rules + 1 settings + 1 auto-approval + 1 api key planned.
    assert result.created == 5
    assert tgt.created_rules == []
    assert tgt.created_settings == []
    assert tgt.created_auto_approval == []
    assert tgt.created_api_keys == []
