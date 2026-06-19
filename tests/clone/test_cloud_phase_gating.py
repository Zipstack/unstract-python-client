"""Tests for capability-probe gating of cloud-only phases.

A cloud phase declares ``probe_path``; the orchestrator probes source then
target before running it. Matrix:
- source absent  → silent skip (no run, no report row, no warning).
- source present, target absent → warn + skip (one ``report.warnings`` entry).
- both present   → run normally.

Core OSS phases (``probe_path is None``) are never probed and always run.
"""

from __future__ import annotations

from unittest.mock import patch

from unstract.clone import orchestrator
from unstract.clone.context import OrgEndpoint
from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, PhaseResult


class _CloudPhase(Phase):
    """Dummy cloud phase; records that it ran on a shared list."""

    invocations: list[str] = []
    name = "dummy_cloud"
    probe_path = "dummy/"

    def run(self, report: CloneReport) -> PhaseResult:
        _CloudPhase.invocations.append(self.name)
        return report.get_phase(self.name)


def _src() -> OrgEndpoint:
    return OrgEndpoint(
        base_url="https://src.example.com",
        organization_id="src_org",
        platform_key="src-key",
    )


def _tgt() -> OrgEndpoint:
    return OrgEndpoint(
        base_url="https://tgt.example.com",
        organization_id="tgt_org",
        platform_key="tgt-key",
    )


def _run_with_probes(*, source_present: bool, target_present: bool) -> CloneReport:
    """Drive clone() with a single cloud phase and scripted per-org probes."""
    _CloudPhase.invocations = []
    scripted = {"src_org": source_present, "tgt_org": target_present}

    def fake_probe(self, path: str) -> bool:
        return scripted[self.endpoint.organization_id]

    with (
        patch.object(orchestrator, "PHASES", [("dummy_cloud", _CloudPhase)]),
        patch.object(orchestrator.PlatformClient, "close"),
        patch.object(orchestrator.PlatformClient, "probe", fake_probe),
    ):
        return orchestrator.clone(_src(), _tgt())


def test_source_absent_skips_silently():
    report = _run_with_probes(source_present=False, target_present=False)
    assert _CloudPhase.invocations == []
    # OSS source must look exactly like today: no phase row, no warning.
    assert report.phases == []
    assert report.warnings == []
    assert "dummy_cloud" not in report.skipped_phases


def test_source_present_target_absent_warns_and_skips():
    report = _run_with_probes(source_present=True, target_present=False)
    assert _CloudPhase.invocations == []
    assert len(report.warnings) == 1
    assert "dummy_cloud" in report.warnings[0]


def test_both_present_runs_phase():
    report = _run_with_probes(source_present=True, target_present=True)
    assert _CloudPhase.invocations == ["dummy_cloud"]
    assert report.warnings == []
