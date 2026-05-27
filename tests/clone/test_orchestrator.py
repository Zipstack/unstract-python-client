"""End-to-end tests for the ``clone()`` orchestrator.

Coverage:
- Phase ordering matches ``PHASES`` declaration.
- ``include`` / ``exclude`` route phases through ``skipped_phases``.
- ``CloneError`` raised by a phase aborts the run; subsequent phases skipped.
- Both ``PlatformClient`` instances are closed even when a phase aborts.
- ``RemapTable`` snapshot lands on the report.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unstract.clone import orchestrator
from unstract.clone.context import CloneOptions, OrgEndpoint
from unstract.clone.exceptions import CloneError
from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, PhaseResult


class _RecordingPhase(Phase):
    """Per-test phase factory; records invocation order on a shared list."""

    invocations: list[str] = []
    name = ""

    def run(self, report: CloneReport) -> PhaseResult:
        _RecordingPhase.invocations.append(self.name)
        result = report.get_phase(self.name)
        result.created += 1
        # Drop a remap entry so we can prove the snapshot lands on the report.
        self.ctx.remap.record(self.name, f"src-{self.name}", f"tgt-{self.name}")
        return result


def _make_phase(phase_name: str) -> type[Phase]:
    return type(
        f"FakePhase_{phase_name}",
        (_RecordingPhase,),
        {"name": phase_name},
    )


@pytest.fixture(autouse=True)
def _reset_invocations():
    _RecordingPhase.invocations = []
    yield
    _RecordingPhase.invocations = []


@pytest.fixture
def fake_phases():
    """Replace PHASES with a small deterministic set for the test run."""
    fake = [
        ("adapter", _make_phase("adapter")),
        ("connector", _make_phase("connector")),
        ("workflow", _make_phase("workflow")),
    ]
    with patch.object(orchestrator, "PHASES", fake):
        yield fake


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


def test_phases_run_in_declared_order(fake_phases):
    with patch.object(orchestrator.PlatformClient, "close") as mock_close:
        report = orchestrator.clone(_src(), _tgt())
    assert _RecordingPhase.invocations == ["adapter", "connector", "workflow"]
    assert [p.name for p in report.phases] == ["adapter", "connector", "workflow"]
    # Both clients must close (source + target) regardless of outcome.
    assert mock_close.call_count == 2


def test_include_filter_only_runs_listed_phases(fake_phases):
    opts = CloneOptions(include=("connector",))
    with patch.object(orchestrator.PlatformClient, "close"):
        report = orchestrator.clone(_src(), _tgt(), opts)
    assert _RecordingPhase.invocations == ["connector"]
    assert set(report.skipped_phases) == {"adapter", "workflow"}


def test_exclude_filter_skips_listed_phases(fake_phases):
    opts = CloneOptions(exclude=("workflow",))
    with patch.object(orchestrator.PlatformClient, "close"):
        report = orchestrator.clone(_src(), _tgt(), opts)
    assert _RecordingPhase.invocations == ["adapter", "connector"]
    assert report.skipped_phases == ["workflow"]


def test_clone_error_aborts_and_skips_subsequent_phases():
    class AbortingPhase(Phase):
        name = "connector"

        def run(self, report: CloneReport) -> PhaseResult:
            raise CloneError("name collision in 'connector'")

    fake = [
        ("adapter", _make_phase("adapter")),
        ("connector", AbortingPhase),
        ("workflow", _make_phase("workflow")),
    ]
    with (
        patch.object(orchestrator, "PHASES", fake),
        patch.object(orchestrator.PlatformClient, "close") as mock_close,
    ):
        report = orchestrator.clone(_src(), _tgt())

    assert _RecordingPhase.invocations == ["adapter"]
    assert report.aborted is True
    assert "name collision" in report.abort_reason
    # Clients still close on abort.
    assert mock_close.call_count == 2


def test_unrelated_exception_propagates_but_still_closes_clients():
    class CrashingPhase(Phase):
        name = "connector"

        def run(self, report: CloneReport) -> PhaseResult:
            raise RuntimeError("boom")

    fake = [
        ("adapter", _make_phase("adapter")),
        ("connector", CrashingPhase),
    ]
    with (
        patch.object(orchestrator, "PHASES", fake),
        patch.object(orchestrator.PlatformClient, "close") as mock_close,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            orchestrator.clone(_src(), _tgt())
    assert mock_close.call_count == 2


def test_remap_snapshot_populated_on_report(fake_phases):
    with patch.object(orchestrator.PlatformClient, "close"):
        report = orchestrator.clone(_src(), _tgt())
    assert report.remap_snapshot == {
        "adapter": {"src-adapter": "tgt-adapter"},
        "connector": {"src-connector": "tgt-connector"},
        "workflow": {"src-workflow": "tgt-workflow"},
    }
