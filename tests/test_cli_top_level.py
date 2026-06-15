"""Tests for the top-level ``unstract`` command group (``unstract.cli``)."""

from __future__ import annotations

from click.testing import CliRunner

from unstract.cli import cli
from unstract.clone.report import CloneReport, Endpoint


def test_clone_invocation_via_top_level_group(monkeypatch):
    captured: dict = {}

    def fake_clone(source, target, options=None):
        captured["source"] = source
        captured["target"] = target
        return CloneReport(
            source=Endpoint(
                base_url=source.base_url, organization_id=source.organization_id
            ),
            target=Endpoint(
                base_url=target.base_url, organization_id=target.organization_id
            ),
        )

    # The clone command's callback resolves run_clone from unstract.clone.cli.
    monkeypatch.setattr("unstract.clone.cli.run_clone", fake_clone)

    result = CliRunner().invoke(
        cli,
        [
            "clone",
            "--source-url",
            "http://src",
            "--source-org",
            "src",
            "--source-key",
            "sk",
            "--target-url",
            "http://tgt",
            "--target-org",
            "tgt",
            "--target-key",
            "tk",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["source"].organization_id == "src"
    assert captured["target"].organization_id == "tgt"
