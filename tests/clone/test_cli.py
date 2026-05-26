"""Tests for the click CLI wiring in ``unstract.clone.cli``.

Coverage:
- ``_parse_size`` accepts bare integers, K/M/G suffixes, decimals.
- ``--max-file-size 0`` propagates as 0 (force every file to manual list),
  not the default cap — distinguished from the unparseable case.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from unstract.clone.cli import _parse_size, cli
from unstract.clone.context import DEFAULT_MAX_FILE_SIZE, CloneOptions
from unstract.clone.report import CloneReport, Endpoint


def test_parse_size_bare_int_is_bytes():
    assert _parse_size("25") == 25


def test_parse_size_accepts_kb_mb_gb_units():
    assert _parse_size("25MB") == 25 * 1024 * 1024
    assert _parse_size("1.5GB") == int(1.5 * 1024 * 1024 * 1024)
    assert _parse_size("512K") == 512 * 1024


def test_parse_size_zero_returns_zero():
    # Regression for `cap_bytes or DEFAULT` — must not coerce 0 to the
    # default. CLI flag --max-file-size 0 means "every file goes to the
    # oversize/manual-upload list".
    assert _parse_size("0") == 0


def test_parse_size_unknown_unit_raises():
    import click

    with pytest.raises(click.BadParameter):
        _parse_size("10XB")


def test_parse_size_unparseable_raises():
    import click

    with pytest.raises(click.BadParameter):
        _parse_size("not-a-size")


def test_cli_max_file_size_zero_propagates_to_options(monkeypatch):
    captured: dict = {}

    def fake_clone(source, target, options=None):
        captured["options"] = options
        return CloneReport(
            source=Endpoint(
                base_url=source.base_url, organization_id=source.organization_id
            ),
            target=Endpoint(
                base_url=target.base_url, organization_id=target.organization_id
            ),
        )

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
            "--max-file-size",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    opts: CloneOptions = captured["options"]
    assert opts.max_file_size == 0


def test_cli_max_file_size_default_when_flag_omitted(monkeypatch):
    captured: dict = {}

    def fake_clone(source, target, options=None):
        captured["options"] = options
        return CloneReport(
            source=Endpoint(
                base_url=source.base_url, organization_id=source.organization_id
            ),
            target=Endpoint(
                base_url=target.base_url, organization_id=target.organization_id
            ),
        )

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
    opts: CloneOptions = captured["options"]
    assert opts.max_file_size == DEFAULT_MAX_FILE_SIZE
