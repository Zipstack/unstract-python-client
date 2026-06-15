"""Click-based CLI for ``unstract.clone``.

Single ``clone`` command, registered on the top-level ``unstract`` group
(``unstract.cli``) — the canonical invocation is ``unstract clone``. The
local group here only backs ``python -m unstract.clone``.

Platform keys can be passed via flags (``--source-key`` / ``--target-key``)
or env vars (``UNSTRACT_SRC_PLATFORM_KEY`` / ``UNSTRACT_TGT_PLATFORM_KEY``)
— env vars are preferred so the key never lands in shell history.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import click

from unstract.clone.context import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_FILE_SIZE,
    CloneOptions,
    OrgEndpoint,
)
from unstract.clone.exceptions import CloneError
from unstract.clone.orchestrator import clone as run_clone

_SIZE_UNITS: dict[str, int] = {
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "M": 1024 * 1024,
    "MB": 1024 * 1024,
    "G": 1024 * 1024 * 1024,
    "GB": 1024 * 1024 * 1024,
}
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$")


def _parse_size(value: str) -> int:
    """Accept ``25``, ``25MB``, ``1.5GB`` etc. Returns bytes."""
    m = _SIZE_RE.match(value)
    if not m:
        raise click.BadParameter(f"can't parse size '{value}'")
    num, unit = m.group(1), m.group(2).upper() or "B"
    if unit not in _SIZE_UNITS:
        raise click.BadParameter(
            f"unknown size unit '{unit}'; use one of {sorted(_SIZE_UNITS)}"
        )
    return int(float(num) * _SIZE_UNITS[unit])


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _split_csv(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(p.strip() for p in value.split(",") if p.strip())


@click.group()
def cli() -> None:
    """Cloning organizations over the Platform API."""


@cli.command("clone")
@click.option("--source-url", required=True, help="Base URL of the source deployment")
@click.option(
    "--source-org", required=True, help="Source organization_id (slug in the URL path)"
)
@click.option(
    "--source-key",
    envvar="UNSTRACT_SRC_PLATFORM_KEY",
    required=True,
    help="Source admin's Platform API key (or env UNSTRACT_SRC_PLATFORM_KEY)",
)
@click.option("--target-url", required=True, help="Base URL of the target deployment")
@click.option(
    "--target-org", required=True, help="Target organization_id (slug in the URL path)"
)
@click.option(
    "--target-key",
    envvar="UNSTRACT_TGT_PLATFORM_KEY",
    required=True,
    help="Target admin's Platform API key (or env UNSTRACT_TGT_PLATFORM_KEY)",
)
@click.option(
    "--dry-run", is_flag=True, help="Plan only — do not POST anything to target"
)
@click.option(
    "--include",
    default=None,
    help="Comma-separated phase names to include (default: all)",
)
@click.option(
    "--exclude",
    default=None,
    help="Comma-separated phase names to exclude",
)
@click.option(
    "--on-name-conflict",
    type=click.Choice(["adopt", "abort"]),
    default="adopt",
    show_default=True,
    help="What to do when a like-named entity exists in target",
)
@click.option(
    "--api-prefix",
    default="api/v1",
    show_default=True,
    help="Backend URL prefix (matches deployment's PATH_PREFIX env)",
)
@click.option(
    "--file-strategy",
    type=click.Choice(["platform_api", "skip"]),
    default="platform_api",
    show_default=True,
    help="How to move Prompt Studio document files. 'skip' = metadata only.",
)
@click.option(
    "--max-file-size",
    default="25MB",
    show_default=True,
    help="Per-file cap for the files phase. Oversize → reported, not aborted.",
)
@click.option(
    "--skip-files",
    is_flag=True,
    help="Alias for --file-strategy=skip.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1, max=32),
    default=DEFAULT_CONCURRENCY,
    show_default=True,
    help="Per-phase worker count. 1 = strictly sequential.",
)
@click.option(
    "--clone-group-members",
    is_flag=True,
    help="Also add group members on target, matched by email. "
    "Members missing on target are skipped and reported.",
)
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
def clone_cmd(
    source_url: str,
    source_org: str,
    source_key: str,
    target_url: str,
    target_org: str,
    target_key: str,
    dry_run: bool,
    include: str | None,
    exclude: str | None,
    on_name_conflict: str,
    api_prefix: str,
    file_strategy: str,
    max_file_size: str,
    skip_files: bool,
    concurrency: int,
    clone_group_members: bool,
    verbose: bool,
) -> None:
    """Clone configured resources from one org to another."""
    _configure_logging(verbose)

    effective_strategy = "skip" if skip_files else file_strategy
    try:
        cap_bytes = _parse_size(max_file_size)
    except click.BadParameter as e:
        raise click.UsageError(str(e)) from e

    options = CloneOptions(
        dry_run=dry_run,
        include=_split_csv(include),
        exclude=_split_csv(exclude) or (),
        on_name_conflict=on_name_conflict,
        verbose=verbose,
        file_strategy=effective_strategy,
        max_file_size=cap_bytes if cap_bytes is not None else DEFAULT_MAX_FILE_SIZE,
        concurrency=concurrency,
        clone_group_members=clone_group_members,
    )

    source = OrgEndpoint(
        base_url=source_url,
        organization_id=source_org,
        platform_key=source_key,
        api_path_prefix=api_prefix,
    )
    target = OrgEndpoint(
        base_url=target_url,
        organization_id=target_org,
        platform_key=target_key,
        api_path_prefix=api_prefix,
    )

    try:
        report = run_clone(source, target, options)
    except CloneError as e:
        click.echo(f"Clone failed: {e}", err=True)
        sys.exit(2)

    click.echo(report.render())
    if report.aborted or any(p.failed for p in report.phases):
        sys.exit(1)


def main(argv: list[str] | None = None) -> Any:
    return cli(args=argv, standalone_mode=True)


if __name__ == "__main__":
    main()
