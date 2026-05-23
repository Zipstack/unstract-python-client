"""Click-based CLI for ``unstract.migration``.

Single ``migrate`` command. Platform keys can be passed via flags
(``--source-key`` / ``--target-key``) or env vars
(``UNSTRACT_SRC_PLATFORM_KEY`` / ``UNSTRACT_TGT_PLATFORM_KEY``) — env vars
are preferred so the key never lands in shell history.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import click

from unstract.migration.context import MigrationOptions, OrgEndpoint
from unstract.migration.exceptions import MigrationError
from unstract.migration.orchestrator import migrate as run_migrate


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
    """Org-to-org data migration over the Platform API."""


@cli.command("migrate")
@click.option("--source-url", required=True, help="Base URL of the source deployment (e.g. https://us.unstract.com)")
@click.option("--source-org", required=True, help="Source organization_id (slug in the URL path)")
@click.option(
    "--source-key",
    envvar="UNSTRACT_SRC_PLATFORM_KEY",
    required=True,
    help="Source admin's Platform API key (or env UNSTRACT_SRC_PLATFORM_KEY)",
)
@click.option("--target-url", required=True, help="Base URL of the target deployment")
@click.option("--target-org", required=True, help="Target organization_id (slug in the URL path)")
@click.option(
    "--target-key",
    envvar="UNSTRACT_TGT_PLATFORM_KEY",
    required=True,
    help="Target admin's Platform API key (or env UNSTRACT_TGT_PLATFORM_KEY)",
)
@click.option("--dry-run", is_flag=True, help="Plan only — do not POST anything to target")
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
@click.option("-v", "--verbose", is_flag=True, help="Debug logging")
def migrate_cmd(
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
    verbose: bool,
) -> None:
    """Migrate configured resources from one org to another."""
    _configure_logging(verbose)

    options = MigrationOptions(
        dry_run=dry_run,
        include=_split_csv(include),
        exclude=_split_csv(exclude) or (),
        on_name_conflict=on_name_conflict,
        verbose=verbose,
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
        report = run_migrate(source, target, options)
    except MigrationError as e:
        click.echo(f"Migration failed: {e}", err=True)
        sys.exit(2)

    click.echo(report.render())
    if report.aborted or any(p.failed for p in report.phases):
        sys.exit(1)


def main(argv: list[str] | None = None) -> Any:
    return cli(args=argv, standalone_mode=True)


if __name__ == "__main__":
    main()
