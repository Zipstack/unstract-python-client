"""Top-level ``unstract`` command group.

Subcommands live in their own subpackages and are registered here so a
single console script (``unstract``) fronts all of them. ``unstract.clone``
keeps its own group + ``main`` so ``python -m unstract.clone`` still works.
"""

from __future__ import annotations

from typing import Any

import click

from unstract.clone.cli import clone_cmd


@click.group(name="unstract")
def cli() -> None:
    """Unstract command-line tools."""


cli.add_command(clone_cmd, name="clone")


def main(argv: list[str] | None = None) -> Any:
    return cli(args=argv, standalone_mode=True)


if __name__ == "__main__":
    main()
