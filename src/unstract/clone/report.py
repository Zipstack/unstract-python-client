"""Structured report produced by ``clone()``.

Tracks per-phase counts (created / adopted / skipped / failed) and a final
remap snapshot. Renders to a rich-formatted table when ``rich`` is
available; falls back to plain text otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    name: str
    created: int = 0
    adopted: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class Endpoint:
    """Endpoint identity for the report header — never carries the API key."""

    base_url: str
    organization_id: str


@dataclass
class CloneReport:
    source: Endpoint | None = None
    target: Endpoint | None = None
    phases: list[PhaseResult] = field(default_factory=list)
    skipped_phases: list[str] = field(default_factory=list)
    remap_snapshot: dict[str, dict[str, str]] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str | None = None
    total_duration_s: float = 0.0
    # Files-phase artifacts. Each entry carries enough context for an
    # operator to act on it without cross-referencing the run log.
    uploaded_files: list[dict[str, Any]] = field(default_factory=list)
    skipped_files: list[dict[str, Any]] = field(default_factory=list)
    oversize_files: list[dict[str, Any]] = field(default_factory=list)
    unsupported_files: list[dict[str, Any]] = field(default_factory=list)
    failed_files: list[dict[str, Any]] = field(default_factory=list)

    def get_phase(self, name: str) -> PhaseResult:
        for p in self.phases:
            if p.name == name:
                return p
        result = PhaseResult(name=name)
        self.phases.append(result)
        return result

    def render(self) -> str:
        """Render as a rich table when available, otherwise plain text."""
        try:
            from io import StringIO

            from rich.console import Console
            from rich.table import Table
        except ImportError:
            return self._render_plain()

        buf = StringIO()
        # force_terminal so ANSI codes survive the StringIO capture; the
        # caller decides whether to strip them when printing to a non-tty.
        console = Console(
            file=buf, force_terminal=True, color_system="truecolor", width=100
        )
        # Actionable summary first so it doesn't scroll past the table.
        self._render_failures_summary(console_print=console.print, rich=True)
        self._render_endpoints(console.print)
        table = Table(title="Clone Report", header_style="bold cyan")
        table.add_column("Phase", style="bold", justify="left")
        for col in ("Created", "Adopted", "Skipped", "Failed", "Time"):
            table.add_column(col, justify="right")

        totals = {"created": 0, "adopted": 0, "skipped": 0, "failed": 0}
        for p in self.phases:
            phase_style = "red" if p.failed else ("yellow" if p.skipped else "green")
            table.add_row(
                f"[{phase_style}]{p.name}[/{phase_style}]",
                self._fmt_count(p.created, "green"),
                self._fmt_count(p.adopted, "green"),
                self._fmt_count(p.skipped, "yellow"),
                self._fmt_count(p.failed, "red"),
                self._fmt_duration(p.duration_s),
            )
            for k in totals:
                totals[k] += getattr(p, k)

        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            self._fmt_count(totals["created"], "green", bold=True),
            self._fmt_count(totals["adopted"], "green", bold=True),
            self._fmt_count(totals["skipped"], "yellow", bold=True),
            self._fmt_count(totals["failed"], "red", bold=True),
            self._fmt_duration(self.total_duration_s, bold=True),
        )
        console.print(table)
        if self.skipped_phases:
            console.print(
                f"[dim]Skipped phases:[/dim] {', '.join(self.skipped_phases)}"
            )
        self._render_files_sections(console)
        self._render_remap_summary(console_print=console.print)
        if self.aborted:
            console.print(f"[bold red]ABORTED:[/bold red] {self.abort_reason}")
        elif totals["failed"]:
            console.print(
                f"[bold red]Completed with {totals['failed']} failure(s)[/bold red] — "
                "see WARNING/ERROR log lines above for details"
            )
        else:
            console.print("[bold green]Completed successfully[/bold green]")
        return buf.getvalue()

    @staticmethod
    def _fmt_count(value: int, color: str, bold: bool = False) -> str:
        """Dim a zero to keep the eye on non-zero cells; colour anything > 0."""
        if value == 0:
            return "[dim]0[/dim]"
        style = f"bold {color}" if bold else color
        return f"[{style}]{value}[/{style}]"

    @staticmethod
    def _fmt_duration(seconds: float, bold: bool = False) -> str:
        if seconds <= 0:
            return "[dim]—[/dim]"
        if seconds < 60:
            text = f"{seconds:.1f}s"
        else:
            mins, secs = divmod(seconds, 60)
            text = f"{int(mins)}m{secs:.0f}s"
        return f"[bold]{text}[/bold]" if bold else text

    @staticmethod
    def _fmt_duration_plain(seconds: float) -> str:
        if seconds <= 0:
            return "—"
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m{secs:.0f}s"

    def _render_plain(self) -> str:
        lines: list[str] = []
        self._render_failures_summary(console_print=lines.append, rich=False)
        lines.extend(["Clone Report", "=" * 60])
        self._render_endpoints(lines.append)
        header = (
            f"{'Phase':<24}{'Created':>10}{'Adopted':>10}"
            f"{'Skipped':>10}{'Failed':>10}{'Time':>10}"
        )
        lines.append(header)
        for p in self.phases:
            lines.append(
                f"{p.name:<24}{p.created:>10}{p.adopted:>10}"
                f"{p.skipped:>10}{p.failed:>10}{self._fmt_duration_plain(p.duration_s):>10}"
            )
        lines.append(
            f"{'TOTAL':<64}{self._fmt_duration_plain(self.total_duration_s):>10}"
        )
        if self.skipped_phases:
            lines.append(f"Skipped phases: {', '.join(self.skipped_phases)}")
        lines.extend(self._files_sections_plain())
        self._render_remap_summary(console_print=lines.append)
        if self.aborted:
            lines.append(f"ABORTED: {self.abort_reason}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": (
                {
                    "base_url": self.source.base_url,
                    "organization_id": self.source.organization_id,
                }
                if self.source
                else None
            ),
            "target": (
                {
                    "base_url": self.target.base_url,
                    "organization_id": self.target.organization_id,
                }
                if self.target
                else None
            ),
            "phases": [
                {
                    "name": p.name,
                    "created": p.created,
                    "adopted": p.adopted,
                    "skipped": p.skipped,
                    "failed": p.failed,
                    "errors": list(p.errors),
                    "duration_s": p.duration_s,
                }
                for p in self.phases
            ],
            "skipped_phases": list(self.skipped_phases),
            "remap_snapshot": self.remap_snapshot,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "total_duration_s": self.total_duration_s,
            "uploaded_files": list(self.uploaded_files),
            "skipped_files": list(self.skipped_files),
            "oversize_files": list(self.oversize_files),
            "unsupported_files": list(self.unsupported_files),
            "failed_files": list(self.failed_files),
        }

    def _render_endpoints(self, console_print: Any) -> None:
        if not self.source and not self.target:
            return
        src = self._fmt_endpoint(self.source)
        tgt = self._fmt_endpoint(self.target)
        console_print(f"Source: {src}")
        console_print(f"Target: {tgt}")

    @staticmethod
    def _fmt_endpoint(ep: Endpoint | None) -> str:
        if ep is None:
            return "?"
        return f"{ep.organization_id} @ {ep.base_url}"

    def _render_remap_summary(self, console_print: Any) -> None:
        """Summarise the remap snapshot. Full map is large and noisy, so
        we only print per-entity counts here; the full mapping is emitted
        at DEBUG and remains in ``as_dict()`` for programmatic consumers.
        """
        if not self.remap_snapshot:
            return
        counts = ", ".join(
            f"{entity}={len(mapping)}"
            for entity, mapping in self.remap_snapshot.items()
            if mapping
        )
        if counts:
            console_print(f"Remap entries: {counts}")
        if logger.isEnabledFor(logging.DEBUG):
            for entity, mapping in self.remap_snapshot.items():
                for src, tgt in mapping.items():
                    logger.debug("remap %s %s -> %s", entity, src, tgt)

    def _render_files_sections(self, console: Any) -> None:
        if self.uploaded_files:
            console.print(f"[green]Files uploaded:[/green] {len(self.uploaded_files)}")
        for header, rows in (
            ("Oversize files (manual upload required)", self.oversize_files),
            ("Unsupported mime files (manual upload required)", self.unsupported_files),
            ("Skipped files (operator action required)", self.skipped_files),
            ("Failed files", self.failed_files),
        ):
            if not rows:
                continue
            console.print(f"[yellow]{header}:[/yellow]")
            for row in rows:
                console.print(f"  - {self._describe_file_row(row)}")

    def _files_sections_plain(self) -> list[str]:
        lines: list[str] = []
        if self.uploaded_files:
            lines.append(f"Files uploaded: {len(self.uploaded_files)}")
        for header, rows in (
            ("Oversize files (manual upload required)", self.oversize_files),
            ("Unsupported mime files (manual upload required)", self.unsupported_files),
            ("Skipped files (operator action required)", self.skipped_files),
            ("Failed files", self.failed_files),
        ):
            if not rows:
                continue
            lines.append(f"{header}:")
            for row in rows:
                lines.append(f"  - {self._describe_file_row(row)}")
        return lines

    # Caps so a long traceback or many failures don't dominate the report.
    _FAILURE_LINE_MAX_CHARS = 200
    _FAILURE_MAX_ROWS = 30

    def _render_failures_summary(self, console_print: Any, rich: bool) -> None:
        rows: list[tuple[str, str]] = []
        for p in self.phases:
            for err in p.errors:
                rows.append((p.name, err))
        if not rows:
            return
        header = "Failures (see WARNING/ERROR log lines above for full detail)"
        if rich:
            console_print(f"[red]{header}:[/red]")
        else:
            console_print(f"{header}:")
        shown = rows[: self._FAILURE_MAX_ROWS]
        for phase_name, err in shown:
            truncated = self._truncate(err, self._FAILURE_LINE_MAX_CHARS)
            if rich:
                console_print(
                    f"  - [bold cyan]{phase_name}[/bold cyan]: {truncated}",
                    highlight=False,
                )
            else:
                console_print(f"  - {phase_name}: {truncated}")
        remaining = len(rows) - len(shown)
        if remaining > 0:
            tail = f"  ... +{remaining} more — see logs"
            if rich:
                console_print(f"[dim]{tail}[/dim]")
            else:
                console_print(tail)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = text.replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    @staticmethod
    def _describe_file_row(row: dict[str, Any]) -> str:
        tool = row.get("tool_name") or row.get("tool_id") or "?"
        name = row.get("file_name", "?")
        extras: list[str] = []
        if "size_bytes" in row:
            extras.append(f"{row['size_bytes']} bytes")
        if "mime_type" in row:
            extras.append(row["mime_type"])
        if "error" in row:
            extras.append(f"error={row['error']}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        return f"tool={tool} file={name}{suffix}"
