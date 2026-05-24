"""Structured report produced by ``migrate()``.

Tracks per-phase counts (created / adopted / skipped / failed) and a final
remap snapshot. Renders to a rich-formatted table when ``rich`` is
available; falls back to plain text otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseResult:
    name: str
    created: int = 0
    adopted: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class MigrationReport:
    phases: list[PhaseResult] = field(default_factory=list)
    skipped_phases: list[str] = field(default_factory=list)
    remap_snapshot: dict[str, dict[str, str]] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str | None = None
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
        console = Console(file=buf, force_terminal=False, width=100)
        table = Table(title="Migration Report")
        for col in ("Phase", "Created", "Adopted", "Skipped", "Failed"):
            table.add_column(col, justify="right" if col != "Phase" else "left")
        for p in self.phases:
            table.add_row(p.name, str(p.created), str(p.adopted), str(p.skipped), str(p.failed))
        console.print(table)
        if self.skipped_phases:
            console.print(f"[dim]Skipped phases:[/dim] {', '.join(self.skipped_phases)}")
        self._render_files_sections(console)
        if self.remap_snapshot:
            remap = Table(title="Source -> Target UUID Map")
            remap.add_column("Entity")
            remap.add_column("Source UUID")
            remap.add_column("Target UUID")
            for entity, mapping in self.remap_snapshot.items():
                for src, tgt in mapping.items():
                    remap.add_row(entity, src, tgt)
            console.print(remap)
        if self.aborted:
            console.print(f"[red]ABORTED:[/red] {self.abort_reason}")
        return buf.getvalue()

    def _render_plain(self) -> str:
        lines = ["Migration Report", "=" * 60]
        header = f"{'Phase':<24}{'Created':>10}{'Adopted':>10}{'Skipped':>10}{'Failed':>10}"
        lines.append(header)
        for p in self.phases:
            lines.append(
                f"{p.name:<24}{p.created:>10}{p.adopted:>10}{p.skipped:>10}{p.failed:>10}"
            )
        if self.skipped_phases:
            lines.append(f"Skipped phases: {', '.join(self.skipped_phases)}")
        lines.extend(self._files_sections_plain())
        if self.remap_snapshot:
            lines.append("")
            lines.append("Source -> Target UUID Map")
            lines.append("-" * 60)
            for entity, mapping in self.remap_snapshot.items():
                for src, tgt in mapping.items():
                    lines.append(f"  {entity:<12} {src} -> {tgt}")
        if self.aborted:
            lines.append(f"ABORTED: {self.abort_reason}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phases": [
                {
                    "name": p.name,
                    "created": p.created,
                    "adopted": p.adopted,
                    "skipped": p.skipped,
                    "failed": p.failed,
                    "errors": list(p.errors),
                }
                for p in self.phases
            ],
            "skipped_phases": list(self.skipped_phases),
            "remap_snapshot": self.remap_snapshot,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "uploaded_files": list(self.uploaded_files),
            "skipped_files": list(self.skipped_files),
            "oversize_files": list(self.oversize_files),
            "unsupported_files": list(self.unsupported_files),
            "failed_files": list(self.failed_files),
        }

    def _render_files_sections(self, console: Any) -> None:
        if self.uploaded_files:
            console.print(
                f"[green]Files uploaded:[/green] {len(self.uploaded_files)}"
            )
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
