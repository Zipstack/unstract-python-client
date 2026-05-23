"""Base class for migration phases."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from unstract.migration.context import MigrationContext
from unstract.migration.report import MigrationReport, PhaseResult

logger = logging.getLogger(__name__)


class Phase(ABC):
    """Abstract phase. One subclass per entity type."""

    name: str = ""

    def __init__(self, ctx: MigrationContext):
        self.ctx = ctx

    @abstractmethod
    def run(self, report: MigrationReport) -> PhaseResult:
        """Migrate all entities of this phase's type. Idempotent across runs."""
        raise NotImplementedError
