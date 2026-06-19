"""Base class for clone phases."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from typing import Any, TypeVar

from unstract.clone.context import CloneContext
from unstract.clone.exceptions import CloneError
from unstract.clone.report import CloneReport, PhaseResult
from unstract.clone.sharing import apply_share_state

T = TypeVar("T")

logger = logging.getLogger(__name__)

# OPTIONS reports any FK/M2M as writable, but the backend overrides these
# server-side on create. Posting them is either noise (silently overwritten)
# or a 400 (when a source-org value doesn't validate against the target org).
# Strip them universally — the phase OPTIONS schema covers the entity-specific
# writable subset. ``shared_users`` stays stripped on create — share state is
# replicated post-create instead (see sharing.py).
SERVER_MANAGED: frozenset[str] = frozenset(
    {
        "id",
        "organization",
        "created_by",
        "created_by_email",
        "modified_by",
        "modified_by_email",
        "created_at",
        "modified_at",
        "shared_users",
        "workflow_owner",
    }
)


def build_post_payload(src: dict[str, Any], writable: frozenset[str]) -> dict[str, Any]:
    """Project ``src`` onto the writable schema, dropping server-managed
    fields, ``None`` values, and empty strings (which DRF treats as blank
    and rejects on required fields).
    """
    keys = writable - SERVER_MANAGED
    # Equality with `(None, "")` matched False and 0 too (Python: False == 0,
    # 0 in (None, "") is False, but `0 not in (...)` falsely returns True).
    # Explicit identity / equality checks preserve falsy-but-meaningful
    # values like ``BooleanField`` False and numeric defaults.
    return {k: src[k] for k in keys if k in src and src[k] is not None and src[k] != ""}


class Phase(ABC):
    """Abstract phase. One subclass per entity type."""

    name: str = ""
    # Share endpoint template for shareable resource types, e.g.
    # "adapter/{id}/share/" ({id} = target pk). None = not shareable.
    share_path_template: str | None = None
    # Capability-gate for cloud-only phases. When set, the orchestrator probes
    # this list endpoint on source/target before running and applies the skip
    # matrix (source absent → silent skip; target absent → warn + skip). Core
    # OSS phases leave it None and always run (no probe call at all).
    probe_path: str | None = None

    def __init__(self, ctx: CloneContext):
        self.ctx = ctx

    @abstractmethod
    def run(self, report: CloneReport) -> PhaseResult:
        """Migrate all entities of this phase's type. Idempotent across runs."""
        raise NotImplementedError

    def apply_share(
        self,
        *,
        src: dict[str, Any],
        tgt_id: str,
        label: str,
        result: PhaseResult,
        lock: threading.Lock,
        src_detail_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """Replicate ``src``'s share state onto the target entity.

        Pass ``src_detail_fn`` when ``src`` may be a stripped list-row —
        the helper fetches the detail only if a share axis is missing.
        No-op for phases without ``share_path_template``; never raises.
        """
        if self.share_path_template is None:
            return
        apply_share_state(
            self.ctx,
            share_path=self.share_path_template.format(id=tgt_id),
            entity_label=f"{self.name} '{label}'",
            src=src,
            result=result,
            lock=lock,
            src_detail_fn=src_detail_fn,
        )

    def parallel_map(
        self,
        items: Iterable[T],
        work_fn: Callable[[T, threading.Lock], None],
    ) -> None:
        """Fan ``work_fn(item, lock)`` across ``ctx.options.concurrency``
        threads. ``work_fn`` must hold ``lock`` while mutating shared
        state. ``CloneError`` from any worker cancels the rest and
        re-raises. ``concurrency <= 1`` skips the executor entirely.
        """
        materialised = list(items)
        if not materialised:
            return

        concurrency = max(1, self.ctx.options.concurrency)
        lock = threading.Lock()

        if concurrency == 1:
            for item in materialised:
                work_fn(item, lock)
            return

        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix=f"clone-{self.name}",
        ) as pool:
            futures: list[Future[None]] = [
                pool.submit(work_fn, item, lock) for item in materialised
            ]
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            clone_err: CloneError | None = None
            other_err: BaseException | None = None
            for fut in done:
                if fut.cancelled():
                    continue
                exc = fut.exception()
                if exc is None:
                    continue
                if isinstance(exc, CloneError) and clone_err is None:
                    clone_err = exc
                elif other_err is None:
                    other_err = exc
            if clone_err is not None or other_err is not None:
                for fut in futures:
                    fut.cancel()
            if clone_err is not None:
                raise clone_err
            if other_err is not None:
                raise other_err
