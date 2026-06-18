"""Replicate a resource's share state onto its cloned counterpart.

Share state is server-managed on create, so it is mirrored post-create via
the resource's share endpoint: groups map through the ``group`` remap (axis
omitted with a warning when the group phase is excluded), the org flag is
copied as-is, and users map by email. Users missing on the target are
skipped with a warning; service accounts and the source owner are skipped
silently. Users/groups listings are memoised per run in
``CloneContext.share_cache``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unstract.clone.context import CloneContext
    from unstract.clone.report import PhaseResult

logger = logging.getLogger(__name__)

SHARE_AXES: tuple[str, ...] = ("shared_users", "shared_groups", "shared_to_org")
# Platform-key identities; they exist per-org and never map across orgs.
SERVICE_ACCOUNT_EMAIL_SUFFIX = "@platform.internal"

_FETCH_FAILED = object()  # cache sentinel so a failing listing isn't re-hit


def is_service_account(row: dict[str, Any]) -> bool:
    """True if a user/member listing row is a service account.

    Email-suffix fallback covers older backends without the flag;
    mis-classification is benign — a service-account email never matches
    across orgs, so worst case is a spurious skip-warning.
    """
    flag = row.get("is_service_account")
    if flag is not None:
        return bool(flag)
    return (row.get("email") or "").lower().endswith(SERVICE_ACCOUNT_EMAIL_SUFFIX)


def _cached(ctx: CloneContext, key: str, build: Callable[[], Any]) -> Any:
    with ctx.share_cache_lock:
        if key in ctx.share_cache:
            return ctx.share_cache[key]
    # Build outside the lock (HTTP call). Two threads may race; the merge
    # below keeps a real result over a failure sentinel (listings are
    # read-only, so two successes are interchangeable).
    try:
        value = build()
    except Exception as e:
        logger.warning("share replication: %s listing failed: %s", key, e)
        value = _FETCH_FAILED
    with ctx.share_cache_lock:
        # Prefer a real result if we raced: a cached failure must not shadow a
        # peer's success, else share replication silently no-ops for the run.
        cached = ctx.share_cache.get(key, _FETCH_FAILED)
        if key not in ctx.share_cache or (
            cached is _FETCH_FAILED and value is not _FETCH_FAILED
        ):
            ctx.share_cache[key] = value
        return ctx.share_cache[key]


def source_user_by_id(ctx: CloneContext) -> dict[str, dict[str, Any]] | None:
    """Map source user pk (as str) -> listing row; ``None`` if the listing failed."""
    value = _cached(
        ctx,
        "source_user_by_id",
        lambda: {str(m["id"]): m for m in ctx.source.list_users() if m.get("email")},
    )
    return None if value is _FETCH_FAILED else value


def target_user_id_by_email(ctx: CloneContext) -> dict[str, int] | None:
    """Map lowercased email -> target user pk; ``None`` if the listing failed."""
    value = _cached(
        ctx,
        "target_user_id_by_email",
        lambda: {
            m["email"].lower(): int(m["id"])
            for m in ctx.target.list_users()
            if m.get("email")
        },
    )
    return None if value is _FETCH_FAILED else value


def _resolve_share_src(
    ctx: CloneContext,
    src: dict[str, Any],
    entity_label: str,
    result: PhaseResult,
    lock: threading.Lock,
    src_detail_fn: Callable[[], dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return the source row carrying share axes, fetching the detail once
    when ``src`` is a stripped list-row. ``None`` signals a fetch failure
    (already warned) — caller should abort.
    """
    if src_detail_fn is not None and not all(k in src for k in SHARE_AXES):
        try:
            return src_detail_fn()
        except Exception as e:
            logger.warning("share %s: source detail fetch failed: %s", entity_label, e)
            with lock:
                result.warnings.append(
                    f"share {entity_label}: source detail fetch failed — "
                    f"share state not replicated: {e}"
                )
            return None
    return src


def _build_share_payload(
    ctx: CloneContext,
    share_src: dict[str, Any],
    entity_label: str,
    result: PhaseResult,
    lock: threading.Lock,
    *,
    include_groups: bool = True,
) -> dict[str, Any] | None:
    """Map ``share_src``'s axes to target ids (groups via remap, users by
    email). Returns the share payload, or ``None`` when there is nothing to
    replicate. ``include_groups=False`` omits the group axis entirely (for
    entities whose serializer has no writable ``shared_groups``).
    """
    shared_to_org = bool(share_src.get("shared_to_org"))
    src_group_ids = list(share_src.get("shared_groups") or [])
    src_user_ids = list(share_src.get("shared_users") or [])
    owner_id = share_src.get("created_by")

    payload: dict[str, Any] = {"shared_to_org": shared_to_org}

    group_warnings: list[str] = []
    if include_groups:
        if src_group_ids and not ctx.options.includes("group"):
            # Axis omitted entirely so the target's group shares are untouched.
            group_warnings.append(
                f"share {entity_label}: group phase excluded — "
                f"{len(src_group_ids)} group share(s) not replicated"
            )
        else:
            mapped_groups: list[int] = []
            for gid in src_group_ids:
                tgt_gid = ctx.remap.resolve("group", str(gid))
                if tgt_gid is None:
                    group_warnings.append(
                        f"share {entity_label}: source group id {gid} has no "
                        "target mapping — skipped"
                    )
                else:
                    mapped_groups.append(int(tgt_gid))
            payload["shared_groups"] = mapped_groups
    elif src_group_ids:
        group_warnings.append(
            f"share {entity_label}: {len(src_group_ids)} group share(s) not "
            "supported by this entity — not replicated"
        )

    user_warnings: list[str] = []
    mapped_users: list[int] = []
    if src_user_ids:
        src_users = source_user_by_id(ctx)
        tgt_by_email = target_user_id_by_email(ctx)
        if src_users is None or tgt_by_email is None:
            user_warnings.append(
                f"share {entity_label}: users listing unavailable — "
                f"{len(src_user_ids)} user share(s) not replicated"
            )
        else:
            for uid in src_user_ids:
                if owner_id is not None and uid == owner_id:
                    continue  # ownership is server-managed on target
                row = src_users.get(str(uid))
                if row is None:
                    user_warnings.append(
                        f"share {entity_label}: source user id {uid} not in "
                        "source users listing — skipped"
                    )
                    continue
                if is_service_account(row):
                    continue
                email = row["email"]
                tgt_uid = tgt_by_email.get(email.lower())
                if tgt_uid is None:
                    user_warnings.append(
                        f"share {entity_label}: user {email} not found in "
                        "target org — skipped"
                    )
                else:
                    mapped_users.append(tgt_uid)
    payload["shared_users"] = mapped_users

    with lock:
        result.warnings.extend(group_warnings)
        result.warnings.extend(user_warnings)

    if not mapped_users and not payload.get("shared_groups") and not shared_to_org:
        logger.debug("share %s: nothing to replicate", entity_label)
        return None
    return payload


def replicate_share(
    ctx: CloneContext,
    *,
    apply_fn: Callable[[dict[str, Any]], Any],
    entity_label: str,
    src: dict[str, Any],
    result: PhaseResult,
    lock: threading.Lock,
    src_detail_fn: Callable[[], dict[str, Any]] | None = None,
    include_groups: bool = True,
) -> None:
    """Map ``src``'s share state and hand the payload to ``apply_fn`` (the
    entity-specific write — a ``/share/`` POST or a detail PATCH). Generic
    over the write mechanism so PATCH-shared cloud entities reuse the same
    user/group mapping. Never raises.
    """
    share_src = _resolve_share_src(ctx, src, entity_label, result, lock, src_detail_fn)
    if share_src is None:
        return
    payload = _build_share_payload(
        ctx, share_src, entity_label, result, lock, include_groups=include_groups
    )
    if payload is None:
        return
    if ctx.options.dry_run:
        logger.info("[dry-run] would share %s: %s", entity_label, payload)
        return
    try:
        apply_fn(payload)
    except Exception as e:
        logger.exception("Failed to apply share state for %s: %s", entity_label, e)
        with lock:
            result.failed += 1
            result.errors.append(f"share {entity_label}: {e}")
        return
    logger.info("shared %s: %s", entity_label, payload)


def apply_share_state(
    ctx: CloneContext,
    *,
    share_path: str,
    entity_label: str,
    src: dict[str, Any],
    result: PhaseResult,
    lock: threading.Lock,
    src_detail_fn: Callable[[], dict[str, Any]] | None = None,
) -> None:
    """Mirror ``src``'s share state onto the target resource via its
    ``/share/`` POST endpoint at ``share_path``. Thin wrapper over
    ``replicate_share`` preserving the original POST behavior.
    """
    replicate_share(
        ctx,
        apply_fn=lambda payload: ctx.target.share_resource(share_path, payload),
        entity_label=entity_label,
        src=src,
        result=result,
        lock=lock,
        src_detail_fn=src_detail_fn,
    )
