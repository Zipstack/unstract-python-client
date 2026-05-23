"""JSON walker that rewrites embedded source UUIDs to target UUIDs.

Used by phases whose payloads carry foreign-key UUIDs inside JSON fields
(e.g. ``tool_instance.metadata``). Unknown UUIDs pass through untouched so
we don't accidentally rewrite an unrelated identifier that just happens
to look like a UUID.
"""

from __future__ import annotations

import re
from typing import Any

from unstract.migration.context import RemapTable

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def remap_uuids(obj: Any, remap: RemapTable) -> Any:
    """Walk a JSON-shaped value; replace any string that looks like a UUID
    AND has a known mapping. Unknown UUIDs pass through untouched.
    """
    if isinstance(obj, dict):
        return {k: remap_uuids(v, remap) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_uuids(v, remap) for v in obj]
    if isinstance(obj, str) and UUID_RE.match(obj):
        return remap.resolve_any(obj) or obj
    return obj
