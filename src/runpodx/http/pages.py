from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runpodx.http.transport import HttpTransport


def collect(
    transport: HttpTransport,
    path: str,
    key: str,
    params: dict[str, Any] | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Every item of a REST v2 list endpoint.

    v2 wraps a list in an object (``{"pods": [...], "pagination": {...}}``) and pages
    it; this follows ``pagination.nextCursor`` until ``hasNextPage`` is false. A bare
    list (v1's shape, or a test double's) is returned as it is.
    """
    query: dict[str, Any] = dict(params or {})
    query.setdefault("limit", limit)
    items: list[dict] = []
    while True:
        raw = transport.request("GET", path, params=query)
        if isinstance(raw, list):
            return items + raw
        items.extend(raw.get(key) or [])
        page = raw.get("pagination") or {}
        cursor = page.get("nextCursor")
        if not page.get("hasNextPage") or not cursor:
            return items
        query = {**query, "cursor": cursor}
