"""Thin Platform API client for the migration subpackage.

One ``PlatformClient`` instance per ``OrgEndpoint``. Methods are entity-
scoped (``list_adapters``, ``create_adapter``, ...) so call sites in phases
read like business logic, not HTTP plumbing.

URL shape: ``{base_url}/{api_path_prefix}/unstract/{organization_id}/<entity>/``
Auth: ``Authorization: Bearer <platform_api_key>``.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from unstract.migration.context import OrgEndpoint
from unstract.migration.exceptions import PlatformAPIError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60


class PlatformClient:
    """HTTP client scoped to a single org via its Platform API key."""

    def __init__(self, endpoint: OrgEndpoint, timeout: int = DEFAULT_TIMEOUT, verify: bool = True):
        self.endpoint = endpoint
        self.timeout = timeout
        self.verify = verify
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {endpoint.platform_key}",
                "Accept": "application/json",
            }
        )

    def _url(self, path: str) -> str:
        base = self.endpoint.base_url.rstrip("/")
        api_prefix = self.endpoint.api_path_prefix.strip("/")
        prefix = f"/{api_prefix}/unstract/{self.endpoint.organization_id}/"
        return base + prefix + path.lstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        url = self._url(path)
        # Redact secrets from logs: only entity path + method, never body.
        logger.debug("%s %s", method, url)
        resp = self._session.request(
            method,
            url,
            params=params,
            json=json,
            timeout=self.timeout,
            verify=self.verify,
        )
        if not 200 <= resp.status_code < 300:
            raise PlatformAPIError(
                f"{method} {path} returned {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text[:2000],
            )
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # ----- adapters -----

    def list_adapters(
        self,
        *,
        name: str | None = None,
        adapter_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List adapters in this org, optionally filtered by name and/or type."""
        params: dict[str, Any] = {}
        if name is not None:
            params["adapter_name"] = name
        if adapter_type is not None:
            params["adapter_type"] = adapter_type
        result = self._request("GET", "adapter/", params=params)
        # DRF ModelViewSet.list returns a bare list (no pagination on this endpoint).
        return result if isinstance(result, list) else result.get("results", [])

    def get_adapter(self, adapter_pk: str) -> dict[str, Any]:
        return self._request("GET", f"adapter/{adapter_pk}/")

    def create_adapter(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "adapter/", json=payload)

    # ----- connectors -----

    def list_connectors(
        self,
        *,
        name: str | None = None,
        connector_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List connectors in this org, optionally filtered by name and/or type."""
        params: dict[str, Any] = {}
        if name is not None:
            params["connector_name"] = name
        if connector_type is not None:
            params["connector_type"] = connector_type
        result = self._request("GET", "connector/", params=params)
        return result if isinstance(result, list) else result.get("results", [])

    def get_connector(self, connector_pk: str) -> dict[str, Any]:
        return self._request("GET", f"connector/{connector_pk}/")

    def create_connector(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "connector/", json=payload)

    # ----- tags -----

    def list_tags(self, *, name: str | None = None) -> list[dict[str, Any]]:
        """List tags in this org, optionally filtered by exact name."""
        params: dict[str, Any] = {}
        if name is not None:
            params["name"] = name
        result = self._request("GET", "tags/", params=params)
        # Tags endpoint uses pagination — accept either bare list or paginated envelope.
        return result if isinstance(result, list) else result.get("results", [])

    def create_tag(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "tags/", json=payload)
