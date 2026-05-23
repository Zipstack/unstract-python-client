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
        # Cache the OPTIONS-derived writable-field set per entity path.
        # Backend serializer is the single source of truth; we read it once.
        self._post_schema_cache: dict[str, frozenset[str]] = {}

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

    def get_post_schema(self, entity_path: str) -> frozenset[str]:
        """Return the set of fields the backend's POST serializer accepts.

        Reads it from a DRF ``OPTIONS`` response (``actions.POST``) once
        per path and caches the result. DRF ``SimpleMetadata`` already
        excludes ``read_only`` fields from ``actions.POST``, so the
        returned set is exactly the writable subset.
        """
        cached = self._post_schema_cache.get(entity_path)
        if cached is not None:
            return cached
        body = self._request("OPTIONS", entity_path)
        actions = (body or {}).get("actions") or {}
        post_block = actions.get("POST") or {}
        writable = frozenset(
            name for name, meta in post_block.items() if not meta.get("read_only")
        )
        self._post_schema_cache[entity_path] = writable
        return writable

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

    # ----- custom tools (prompt studio) -----

    def list_custom_tools(self) -> list[dict[str, Any]]:
        """List all prompt-studio projects in this org. No name filter."""
        result = self._request("GET", "prompt-studio/")
        return result if isinstance(result, list) else result.get("results", [])

    def get_custom_tool(self, tool_id: str) -> dict[str, Any]:
        """Tool detail; response includes embedded ``prompts`` + ``default_profile``."""
        return self._request("GET", f"prompt-studio/{tool_id}/")

    def create_custom_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a custom tool. Backend also auto-creates one default ProfileManager."""
        return self._request("POST", "prompt-studio/", json=payload)

    def export_custom_tool(self, tool_id: str, *, force: bool = True) -> Any:
        """Republish ``PromptStudioRegistry`` from the tool's current target state.

        Used after profile+prompt reconciliation so the registry row is
        rebuilt without the SDK ever carrying ``tool_metadata`` across orgs.
        """
        return self._request(
            "POST",
            f"prompt-studio/export/{tool_id}",
            json={
                "is_shared_with_org": False,
                "user_id": [],
                "force_export": force,
            },
        )

    # ----- profile managers -----

    def list_profiles(self, tool_id: str) -> list[dict[str, Any]]:
        """List ProfileManager rows for a tool via the per-tool list action."""
        result = self._request(
            "GET", f"prompt-studio/prompt-studio-profile/{tool_id}/"
        )
        return result if isinstance(result, list) else result.get("results", [])

    def create_profile(self, tool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to ``prompt-studio/profilemanager/{tool_id}`` (no trailing slash)."""
        return self._request(
            "POST", f"prompt-studio/profilemanager/{tool_id}", json=payload
        )

    def delete_profile(self, profile_id: str) -> None:
        self._request("DELETE", f"profile-manager/{profile_id}/")

    def set_default_profile(self, tool_id: str, profile_id: str) -> Any:
        """Mark a single profile as default for this tool (zeros the rest)."""
        return self._request(
            "PATCH",
            f"prompt-studio/prompt-studio-profile/{tool_id}/",
            json={"default_profile": profile_id},
        )

    # ----- prompts -----

    def list_prompts(self, *, tool_id: str) -> list[dict[str, Any]]:
        """List prompts filtered by tool_id (FilterHelper-backed)."""
        result = self._request("GET", "prompt/", params={"tool_id": tool_id})
        return result if isinstance(result, list) else result.get("results", [])

    def create_prompt(self, tool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to ``prompt-studio/prompt-studio-prompt/{tool_id}/`` (create_prompt action)."""
        return self._request(
            "POST", f"prompt-studio/prompt-studio-prompt/{tool_id}/", json=payload
        )

    # ----- workflows -----

    def list_workflows(self, *, name: str | None = None) -> list[dict[str, Any]]:
        """List workflows in this org, optionally filtered by exact name."""
        params: dict[str, Any] = {}
        if name is not None:
            params["workflow_name"] = name
        result = self._request("GET", "workflow/", params=params)
        return result if isinstance(result, list) else result.get("results", [])

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        return self._request("GET", f"workflow/{workflow_id}/")

    def create_workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a workflow. Backend auto-creates empty WorkflowEndpoints for it."""
        return self._request("POST", "workflow/", json=payload)
