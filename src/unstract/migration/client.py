"""Thin Platform API client for the migration subpackage.

One ``PlatformClient`` instance per ``OrgEndpoint``. Methods are entity-
scoped (``list_adapters``, ``create_adapter``, ...) so call sites in phases
read like business logic, not HTTP plumbing.

URL shape: ``{base_url}/{api_path_prefix}/unstract/{organization_id}/<entity>/``
Auth: ``Authorization: Bearer <platform_api_key>``.
"""

from __future__ import annotations

import json as json_lib
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
        files: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        url = self._url(path)
        # Redact secrets from logs: only entity path + method, never body.
        logger.debug("%s %s", method, url)
        resp = self._session.request(
            method,
            url,
            params=params,
            json=json,
            files=files,
            data=data,
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

    def list_profiles(self, tool_id: str) -> list[dict[str, Any]]:
        """List ProfileManager rows for a tool.

        Migration reads this on the source only — to discover the
        default profile's adapter UUIDs so they can be remapped to
        target adapter ids for ``import_project``.
        """
        result = self._request(
            "GET", f"prompt-studio/prompt-studio-profile/{tool_id}/"
        )
        return result if isinstance(result, list) else result.get("results", [])

    def export_project(self, tool_id: str) -> dict[str, Any]:
        """Export a prompt-studio project as a portable JSON blob.

        Bundles ``tool_metadata``, ``tool_settings``,
        ``default_profile_settings``, ``prompts``, ``export_metadata`` in
        one shot — feed straight into ``import_project`` or
        ``sync_prompts`` on the target.
        """
        return self._request("GET", f"prompt-studio/project-transfer/{tool_id}")

    def import_project(
        self,
        export_data: dict[str, Any],
        adapter_ids: dict[str, str | None] | None = None,
    ) -> dict[str, Any]:
        """Import a prompt-studio project from an export blob.

        Backend creates the tool, builds the default ProfileManager from
        the supplied target-org adapter ids, and imports all prompts in
        one call. On name collision the backend silently uniquifies the
        new tool's name — callers should pre-check via
        ``list_custom_tools`` to avoid that.

        ``adapter_ids`` keys are the backend's form fields:
        ``llm_adapter_id``, ``vector_db_adapter_id``,
        ``embedding_adapter_id``, ``x2text_adapter_id``. All four
        required to wire the profile; otherwise backend falls back to
        a profile without adapters and flags ``needs_adapter_config``.
        """
        tool_name = (
            export_data.get("tool_metadata", {}).get("tool_name") or "export"
        )
        content = json_lib.dumps(export_data).encode()
        files = {"file": (f"{tool_name}.json", content, "application/json")}
        data: dict[str, Any] = {}
        if adapter_ids:
            for key in (
                "llm_adapter_id",
                "vector_db_adapter_id",
                "embedding_adapter_id",
                "x2text_adapter_id",
            ):
                val = adapter_ids.get(key)
                if val:
                    data[key] = val
        return self._request(
            "POST",
            "prompt-studio/project-transfer/",
            files=files,
            data=data,
        )

    def sync_prompts(
        self,
        tool_id: str,
        export_data: dict[str, Any],
        *,
        create_copy: bool = False,
    ) -> dict[str, Any]:
        """Rip-and-replace prompts on an existing target tool.

        Adopt path: target tool already exists with its own
        adapter-bound profiles. This overwrites its prompt set (and
        ``tool_settings``) from source; profiles and uploaded documents
        are left untouched.
        """
        payload = {"data": export_data, "create_copy": create_copy}
        return self._request(
            "POST", f"prompt-studio/{tool_id}/sync-prompts/", json=payload
        )

    def list_prompt_documents(self, tool_id: str) -> list[dict[str, Any]]:
        """List DocumentManager rows for a tool.

        Used by FilesPhase for target-side idempotency and source-side
        enumeration. Response items carry ``document_id``,
        ``document_name``, and ``tool`` (per the serializer's
        ``to_representation`` filter).
        """
        result = self._request(
            "GET", "prompt-document/", params={"tool_id": tool_id}
        )
        return result if isinstance(result, list) else result.get("results", [])

    def download_prompt_file(
        self, tool_id: str, file_name: str
    ) -> dict[str, Any]:
        """GET a Prompt Studio document by tool + filename.

        Returns the backend's ``{"data": ..., "mime_type": ...}`` envelope
        verbatim. PDFs come back as base64; text/csv as decoded utf-8;
        Excel returns a placeholder string (not real bytes) — callers must
        treat unsupported mime types as needing manual re-upload.
        """
        return self._request(
            "GET",
            f"prompt-studio/file/{tool_id}",
            params={"file_name": file_name},
        )

    def upload_prompt_file(
        self,
        tool_id: str,
        file_name: str,
        data: bytes,
        mime_type: str,
    ) -> dict[str, Any]:
        """Upload a file into a target Prompt Studio tool.

        Backend writes bytes to storage and creates a ``DocumentManager``
        row. The DM model has ``UniqueConstraint(document_name, tool)``,
        so callers must pre-check via ``list_prompt_documents`` to avoid
        an IntegrityError → 500 on re-runs.
        """
        files = {"file": (file_name, data, mime_type)}
        return self._request(
            "POST", f"prompt-studio/file/{tool_id}/", files=files
        )

    def export_custom_tool(self, tool_id: str, *, force: bool = True) -> Any:
        """Republish ``PromptStudioRegistry`` from the tool's current state.

        Called after import/sync so the registry row reflects the
        freshly landed prompts. Required for ToolInstancePhase to find
        a target registry id to remap.
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

    # ----- prompt studio registry -----

    def list_registries(
        self, *, custom_tool: str | None = None
    ) -> list[dict[str, Any]]:
        """List PromptStudioRegistry rows. The list endpoint returns nothing
        unless a filter is supplied; pass ``custom_tool`` to look up the
        registry id for a given tool.
        """
        params: dict[str, Any] = {}
        if custom_tool is not None:
            params["custom_tool"] = custom_tool
        result = self._request("GET", "prompt-studio/registry/", params=params)
        return result if isinstance(result, list) else result.get("results", [])

    # ----- tool instances -----

    def list_tool_instances(
        self, *, workflow_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List ToolInstance rows, optionally scoped to a workflow."""
        params: dict[str, Any] = {}
        if workflow_id is not None:
            params["workflow"] = workflow_id
        result = self._request("GET", "tool_instance/", params=params)
        return result if isinstance(result, list) else result.get("results", [])

    def create_tool_instance(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a tool instance (max 1 per workflow). The backend overwrites
        the ``metadata`` field with tool defaults — caller must PATCH after
        create to transfer source metadata.
        """
        return self._request("POST", "tool_instance/", json=payload)

    def update_tool_instance_metadata(
        self, instance_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """PATCH a tool instance's metadata. Backend resolves adapter names
        in the payload to local UUIDs via ``update_instance_metadata``.
        """
        return self._request(
            "PATCH", f"tool_instance/{instance_id}/", json={"metadata": metadata}
        )

    # ----- workflow endpoints -----

    def list_workflow_endpoints(
        self, *, workflow_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List workflow endpoints, optionally filtered by workflow id.

        The backend auto-creates one SOURCE and one DESTINATION endpoint
        per workflow, so a workflow filter typically returns exactly two
        rows.
        """
        params: dict[str, Any] = {}
        if workflow_id is not None:
            params["workflow"] = workflow_id
        result = self._request("GET", "workflow/endpoint/", params=params)
        return result if isinstance(result, list) else result.get("results", [])

    def update_workflow_endpoint(
        self, endpoint_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "PATCH", f"workflow/endpoint/{endpoint_id}/", json=payload
        )

    # ----- pipelines (ETL / TASK) -----

    def list_pipelines(
        self,
        *,
        name: str | None = None,
        pipeline_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List pipelines in this org, optionally filtered by exact name
        and/or pipeline_type (``ETL`` / ``TASK`` / ``APP``).
        """
        params: dict[str, Any] = {}
        if name is not None:
            params["pipeline_name"] = name
        if pipeline_type is not None:
            params["type"] = pipeline_type
        result = self._request("GET", "pipeline/", params=params)
        return result if isinstance(result, list) else result.get("results", [])

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        return self._request("GET", f"pipeline/{pipeline_id}/")

    def create_pipeline(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a pipeline. Backend force-sets ``active=True`` and auto-creates
        a single active API key on the new pipeline.
        """
        return self._request("POST", "pipeline/", json=payload)

    def update_pipeline(
        self, pipeline_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request("PATCH", f"pipeline/{pipeline_id}/", json=payload)

    # ----- API deployments -----

    def list_api_deployments(
        self,
        *,
        api_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List API deployments in this org, optionally filtered by exact api_name."""
        params: dict[str, Any] = {}
        if api_name is not None:
            params["api_name"] = api_name
        result = self._request("GET", "api/deployment/", params=params)
        return result if isinstance(result, list) else result.get("results", [])

    def get_api_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"api/deployment/{deployment_id}/")

    def create_api_deployment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create an API deployment. Backend auto-creates a single active key
        and returns it in the response under ``api_key``.
        """
        return self._request("POST", "api/deployment/", json=payload)

    def update_api_deployment(
        self, deployment_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request(
            "PATCH", f"api/deployment/{deployment_id}/", json=payload
        )

    # ----- API keys (per pipeline / deployment) -----

    def list_pipeline_keys(self, pipeline_id: str) -> list[dict[str, Any]]:
        """List API keys belonging to a pipeline."""
        result = self._request("GET", f"api/keys/pipeline/{pipeline_id}/")
        return result if isinstance(result, list) else result.get("results", [])

    def list_api_deployment_keys(self, deployment_id: str) -> list[dict[str, Any]]:
        """List API keys belonging to an API deployment."""
        result = self._request("GET", f"api/keys/api/{deployment_id}/")
        return result if isinstance(result, list) else result.get("results", [])

    def create_api_key(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create an extra API key tied to a pipeline or deployment.

        Used to mirror non-default keys (e.g. an additional rotated key)
        on the target. The ``api_key`` UUID itself is server-generated
        and cannot be carried over from source.
        """
        return self._request("POST", "api/keys/api/", json=payload)
