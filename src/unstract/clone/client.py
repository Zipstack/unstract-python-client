"""Thin Platform API client for the clone subpackage.

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

from unstract.clone.context import OrgEndpoint
from unstract.clone.exceptions import PlatformAPIError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60


class PlatformClient:
    """HTTP client scoped to a single org via its Platform API key."""

    def __init__(
        self, endpoint: OrgEndpoint, timeout: int = DEFAULT_TIMEOUT, verify: bool = True
    ):
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

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._session.close()

    def __enter__(self) -> "PlatformClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

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
        """Return the set of fields the backend's POST accepts.

        Reads it from an ``OPTIONS`` response (``actions.POST``) once per
        path and caches the result. Read-only fields are already absent
        from ``actions.POST``, so the returned set is exactly the writable
        subset.
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

    def probe(self, path: str) -> bool:
        """Capability probe: is this feature's route installed on this deployment?

        GET ``path`` and return True on 200, False on 404 (route absent =
        feature not built into this deployment). Any other status / transport
        error re-raises — a real failure must not look like "feature missing".
        """
        try:
            self._request("GET", path)
        except PlatformAPIError as e:
            if e.status_code == 404:
                return False
            raise
        return True

    # ----- org users & groups -----

    def list_users(self) -> list[dict[str, Any]]:
        """List org member rows (each carries ``id`` and ``email``)."""
        result = self._request("GET", "users/")
        return (result or {}).get("members", [])

    def list_groups(self) -> list[dict[str, Any]]:
        """List org groups; no server-side name filter — callers match in memory."""
        result = self._request("GET", "groups/")
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def create_group(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a group; response has no ``id`` — re-list to learn the pk."""
        return self._request("POST", "groups/", json=payload)

    def list_group_members(self, group_id: Any) -> list[dict[str, Any]]:
        """List a group's member rows (each carries ``email``)."""
        result = self._request("GET", f"groups/{group_id}/members/")
        return result if isinstance(result, list) else result.get("results", [])

    def add_group_members(self, group_id: Any, user_ids: list[int]) -> Any:
        """Bulk-add members by user pk; idempotent server-side."""
        return self._request(
            "POST", f"groups/{group_id}/members/", json={"user_ids": user_ids}
        )

    # ----- sharing -----

    def share_resource(self, share_path: str, payload: dict[str, Any]) -> Any:
        """Replace-style share update; axes omitted from ``payload`` are untouched."""
        return self._request("POST", share_path, json=payload)

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
        """Fetch a single prompt-studio project.

        Notably includes ``output`` (the default document id the UI
        selects on load).
        """
        return self._request("GET", f"prompt-studio/{tool_id}/")

    def update_custom_tool(self, tool_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """PATCH a prompt-studio project. Used to set ``output`` (the
        default doc id) after the files phase uploads documents."""
        return self._request("PATCH", f"prompt-studio/{tool_id}/", json=body)

    def list_profiles(self, tool_id: str) -> list[dict[str, Any]]:
        """List the adapter profiles for a tool.

        The clone reads this on the source only — to discover the
        default profile's adapter UUIDs so they can be remapped to
        target adapter ids for ``import_project``.
        """
        result = self._request("GET", f"prompt-studio/prompt-studio-profile/{tool_id}/")
        return result if isinstance(result, list) else result.get("results", [])

    def list_prompts(self, tool_id: str) -> list[dict[str, Any]]:
        """List a tool's prompts (``prompt_id`` + ``prompt_key`` per row).

        Used to map source prompt ids to the target prompts created by
        ``import_project`` / ``sync_prompts`` (matched by ``prompt_key``),
        so prompt-scoped cloud config can remap its FKs.
        """
        result = self._request(
            "GET", "prompt-studio/prompt/", params={"tool_id": tool_id}
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

        Creates the tool, the default adapter profile from the supplied
        target-org adapter ids, and all prompts in one call. On name
        collision the new tool comes back with a uniquified name — callers
        should pre-check via ``list_custom_tools`` to avoid that.

        ``adapter_ids`` keys are the form fields ``llm_adapter_id``,
        ``vector_db_adapter_id``, ``embedding_adapter_id``,
        ``x2text_adapter_id``. All four required to wire the profile;
        otherwise the response flags ``needs_adapter_config``.
        """
        tool_name = export_data.get("tool_metadata", {}).get("tool_name") or "export"
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
        """List a tool's prompt documents.

        Used by FilesPhase for target-side idempotency and source-side
        enumeration. Response items carry ``document_id``,
        ``document_name``, and ``tool``.
        """
        result = self._request(
            "GET", "prompt-studio/prompt-document/", params={"tool_id": tool_id}
        )
        return result if isinstance(result, list) else result.get("results", [])

    def download_prompt_file(self, tool_id: str, document_id: str) -> dict[str, Any]:
        """GET a Prompt Studio document by tool + document id.

        The endpoint resolves the filename from the document id, so the
        SDK passes the ``document_id`` it already has from
        ``list_prompt_documents`` rather than reposting the filename.
        Returns ``{"data": ..., "mime_type": ...}`` — PDFs base64,
        text/csv utf-8, Excel placeholder.
        """
        return self._request(
            "GET",
            f"prompt-studio/file/{tool_id}",
            params={"document_id": document_id},
        )

    def upload_prompt_file(
        self,
        tool_id: str,
        file_name: str,
        data: bytes,
        mime_type: str,
    ) -> dict[str, Any]:
        """Upload a file into a target Prompt Studio tool.

        Filenames are unique per tool, so callers must pre-check via
        ``list_prompt_documents`` to avoid a duplicate-name error on
        re-runs.
        """
        files = {"file": (file_name, data, mime_type)}
        return self._request("POST", f"prompt-studio/file/{tool_id}", files=files)

    def export_custom_tool(self, tool_id: str, *, force: bool = True) -> Any:
        """Republish the tool's registry entry from its current state.

        Called after import/sync so the registry reflects the freshly
        landed prompts. Required for ToolInstancePhase to find a target
        registry id to remap.
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
        """List prompt-studio registry rows. The list endpoint returns nothing
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
        """Create a tool instance (max 1 per workflow). The created row comes
        back with default ``metadata`` — caller must PATCH after create to
        transfer source metadata.
        """
        return self._request("POST", "tool_instance/", json=payload)

    def update_tool_instance_metadata(
        self, instance_id: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """PATCH a tool instance's metadata. Adapter names in the payload are
        resolved to local UUIDs server-side.
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
        return self._request("PATCH", f"workflow/endpoint/{endpoint_id}/", json=payload)

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
        """Create a pipeline. The new pipeline comes back active with a single
        active API key auto-provisioned.
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
        return self._request("PATCH", f"api/deployment/{deployment_id}/", json=payload)

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

    # ----- lookups (cloud-only) -----

    def list_lookup_definitions(self) -> list[dict[str, Any]]:
        """List lookup definitions in this org. Also the capability-probe path."""
        result = self._request("GET", "lookups/definitions/")
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def get_lookup_definition(self, lookup_id: str) -> dict[str, Any]:
        """Fetch a lookup definition's detail.

        Detail inlines the draft content: ``prompt_template``,
        ``draft_version_id``, ``input_vars``, and ``adapters`` (a dict with
        ``llm`` / ``x2text`` adapter UUIDs, either possibly ``None``).
        """
        return self._request("GET", f"lookups/definitions/{lookup_id}/")

    def create_lookup_definition(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a lookup definition. The new definition comes with an empty
        DRAFT version and default adapters; populate it via the
        draft/adapters/file endpoints below.
        """
        return self._request("POST", "lookups/definitions/", json=payload)

    def update_lookup_draft_template(
        self, lookup_id: str, prompt_template: str
    ) -> dict[str, Any]:
        """Set the draft version's prompt template."""
        return self._request(
            "PATCH",
            f"lookups/definitions/{lookup_id}/draft/",
            json={"prompt_template": prompt_template},
        )

    def update_lookup_draft_adapters(
        self, lookup_id: str, adapters: dict[str, str]
    ) -> dict[str, Any]:
        """Set the draft version's LLM and/or X2Text adapters by target UUID.

        ``adapters`` may carry either or both of ``llm`` / ``x2text``; absent
        keys leave the existing draft adapter untouched.
        """
        return self._request(
            "PATCH",
            f"lookups/definitions/{lookup_id}/adapters/",
            json=adapters,
        )

    def list_lookup_files(self, lookup_id: str) -> list[dict[str, Any]]:
        """List a lookup's draft reference files (rows carry ``file_id``,
        ``file_name``, ``file_size``).
        """
        result = self._request("GET", f"lookups/definitions/{lookup_id}/files/")
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def download_lookup_file(self, lookup_id: str, file_id: str) -> bytes:
        """Download a reference file's original bytes.

        Returns raw bytes — the content route serves an ``HttpResponse`` body
        (not a JSON envelope), so this bypasses the JSON-decoding request path.
        """
        url = self._url(f"lookups/definitions/{lookup_id}/files/{file_id}/content/")
        logger.debug("GET %s", url)
        resp = self._session.get(url, timeout=self.timeout, verify=self.verify)
        if not 200 <= resp.status_code < 300:
            raise PlatformAPIError(
                f"GET lookups/definitions/{lookup_id}/files/{file_id}/content/ "
                f"returned {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text[:2000],
            )
        return resp.content

    def upload_lookup_file(
        self, lookup_id: str, file_name: str, data: bytes, mime_type: str
    ) -> dict[str, Any]:
        """Upload a reference file into a lookup's draft version.

        Filenames are unique per draft version, so callers pre-check via
        ``list_lookup_files`` to avoid a 409.
        """
        files = {"file": (file_name, data, mime_type)}
        return self._request(
            "POST", f"lookups/definitions/{lookup_id}/files/", files=files
        )

    def list_lookup_assignments(self) -> list[dict[str, Any]]:
        """List prompt-lookup assignment rows in this org.

        Each row carries ``assignment_id``, ``prompt`` (source prompt uuid),
        ``version`` (source lookup-version uuid), ``lookup_definition``
        (source lookup_id), ``is_draft_version``, and ``variable_mappings``.
        """
        result = self._request("GET", "lookups/assignments/")
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def create_lookup_assignment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a prompt-lookup assignment.

        Writable: ``prompt``, ``lookup_definition`` (required), ``version``,
        ``variable_mappings``. At most one assignment per prompt, so callers
        pre-check target assignments.
        """
        return self._request("POST", "lookups/assignments/", json=payload)

    def update_lookup_share(
        self, lookup_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Replicate share state onto a lookup via its detail PATCH.

        ``payload`` carries ``shared_to_org`` + ``shared_users`` (target user
        pks). Lookups expose no group-sharing axis, so no ``shared_groups``.
        """
        return self._request(
            "PATCH", f"lookups/definitions/{lookup_id}/", json=payload
        )

    def list_lookup_versions(self, lookup_id: str) -> list[dict[str, Any]]:
        """List a lookup's versions (draft + published).

        Rows carry ``version_id``, ``is_draft``, ``version_number``,
        ``version_name``; the detail (``get_lookup_version``) inlines content.
        """
        result = self._request(
            "GET", f"lookups/definitions/{lookup_id}/versions/"
        )
        if isinstance(result, list):
            return result
        # This endpoint wraps rows as {"versions": [...], "next_version_number"}.
        return (result or {}).get("versions", (result or {}).get("results", []))

    def get_lookup_version(
        self, lookup_id: str, version_id: str
    ) -> dict[str, Any]:
        """Fetch a version's detail (``prompt_template``, adapters, files)."""
        return self._request(
            "GET", f"lookups/definitions/{lookup_id}/versions/{version_id}/"
        )

    def download_lookup_version_file(
        self, lookup_id: str, version_id: str, file_id: str
    ) -> bytes:
        """Download a published version's reference-file bytes (raw body)."""
        path = (
            f"lookups/definitions/{lookup_id}/versions/{version_id}/"
            f"files/{file_id}/content/"
        )
        url = self._url(path)
        logger.debug("GET %s", url)
        resp = self._session.get(url, timeout=self.timeout, verify=self.verify)
        if not 200 <= resp.status_code < 300:
            raise PlatformAPIError(
                f"GET {path} returned {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text[:2000],
            )
        return resp.content

    def publish_lookup_version(
        self, lookup_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Freeze the current draft into a published version.

        ``payload`` carries ``version_name`` (+ optional ``rebind_assignments``).
        Returns the new published version (``version_id``). Used to replay a
        source lookup's published-version history onto the target.
        """
        return self._request(
            "POST", f"lookups/definitions/{lookup_id}/versions/", json=payload
        )

    # ----- manual review / HITL (cloud-only) -----
    #
    # Each workflow holds one review-rule row per ``rule_type`` (DB / API)
    # and one HITL-settings row. The workflow-scoped GET routes take the
    # workflow id in the URL path and wrap the row in ``{"data": ...}``;
    # they 404 (rules) / 500 (settings) when none exists — callers treat a
    # missing row as "nothing to clone", not an error.

    MR_RULE_TYPES: tuple[str, ...] = ("DB", "API")

    def get_review_rule(
        self, workflow_id: str, rule_type: str
    ) -> dict[str, Any] | None:
        """Fetch a workflow's review rule for one ``rule_type``.

        Returns the rule dict (with nested ``confidence_filters``) or ``None``
        when no rule of that type exists (backend answers 404).
        """
        try:
            body = self._request(
                "GET",
                f"manual_review/rule_engine/workflow/{workflow_id}/",
                params={"rule_type": rule_type},
            )
        except PlatformAPIError as e:
            if e.status_code == 404:
                return None
            raise
        return (body or {}).get("data")

    def create_review_rule(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a review rule (+ nested ``confidence_filters``).

        Writable: ``workflow`` (required), ``rule_type``, ``percentage``,
        ``rule_string``, ``rule_json``, ``rule_logic``, ``confidence_filters``.
        Unique per workflow + ``rule_type`` within the org.
        """
        return self._request("POST", "manual_review/rule_engine/", json=payload)

    def get_review_settings(self, workflow_id: str) -> dict[str, Any] | None:
        """Fetch a workflow's review settings, or ``None`` if absent.

        The route answers 500 (not 404) when no row exists, so only a 500 is
        treated as "no settings to clone". Other errors (401/403/429) must
        surface — suppressing them would silently drop configured settings.
        """
        try:
            body = self._request(
                "GET", f"manual_review/settings/workflow/{workflow_id}/"
            )
        except PlatformAPIError as e:
            if e.status_code == 500:
                return None
            raise
        return (body or {}).get("data")

    def create_review_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a review settings row.

        Writable: ``workflow`` (one per workflow, required), ``sync_with``,
        ``ttl_hours``.
        """
        return self._request("POST", "manual_review/settings/", json=payload)

    def list_auto_approval_settings(self) -> list[dict[str, Any]]:
        """List org-level auto-approval settings (0 or 1 per org).

        Returns 200 bare with no query params, so it doubles as the
        manual-review capability probe path.
        """
        result = self._request("GET", "manual_review/auto_approval_settings/")
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def create_auto_approval_settings(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Create org-level auto-approval settings.

        Writable: ``auto_approved_document_classes``, ``auto_approved_users``.
        ``organization`` is server-set. Unique per org.
        """
        return self._request(
            "POST", "manual_review/auto_approval_settings/", json=payload
        )

    def list_review_api_keys(self) -> list[dict[str, Any]]:
        """List review API keys in this org."""
        result = self._request("GET", "manual_review/api/keys/")
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def create_review_api_key(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a review API key. The ``api_key`` secret is server-minted
        and cannot be carried over from source.
        Writable: ``class_name``, ``description``, ``is_active``.
        """
        return self._request("POST", "manual_review/api/key/", json=payload)

    # ----- agentic studio (cloud-only) -----

    def list_agentic_projects(self) -> list[dict[str, Any]]:
        """List agentic projects in this org. Also the capability-probe path.

        Rows carry ``id``, ``name``, ``description``, the four adapter FK ids
        (``llm_connector_id`` / ``agent_llm_connector_id`` /
        ``lightweight_llm_connector_id`` / ``text_extractor_connector_id``),
        and ``canary_fields``.
        """
        result = self._request("GET", "agentic/projects/")
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def create_agentic_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create an agentic project. Returns the created row (carries ``id``)."""
        return self._request("POST", "agentic/projects/", json=payload)

    def list_agentic_prompt_versions(
        self, *, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List agentic prompt versions, optionally scoped to a project.

        Rows carry ``id``, ``project``, ``version``, ``prompt_text``,
        ``accuracy``, ``is_active``, and the self-FK ``parent_version``.
        """
        params: dict[str, Any] = {}
        if project_id is not None:
            params["project_id"] = project_id
        result = self._request("GET", "agentic/prompt-versions/", params=params)
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def create_agentic_prompt_version(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create an agentic prompt version (flat endpoint, ``project`` in body)."""
        return self._request("POST", "agentic/prompt-versions/", json=payload)

    def list_agentic_schemas(
        self, *, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List agentic schemas, optionally scoped to a project.

        Rows carry ``id``, ``project``, ``json_schema``, ``version``,
        ``is_active``.
        """
        params: dict[str, Any] = {}
        if project_id is not None:
            params["project_id"] = project_id
        result = self._request("GET", "agentic/schemas/", params=params)
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def create_agentic_schema(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create an agentic schema (flat endpoint, ``project`` in body)."""
        return self._request("POST", "agentic/schemas/", json=payload)

    def list_agentic_settings(self) -> list[dict[str, Any]]:
        """List agentic settings. Org-wide key/value rows (no project FK)."""
        result = self._request("GET", "agentic/settings/")
        return result if isinstance(result, list) else (result or {}).get("results", [])

    def create_agentic_setting(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create an org-wide agentic setting."""
        return self._request("POST", "agentic/settings/", json=payload)

    def update_agentic_setting(
        self, setting_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """PATCH an existing agentic setting by id."""
        return self._request("PATCH", f"agentic/settings/{setting_id}/", json=payload)

    def export_agentic_project(self, project_id: str, *, force: bool = True) -> Any:
        """Republish the project's registry entry from its active schema +
        prompt. Mirror of ``export_custom_tool``.

        Requires an active schema and active prompt; ``force_export``
        bypasses the completion check. Caller re-reads the registry to learn
        the new id.
        """
        return self._request(
            "POST",
            f"agentic/projects/{project_id}/export/",
            json={
                "is_shared_with_org": False,
                "user_ids": [],
                "force_export": force,
            },
        )

    def list_agentic_registries(
        self, *, agentic_project: str | None = None
    ) -> list[dict[str, Any]]:
        """List agentic registry rows. The list endpoint returns nothing
        unless a filter is supplied; pass ``agentic_project`` to look up the
        registry id for a given project.
        """
        params: dict[str, Any] = {}
        if agentic_project is not None:
            params["agentic_project"] = agentic_project
        result = self._request("GET", "agentic-studio-registry/", params=params)
        return result if isinstance(result, list) else (result or {}).get("results", [])
