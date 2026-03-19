"""Client for Unstract Prompt Studio project promotion.

Provides methods to export, import, and sync Prompt Studio projects
across environments using Platform API Key (Bearer token) authentication.

Typical usage for promoting a project from dev to prod::

    from unstract.prompt_studio import PromptStudioClient

    source = PromptStudioClient(
        base_url="https://dev.unstract.com",
        api_key="<dev-api-key>",
        org_id="org_abc123",
    )
    target = PromptStudioClient(
        base_url="https://prod.unstract.com",
        api_key="<prod-api-key>",
        org_id="org_xyz789",
    )

    # Export from source
    export_data = source.export_project("<tool_id>")

    # Option A: Import as new project on target
    result = target.import_project(export_data, adapters={
        "llm_adapter_id": 42,
        "embedding_adapter_id": 15,
    })

    # Option B: Sync into existing project on target
    result = target.sync_prompts("<target_tool_id>", export_data)
"""

import json
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class PromptStudioClientError(Exception):
    """Raised when a Prompt Studio API call fails."""

    def __init__(self, message: str, status_code: int | None = None, response=None):
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class PromptStudioClient:
    """Client for Prompt Studio project promotion APIs.

    Args:
        base_url: Unstract instance URL (e.g., ``https://app.unstract.com``).
        api_key: Platform API Key UUID with ``read_write`` permission.
        org_id: Organization ID (e.g., ``org_abc123`` or ``mock_org``).
        timeout: Request timeout in seconds.
        verify: Whether to verify SSL certificates.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        org_id: str,
        timeout: int = 120,
        verify: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.org_id = org_id
        self.timeout = timeout
        self.verify = verify
        self._api_base = f"{self.base_url}/api/v1/unstract/{self.org_id}"

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _url(self, path: str) -> str:
        return f"{self._api_base}/{path.lstrip('/')}"

    def _request(
        self, method: str, path: str, **kwargs
    ) -> requests.Response:
        """Make an authenticated request and raise on HTTP errors."""
        url = self._url(path)
        merged_headers = {**self._headers, **kwargs.pop("headers", {})}
        kwargs["headers"] = merged_headers
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify)

        response = requests.request(method, url, **kwargs)

        if not response.ok:
            try:
                detail = response.json()
            except (ValueError, requests.JSONDecodeError):
                detail = response.text
            raise PromptStudioClientError(
                f"{method} {url} returned {response.status_code}: {detail}",
                status_code=response.status_code,
                response=response,
            )
        return response

    # ------------------------------------------------------------------
    # Core APIs
    # ------------------------------------------------------------------

    def list_projects(self) -> list[dict]:
        """List all Prompt Studio projects in the organization.

        Returns:
            List of project dicts with keys like ``tool_id``, ``tool_name``, etc.
        """
        resp = self._request("GET", "prompt-studio/")
        return resp.json()

    def get_project(self, tool_id: str) -> dict:
        """Get details of a single project.

        Args:
            tool_id: UUID of the Prompt Studio project.

        Returns:
            Project dict with full details including prompts.
        """
        resp = self._request("GET", f"prompt-studio/{tool_id}/")
        return resp.json()

    def export_project(self, tool_id: str) -> dict:
        """Export a project's full configuration as JSON.

        This is the ``project-transfer`` export — includes tool metadata,
        settings, prompts, and default profile settings. Suitable for
        importing on another environment.

        Args:
            tool_id: UUID of the project to export.

        Returns:
            Export JSON dict with keys: ``tool_metadata``, ``tool_settings``,
            ``default_profile_settings``, ``prompts``, ``export_metadata``.
        """
        resp = self._request("GET", f"prompt-studio/project-transfer/{tool_id}")
        return resp.json()

    def import_project(
        self,
        export_data: dict | str | Path,
        adapters: dict | None = None,
    ) -> dict:
        """Import a project from export JSON.

        Creates a new project on this environment. If a project with the
        same name exists, a unique name is generated.

        Args:
            export_data: Export JSON as a dict, a JSON string, a file path,
                or a ``Path`` object pointing to the export file.
            adapters: Optional dict of adapter IDs for the target environment::

                {
                    "llm_adapter_id": 42,
                    "vector_db_adapter_id": 7,
                    "embedding_adapter_id": 15,
                    "x2text_adapter_id": 3,
                }

        Returns:
            Import result dict with ``tool_id``, ``message``,
            ``needs_adapter_config``, and optional ``warning``.
        """
        # Resolve export_data to bytes for the multipart upload.
        # Read eagerly to avoid file handle leaks.
        if isinstance(export_data, Path):
            if not export_data.is_file():
                raise FileNotFoundError(f"Export file not found: {export_data}")
            with open(export_data, "rb") as f:
                content = f.read()
            filename = export_data.name
        elif isinstance(export_data, str) and Path(export_data).is_file():
            with open(export_data, "rb") as f:
                content = f.read()
            filename = Path(export_data).name
        elif isinstance(export_data, dict):
            content = json.dumps(export_data).encode()
            tool_name = (
                export_data.get("tool_metadata", {}).get("tool_name", "export")
            )
            filename = f"{tool_name}.json"
        elif isinstance(export_data, str):
            content = export_data.encode()
            filename = "export.json"
        else:
            raise PromptStudioClientError(
                "export_data must be a dict, JSON string, or file path"
            )

        files = {"file": (filename, content, "application/json")}
        data = {}
        if adapters:
            for key in (
                "llm_adapter_id",
                "vector_db_adapter_id",
                "embedding_adapter_id",
                "x2text_adapter_id",
            ):
                if key in adapters:
                    data[key] = adapters[key]

        resp = self._request("POST", "prompt-studio/project-transfer/", files=files, data=data)
        return resp.json()

    def sync_prompts(
        self,
        tool_id: str,
        export_data: dict,
        create_copy: bool = False,
    ) -> dict:
        """Sync prompts into an existing project.

        Rip-and-replace: deletes all existing prompts and recreates them
        from the export data. Tool settings are updated. Profiles, adapters,
        and uploaded documents are left untouched.

        Args:
            tool_id: UUID of the target project to sync into.
            export_data: Export JSON dict (must contain ``prompts`` key).
            create_copy: If ``True``, creates a backup clone before syncing.

        Returns:
            Sync result dict with ``prompts_created``, ``prompts_deleted``,
            ``tool_settings_updated``, and optional backup info.
        """
        payload = {"data": export_data, "create_copy": create_copy}
        resp = self._request(
            "POST",
            f"prompt-studio/{tool_id}/sync-prompts/",
            json=payload,
        )
        return resp.json()

    def check_deployment_usage(self, tool_id: str) -> dict:
        """Check if a project is used in any deployments.

        Useful before syncing to understand the blast radius.

        Args:
            tool_id: UUID of the project to check.

        Returns:
            Dict with ``is_used``, ``deployment_types``, and ``message``.
        """
        resp = self._request(
            "GET", f"prompt-studio/{tool_id}/check_deployment_usage/"
        )
        return resp.json()

    def upload_file(self, tool_id: str, file_path: str | Path) -> dict:
        """Upload a document to a Prompt Studio project.

        Args:
            tool_id: UUID of the project.
            file_path: Path to the file to upload.

        Returns:
            Upload response dict.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f.read())}
        resp = self._request(
            "POST", f"prompt-studio/file/{tool_id}", files=files
        )
        return resp.json()

    def get_default_triad(self) -> dict:
        """Get the default adapter triad for the current user.

        Returns:
            Dict with default adapter IDs (``llm``, ``vector_store``,
            ``embedding_model``, ``x2text``), or empty dict if not configured.
        """
        resp = self._request("GET", "adapter/default_triad/")
        return resp.json()

    def create_profile(
        self,
        tool_id: str,
        llm: str | None = None,
        vector_store: str | None = None,
        embedding_model: str | None = None,
        x2text: str | None = None,
        profile_name: str = "default",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        retrieval_strategy: str = "simple",
        similarity_top_k: int = 3,
        is_default: bool = True,
    ) -> dict:
        """Create a profile for a Prompt Studio project.

        If adapter IDs are not provided, the user's default triad is used.
        If this is the first profile on the project, it automatically becomes
        the default profile and is assigned to all prompts.

        Args:
            tool_id: UUID of the project.
            llm: LLM adapter instance ID. Falls back to default triad.
            vector_store: Vector DB adapter instance ID. Falls back to default.
            embedding_model: Embedding adapter instance ID. Falls back to default.
            x2text: X2Text adapter instance ID. Falls back to default.
            profile_name: Name for the profile.
            chunk_size: Chunk size for indexing.
            chunk_overlap: Chunk overlap for indexing.
            retrieval_strategy: Retrieval strategy (simple, subquestion, etc.).
            similarity_top_k: Number of top embeddings for context.
            is_default: Whether this profile should be the default.

        Returns:
            Created profile dict.
        """
        # Fill missing adapters from default triad
        if not all([llm, vector_store, embedding_model, x2text]):
            defaults = self.get_default_triad()
            llm = llm or defaults.get("default_llm_adapter")
            vector_store = vector_store or defaults.get("default_vector_db_adapter")
            embedding_model = embedding_model or defaults.get("default_embedding_adapter")
            x2text = x2text or defaults.get("default_x2text_adapter")

        missing = []
        if not llm:
            missing.append("llm")
        if not vector_store:
            missing.append("vector_store")
        if not embedding_model:
            missing.append("embedding_model")
        if not x2text:
            missing.append("x2text")
        if missing:
            raise PromptStudioClientError(
                f"Missing adapter IDs and no default triad configured: {missing}"
            )

        payload = {
            "prompt_studio_tool": tool_id,
            "profile_name": profile_name,
            "llm": llm,
            "vector_store": vector_store,
            "embedding_model": embedding_model,
            "x2text": x2text,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_strategy": retrieval_strategy,
            "similarity_top_k": similarity_top_k,
            "is_default": is_default,
        }
        resp = self._request(
            "POST", f"prompt-studio/profilemanager/{tool_id}", json=payload
        )
        return resp.json()

    def export_tool(self, tool_id: str) -> dict:
        """Export a tool for deployment (registry export).

        Always performs a force export to ensure the registry is up to date
        with the latest project state.

        Args:
            tool_id: UUID of the project to export for deployment.

        Returns:
            Export result dict.
        """
        resp = self._request(
            "POST",
            f"prompt-studio/export/{tool_id}",
            json={"force_export": True, "is_shared_with_org": True},
        )
        return resp.json()

    # ------------------------------------------------------------------
    # High-level promotion
    # ------------------------------------------------------------------

    def promote(
        self,
        tool_id: str,
        target: "PromptStudioClient",
        target_tool_id: str,
        create_copy: bool = True,
        export: bool = False,
    ) -> dict:
        """Promote a project from this environment to a target environment.

        Syncs prompts from a source project into an existing target project.
        The target project must already exist with a default profile
        configured (use ``import_project`` + ``create_profile`` for
        one-time setup).

        Orchestrates the promotion flow:

        1. **Export** the project from this (source) environment.
        2. **Sync** prompts into the target project (rip-and-replace).
        3. **Export for deployment** (optional): if ``export=True``, runs
           a force export on the target to update the tool registry.

        Args:
            tool_id: UUID of the source project to promote.
            target: A ``PromptStudioClient`` connected to the target env.
            target_tool_id: UUID of the existing target project to sync into.
            create_copy: If ``True`` (default), creates a backup clone
                on the target before syncing.
            export: If ``True``, export the tool for deployment on the
                target after syncing. Always uses force export.

        Returns:
            Dict with promotion result::

                {
                    "tool_id": "UUID of the target project",
                    "prompts_created": N,
                    "prompts_deleted": N,
                    "tool_settings_updated": true,
                    "backup_tool_id": "...",  # only if create_copy=True
                    "export_result": { ... }  # only if export=True
                }
        """
        # Step 1: Export from source
        logger.info("Exporting project %s from %s", tool_id, self.base_url)
        export_data = self.export_project(tool_id)
        tool_name = export_data.get("tool_metadata", {}).get("tool_name", "?")
        prompt_count = len(export_data.get("prompts", []))
        logger.info(
            "Exported '%s' with %d prompts", tool_name, prompt_count
        )

        # Step 2: Sync prompts into target
        logger.info(
            "Syncing prompts into %s on %s (backup=%s)",
            target_tool_id,
            target.base_url,
            create_copy,
        )
        result = target.sync_prompts(
            target_tool_id, export_data, create_copy=create_copy
        )
        result["tool_id"] = target_tool_id

        logger.info("Promotion complete: %s", result.get("message", ""))

        # Step 3: Optionally export for deployment
        if export:
            logger.info(
                "Exporting tool %s for deployment on %s",
                target_tool_id,
                target.base_url,
            )
            result["export_result"] = target.export_tool(target_tool_id)

        return result
