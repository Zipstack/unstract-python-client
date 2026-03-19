"""Tests for PromptStudioClient."""

import json
from unittest.mock import MagicMock, patch

import pytest

from unstract.prompt_studio.client import PromptStudioClient, PromptStudioClientError

MOCK_BASE_URL = "https://test.unstract.com"
MOCK_API_KEY = "test-api-key-uuid"
MOCK_ORG_ID = "org_test123"
MOCK_TOOL_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

MOCK_EXPORT_DATA = {
    "tool_metadata": {
        "tool_name": "Test Project",
        "description": "A test project",
        "author": "tester",
        "icon": None,
    },
    "tool_settings": {
        "preamble": "You are a helpful assistant.",
        "postamble": "Return only extracted info.",
    },
    "default_profile_settings": {
        "chunk_size": 1000,
        "chunk_overlap": 100,
    },
    "prompts": [
        {
            "prompt_key": "name",
            "prompt": "What is the name?",
            "active": True,
            "enforce_type": "text",
            "sequence_number": 1,
            "prompt_type": "PROMPT",
        },
        {
            "prompt_key": "date",
            "prompt": "What is the date?",
            "active": True,
            "enforce_type": "date",
            "sequence_number": 2,
            "prompt_type": "PROMPT",
        },
    ],
    "export_metadata": {
        "exported_at": "2026-03-19T00:00:00Z",
        "tool_id": MOCK_TOOL_ID,
    },
}


@pytest.fixture
def client():
    return PromptStudioClient(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        org_id=MOCK_ORG_ID,
    )


class TestClientInit:
    def test_url_construction(self, client):
        assert client._api_base == f"{MOCK_BASE_URL}/api/v1/unstract/{MOCK_ORG_ID}"

    def test_trailing_slash_stripped(self):
        c = PromptStudioClient(
            base_url="https://test.com/", api_key="k", org_id="o"
        )
        assert c.base_url == "https://test.com"

    def test_headers(self, client):
        assert client._headers == {"Authorization": f"Bearer {MOCK_API_KEY}"}


class TestListProjects:
    @patch("unstract.prompt_studio.client.requests.request")
    def test_list_projects(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = [
            {"tool_id": MOCK_TOOL_ID, "tool_name": "Project 1"}
        ]
        mock_request.return_value = mock_response

        result = client.list_projects()

        assert len(result) == 1
        assert result[0]["tool_id"] == MOCK_TOOL_ID
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "GET"
        assert "prompt-studio/" in args[1]


class TestExportProject:
    @patch("unstract.prompt_studio.client.requests.request")
    def test_export_project(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = MOCK_EXPORT_DATA
        mock_request.return_value = mock_response

        result = client.export_project(MOCK_TOOL_ID)

        assert result["tool_metadata"]["tool_name"] == "Test Project"
        assert len(result["prompts"]) == 2
        args, _ = mock_request.call_args
        assert f"project-transfer/{MOCK_TOOL_ID}" in args[1]


class TestImportProject:
    @patch("unstract.prompt_studio.client.requests.request")
    def test_import_from_dict(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "message": "Project imported successfully as 'Test Project'",
            "tool_id": "new-tool-id",
            "needs_adapter_config": True,
        }
        mock_request.return_value = mock_response

        result = client.import_project(MOCK_EXPORT_DATA, adapters={
            "llm_adapter_id": 42,
            "embedding_adapter_id": 15,
        })

        assert result["tool_id"] == "new-tool-id"
        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert "project-transfer/" in args[1]
        assert "files" in kwargs
        assert kwargs["data"]["llm_adapter_id"] == 42

    @patch("unstract.prompt_studio.client.requests.request")
    def test_import_from_file(self, mock_request, client, tmp_path):
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps(MOCK_EXPORT_DATA))

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"tool_id": "new-id"}
        mock_request.return_value = mock_response

        result = client.import_project(export_file)

        assert result["tool_id"] == "new-id"
        args, kwargs = mock_request.call_args
        assert "files" in kwargs


class TestSyncPrompts:
    @patch("unstract.prompt_studio.client.requests.request")
    def test_sync_prompts(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "message": "Synced 2 prompts into 'Target'",
            "prompts_deleted": 1,
            "prompts_created": 2,
            "tool_settings_updated": True,
        }
        mock_request.return_value = mock_response

        result = client.sync_prompts(MOCK_TOOL_ID, MOCK_EXPORT_DATA)

        assert result["prompts_created"] == 2
        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert "sync-prompts/" in args[1]
        body = kwargs["json"]
        assert body["data"] == MOCK_EXPORT_DATA
        assert body["create_copy"] is False

    @patch("unstract.prompt_studio.client.requests.request")
    def test_sync_with_backup(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "prompts_created": 2,
            "backup_tool_id": "backup-id",
        }
        mock_request.return_value = mock_response

        result = client.sync_prompts(
            MOCK_TOOL_ID, MOCK_EXPORT_DATA, create_copy=True
        )

        assert result["backup_tool_id"] == "backup-id"
        body = mock_request.call_args.kwargs["json"]
        assert body["create_copy"] is True


class TestErrorHandling:
    @patch("unstract.prompt_studio.client.requests.request")
    def test_http_error_raises(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 403
        mock_response.json.return_value = {"message": "Forbidden"}
        mock_request.return_value = mock_response

        with pytest.raises(PromptStudioClientError) as exc_info:
            client.list_projects()

        assert exc_info.value.status_code == 403

    @patch("unstract.prompt_studio.client.requests.request")
    def test_non_json_error(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError
        mock_response.text = "Internal Server Error"
        mock_request.return_value = mock_response

        with pytest.raises(PromptStudioClientError) as exc_info:
            client.list_projects()

        assert "500" in str(exc_info.value)


class TestExportTool:
    @patch("unstract.prompt_studio.client.requests.request")
    def test_export_tool_force(self, mock_request, client):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"status": "exported"}
        mock_request.return_value = mock_response

        result = client.export_tool(MOCK_TOOL_ID)

        assert result["status"] == "exported"
        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert f"prompt-studio/export/{MOCK_TOOL_ID}" in args[1]
        assert kwargs["json"]["force_export"] is True


class TestPromote:
    @patch("unstract.prompt_studio.client.requests.request")
    def test_promote_sync(self, mock_request):
        source = PromptStudioClient(
            base_url="https://dev.unstract.com",
            api_key="source-key",
            org_id="org_dev",
        )
        target = PromptStudioClient(
            base_url="https://prod.unstract.com",
            api_key="target-key",
            org_id="org_prod",
        )

        export_resp = MagicMock()
        export_resp.ok = True
        export_resp.json.return_value = MOCK_EXPORT_DATA

        sync_resp = MagicMock()
        sync_resp.ok = True
        sync_resp.json.return_value = {
            "message": "Synced 2 prompts",
            "prompts_created": 2,
        }

        mock_request.side_effect = [export_resp, sync_resp]

        result = source.promote(
            MOCK_TOOL_ID,
            target,
            target_tool_id="existing-prod-tool",
            create_copy=True,
        )

        assert result["tool_id"] == "existing-prod-tool"
        assert result["prompts_created"] == 2
        assert mock_request.call_count == 2

    @patch("unstract.prompt_studio.client.requests.request")
    def test_promote_with_export(self, mock_request):
        source = PromptStudioClient(
            base_url="https://dev.unstract.com",
            api_key="source-key",
            org_id="org_dev",
        )
        target = PromptStudioClient(
            base_url="https://prod.unstract.com",
            api_key="target-key",
            org_id="org_prod",
        )

        export_resp = MagicMock()
        export_resp.ok = True
        export_resp.json.return_value = MOCK_EXPORT_DATA

        sync_resp = MagicMock()
        sync_resp.ok = True
        sync_resp.json.return_value = {
            "message": "Synced 2 prompts",
            "prompts_created": 2,
        }

        tool_export_resp = MagicMock()
        tool_export_resp.ok = True
        tool_export_resp.json.return_value = {"status": "exported"}

        mock_request.side_effect = [export_resp, sync_resp, tool_export_resp]

        result = source.promote(
            MOCK_TOOL_ID,
            target,
            target_tool_id="existing-prod-tool",
            export=True,
        )

        assert result["export_result"]["status"] == "exported"
        assert mock_request.call_count == 3
        # Verify the export call used force_export
        export_call_kwargs = mock_request.call_args_list[2].kwargs
        assert export_call_kwargs["json"]["force_export"] is True
