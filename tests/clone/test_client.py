"""Tests for ``PlatformClient`` HTTP layer.

Coverage:
- URL composition honours base_url, api_path_prefix, organization_id.
- Bearer auth header present on every request.
- Non-2xx response raises ``PlatformAPIError`` with status_code + body.
- 204 / empty body returns ``None`` instead of raising on .json().
- ``get_post_schema`` parses DRF ``actions.POST`` and caches per path.
- ``close()`` shuts the underlying session; context manager works.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from unstract.clone.client import PlatformClient
from unstract.clone.context import OrgEndpoint
from unstract.clone.exceptions import PlatformAPIError


def _endpoint() -> OrgEndpoint:
    return OrgEndpoint(
        base_url="https://api.example.com",
        organization_id="org_abc",
        platform_key="plat-key-xyz",
    )


def _fake_response(status: int, payload=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.content = b"" if payload is None and not text else b"x"
    resp.json.return_value = payload
    return resp


def _client_with_mock(
    payload=None, status: int = 200, text: str = ""
) -> tuple[PlatformClient, MagicMock]:
    client = PlatformClient(_endpoint())
    mock_request = MagicMock(return_value=_fake_response(status, payload, text))
    client._session.request = mock_request
    return client, mock_request


def test_url_composition_includes_org_and_api_prefix():
    client, mock_request = _client_with_mock(payload=[])
    client.list_adapters()
    call = mock_request.call_args
    assert call.args[0] == "GET"
    assert call.args[1] == "https://api.example.com/api/v1/unstract/org_abc/adapter/"


def test_bearer_token_sent_on_session():
    client, _ = _client_with_mock(payload=[])
    assert client._session.headers["Authorization"] == "Bearer plat-key-xyz"
    assert client._session.headers["Accept"] == "application/json"


def test_non_2xx_raises_platform_api_error_with_status_and_body():
    client, _ = _client_with_mock(status=404, text="not found")
    with pytest.raises(PlatformAPIError) as exc_info:
        client.list_adapters()
    err = exc_info.value
    assert err.status_code == 404
    assert "not found" in err.body


def test_500_with_long_body_truncated_to_2000_chars():
    big = "x" * 5000
    client, _ = _client_with_mock(status=500, text=big)
    with pytest.raises(PlatformAPIError) as exc_info:
        client.list_adapters()
    assert len(exc_info.value.body) == 2000


def test_204_no_content_returns_none():
    client = PlatformClient(_endpoint())
    resp = MagicMock()
    resp.status_code = 204
    resp.content = b""
    client._session.request = MagicMock(return_value=resp)
    assert client._request("DELETE", "tag/abc/") is None


def test_get_post_schema_parses_options_and_caches():
    options_body = {
        "actions": {
            "POST": {
                "name": {"read_only": False},
                "id": {"read_only": True},
                "shared_to_org": {"read_only": False},
                # No read_only key → treated as writable.
                "description": {},
            }
        }
    }
    client, mock_request = _client_with_mock(payload=options_body)
    writable = client.get_post_schema("adapter/")
    assert writable == frozenset({"name", "shared_to_org", "description"})
    # second call hits cache — no extra HTTP.
    writable2 = client.get_post_schema("adapter/")
    assert writable2 is writable
    assert mock_request.call_count == 1


def test_get_post_schema_handles_missing_actions_block():
    client, _ = _client_with_mock(payload={})
    assert client.get_post_schema("connector/") == frozenset()


def test_close_shuts_session():
    client = PlatformClient(_endpoint())
    sess = client._session
    sess.close = MagicMock()
    client.close()
    sess.close.assert_called_once()


def test_context_manager_closes_on_exit():
    with PlatformClient(_endpoint()) as client:
        client._session.close = MagicMock()
        sess_close = client._session.close
    sess_close.assert_called_once()


def test_list_endpoint_unwraps_paginated_envelope():
    client, _ = _client_with_mock(payload={"results": [{"id": "a"}, {"id": "b"}]})
    items = client.list_tags()
    assert [i["id"] for i in items] == ["a", "b"]


def test_list_endpoint_accepts_bare_list():
    client, _ = _client_with_mock(payload=[{"id": "a"}])
    items = client.list_tags()
    assert items == [{"id": "a"}]


def test_options_response_with_null_body_still_yields_empty_schema():
    # Some deployments return 200 with no body on OPTIONS.
    client, _ = _client_with_mock(payload=None, text="")
    assert client.get_post_schema("pipeline/") == frozenset()
