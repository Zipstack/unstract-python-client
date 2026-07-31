"""Tests for ``PlatformClient`` HTTP layer.

Coverage:
- URL composition honours base_url, api_path_prefix, organization_id.
- Bearer auth header present on every request.
- Non-2xx response raises ``PlatformAPIError`` with status_code + body.
- 204 / empty body returns ``None`` instead of raising on .json().
- ``get_post_schema`` parses DRF ``actions.POST`` and caches per path.
- ``close()`` shuts the underlying session; context manager works.
- ``_paginate`` follows ``next`` to exhaustion and refuses to return a short read.
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


def test_get_review_settings_500_treated_as_absent():
    # Backend raises DoesNotExist (-> 500) when no HITLSettings row exists.
    client, _ = _client_with_mock(status=500, text="DoesNotExist")
    assert client.get_review_settings("wf-1") is None


def test_get_review_settings_reraises_non_500():
    # Auth / rate-limit errors must surface, not masquerade as "no settings".
    client, _ = _client_with_mock(status=403, text="forbidden")
    with pytest.raises(PlatformAPIError) as exc_info:
        client.get_review_settings("wf-1")
    assert exc_info.value.status_code == 403


def _client_with_pages(*payloads) -> tuple[PlatformClient, MagicMock]:
    """Client whose session returns each payload in turn, one per request."""
    client = PlatformClient(_endpoint())
    mock_request = MagicMock(
        side_effect=[_fake_response(200, p) for p in payloads],
    )
    client._session.request = mock_request
    return client, mock_request


def test_paginate_follows_next_across_pages():
    page1 = {
        "count": 3,
        "next": "https://api.example.com/next?page=2",
        "results": [1, 2],
    }
    page2 = {"count": 3, "next": None, "results": [3]}
    client, mock_request = _client_with_pages(page1, page2)

    assert client.list_tags() == [1, 2, 3]
    # Second hop must GET the absolute ``next`` URL verbatim, not an org path.
    assert mock_request.call_args.args[1] == "https://api.example.com/next?page=2"


def test_paginate_raises_on_short_read():
    # A page set that doesn't add up means rows were dropped; a clone that
    # silently copies a subset is worse than one that fails.
    truncated = {"count": 9, "next": None, "results": [1, 2]}
    client, _ = _client_with_pages(truncated)
    with pytest.raises(PlatformAPIError, match="count=9"):
        client.list_tags()


def test_paginate_raises_on_cyclic_next():
    looping = {"count": 2, "next": "https://api.example.com/loop", "results": [1]}
    client, _ = _client_with_pages(looping, looping, looping)
    with pytest.raises(PlatformAPIError, match="looped"):
        client.list_tags()


def test_paginate_rejects_offsite_next():
    # A ``next`` pointing at another host must not receive the bearer key.
    page1 = {"count": 3, "next": "https://evil.example.com/next", "results": [1, 2]}
    client, _ = _client_with_pages(page1)
    with pytest.raises(PlatformAPIError, match="left the platform host"):
        client.list_tags()


def test_paginate_follows_equivalent_origin_next():
    # Same host with uppercase + explicit default port must be followed,
    # not rejected as off-site.
    page1 = {
        "count": 3,
        "next": "https://API.EXAMPLE.COM:443/next?page=2",
        "results": [1, 2],
    }
    page2 = {"count": 3, "next": None, "results": [3]}
    client, _ = _client_with_pages(page1, page2)
    assert client.list_tags() == [1, 2, 3]


def test_paginate_follows_next_with_different_scheme_or_port():
    # A TLS-terminating proxy emits an http:// (and/or off-port) next link for
    # an https:// client. Same host → must be followed, not rejected.
    page1 = {
        "count": 3,
        "next": "http://api.example.com:8080/next?page=2",
        "results": [1, 2],
    }
    page2 = {"count": 3, "next": None, "results": [3]}
    client, _ = _client_with_pages(page1, page2)
    assert client.list_tags() == [1, 2, 3]


def test_paginate_empty_body_returns_empty_list():
    # A 204 / empty first page means "no rows", not a malformed payload — it
    # must return [] like the pre-pagination ``(result or {}).get`` guard did.
    client = PlatformClient(_endpoint())
    empty = MagicMock()
    empty.status_code = 204
    empty.content = b""
    client._session.request = MagicMock(return_value=empty)
    assert client.list_tags() == []


def test_paginate_raises_on_malformed_port_in_next():
    # A `next` URL with a non-numeric port makes urlparse raise ValueError on
    # `.port`; it must surface as PlatformAPIError, not an incidental traceback.
    page1 = {
        "count": 3,
        "next": "https://api.example.com:notaport/next",
        "results": [1, 2],
    }
    client, _ = _client_with_pages(page1)
    with pytest.raises(PlatformAPIError, match="malformed URL"):
        client.list_tags()


def test_paginate_raises_on_nonlist_results():
    # `results` present but not a list must fail loudly, not corrupt rows via
    # extend (character-by-character for a string, TypeError for an int).
    bad = {"count": 1, "next": None, "results": "oops"}
    client, _ = _client_with_pages(bad)
    with pytest.raises(PlatformAPIError, match="unrecognised list payload"):
        client.list_tags()


def test_paginate_raises_on_malformed_later_page():
    # A later page that isn't a DRF envelope must fail loudly, not raise an
    # incidental AttributeError on the next loop turn.
    page1 = {"count": 3, "next": "https://api.example.com/next", "results": [1, 2]}
    page2 = [3]  # bare list where an envelope was expected
    client, _ = _client_with_pages(page1, page2)
    with pytest.raises(PlatformAPIError, match="unrecognised list payload"):
        client.list_tags()
