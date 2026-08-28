"""Parity tests against the last released client.

The transport underneath ``APIDeploymentsClient`` changed; its published
behaviour must not. These tests pin the seams where that could silently break:
the constructor and method signatures, what goes out on the wire, which
exceptions come back out, and the exact dict each method returns — the last one
by running the released client side by side over the same responses.

This suite exists for the transport migration, not forever: once the released
client it compares against is old enough that no caller is upgrading from it,
it should be dropped or re-baselined deliberately (``tools/refresh_baseline.sh``)
rather than edited case by case until it passes.

Differences from the baseline that are accepted rather than fixed are asserted
here explicitly, each in the test that would otherwise be blind to it — the
``User-Agent``, the per-part ``Content-Type`` on form fields, and the error text
now reported for non-2xx bodies the baseline dropped.
"""

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import re
import socket
import threading
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import requests
from requests.exceptions import (
    ConnectionError,
    ConnectTimeout,
    ContentDecodingError,
    InvalidHeader,
    MissingSchema,
    ProxyError,
    ReadTimeout,
    RequestException,
    Timeout,
    TooManyRedirects,
)

from unstract.api_deployments.client import (
    _EXECUTE_SEND_ONLY,
    _STATUS_SEND_ONLY,
    APIDeploymentsClient,
    APIDeploymentsClientException,
)
from unstract.api_deployments.sdk_docstudio.types import UNSET

BASELINE_VERSION = "1.5.3"
BASELINE_PATH = Path(__file__).parent / "baseline" / "client_1_5_3.py"
BASELINE_SHA256 = "45201bb0de000e8f3a0e65f40cb0b08fec389514f7a17c8bb3410a3dc59229df"
SPEC_PATH = Path(__file__).parents[1] / "specs" / "docstudio-oss.json"

API_URL = "https://api.example.com/deployment/api/testorg/testapi/"
STATUS_ENDPOINT = "/deployment/api/testorg/testapi/?execution_id=exec-123"

#: Operations the facade wraps. The spec declares exactly these, and a new one
#: has to be added here deliberately rather than arriving unnoticed.
WRAPPED_OPERATIONS = frozenset({"execute", "status"})


def _load_baseline():
    """Import the vendored released client under its own module name."""
    spec = importlib.util.spec_from_file_location("baseline_client", BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = _load_baseline()


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"hello")
    return str(path)


def _client(**kwargs):
    kwargs.setdefault("api_url", API_URL)
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("logging_level", "ERROR")
    kwargs.setdefault("max_retries", 0)
    return APIDeploymentsClient(**kwargs)


def _baseline_client(**kwargs):
    kwargs.setdefault("api_url", API_URL)
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("logging_level", "ERROR")
    kwargs.setdefault("max_retries", 0)
    return baseline.APIDeploymentsClient(**kwargs)


def _httpx_response(status_code=200, json_data=None, text=None):
    if text is not None:
        return httpx.Response(status_code, text=text)
    return httpx.Response(status_code, json=json_data)


def _requests_response(status_code=200, json_data=None, text=None):
    response = MagicMock()
    response.status_code = status_code
    if text is not None:
        response.text = text
        response.json.side_effect = requests.exceptions.JSONDecodeError(
            "no json", text, 0
        )
    else:
        response.text = json.dumps(json_data)
        response.json.return_value = json_data
    response.headers = {}
    return response


# --------------------------------------------------------------------------
# Transport error translation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (httpx.ConnectTimeout("connect timed out"), ConnectTimeout),
        (httpx.ReadTimeout("read timed out"), ReadTimeout),
        # Neither had a Timeout equivalent: a send that failed and a pool that
        # could not hand out a connection both surfaced as ConnectionError.
        (httpx.WriteTimeout("write timed out"), ConnectionError),
        (httpx.PoolTimeout("pool timed out"), ConnectionError),
        (httpx.ConnectError("refused"), ConnectionError),
        (httpx.ReadError("reset"), ConnectionError),
        (httpx.WriteError("broken pipe"), ConnectionError),
        (httpx.ProtocolError("bad framing"), ConnectionError),
        (httpx.ProxyError("proxy exploded"), ProxyError),
        (httpx.UnsupportedProtocol("no scheme"), MissingSchema),
        (httpx.TooManyRedirects("looping"), TooManyRedirects),
        (httpx.DecodingError("bad gzip"), ContentDecodingError),
        # A request that cannot be written as composed -- an api_key carrying a
        # newline is the everyday cause -- is a client-side fault, not the
        # server-side framing failure its parent ProtocolError stands for.
        (httpx.LocalProtocolError("illegal header value"), InvalidHeader),
    ],
)
def test_transport_errors_are_translated(raised, expected):
    """Callers catch the ``requests`` classes; httpx's are not subclasses.

    ``ConnectTimeout`` is the case that makes ordering load-bearing, and it is
    also both a ``ConnectionError`` and a ``Timeout`` — the plain ``Timeout``
    that httpx's hierarchy implies would stop matching half the callers. The
    exact class matters too: a caller catching ``ReadTimeout`` sees nothing if
    a broader ``Timeout`` is raised in its place.
    """
    client = _client()
    with patch.object(
        client._transport.get_httpx_client(), "request", side_effect=raised
    ):
        with pytest.raises(expected) as caught:
            client._send("get", "/anything")
    assert type(caught.value) is expected


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        ("::::", APIDeploymentsClientException),
        ("/deployment/api/testorg/testapi/", MissingSchema),
        # Parity: the released client raised this one too.
        ("http://[bad", ValueError),
    ],
)
def test_a_malformed_deployment_url_raises_the_class_this_client_chose(
    api_url, expected
):
    """A deliberate divergence, pinned here so it stays deliberate.

    The released client answered the first two with ``InvalidSchema``, which said
    nothing about which URL was wrong. This is a configuration typo that never
    reaches the wire, so the clearer class is worth the difference -- but a
    caller wrapping construction in ``except RequestException`` no longer catches
    the first, which is the part that has to be visible.
    """
    with pytest.raises(expected) as caught:
        client = _client(api_url=api_url)
        client.check_execution_status(STATUS_ENDPOINT)
    # MissingSchema is itself a ValueError, so the parity row can only tell the
    # two apart by the exact class.
    assert type(caught.value) is expected


def _httpx_request_errors():
    """Every httpx request failure, discovered rather than listed.

    A hand-written list is exactly as complete as it was the day it was
    written; this one grows when httpx does.
    """
    found, stack = [], [httpx.RequestError]
    while stack:
        cls = stack.pop()
        found.append(cls)
        stack.extend(cls.__subclasses__())
    return sorted(found, key=lambda cls: cls.__name__)


@pytest.mark.parametrize("cls", _httpx_request_errors(), ids=lambda cls: cls.__name__)
def test_no_httpx_failure_escapes_untranslated(cls):
    """An httpx class reaching a caller is a class no caller catches."""
    client = _client()
    with patch.object(
        client._transport.get_httpx_client(), "request", side_effect=cls("boom")
    ):
        with pytest.raises(RequestException):
            client._send("get", "/anything")


@pytest.mark.parametrize(
    ("raised", "retried"),
    [
        (httpx.PoolTimeout("pool timed out"), True),
        (httpx.ProxyError("proxy exploded"), True),
        # Retrying these cannot start working: the URL stays malformed, the
        # redirect chain stays a loop, the body stays undecodable, the header
        # stays illegal.
        (httpx.UnsupportedProtocol("no scheme"), False),
        (httpx.TooManyRedirects("looping"), False),
        (httpx.DecodingError("bad gzip"), False),
        (httpx.LocalProtocolError("illegal header value"), False),
    ],
)
def test_translation_decides_what_gets_retried(raised, retried):
    client = _client(max_retries=2, initial_delay=0, max_delay=0, jitter=0)
    with patch.object(
        client._transport.get_httpx_client(), "request", side_effect=raised
    ) as request:
        with pytest.raises(RequestException):
            client._request_with_retry("get", "/anything")
    assert (request.call_count > 1) is retried


def test_a_connect_timeout_is_still_a_connection_error():
    client = _client()
    with patch.object(
        client._transport.get_httpx_client(),
        "request",
        side_effect=httpx.ConnectTimeout("connect timed out"),
    ):
        with pytest.raises(ConnectionError):
            client._send("get", "/anything")
    with patch.object(
        client._transport.get_httpx_client(),
        "request",
        side_effect=httpx.ConnectTimeout("connect timed out"),
    ):
        with pytest.raises(Timeout):
            client._send("get", "/anything")


def test_translated_errors_keep_the_original_cause():
    client = _client()
    original = httpx.ConnectError("refused")
    with patch.object(
        client._transport.get_httpx_client(), "request", side_effect=original
    ):
        with pytest.raises(ConnectionError) as excinfo:
            client._send("get", "/anything")
    assert excinfo.value.__cause__ is original


def test_structure_file_raises_translated_error(sample_file):
    client = _client()
    with patch.object(
        client._transport.get_httpx_client(),
        "request",
        side_effect=httpx.ConnectError("refused"),
    ):
        with pytest.raises(ConnectionError):
            client.structure_file([sample_file])


def test_check_execution_status_raises_translated_error():
    client = _client()
    with patch.object(
        client._transport.get_httpx_client(),
        "request",
        side_effect=httpx.ReadTimeout("read timed out"),
    ):
        with pytest.raises(Timeout):
            client.check_execution_status(STATUS_ENDPOINT)


def test_translation_happens_inside_the_retried_call(sample_file):
    """Retry counts the transport failures, which requires translation first.

    ``_request_with_retry`` retries on the ``requests`` exception types. If the
    httpx exception escaped the retried callable untranslated it would never
    match, and transport-error retry would quietly stop working.
    """
    client = _client(api_timeout=0, max_retries=2, initial_delay=0.001, max_delay=0.002)
    with patch.object(
        client._transport.get_httpx_client(),
        "request",
        side_effect=httpx.ConnectError("refused"),
    ) as mock_request:
        with pytest.raises(ConnectionError):
            client.structure_file([sample_file])
    assert mock_request.call_count == 3


# --------------------------------------------------------------------------
# api_timeout is an execution mode, never a transport timeout
# --------------------------------------------------------------------------


@pytest.mark.parametrize("api_timeout", [-1, 0, 1, 300])
def test_api_timeout_never_configures_the_transport(api_timeout):
    """``api_timeout`` selects a backend execution mode.

    ``-1``/``0`` mean async;
    handing either to the transport fails inside the connection layer.
    """
    client = _client(api_timeout=api_timeout)
    assert client._transport.get_httpx_client().timeout == httpx.Timeout(None)


@pytest.mark.parametrize("api_timeout", [-1, 0, 300])
def test_api_timeout_never_reaches_the_transport_call(sample_file, api_timeout):
    client = _client(api_timeout=api_timeout)
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"message": {}})
        client.structure_file([sample_file])
    _, kwargs = mock_send.call_args
    assert "timeout" not in kwargs


# --------------------------------------------------------------------------
# What goes out on the wire
# --------------------------------------------------------------------------


def _captured_execute_kwargs(client, file_path, **request_params):
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"message": {}})
        client.structure_file([file_path], **request_params)
    return mock_send.call_args


def _execute_parts(client, file_path, **request_params):
    _, kwargs = _captured_execute_kwargs(client, file_path, **request_params)
    return {name: value for name, value in kwargs["files"]}


def test_execute_sends_only_the_fields_the_client_sets(sample_file):
    """A spec default written into the request pins a value the server would
    otherwise choose, and the two diverge the moment the server's default
    moves."""
    _, kwargs = _captured_execute_kwargs(_client(api_timeout=300), sample_file)
    assert {name for name, _ in kwargs["files"]} == {
        "files",
        "include_metadata",
        "timeout",
    }
    assert _EXECUTE_SEND_ONLY == {
        "files",
        "include_metadata",
        "timeout",
        "additional_properties",
    }


def test_execute_multipart_values_match_the_released_client(sample_file):
    _, kwargs = _captured_execute_kwargs(
        _client(api_timeout=300, include_metadata=True), sample_file
    )
    parts = {name: value for name, value in kwargs["files"]}
    assert parts["timeout"][1] == b"300"
    assert parts["include_metadata"][1] == b"True"
    assert parts["files"][0] == "sample.txt"
    assert parts["files"][2] == "application/octet-stream"


# --------------------------------------------------------------------------
# Request parameters, added as keyword-only arguments
# --------------------------------------------------------------------------


def _request_param_names():
    return [
        name
        for name, p in inspect.signature(
            APIDeploymentsClient.structure_file
        ).parameters.items()
        if p.kind is p.KEYWORD_ONLY
    ]


def test_request_parameters_are_named_as_the_spec_names_them():
    """A rename here would need a translation table in every caller."""
    spec = json.loads(SPEC_PATH.read_text())
    declared = set(spec["components"]["schemas"]["ExecuteRequest"]["properties"])
    # ``files`` is built from ``file_paths``, not passed through.
    assert set(_request_param_names()) == declared - {"files"}


def test_request_parameters_are_keyword_only(sample_file):
    with pytest.raises(TypeError):
        _client().structure_file([sample_file], 300)


def test_an_unset_parameter_is_not_sent(sample_file):
    """Sending a default pins a value the server would otherwise choose."""
    parts = _execute_parts(_client(api_timeout=300), sample_file)
    assert set(parts) == {"files", "include_metadata", "timeout"}


def test_a_requested_parameter_is_sent(sample_file):
    parts = _execute_parts(
        _client(api_timeout=300),
        sample_file,
        tags="a,b",
        llm_profile_id="profile-1",
        use_file_history=True,
    )
    assert parts["tags"][1] == b"a,b"
    assert parts["llm_profile_id"][1] == b"profile-1"
    assert parts["use_file_history"][1] == b"True"


@pytest.mark.parametrize(
    ("param", "value", "expected"),
    [
        ("timeout", 0, b"0"),
        ("include_metrics", False, b"False"),
        ("include_extracted_text", False, b"False"),
        ("tags", "", b""),
    ],
)
def test_a_falsy_parameter_is_still_sent(sample_file, param, value, expected):
    """``False``/``0``/``""`` are choices, not absences; a truthiness filter
    eats them and silently hands the decision back to the server."""
    parts = _execute_parts(_client(api_timeout=300), sample_file, **{param: value})
    assert parts[param][1] == expected


@pytest.mark.parametrize(
    "param", ["llm_profile_id", "hitl_queue_name", "hitl_packet_id"]
)
def test_an_explicit_none_is_not_sent(sample_file, param):
    """``None`` is how a caller forwards "no override" from its own optional
    config. A form field carries no null, so one sent at all goes out as the
    literal string ``"None"`` for the service to look up."""
    parts = _execute_parts(_client(api_timeout=300), sample_file, **{param: None})
    assert param not in parts


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"a": 1, "b": "x"}, b'{"a": 1, "b": "x"}'),
        ([1, 2], b"[1, 2]"),
        (True, b"true"),
        # Already a string: passed through, so a caller serialising its own
        # payload does not get it JSON-encoded a second time.
        ('{"a": 1}', b'{"a": 1}'),
    ],
)
def test_custom_data_goes_out_as_json(sample_file, value, expected):
    """The generated encoder writes ``str(value)``. For anything but a string
    that is a Python repr, which the server's JSON field cannot parse."""
    parts = _execute_parts(_client(api_timeout=300), sample_file, custom_data=value)
    assert parts["custom_data"][1] == expected


def test_custom_data_is_json_on_the_wire(tmp_path):
    """Read off the socket, not out of the kwargs: the encoding step between
    the two is where a repr would survive unnoticed."""
    path = tmp_path / "sample.txt"
    path.write_bytes(b"hello")

    (raw,) = _wire_requests(
        lambda url: _client(api_url=url, api_timeout=300).structure_file(
            [str(path)], custom_data={"a": 1, "b": "x"}
        )
    )
    sent = {
        field: content for field, _filename, content, _headers in _multipart_parts(raw)
    }
    assert json.loads(sent["custom_data"]) == {"a": 1, "b": "x"}


def test_a_requested_parameter_overrides_the_constructor(sample_file):
    parts = _execute_parts(
        _client(api_timeout=300, include_metadata=False),
        sample_file,
        timeout=-1,
        include_metadata=True,
    )
    assert parts["timeout"][1] == b"-1"
    assert parts["include_metadata"][1] == b"True"


@pytest.mark.parametrize("timeout", [0, -1, 300])
def test_a_requested_timeout_selects_the_execution_mode(sample_file, timeout):
    """``timeout`` is an execution mode: at or below zero the request only
    queues, so a 5xx is safe to retry; above it the request runs the execution
    and a retry would run it twice.

    ``-1`` is the API's own default for queue-only, so it belongs on the retried
    side even though the released client tested for exactly ``0``. Passing the
    mode per request has to move the decision with it.
    """
    queues = timeout <= 0
    with patch.object(APIDeploymentsClient, "_request_with_retry") as retried:
        with patch.object(APIDeploymentsClient, "_send") as sent:
            retried.return_value = sent.return_value = _httpx_response(
                200, {"message": {}}
            )
            # Constructor set the other way round, so only the per-request
            # value can be what decided this.
            _client(api_timeout=300 if queues else 0).structure_file(
                [sample_file], timeout=timeout
            )
            assert retried.called is queues
            assert sent.called is not queues


@pytest.mark.parametrize("api_timeout", [0, -1])
def test_a_queue_only_execution_is_retried(sample_file, api_timeout):
    """A 5xx on a queue-only request means queuing failed; retrying cannot
    duplicate work, and not retrying hands back a finished-looking result with
    no execution behind it."""
    client = _client(api_timeout=api_timeout, max_retries=2, initial_delay=0, jitter=0)
    with patch.object(APIDeploymentsClient, "_send") as sent:
        sent.return_value = _httpx_response(503, {"message": {}})
        client.structure_file([sample_file])
    assert sent.call_count == 3


def test_multipart_boundary_is_random_and_matches_the_body(sample_file):
    """The generated builder pins ``boundary=+++`` in the header. A PDF
    containing those bytes would corrupt the encoding, so the header is dropped
    and the transport picks the boundary — as the released client did.

    Encoding happens inside the send, while the file handles are still
    open.
    """
    encoded = []

    def encode(method, url, **kwargs):
        assert "Content-Type" not in kwargs.get("headers", {})
        transport = httpx.Client(base_url="https://api.example.com")
        request = transport.build_request(method, url, **kwargs)
        encoded.append((request.headers["content-type"], request.read()))
        return _httpx_response(200, {"message": {}})

    with patch.object(APIDeploymentsClient, "_send", side_effect=encode):
        _client().structure_file([sample_file])
        _client().structure_file([sample_file])

    boundaries = []
    for content_type, body in encoded:
        header_boundary = content_type.split("boundary=")[1]
        assert body.split(b"\r\n")[0] == b"--" + header_boundary.encode()
        assert header_boundary != "+++"
        boundaries.append(header_boundary)
    assert boundaries[0] != boundaries[1]


def _captured_status_params(client=None, endpoint=STATUS_ENDPOINT, **request_params):
    client = client or _client()
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"status": "COMPLETED"})
        client.check_execution_status(endpoint, **request_params)
    return mock_send.call_args[1]["params"]


def test_status_sends_only_the_fields_the_client_sets():
    assert set(_captured_status_params()) == {"execution_id", "include_metadata"}


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        (API_URL, "https://api.example.com/deployment/api/testorg/testapi/"),
        (
            "https://api.example.com/unstract/deployment/api/testorg/testapi/",
            "https://api.example.com/unstract/deployment/api/testorg/testapi/",
        ),
    ],
)
def test_status_is_polled_under_the_deployment_urls_own_prefix(api_url, expected):
    """A poll that misses is a paid execution whose result is never collected."""
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"status": "COMPLETED"})
        _client(api_url=api_url).check_execution_status(STATUS_ENDPOINT)
    assert mock_send.call_args[0][1] == expected


def test_the_status_endpoints_own_query_parameters_are_forwarded():
    """The endpoint is the service's instruction for reaching this execution.

    A region hint or a signature dropped from it polls somewhere the execution
    is not, and the execution has already been paid for.
    """
    params = _captured_status_params(
        client=_client(),
        endpoint=STATUS_ENDPOINT + "&region=eu&sig=abc123",
    )
    assert params["region"] == "eu"
    assert params["sig"] == "abc123"
    assert params["execution_id"] == "exec-123"


def test_a_deployment_url_without_the_spec_route_polls_the_endpoint_as_returned():
    """No prefix can be derived from a rewritten URL, and a guessed path polls
    nothing."""
    client = _client(api_url="https://gw.example.com/v1/testorg/testapi/")
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"status": "COMPLETED"})
        client.check_execution_status("/v1/testorg/testapi/?execution_id=exec-123")
    assert mock_send.call_args[0][1] == "https://gw.example.com/v1/testorg/testapi/"


def test_status_request_parameters_are_named_as_the_spec_names_them():
    """A rename here would need a translation table in every caller."""
    spec = json.loads(SPEC_PATH.read_text())
    execute = "/deployment/api/{org_name}/{api_name}/"
    declared = {
        p["name"]
        for p in spec["paths"][execute]["get"]["parameters"]
        if p["in"] == "query"
    }
    accepted = {
        name
        for name, p in inspect.signature(
            APIDeploymentsClient.check_execution_status
        ).parameters.items()
        if p.kind is p.KEYWORD_ONLY
    }
    # `execution_id` is read out of the endpoint URL the server handed back.
    assert accepted == declared - {"execution_id"}


def test_status_request_parameters_are_keyword_only():
    with pytest.raises(TypeError):
        _client().check_execution_status(STATUS_ENDPOINT, True)


def test_a_requested_status_parameter_is_sent():
    params = _captured_status_params(include_metrics=True, include_extracted_text=False)
    assert params["include_metrics"] == "True"
    # False is a choice; a truthiness filter would drop it and hand the decision
    # back to the server.
    assert params["include_extracted_text"] == "False"


def test_a_requested_status_parameter_overrides_the_constructor():
    params = _captured_status_params(
        _client(include_metadata=False), include_metadata=True
    )
    assert params["include_metadata"] == "True"
    assert _STATUS_SEND_ONLY == {"execution_id", "include_metadata"}


def test_status_url_matches_the_released_client():
    """The status URL is rebuilt from the spec route plus the execution id
    instead of concatenating the server-supplied path.

    Same request either way, which is what this pins.
    """
    client = _client(include_metadata=True)
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"status": "COMPLETED"})
        client.check_execution_status(STATUS_ENDPOINT)
    args, kwargs = mock_send.call_args
    # Encoded by the transport that will send it, rather than stringified here:
    # httpx renders a bool as `true` where urlencoding one gives `True`, and
    # normalising both sides is how a comparison stops seeing the difference.
    sent = httpx.URL(client.base_url).join(args[1]).copy_merge_params(kwargs["params"])
    ours = urlparse(str(sent))
    ours_query = parse_qs(ours.query)

    published = urlparse(client.base_url + STATUS_ENDPOINT)
    published_query = {
        **parse_qs(published.query),
        "include_metadata": [str(client.include_metadata)],
    }

    assert args[0].lower() == "get"
    assert (ours.scheme, ours.netloc, ours.path) == (
        published.scheme,
        published.netloc,
        published.path,
    )
    assert ours_query == published_query


@pytest.mark.parametrize(
    "endpoint",
    [
        # What the server actually returns, and the only spelling the released
        # client handled: it concatenated base URL and endpoint, so an absolute
        # one produced `https://hosthttps://host/...` and a relative one with
        # no leading slash produced `https://hostdeployment/...`.
        "/deployment/api/testorg/testapi/?execution_id=exec-123",
        "https://api.example.com/deployment/api/testorg/testapi/?execution_id=exec-123",
        "deployment/api/testorg/testapi/?execution_id=exec-123",
    ],
)
def test_the_status_endpoint_is_read_not_concatenated(endpoint):
    """Only the execution id is taken from the server's endpoint; the route
    comes from the spec. Every spelling therefore resolves to one request."""
    client = _client()
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"status": "COMPLETED"})
        client.check_execution_status(endpoint)
    args, kwargs = mock_send.call_args
    assert args[0].lower() == "get"
    assert str(httpx.URL(client.base_url).join(args[1])) == API_URL
    assert kwargs["params"]["execution_id"] == "exec-123"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://attacker.example/deployment/api/testorg/testapi/",
        # A host can be spelled without a scheme, and a path beginning `//` is
        # read as one by anything that resolves a reference.
        "//attacker.example/deployment/api/testorg/testapi/",
        "///attacker.example/deployment/api/testorg/testapi/",
        "////attacker.example/deployment/api/testorg/testapi/",
        "https:////attacker.example/deployment/api/testorg/testapi/",
        "https://api.example.com//attacker.example/deployment/api/testorg/testapi/",
    ],
)
def test_a_status_endpoint_naming_another_host_is_not_polled(endpoint):
    """The reply names the path to poll. It does not get to name the host the
    deployment key is sent to.

    The deployment URL here carries a prefix the spec route cannot account for,
    which is the branch that reads the endpoint rather than rebuilding it.
    """
    client = _client(api_url="https://api.example.com/other/testorg/testapi/")
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"status": "COMPLETED"})
        client.check_execution_status(f"{endpoint}?execution_id=exec-123")
    args, _ = mock_send.call_args
    sent = httpx.URL(client.base_url).join(args[1])

    assert sent.host == "api.example.com"


def test_the_key_is_read_at_call_time():
    """A rotated key reaches the next request.

    The released client built its header on every call, so assigning ``api_key``
    took effect immediately; a transport that captured it once would answer with
    the old one until the client was rebuilt.
    """
    client = _client()
    with patch.object(client._transport.get_httpx_client(), "request") as request:
        request.return_value = _httpx_response(200, {})
        client._send("GET", API_URL)
        before = request.call_args.kwargs["headers"]["Authorization"]
        client.api_key = "rotated-key"
        client._send("GET", API_URL)
        after = request.call_args.kwargs["headers"]["Authorization"]

    assert before == "Bearer test-key"
    assert after == "Bearer rotated-key"


@pytest.mark.parametrize(
    "api_url",
    [
        API_URL,
        # Nothing normalises the deployment URL on the way in, so whatever the
        # caller registered is what the released client posted to.
        API_URL.rstrip("/"),
        "https://api.example.com/unstract/deployment/api/testorg/testapi/",
        "https://api.example.com/deployment/api/TestOrg/testapi/",
    ],
)
def test_execute_url_matches_the_deployment_url(api_url):
    """The deployment URL goes back out as given.

    A path prefix — an ingress route, an on-prem reverse proxy — is part of it
    and no route template can carry it, so the URL cannot be rebuilt from one.
    """
    client = _client(api_url=api_url)
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"message": {}})
        with patch("builtins.open", return_value=io.BytesIO(b"x")):
            client.structure_file(["sample.txt"])
    args, _ = mock_send.call_args

    with patch.object(baseline, "requests") as mock_requests:
        mock_requests.post.return_value = _requests_response(200, {"message": {}})
        with patch("builtins.open", return_value=io.BytesIO(b"x")):
            _baseline_client(api_url=api_url).structure_file(["sample.txt"])

    assert args[0].lower() == "post"
    assert args[1] == api_url == mock_requests.post.call_args[0][0]


def _wire_requests(*calls, reply=b'{"status":"COMPLETED","message":{}}'):
    """Run each call against a loopback server and return the raw requests.

    Below the client, the transport adds headers of its own -- and drops none
    of them into any object the client can be asked for. A socket is the only
    place both clients can be compared on what they actually send. One server
    serves every call, so the ``Host`` header is the same for all of them.
    """
    raw = []
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(len(calls))

    def serve():
        for _ in calls:
            conn, _address = server.accept()
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            # The body has to be drained too: a client whose upload is never
            # read can block on the socket instead of returning.
            head, _, rest = data.partition(b"\r\n\r\n")
            declared = _header_value(head, "content-length")
            while declared and len(rest) < int(declared):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                rest += chunk
            raw.append(head + b"\r\n\r\n" + rest)
            body = reply
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
            )
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.getsockname()[1]}/deployment/api/org/name/"
        for call in calls:
            call(url)
    finally:
        thread.join(timeout=10)
        server.close()

    return raw


def _headers(head: bytes) -> dict[str, str]:
    return {
        name.lower(): value.strip()
        for name, _, value in (
            line.partition(":") for line in head.decode().split("\r\n")[1:]
        )
    }


def _header_value(head: bytes, name: str) -> str:
    return _headers(head).get(name, "")


def _wire_heads(*calls):
    """The request headers each call put on the wire."""
    return [_headers(raw.split(b"\r\n\r\n")[0]) for raw in _wire_requests(*calls)]


def _multipart_parts(raw: bytes) -> list[tuple[str, str, bytes, tuple]]:
    """``(field, filename, content, headers)`` for every part of a multipart
    request.

    The headers are carried too: a difference in them is invisible to a
    comparison of what the parts contain, and the parts of this request do
    differ from the released client's there.

    The boundary itself is deliberately not compared: it is random per request
    in both clients, so only what it delimits can be.
    """
    head, _, body = raw.partition(b"\r\n\r\n")
    boundary = _header_value(head, "content-type").partition("boundary=")[2]
    parts = []
    for chunk in body.split(b"--" + boundary.encode()):
        headers, _, content = chunk.partition(b"\r\n\r\n")
        disposition = headers.decode("utf-8", errors="replace")
        if "content-disposition" not in disposition.lower():
            continue
        field = re.search(r'name="([^"]*)"', disposition)
        filename = re.search(r'filename="([^"]*)"', disposition)
        parts.append(
            (
                field.group(1) if field else "",
                filename.group(1) if filename else "",
                content.removesuffix(b"\r\n"),
                tuple(
                    line.split(":", 1)[0].strip().lower()
                    for line in disposition.strip().splitlines()
                    if ":" in line
                ),
            )
        )
    return parts


def test_wire_headers_match_the_released_client():
    """The headers no caller sets are still on the wire.

    ``Accept-Encoding`` is the load-bearing one: it decides whether responses
    come back compressed at all.
    """
    ours, theirs = _wire_heads(
        lambda url: _client(api_url=url).check_execution_status(STATUS_ENDPOINT),
        lambda url: _baseline_client(api_url=url).check_execution_status(
            STATUS_ENDPOINT
        ),
    )

    assert {name: ours[name] for name in theirs if name != "user-agent"} == {
        name: value for name, value in theirs.items() if name != "user-agent"
    }
    assert ours.keys() == theirs.keys()
    # The one accepted difference: the transport names itself, and nothing on
    # the wire branches on it.
    assert ours["user-agent"].startswith("python-httpx/")


def test_a_multi_file_upload_matches_the_released_client(tmp_path):
    """The method takes a list, and the second file is where a transport swap
    diverges: one part written, one dropped, or two parts sharing a name the
    server then reads as one."""
    paths = []
    for name, content in (("first.txt", b"one"), ("second.txt", b"two")):
        path = tmp_path / name
        path.write_bytes(content)
        paths.append(str(path))

    ours, theirs = _wire_requests(
        lambda url: _client(api_url=url, api_timeout=300).structure_file(paths),
        lambda url: _baseline_client(api_url=url, api_timeout=300).structure_file(
            paths
        ),
    )

    # Sorted: the two clients order the fields differently, which no multipart
    # parser reads as meaning. The order of the files among themselves is the
    # part that carries meaning, and it is pinned below.
    assert sorted(part[:3] for part in _multipart_parts(ours)) == sorted(
        part[:3] for part in _multipart_parts(theirs)
    )
    uploaded = [part for part in _multipart_parts(ours) if part[0] == "files"]
    assert [(part[1], part[2]) for part in uploaded] == [
        ("first.txt", b"one"),
        ("second.txt", b"two"),
    ]

    # The one accepted difference in the parts themselves: the released client
    # sent scalars through `requests`' `data=`, which writes only a
    # Content-Disposition, while the generated encoder types every scalar as
    # text/plain. Django routes on the presence of `filename`, so nothing
    # downstream reads the extra header -- but it is a real difference and it
    # is asserted rather than left invisible.
    scalar_headers = {
        part[0]: part[3] for part in _multipart_parts(ours) if part[0] != "files"
    }
    assert scalar_headers
    assert all(
        headers == ("content-disposition", "content-type")
        for headers in scalar_headers.values()
    )
    assert all(
        part[3] == ("content-disposition",)
        for part in _multipart_parts(theirs)
        if part[0] != "files"
    )


def test_redirects_are_followed():
    """The released client followed them on both verbs.

    Not following one turns a load balancer's 307 into a body the poll loop
    reads as a finished execution with no status.
    """
    assert _client()._transport.get_httpx_client().follow_redirects is True


def test_a_status_endpoint_without_an_execution_id_is_refused():
    """Polling with a blank id asks the service about an execution nobody has;
    what it answers is not this execution's state."""
    with pytest.raises(APIDeploymentsClientException):
        _client().check_execution_status("/deployment/api/testorg/testapi/")


def test_the_transport_is_untimed_by_default():
    """A connection that stalls forever is what the released client did.

    Bounding it by default would turn a hang into an exception callers have
    never had to handle, and ``api_timeout`` cannot serve: it is an execution
    mode the backend reads, not a socket timeout.
    """
    assert _client()._transport.get_httpx_client().timeout == httpx.Timeout(None)


def test_transport_timeout_is_what_the_transport_uses():
    assert _client(transport_timeout=5)._transport.get_httpx_client().timeout == (
        httpx.Timeout(5)
    )


def test_transport_timeout_bounds_a_stalled_connection():
    """The call is made off the test thread, so the failure this pins -- a
    request that never returns -- fails the test instead of hanging the run."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    accepted, outcome = [], []

    def stall():
        conn, _address = server.accept()
        accepted.append(conn)  # held open, and answered by nobody

    def call():
        url = f"http://127.0.0.1:{server.getsockname()[1]}/deployment/api/org/name/"
        client = _client(api_url=url, transport_timeout=0.2)
        try:
            client.check_execution_status(STATUS_ENDPOINT)
            outcome.append(None)
        except BaseException as e:  # noqa: BLE001 - reported, not handled
            outcome.append(e)

    stalling = threading.Thread(target=stall, daemon=True)
    calling = threading.Thread(target=call, daemon=True)
    stalling.start()
    calling.start()
    try:
        calling.join(timeout=10)
        assert outcome, "the request never returned"
        assert isinstance(outcome[0], ReadTimeout)
    finally:
        stalling.join(timeout=5)
        for conn in accepted:
            conn.close()
        server.close()


def test_deployment_route_rejects_an_unusable_url():
    client = _client(api_url="https://api.example.com/onlyone")
    with pytest.raises(APIDeploymentsClientException):
        _ = client._deployment_route


@pytest.mark.parametrize(
    "api_url", ["https://gw.example.com/extract/", "https://api.example.com/myapi/"]
)
def test_execute_does_not_need_a_derivable_route(sample_file, api_url):
    """The route is only ever spent on a URL execute then discards, so requiring
    one would reject deployment URLs the released client posted to -- an ingress
    or reverse-proxy rewrite short enough to have no org/API pair in it.

    Only ``check_execution_status``, which rebuilds the poll path, needs it.
    """
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"message": {}})
        _client(api_url=api_url).structure_file([sample_file])

    with patch.object(baseline, "requests") as mock_requests:
        mock_requests.post.return_value = _requests_response(200, {"message": {}})
        _baseline_client(api_url=api_url).structure_file([sample_file])

    assert mock_send.call_args[0][1] == mock_requests.post.call_args[0][0] == api_url


def test_the_client_releases_its_connections():
    """The transport is pooled and kept between calls; nothing else closes it,
    so a client built per job would accumulate pools until collected."""
    client = _client()
    httpx_client = client._transport.get_httpx_client()
    client.close()
    assert httpx_client.is_closed
    assert client._transport_client is None
    # Idempotent, and the next call builds a fresh transport rather than
    # reaching into a closed pool.
    client.close()
    assert client._transport.get_httpx_client() is not httpx_client


def test_the_client_is_a_context_manager():
    with _client() as client:
        httpx_client = client._transport.get_httpx_client()
    assert httpx_client.is_closed


def test_the_transport_is_built_once_under_contention():
    """Two threads racing the first call would otherwise each build a pool, and
    the one that lost would be dropped still holding its sockets."""
    client = _client()
    built = []
    barrier = threading.Barrier(8)

    def race():
        barrier.wait()
        built.append(client._transport)

    threads = [threading.Thread(target=race) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(set(map(id, built))) == 1


# --------------------------------------------------------------------------
# Return shape, compared against the released client over the same responses
# --------------------------------------------------------------------------


def _message(**fields):
    return {"message": fields}


EXECUTE_CASES = [
    ("pending", 200, _message(execution_status="PENDING", status_api=STATUS_ENDPOINT)),
    (
        "executing",
        200,
        _message(execution_status="EXECUTING", status_api=STATUS_ENDPOINT),
    ),
    ("success", 200, _message(execution_status="SUCCESS", result=[{"file": "a"}])),
    (
        "success_without_result",
        200,
        _message(execution_status="SUCCESS", status_api=STATUS_ENDPOINT),
    ),
    ("error", 200, _message(execution_status="ERROR", error="boom")),
    ("unauthorized", 401, {"errors": [{"detail": "Invalid token"}]}),
    ("unprocessable", 422, _message(execution_status="ERROR", error="bad input")),
    ("server_error", 500, _message(execution_status="ERROR", error="oops")),
]


@pytest.mark.parametrize(
    ("name", "status_code", "body"), EXECUTE_CASES, ids=[c[0] for c in EXECUTE_CASES]
)
def test_structure_file_returns_what_the_released_client_returned(
    sample_file, name, status_code, body
):
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(status_code, body)
        ours = _client(api_timeout=300).structure_file([sample_file])

    with patch.object(baseline, "requests") as mock_requests:
        mock_requests.post.return_value = _requests_response(status_code, body)
        theirs = _baseline_client(api_timeout=300).structure_file([sample_file])

    assert ours == theirs


def test_structure_file_matches_on_a_non_json_body(sample_file):
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(502, text="<html>gateway</html>")
        ours = _client(api_timeout=300).structure_file([sample_file])

    with patch.object(baseline, "requests") as mock_requests:
        mock_requests.post.return_value = _requests_response(
            502, text="<html>gateway</html>"
        )
        theirs = _baseline_client(api_timeout=300).structure_file([sample_file])

    assert ours == theirs


def test_structure_file_missing_file_still_raises(sample_file):
    with pytest.raises(APIDeploymentsClientException) as caught:
        _client().structure_file(["/nonexistent/file.txt"])
    assert "File not found" in str(caught.value)
    with pytest.raises(baseline.APIDeploymentsClientException):
        _baseline_client().structure_file(["/nonexistent/file.txt"])


def test_an_unreadable_file_raises_the_documented_exception(tmp_path, sample_file):
    """Not only a missing one: a directory or an unreadable path reached the
    caller as a raw builtin, past the handles already opened for the files
    before it."""
    opened = []
    real_open = open

    def tracking_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    directory = tmp_path / "adir"
    directory.mkdir()
    with patch("builtins.open", side_effect=tracking_open):
        with pytest.raises(APIDeploymentsClientException) as caught:
            _client().structure_file([sample_file, str(directory)])

    assert "File not found" not in str(caught.value)
    assert opened and all(handle.closed for handle in opened)


# Error shapes the API answers with. The handler-routed statuses answer with
# drf-standardized-errors; a few are still built by hand. The baseline read the
# reason out of neither, so a caller saw an empty error next to a status code —
# an accepted difference from parity, and the reason for these tests.
ERROR_BODY_CASES = [
    (
        "standardized",
        400,
        {
            "type": "validation_error",
            "errors": [
                {
                    "code": "invalid",
                    "detail": "Queue 'nope' does not exist",
                    "attr": "hitl_queue_name",
                }
            ],
        },
        "Queue 'nope' does not exist",
    ),
    (
        "standardized_multiple",
        400,
        {
            "type": "validation_error",
            "errors": [
                {"code": "invalid", "detail": "first", "attr": "a"},
                {"code": "invalid", "detail": "second", "attr": "b"},
            ],
        },
        "first; second",
    ),
    (
        "hand_built",
        404,
        {"status": "ERROR", "message": "API deployment not found"},
        "API deployment not found",
    ),
    ("unauthorized", 401, {"errors": [{"detail": "Invalid token"}]}, "Invalid token"),
]

#: Bodies carrying nothing readable. Neither shape may crash, and the body
#: itself is more use to a caller than an empty string.
UNREADABLE_ERROR_BODIES = [
    ("no_detail", 409, {"type": "conflict", "errors": []}),
    ("a_list", 502, [1, 2]),
    ("a_string", 403, "forbidden"),
]


@pytest.mark.parametrize(
    ("name", "status_code", "body", "expected"),
    ERROR_BODY_CASES,
    ids=[c[0] for c in ERROR_BODY_CASES],
)
def test_structure_file_reports_the_reason_an_error_carries(
    sample_file, name, status_code, body, expected
):
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(status_code, body)
        result = _client(api_timeout=300).structure_file([sample_file])

    assert result["error"] == expected
    assert result["status_code"] == status_code
    assert result["pending"] is False


@pytest.mark.parametrize(
    ("name", "status_code", "body", "expected"),
    ERROR_BODY_CASES,
    ids=[c[0] for c in ERROR_BODY_CASES],
)
def test_check_execution_status_reports_the_reason_an_error_carries(
    name, status_code, body, expected
):
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(status_code, body)
        result = _client().check_execution_status(STATUS_ENDPOINT)

    assert result["error"] == expected
    assert result["status_code"] == status_code


@pytest.mark.parametrize(
    ("name", "status_code", "body"),
    UNREADABLE_ERROR_BODIES,
    ids=[c[0] for c in UNREADABLE_ERROR_BODIES],
)
def test_an_unreadable_error_body_is_reported_not_raised(
    sample_file, name, status_code, body
):
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(status_code, body)
        executed = _client(api_timeout=300).structure_file([sample_file])

    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(status_code, body)
        polled = _client().check_execution_status(STATUS_ENDPOINT)

    for result in (executed, polled):
        assert result["status_code"] == status_code
        assert json.loads(result["error"]) == body


def _declared_responses(operation_id: str) -> dict[int, str]:
    """``{status: schema name}`` for one operation, read out of the spec."""
    spec = json.loads(SPEC_PATH.read_text())
    for path in spec["paths"].values():
        for method, operation in path.items():
            if method not in {"get", "post"}:
                continue
            if operation["operationId"] != operation_id:
                continue
            return {
                int(code): body["content"]["application/json"]["schema"]["$ref"].split(
                    "/"
                )[-1]
                for code, body in operation["responses"].items()
            }
    raise AssertionError(f"{operation_id} not declared in the spec")


def _body_for(schema: str) -> dict:
    """A response body of the shape the spec declares, for an error status.

    Built from the schema name rather than hardcoded per status, so a spec that
    re-points a status at the other error shape is exercised as the new shape
    without this table being touched.
    """
    if schema == "ErrorResponse":
        return {
            "type": "client_error",
            # `code` is free-form here, deliberately not an enum: the server
            # emits "error" for statuses the handler routes without a subtype.
            "errors": [{"code": "error", "detail": "the reason", "attr": None}],
        }
    # The shapes still built by hand: 406 acknowledges, 422/500 report setup
    # failures through the same envelope a success uses.
    return {"status": "ERROR", "message": "the reason"}


@pytest.mark.parametrize("operation", sorted(WRAPPED_OPERATIONS))
def test_every_error_status_the_spec_declares_is_reported(sample_file, operation):
    """The spec and this client have to agree on the error body, and they
    disagreed once: the facade read one shape while the spec declared another.

    Both shapes are declared now, both are read here, and a status the spec adds
    arrives as a case rather than as an empty ``error`` in production.
    """
    declared = _declared_responses(operation)
    errors = {code: schema for code, schema in declared.items() if code != 200}
    assert errors, operation
    # Both shapes are actually in play; a regression to one of them is a
    # narrowing this would otherwise not notice.
    assert set(errors.values()) > {"ErrorResponse"}

    for status_code, schema in errors.items():
        body = _body_for(schema)
        with patch.object(APIDeploymentsClient, "_send") as mock_send:
            mock_send.return_value = _httpx_response(status_code, body)
            if operation == "execute":
                result = _client(api_timeout=300).structure_file([sample_file])
            else:
                result = _client().check_execution_status(STATUS_ENDPOINT)

        assert result["status_code"] == status_code, (operation, status_code)
        assert result["error"] == "the reason", (operation, status_code, schema)


def test_the_generated_models_read_the_bodies_the_server_sends():
    """The generated models are an implementation detail, but a wrong one is a
    trap for anyone who imports them: the facade would keep working while the
    models silently lost the payload."""
    from unstract.api_deployments.sdk_docstudio.models import (
        ErrorResponse,
        ExecutionMessage,
        FileResult,
    )

    error = ErrorResponse.from_dict(_body_for("ErrorResponse"))
    assert [detail.detail for detail in error.errors] == ["the reason"]
    assert error.type_ == "client_error"
    assert not error.additional_properties

    # `error` and `status_api` are absent on a success and null on a failure;
    # declared required, either one raised a bare KeyError.
    message = ExecutionMessage.from_dict(
        {"execution_id": "exec-123", "execution_status": "SUCCESS"}
    )
    assert message.error is UNSET and message.status_api is UNSET

    result = FileResult.from_dict(
        {"file": "a.pdf", "status": "SUCCESS", "extracted_text": "hello"}
    )
    assert result.extracted_text == "hello"
    assert not result.additional_properties
    # Dropped from the spec: the server never sent it.
    assert not hasattr(result, "metrics")


def test_a_success_envelope_is_never_read_as_an_error(sample_file):
    """The fallback only fires where the envelope had nothing: a non-2xx that
    still carries one is read out of it, the way the released client did."""
    body = {"message": {"execution_status": "ERROR", "error": "bad input"}}
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(422, body)
        ours = _client(api_timeout=300).structure_file([sample_file])

    with patch.object(baseline, "requests") as mock_requests:
        mock_requests.post.return_value = _requests_response(422, body)
        theirs = _baseline_client(api_timeout=300).structure_file([sample_file])

    assert ours == theirs
    assert ours["error"] == "bad input"


def test_a_status_endpoint_without_an_execution_id_does_not_report_its_query():
    """The documented usage prints this exception straight to a log, and the
    query is the service's to shape."""
    with pytest.raises(APIDeploymentsClientException) as caught:
        _client().check_execution_status(
            "/deployment/api/testorg/testapi/?token=s3cret-signature"
        )
    assert "s3cret-signature" not in str(caught.value)
    assert "/deployment/api/testorg/testapi/" in str(caught.value)


def test_structure_file_closes_its_handles(sample_file):
    """The released client leaked these; closing them is invisible to
    callers."""
    opened = []
    real_open = open

    def tracking_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"message": {}})
        with patch("builtins.open", side_effect=tracking_open):
            _client().structure_file([sample_file])

    assert opened and all(handle.closed for handle in opened)


STATUS_CASES = [
    ("completed", 200, {"status": "COMPLETED", "message": [{"file": "a"}]}),
    ("executing", 200, {"status": "EXECUTING", "message": ""}),
    ("queued", 200, {"status": "QUEUED", "message": ""}),
    ("error", 200, {"status": "ERROR", "error": "boom", "message": ""}),
    ("already_acknowledged", 406, {"status": "", "error": "already acknowledged"}),
    ("server_error", 500, {"status": "", "error": "oops"}),
]


@pytest.mark.parametrize(
    ("name", "status_code", "body"), STATUS_CASES, ids=[c[0] for c in STATUS_CASES]
)
def test_check_execution_status_returns_what_the_released_client_returned(
    name, status_code, body
):
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(status_code, body)
        ours = _client().check_execution_status(STATUS_ENDPOINT)

    with patch.object(baseline, "requests") as mock_requests:
        mock_requests.request.return_value = _requests_response(status_code, body)
        theirs = _baseline_client().check_execution_status(STATUS_ENDPOINT)

    assert ours == theirs


def test_check_execution_status_matches_on_a_non_json_body():
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(502, text="<html>gateway</html>")
        ours = _client().check_execution_status(STATUS_ENDPOINT)

    with patch.object(baseline, "requests") as mock_requests:
        mock_requests.request.return_value = _requests_response(
            502, text="<html>gateway</html>"
        )
        theirs = _baseline_client().check_execution_status(STATUS_ENDPOINT)

    assert ours == theirs


# --------------------------------------------------------------------------
# Construction and surface
# --------------------------------------------------------------------------


def _baseline_class_node():
    tree = ast.parse(BASELINE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "APIDeploymentsClient":
            return node
    raise AssertionError("APIDeploymentsClient not found in the baseline")


def _baseline_init_params():
    """Constructor parameters and defaults, read out of the baseline source.

    Parsed rather than imported so the comparison is against the
    released text, not against whatever a shared import happened to
    bind.
    """
    for node in _baseline_class_node().body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args.args[1:]
            defaults = [None] * (len(args) - len(node.args.defaults)) + [
                ast.literal_eval(d) for d in node.args.defaults
            ]
            return list(zip([a.arg for a in args], defaults))
    raise AssertionError("__init__ not found in the baseline")


def test_constructor_parameters_are_unchanged():
    """Names, order and defaults all matter: callers pass some positionally."""
    live = inspect.signature(APIDeploymentsClient.__init__).parameters
    live_params = [
        (name, None if p.default is inspect.Parameter.empty else p.default)
        for name, p in live.items()
        # Keyword-only parameters are excluded: they cannot be reached by any
        # existing call, so adding one leaves every released call shape intact.
        if name != "self" and p.kind is not p.KEYWORD_ONLY
    ]
    assert live_params == _baseline_init_params()


def test_public_methods_are_unchanged():
    baseline_methods = {
        node.name: node
        for node in _baseline_class_node().body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert baseline_methods

    for name, node in baseline_methods.items():
        live = getattr(APIDeploymentsClient, name, None)
        assert live is not None, f"{name} disappeared from the client"
        # Keyword-only parameters are excluded: they cannot be reached by any
        # existing call, so adding one leaves every released call shape intact.
        live_args = [
            arg
            for arg, p in inspect.signature(live).parameters.items()
            if arg not in ("self", "cls") and p.kind is not p.KEYWORD_ONLY
        ]
        assert live_args == [a.arg for a in node.args.args[1:]], name


def test_class_attributes_are_unchanged():
    compared = 0
    for node in _baseline_class_node().body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue  # logger and friends: identity, not value
        for target in node.targets:
            assert getattr(APIDeploymentsClient, target.id) == value, target.id
            compared += 1
    # A parse that matches no node asserts nothing and still reports green.
    assert compared


def test_module_level_names_are_unchanged():
    import unstract.api_deployments.client as live

    tree = ast.parse(BASELINE_PATH.read_text())
    names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]
    assert names
    for name in names:
        assert hasattr(live, name), name


#: What an install of this package puts on the path. A removal here is a
#: breaking change for anyone whose deploy runs the command or imports the
#: module, so it has to be made in the same diff that moves this list.
INSTALLED_CONSOLE_SCRIPTS: dict[str, str] = {}
INSTALLED_TOP_LEVEL_MODULES = {"unstract.api_deployments", "unstract.clone"}


def test_the_packaging_surface_is_what_it_claims():
    """The wire behaviour above is all pinned against the baseline and stays
    green through a console script being deleted, which is the one published
    contract this change does break."""
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    assert pyproject["project"].get("scripts", {}) == INSTALLED_CONSOLE_SCRIPTS

    src = Path(__file__).parents[1] / "src" / "unstract"
    packaged = {
        f"unstract.{path.name}"
        for path in src.iterdir()
        if (path / "__init__.py").exists()
    }
    assert packaged == INSTALLED_TOP_LEVEL_MODULES

    for module in INSTALLED_TOP_LEVEL_MODULES:
        assert importlib.util.find_spec(module) is not None, module


def test_every_declared_operation_is_wrapped():
    """A new spec operation shows up here as a failure, not as silence.

    Compared whole rather than after subtracting an exception list: an entry
    excusing an operation the spec no longer declares keeps passing forever, and
    nothing about a green run says the list is still describing anything.
    """
    spec = json.loads(SPEC_PATH.read_text())
    declared = {
        operation["operationId"]
        for path in spec["paths"].values()
        for method, operation in path.items()
        if method in {"get", "post", "put", "patch", "delete"}
    }
    assert declared == WRAPPED_OPERATIONS


def test_the_baseline_is_the_released_client_unmodified():
    # A digest, not a version string in a comment: an edited baseline can claim
    # any provenance it likes, and every parity test here would still pass.
    assert BASELINE_PATH.name == f"client_{BASELINE_VERSION.replace('.', '_')}.py"
    assert hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest() == BASELINE_SHA256
