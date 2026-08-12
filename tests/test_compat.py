"""Parity tests against the last released client.

The transport underneath ``APIDeploymentsClient`` changed; its published
behaviour must not. These tests pin the seams where that could silently break:
the constructor and method signatures, what goes out on the wire, which
exceptions come back out, and the exact dict each method returns — the last one
by running the released client side by side over the same responses.
"""

import ast
import importlib.util
import inspect
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import requests
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout, Timeout

from unstract.api_deployments.client import (
    _EXECUTE_SEND_ONLY,
    _STATUS_SEND_ONLY,
    APIDeploymentsClient,
)

BASELINE_VERSION = "1.5.3"
BASELINE_PATH = Path(__file__).parent / "baseline" / "client_1_5_3.py"
SPEC_PATH = Path(__file__).parents[1] / "specs" / "docstudio.json"

API_URL = "https://api.example.com/deployment/api/testorg/testapi/"
STATUS_ENDPOINT = "/deployment/api/testorg/testapi/?execution_id=exec-123"

# Operations the spec declares that the facade deliberately does not wrap. The
# CLI has no use for them yet; listing them here keeps the coverage check honest
# instead of silently passing on whatever happens to be implemented.
UNWRAPPED_OPERATIONS = frozenset({"mcp_retrieve", "mcp_create"})


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
        (httpx.WriteTimeout("write timed out"), Timeout),
        (httpx.PoolTimeout("pool timed out"), Timeout),
        (httpx.ConnectError("refused"), ConnectionError),
        (httpx.ReadError("reset"), ConnectionError),
        (httpx.WriteError("broken pipe"), ConnectionError),
        (httpx.ProtocolError("bad framing"), ConnectionError),
        (httpx.ProxyError("proxy exploded"), ConnectionError),
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
    """``api_timeout`` selects a backend execution mode. ``-1``/``0`` mean async;
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


def _captured_execute_kwargs(client, file_path):
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"message": {}})
        client.structure_file([file_path])
    return mock_send.call_args


def test_execute_sends_only_the_fields_the_client_sets(sample_file):
    """A spec default written into the request pins a value the server would
    otherwise choose, and the two diverge the moment the server's default moves.
    """
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


def test_multipart_boundary_is_random_and_matches_the_body(sample_file):
    """The generated builder pins ``boundary=+++`` in the header. A PDF
    containing those bytes would corrupt the encoding, so the header is dropped
    and the transport picks the boundary — as the released client did.

    Encoding happens inside the send, while the file handles are still open.
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


def test_status_sends_only_the_fields_the_client_sets():
    client = _client()
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"status": "COMPLETED"})
        client.check_execution_status(STATUS_ENDPOINT)
    _, kwargs = mock_send.call_args
    assert set(kwargs["params"]) == {"execution_id", "include_metadata"}
    assert _STATUS_SEND_ONLY == {"execution_id", "include_metadata"}


def test_status_url_matches_the_released_client():
    """The status URL is rebuilt from the spec route plus the execution id
    instead of concatenating the server-supplied path. Same request either way,
    which is what this pins.
    """
    client = _client(include_metadata=True)
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"status": "COMPLETED"})
        client.check_execution_status(STATUS_ENDPOINT)
    args, kwargs = mock_send.call_args
    ours = urlparse(str(httpx.URL(client.base_url).join(args[1])))
    ours_query = {
        **parse_qs(ours.query),
        **{k: [str(v)] for k, v in kwargs["params"].items()},
    }

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


def test_execute_url_matches_the_deployment_url():
    client = _client()
    with patch.object(APIDeploymentsClient, "_send") as mock_send:
        mock_send.return_value = _httpx_response(200, {"message": {}})
        with patch("builtins.open", return_value=io.BytesIO(b"x")):
            client.structure_file(["sample.txt"])
    args, _ = mock_send.call_args
    assert args[0].lower() == "post"
    assert str(httpx.URL(client.base_url).join(args[1])) == API_URL


def test_deployment_route_rejects_an_unusable_url():
    from unstract.api_deployments.client import APIDeploymentsClientException

    client = _client(api_url="https://api.example.com/onlyone")
    with pytest.raises(APIDeploymentsClientException):
        _ = client._deployment_route


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
    from unstract.api_deployments.client import APIDeploymentsClientException

    with pytest.raises(APIDeploymentsClientException):
        _client().structure_file(["/nonexistent/file.txt"])
    with pytest.raises(baseline.APIDeploymentsClientException):
        _baseline_client().structure_file(["/nonexistent/file.txt"])


def test_structure_file_closes_its_handles(sample_file):
    """The released client leaked these; closing them is invisible to callers."""
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

    Parsed rather than imported so the comparison is against the released text,
    not against whatever a shared import happened to bind.
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
        if name != "self"
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
        live_args = [
            p for p in inspect.signature(live).parameters if p not in ("self", "cls")
        ]
        assert live_args == [a.arg for a in node.args.args[1:]], name


def test_class_attributes_are_unchanged():
    for node in _baseline_class_node().body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue  # logger and friends: identity, not value
        for target in node.targets:
            assert getattr(APIDeploymentsClient, target.id) == value, target.id


def test_module_level_names_are_unchanged():
    import unstract.api_deployments.client as live

    tree = ast.parse(BASELINE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            assert hasattr(live, node.name), node.name


def test_every_wrapped_operation_is_covered():
    """A new spec operation shows up here as a failure, not as silence."""
    spec = json.loads(SPEC_PATH.read_text())
    declared = {
        operation["operationId"]
        for path in spec["paths"].values()
        for method, operation in path.items()
        if method in {"get", "post", "put", "patch", "delete"}
    }
    covered = {"execute", "status"}
    assert declared - UNWRAPPED_OPERATIONS == covered


def test_the_baseline_is_a_released_version():
    assert BASELINE_PATH.name == f"client_{BASELINE_VERSION.replace('.', '_')}.py"
    assert "DO NOT EDIT" in BASELINE_PATH.read_text(encoding="utf-8")
