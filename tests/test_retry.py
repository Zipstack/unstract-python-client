"""Tests for the exponential backoff retry logic in APIDeploymentsClient."""

import io
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import ConnectionError, Timeout

from unstract.api_deployments.client import APIDeploymentsClient


@pytest.fixture
def client():
    """Create a client with fast retry settings for testing."""
    return APIDeploymentsClient(
        api_url="https://api.example.com/v1/deploy",
        api_key="test-key",
        api_timeout=30,
        logging_level="WARNING",
        max_retries=3,
        initial_delay=0.01,
        max_delay=0.1,
        backoff_factor=2.0,
    )


@pytest.fixture
def client_no_retry():
    """Create a client with retries disabled."""
    return APIDeploymentsClient(
        api_url="https://api.example.com/v1/deploy",
        api_key="test-key",
        max_retries=0,
    )


def _mock_response(status_code=200, json_data=None, headers=None, text=""):
    """Helper to create a mock response object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    resp.json.return_value = json_data or {}
    return resp


# ── Constructor and backward compatibility ──


class TestConstructorDefaults:
    def test_default_retry_params(self):
        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
        )
        assert c.max_retries == 4
        assert c.initial_delay == 2.0
        assert c.max_delay == 60.0
        assert c.backoff_factor == 2.0

    def test_custom_retry_params(self):
        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            backoff_factor=3.0,
        )
        assert c.max_retries == 5
        assert c.initial_delay == 0.5
        assert c.max_delay == 30.0
        assert c.backoff_factor == 3.0

    def test_backward_compat_no_retry_args(self):
        """Existing code that doesn't pass retry params still works."""
        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            api_timeout=10,
            logging_level="DEBUG",
            include_metadata=False,
        )
        assert c.api_timeout == 10
        assert c.max_retries == 4


class TestIsRetryableStatus:
    def test_default_retries_429(self, client):
        assert client._is_retryable_status(429) is True

    def test_default_retries_500(self, client):
        assert client._is_retryable_status(500) is True

    def test_default_retries_502(self, client):
        assert client._is_retryable_status(502) is True

    def test_default_retries_503(self, client):
        assert client._is_retryable_status(503) is True

    def test_default_retries_504(self, client):
        assert client._is_retryable_status(504) is True

    def test_default_retries_any_5xx(self, client):
        """Any 5xx status code should be retried by default."""
        for code in (500, 501, 502, 503, 504, 507, 511, 599):
            assert (
                client._is_retryable_status(code) is True
            ), f"Expected {code} retryable"

    def test_default_no_retry_4xx_except_429(self, client):
        """4xx codes (except 429) should not be retried by default."""
        for code in (400, 401, 403, 404, 405, 408, 422):
            assert (
                client._is_retryable_status(code) is False
            ), f"Expected {code} not retryable"

    def test_default_no_retry_2xx(self, client):
        assert client._is_retryable_status(200) is False
        assert client._is_retryable_status(201) is False


# ── _calculate_delay ──


class TestCalculateDelay:
    def test_delay_within_bounds(self, client):
        for attempt in range(5):
            delay = client._calculate_delay(attempt)
            assert 0 <= delay <= client.max_delay

    def test_delay_exponential_growth_upper_bound(self, client):
        """Upper bound of jitter grows exponentially."""
        upper_0 = client.initial_delay * (client.backoff_factor**0)
        upper_1 = client.initial_delay * (client.backoff_factor**1)
        assert upper_1 > upper_0

    def test_delay_capped_at_max(self):
        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            initial_delay=10.0,
            max_delay=5.0,
            backoff_factor=10.0,
        )
        for _ in range(100):
            delay = c._calculate_delay(10)
            assert delay <= c.max_delay

    def test_delay_zero_attempt(self, client):
        """Attempt 0: delay between 0 and initial_delay."""
        for _ in range(50):
            delay = client._calculate_delay(0)
            assert 0 <= delay <= client.initial_delay


# ── _request_with_retry: successful requests ──


class TestRequestWithRetrySuccess:
    @patch("unstract.api_deployments.client.requests.request")
    def test_success_on_first_try(self, mock_request, client):
        mock_request.return_value = _mock_response(200)
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 1

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_retry_on_503_then_success(self, mock_request, mock_sleep, client):
        mock_request.side_effect = [
            _mock_response(503),
            _mock_response(200),
        ]
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_retry_on_500_then_success(self, mock_request, mock_sleep, client):
        mock_request.side_effect = [
            _mock_response(500),
            _mock_response(500),
            _mock_response(200),
        ]
        resp = client._request_with_retry("POST", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_retry_on_429_then_success(self, mock_request, mock_sleep, client):
        mock_request.side_effect = [
            _mock_response(429),
            _mock_response(200),
        ]
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 2

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_retry_on_502_then_success(self, mock_request, mock_sleep, client):
        mock_request.side_effect = [
            _mock_response(502),
            _mock_response(200),
        ]
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 2

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_retry_on_504_then_success(self, mock_request, mock_sleep, client):
        mock_request.side_effect = [
            _mock_response(504),
            _mock_response(200),
        ]
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 2


# ── _request_with_retry: connection errors ──


class TestRequestWithRetryConnectionErrors:
    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_retry_on_connection_error_then_success(
        self, mock_request, mock_sleep, client
    ):
        mock_request.side_effect = [
            ConnectionError("Connection refused"),
            _mock_response(200),
        ]
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 2

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_retry_on_timeout_then_success(self, mock_request, mock_sleep, client):
        mock_request.side_effect = [
            Timeout("Request timed out"),
            _mock_response(200),
        ]
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 2

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_connection_error_exhausted_raises(self, mock_request, mock_sleep, client):
        mock_request.side_effect = ConnectionError("Connection refused")
        with pytest.raises(ConnectionError):
            client._request_with_retry("GET", "https://api.example.com/test")
        assert mock_request.call_count == client.max_retries + 1

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_timeout_exhausted_raises(self, mock_request, mock_sleep, client):
        mock_request.side_effect = Timeout("Request timed out")
        with pytest.raises(Timeout):
            client._request_with_retry("GET", "https://api.example.com/test")
        assert mock_request.call_count == client.max_retries + 1


# ── _request_with_retry: exhaustion returns last response ──


class TestRequestWithRetryExhaustion:
    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_all_retries_exhausted_returns_last_response(
        self, mock_request, mock_sleep, client
    ):
        mock_request.return_value = _mock_response(503)
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 503
        assert mock_request.call_count == client.max_retries + 1


# ── No retry on non-retryable status codes ──


class TestNoRetryOnNonRetryable:
    @patch("unstract.api_deployments.client.requests.request")
    def test_no_retry_on_200(self, mock_request, client):
        mock_request.return_value = _mock_response(200)
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        assert mock_request.call_count == 1

    @patch("unstract.api_deployments.client.requests.request")
    def test_no_retry_on_400(self, mock_request, client):
        mock_request.return_value = _mock_response(400)
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 400
        assert mock_request.call_count == 1

    @patch("unstract.api_deployments.client.requests.request")
    def test_no_retry_on_401(self, mock_request, client):
        mock_request.return_value = _mock_response(401)
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 401
        assert mock_request.call_count == 1

    @patch("unstract.api_deployments.client.requests.request")
    def test_no_retry_on_404(self, mock_request, client):
        mock_request.return_value = _mock_response(404)
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 404
        assert mock_request.call_count == 1

    @patch("unstract.api_deployments.client.requests.request")
    def test_no_retry_on_422(self, mock_request, client):
        mock_request.return_value = _mock_response(422)
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 422
        assert mock_request.call_count == 1


# ── Timeout passed to requests ──


class TestTimeoutPassed:
    @patch("unstract.api_deployments.client.requests.request")
    def test_no_default_timeout_set(self, mock_request, client):
        """api_timeout is a server-side parameter, not an HTTP socket timeout.

        _request_with_retry should NOT inject a default timeout kwarg.
        """
        mock_request.return_value = _mock_response(200)
        client._request_with_retry("GET", "https://api.example.com/test")
        _, kwargs = mock_request.call_args
        assert "timeout" not in kwargs

    @patch("unstract.api_deployments.client.requests.request")
    def test_explicit_timeout_not_overridden(self, mock_request, client):
        """Callers can still pass an explicit HTTP socket timeout."""
        mock_request.return_value = _mock_response(200)
        client._request_with_retry("GET", "https://api.example.com/test", timeout=99)
        _, kwargs = mock_request.call_args
        assert kwargs["timeout"] == 99


# ── Retry-After header on 429 ──


class TestRetryAfterHeader:
    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_429_respects_retry_after_header(self, mock_request, mock_sleep, client):
        mock_request.side_effect = [
            _mock_response(429, headers={"Retry-After": "5"}),
            _mock_response(200),
        ]
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        mock_sleep.assert_called_once_with(5.0)

    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_429_invalid_retry_after_falls_back(self, mock_request, mock_sleep, client):
        mock_request.side_effect = [
            _mock_response(429, headers={"Retry-After": "not-a-number"}),
            _mock_response(200),
        ]
        resp = client._request_with_retry("GET", "https://api.example.com/test")
        assert resp.status_code == 200
        # Should have used _calculate_delay fallback (some float >= 0)
        assert mock_sleep.call_count == 1
        delay_used = mock_sleep.call_args[0][0]
        assert delay_used >= 0


# ── File seek on retry ──


class TestFileSeekOnRetry:
    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_file_objects_rewound_on_retry(self, mock_request, mock_sleep, client):
        file_obj = io.BytesIO(b"test data")
        files = [("files", ("test.pdf", file_obj, "application/octet-stream"))]
        mock_request.side_effect = [
            _mock_response(503),
            _mock_response(200),
        ]
        resp = client._request_with_retry(
            "POST", "https://api.example.com/test", files=files
        )
        assert resp.status_code == 200
        # File should have been rewound before second attempt
        assert file_obj.tell() == 0 or mock_request.call_count == 2


# ── max_retries=0 disables retry ──


class TestDisabledRetry:
    @patch("unstract.api_deployments.client.requests.request")
    def test_no_retry_when_max_retries_zero(self, mock_request, client_no_retry):
        mock_request.return_value = _mock_response(503)
        resp = client_no_retry._request_with_retry(
            "GET", "https://api.example.com/test"
        )
        assert resp.status_code == 503
        assert mock_request.call_count == 1

    @patch("unstract.api_deployments.client.requests.request")
    def test_connection_error_raises_immediately_when_disabled(
        self, mock_request, client_no_retry
    ):
        mock_request.side_effect = ConnectionError("fail")
        with pytest.raises(ConnectionError):
            client_no_retry._request_with_retry("GET", "https://api.example.com/test")
        assert mock_request.call_count == 1


# ── check_execution_status pending bug fix ──


class TestCheckExecutionStatusPendingFix:
    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_503_after_exhaustion_sets_pending_true(
        self, mock_request, mock_sleep, client
    ):
        mock_request.return_value = _mock_response(
            503, json_data={"status": "", "error": "Service Unavailable", "message": ""}
        )
        result = client.check_execution_status("/api/v1/status/123")
        assert result["pending"] is True
        assert result["status_code"] == 503

    @patch("unstract.api_deployments.client.requests.request")
    def test_200_with_pending_status_sets_pending_true(self, mock_request, client):
        mock_request.return_value = _mock_response(
            200, json_data={"status": "EXECUTING", "error": "", "message": ""}
        )
        result = client.check_execution_status("/api/v1/status/123")
        assert result["pending"] is True
        assert result["status_code"] == 200

    @patch("unstract.api_deployments.client.requests.request")
    def test_200_with_completed_status_sets_pending_false(self, mock_request, client):
        mock_request.return_value = _mock_response(
            200,
            json_data={
                "status": "COMPLETED",
                "error": "",
                "message": '{"result": "data"}',
            },
        )
        result = client.check_execution_status("/api/v1/status/123")
        assert result["pending"] is False

    @patch("unstract.api_deployments.client.requests.request")
    def test_422_with_executing_status_sets_pending_true(self, mock_request, client):
        """HTTP 422 is currently returned by Unstract for in-progress statuses.

        Per the Status API migration guide, the server will change from 422 to
        200 for EXECUTING/PENDING statuses. We determine pending state solely
        from the response body ``status`` field (Option 1 from the docs), making
        this client future-proof regardless of the HTTP status code.
        """
        mock_request.return_value = _mock_response(
            422, json_data={"status": "EXECUTING", "error": "", "message": ""}
        )
        result = client.check_execution_status("/api/v1/status/123")
        assert result["pending"] is True
        assert result["status_code"] == 422

    @patch("unstract.api_deployments.client.requests.request")
    def test_422_with_pending_status_sets_pending_true(self, mock_request, client):
        """HTTP 422 with PENDING body status — still detected via body
        check."""
        mock_request.return_value = _mock_response(
            422, json_data={"status": "PENDING", "error": "", "message": ""}
        )
        result = client.check_execution_status("/api/v1/status/123")
        assert result["pending"] is True
        assert result["status_code"] == 422

    @patch("unstract.api_deployments.client.requests.request")
    def test_400_does_not_set_pending(self, mock_request, client):
        mock_request.return_value = _mock_response(
            400, json_data={"status": "", "error": "Bad request", "message": ""}
        )
        result = client.check_execution_status("/api/v1/status/123")
        assert result["pending"] is False


# ── structure_file uses retry ──


class TestStructureFileUsesRetry:
    @patch("unstract.api_deployments.client.time.sleep")
    @patch("unstract.api_deployments.client.requests.request")
    def test_structure_file_retries_on_503_async_mode(
        self, mock_request, mock_sleep, tmp_path
    ):
        """In async mode (api_timeout=0), POST is retried on 5xx."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            api_timeout=0,
            max_retries=2,
            initial_delay=0.01,
            max_delay=0.1,
        )
        mock_request.side_effect = [
            _mock_response(503),
            _mock_response(
                200,
                json_data={
                    "message": {
                        "execution_status": "SUCCESS",
                        "error": "",
                        "result": '{"key": "value"}',
                        "status_api": "/api/status/1",
                    }
                },
            ),
        ]
        result = c.structure_file([str(test_file)])
        assert result["status_code"] == 200
        assert mock_request.call_count == 2


class TestStructureFileNoRetryInSyncMode:
    @patch("unstract.api_deployments.client.requests.post")
    def test_structure_file_no_retry_on_503_sync_mode(self, mock_post, tmp_path):
        """In sync mode (api_timeout>0), POST is NOT retried on 5xx."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            api_timeout=300,
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.1,
        )
        mock_post.return_value = _mock_response(
            503,
            json_data={
                "message": {
                    "execution_status": "",
                    "error": "Service Unavailable",
                    "result": "",
                }
            },
        )
        result = c.structure_file([str(test_file)])
        assert result["status_code"] == 503
        # Only one call — no retries in sync mode
        assert mock_post.call_count == 1

    @patch("unstract.api_deployments.client.requests.post")
    def test_structure_file_sync_mode_default_timeout(self, mock_post, tmp_path):
        """Default api_timeout=300 means sync mode — no POST retry."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.1,
        )
        mock_post.return_value = _mock_response(
            500,
            json_data={
                "message": {
                    "execution_status": "",
                    "error": "Internal Server Error",
                    "result": "",
                }
            },
        )
        result = c.structure_file([str(test_file)])
        assert result["status_code"] == 500
        assert mock_post.call_count == 1


# ── structure_file: POST 422 does NOT set pending ──
# The POST endpoint returns 200 for successful queuing (PENDING/EXECUTING)
# and 422 only on setup errors. A 422 should never trigger polling.


class TestStructureFile422DoesNotSetPending:
    @patch("unstract.api_deployments.client.requests.post")
    def test_422_pending_does_not_set_pending(self, mock_post, tmp_path):
        """POST 422 with PENDING status should NOT set pending=True.

        The POST endpoint returns 422 only on setup errors, so the
        client must not start polling even if execution_status says
        PENDING.
        """
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            max_retries=0,
            logging_level="WARNING",
        )
        mock_post.return_value = _mock_response(
            422,
            json_data={
                "message": {
                    "execution_status": "PENDING",
                    "error": "",
                    "result": "",
                    "status_api": "/api/status/abc",
                }
            },
        )
        result = c.structure_file([str(test_file)])
        assert result["pending"] is False
        assert result["status_code"] == 422

    @patch("unstract.api_deployments.client.requests.post")
    def test_422_executing_does_not_set_pending(self, mock_post, tmp_path):
        """POST 422 with EXECUTING status should NOT set pending=True.

        Same rationale as above — 422 from POST means a setup error, not
        a legitimately queued execution.
        """
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            max_retries=0,
            logging_level="WARNING",
        )
        mock_post.return_value = _mock_response(
            422,
            json_data={
                "message": {
                    "execution_status": "EXECUTING",
                    "error": "",
                    "result": "",
                    "status_api": "/api/status/abc",
                }
            },
        )
        result = c.structure_file([str(test_file)])
        assert result["pending"] is False
        assert result["status_code"] == 422

    @patch("unstract.api_deployments.client.requests.post")
    def test_200_pending_sets_pending_true(self, mock_post, tmp_path):
        """POST 200 + PENDING correctly sets pending=True for polling."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        c = APIDeploymentsClient(
            api_url="https://api.example.com/v1/deploy",
            api_key="test-key",
            max_retries=0,
            logging_level="WARNING",
        )
        mock_post.return_value = _mock_response(
            200,
            json_data={
                "message": {
                    "execution_status": "PENDING",
                    "error": "",
                    "result": "",
                    "status_api": "/api/status/abc",
                }
            },
        )
        result = c.structure_file([str(test_file)])
        assert result["pending"] is True
        assert result["status_code"] == 200
