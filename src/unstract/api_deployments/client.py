"""This module provides an API client to invoke APIs deployed on the Unstract
platform.

Classes:
    APIDeploymentsClient: A class to invoke APIs deployed on the Unstract platform.
    APIDeploymentsClientException: A class to handle exceptions raised by the
        APIDeploymentsClient class.
"""

import logging
import ntpath
import os
import time
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import attrs
import httpx

# `requests` remains a dependency for its exception classes. Downstream code
# catches ConnectionError and Timeout by name around these calls, and the httpx
# equivalents are not subclasses, so they are translated at the transport seam.
from requests.exceptions import (
    ConnectionError,
    ConnectTimeout,
    ContentDecodingError,
    InvalidURL,
    MissingSchema,
    ProxyError,
    ReadTimeout,
    Timeout,
    TooManyRedirects,
)
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from unstract.api_deployments.sdk_docstudio import AuthenticatedClient
from unstract.api_deployments.sdk_docstudio.api.deployment import execute, status
from unstract.api_deployments.sdk_docstudio.models import ExecuteRequest
from unstract.api_deployments.sdk_docstudio.types import UNSET, File, Unset
from unstract.api_deployments.utils import UnstractUtils


def _translate_transport_errors(fn, *args, **kwargs):
    """Re-raise httpx transport failures as their ``requests`` equivalents.

    Callers document and catch the ``requests`` classes, and the retry policy
    keys off them too, so the class chosen here decides whether a failure is
    retried. Every branch is ordered before the base class it derives from.
    ``RequestError`` is the catch-all for the transport subtree, which is where
    a novel failure appears. httpx puts three families outside it: ``InvalidURL``,
    translated here because ``requests`` raised its own, and ``StreamError`` and
    ``CookieConflict``, which propagate as themselves.
    """
    try:
        return fn(*args, **kwargs)
    except httpx.ConnectTimeout as e:
        # ConnectTimeout is both a ConnectionError and a Timeout; the plain
        # Timeout httpx implies would stop matching half the callers.
        raise ConnectTimeout(str(e)) from e
    except httpx.ReadTimeout as e:
        raise ReadTimeout(str(e)) from e
    except (httpx.WriteTimeout, httpx.PoolTimeout) as e:
        # Neither had a Timeout equivalent: a send that failed and a pool that
        # could not hand out a connection both surfaced as ConnectionError.
        raise ConnectionError(str(e)) from e
    except httpx.TimeoutException as e:
        raise Timeout(str(e)) from e
    except httpx.UnsupportedProtocol as e:
        # A URL rejected before any socket is opened. Deliberately not a
        # ConnectionError: retrying a malformed URL cannot start working.
        raise MissingSchema(str(e)) from e
    except httpx.ProxyError as e:
        raise ProxyError(str(e)) from e
    except httpx.ConnectError as e:
        raise ConnectionError(str(e)) from e
    except httpx.TooManyRedirects as e:
        raise TooManyRedirects(str(e)) from e
    except httpx.DecodingError as e:
        raise ContentDecodingError(str(e)) from e
    except httpx.InvalidURL as e:
        raise InvalidURL(str(e)) from e
    except httpx.RequestError as e:
        raise ConnectionError(str(e)) from e


def _query_value(url: str, key: str) -> str:
    """Read one required query parameter out of a URL, absolute or relative.

    Empty is not a usable value here: it polls for an execution the service
    cannot identify and reports whatever it makes of a blank id.
    """
    value = parse_qs(urlparse(url).query).get(key, [""])[0]
    if not value:
        raise APIDeploymentsClientException(
            f"No {key} in {url!r}. The status endpoint the service returned "
            "carries it; pass that endpoint unmodified."
        )
    return value


def _forwarded_query(url: str) -> dict[str, str]:
    """Everything else the service put on the status endpoint.

    A region hint, a signature, a cursor: the endpoint is the service's
    instruction for reaching this execution, and dropping part of it polls
    somewhere the execution is not.
    """
    return {
        key: values[-1]
        for key, values in parse_qs(urlparse(url).query, keep_blank_values=True).items()
        if key != "execution_id"
    }


class APIDeploymentsClientException(Exception):
    """A class to handle exceptions raised by the APIClient class."""

    def __init__(self, message):
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return repr(self.value)

        def error_message(self):
            return self.value


class _WaitRetryAfterOrExponentialJitter(wait_base):
    """Wait strategy that respects Retry-After on 429, else exponential jitter.

    For 429 responses with a valid ``Retry-After`` header the server-requested
    delay is used.  In every other case the strategy delegates to
    ``wait_exponential_jitter`` (additive jitter).
    """

    def __init__(
        self,
        initial: float,
        max: float,
        exp_base: float,
        jitter: float,
    ) -> None:
        super().__init__()
        self._exp_jitter = wait_exponential_jitter(
            initial=initial, max=max, exp_base=exp_base, jitter=jitter
        )

    def __call__(self, retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        if outcome and not outcome.failed:
            response = outcome.result()
            if response is not None and getattr(response, "status_code", None) == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        return float(retry_after)
                    except (ValueError, TypeError):
                        pass
        return self._exp_jitter(retry_state)


#: Request fields this client sets itself. Anything outside these sets is reset
#: to UNSET (body) or filtered out (query) before the request goes out, so no
#: parameter is sent that the caller did not ask for. A new parameter must be
#: added here to be sent at all.
_EXECUTE_SEND_ONLY = frozenset(
    {"timeout", "include_metadata", "files", "additional_properties"}
)
_STATUS_SEND_ONLY = frozenset({"execution_id", "include_metadata"})


class APIDeploymentsClient:
    """A class to invoke APIs deployed on the Unstract platform."""

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    log_stream_handler = logging.StreamHandler()
    log_stream_handler.setFormatter(formatter)
    logger.addHandler(log_stream_handler)

    api_key = ""
    api_timeout = 300
    in_progress_statuses = ["PENDING", "EXECUTING", "READY", "QUEUED", "INITIATED"]

    def __init__(
        self,
        api_url: str,
        api_key: str,
        api_timeout: int = 300,
        logging_level: str = "INFO",
        include_metadata: bool = False,
        verify: bool = True,
        max_retries: int = 4,
        initial_delay: float = 2.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: float = 1.0,
        *,
        transport_timeout: float | None = None,
    ):
        """Initializes the APIClient class.

        Args:
            api_key (str): The API key to authenticate the API request.
            api_timeout (int): The timeout to wait for the API response.
            logging_level (str): The logging level to log messages.
            max_retries (int): Maximum number of retry attempts for failed requests.
            initial_delay (float): Initial delay in seconds before the first retry.
            max_delay (float): Maximum delay in seconds between retries.
            backoff_factor (float): Multiplier applied to delay for each retry.
            jitter (float): Maximum additive jitter in seconds added to each delay.
            transport_timeout (float | None): Socket timeout in seconds. Unset
                means a stalled connection blocks forever, which is what the
                released client did; ``api_timeout`` cannot serve here because
                it is an execution mode, not a socket timeout.
        """
        if logging_level == "":
            logging_level = os.getenv("UNSTRACT_API_CLIENT_LOGGING_LEVEL", "INFO")
        if logging_level == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif logging_level == "INFO":
            self.logger.setLevel(logging.INFO)
        elif logging_level == "WARNING":
            self.logger.setLevel(logging.WARNING)
        elif logging_level == "ERROR":
            self.logger.setLevel(logging.ERROR)

        # self.logger.setLevel(logging_level)
        self.logger.debug("Logging level set to: " + logging_level)

        if api_key == "":
            self.api_key = os.getenv("UNSTRACT_API_DEPLOYMENT_KEY", "")
        else:
            self.api_key = api_key
        self.logger.debug("API key set to: " + UnstractUtils.redact_key(self.api_key))

        self.api_timeout = api_timeout
        self.api_url = api_url
        self.__save_base_url(api_url)
        self.include_metadata = include_metadata
        self.verify = verify
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.transport_timeout = transport_timeout

    def _is_retryable_status(self, status_code: int) -> bool:
        """Checks whether a status code should trigger a retry.

        Args:
            status_code (int): The HTTP status code to check.

        Returns:
            bool: True if the request should be retried.
        """
        return status_code >= 500 or status_code == 429

    def __save_base_url(self, full_url: str):
        """Extracts the base URL from the full URL and saves it.

        Args:
            full_url (str): The full URL of the API.
        """
        parsed_url = urlparse(full_url)
        self.base_url = parsed_url.scheme + "://" + parsed_url.netloc
        self.logger.debug("Base URL: " + self.base_url)

    @property
    def _transport(self):
        """The HTTP client, built on first use.

        Untimed by default, matching the previous behaviour. ``api_timeout`` is
        a backend execution mode (0 selects async execution), never a socket
        timeout; feeding it to the transport fails deep in the connection layer
        for the negative values the API accepts. ``transport_timeout`` is the
        way to bound a stalled connection.
        """
        if getattr(self, "_transport_client", None) is None:
            self._transport_client = AuthenticatedClient(
                base_url=self.base_url,
                token=self.api_key,
                verify_ssl=self.verify,
                timeout=httpx.Timeout(self.transport_timeout),
                raise_on_unexpected_status=False,
                # The previous transport followed redirects. Without this a 30x
                # from a load balancer is read as a terminal result with no
                # status, which a poll loop reports as a finished-and-empty job.
                follow_redirects=True,
            )
        return self._transport_client

    @property
    def _deployment_route(self) -> tuple[str, str]:
        """Organisation and API name, from the deployment URL's last two
        segments."""
        segments = urlparse(self.api_url).path.strip("/").split("/")
        if len(segments) < 2:
            raise APIDeploymentsClientException(
                f"Cannot derive organisation and API name from api_url: {self.api_url}"
            )
        return segments[-2], segments[-1]

    def _status_url(self, endpoint: str, path: str) -> str:
        """Absolute URL to poll, under the deployment's own path prefix.

        ``base_url`` is scheme and host only, so a deployment served under a path
        prefix would execute -- the execute call sends the caller's URL verbatim
        -- and then never poll. The prefix is whatever precedes the spec route
        inside the deployment URL. Where the two do not line up there is no
        prefix to derive, and the endpoint the service returned is used as it
        came: a guessed path polls nothing, and the execution behind it has
        already been paid for.
        """
        route = path.rstrip("/")
        prefix = urlparse(self.api_url).path.rstrip("/")
        if route and prefix.endswith(route):
            return self.base_url + prefix[: -len(route)] + path
        # Joined rather than concatenated: the query travels as params, and an
        # absolute endpoint has to stay the URL it already is.
        return urljoin(self.base_url, urlparse(endpoint)._replace(query="").geturl())

    def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Issue one request, translating transport failures on the way out.

        Translation happens here rather than around the retry loop, so
        the retry policy still sees the exception types it is configured
        to retry.
        """
        return _translate_transport_errors(
            self._transport.get_httpx_client().request, method, url, **kwargs
        )

    @staticmethod
    def _read_body(response):
        """Read the JSON body directly, never the generated response model.

        A model is only built for the statuses the spec declares, and error
        bodies are typed loosely, so an undeclared status or any error response
        has no usable model. ``None`` means the body was not JSON.
        """
        try:
            return response.json()
        except ValueError:
            return None

    @staticmethod
    def _rewind_files(files):
        """Rewinds file objects so they can be re-sent on retry."""
        for file_tuple in files:
            file_obj = file_tuple[1]
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            elif isinstance(file_obj, tuple) and len(file_obj) >= 2:
                if hasattr(file_obj[1], "seek"):
                    file_obj[1].seek(0)

    def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Makes an HTTP request with exponential backoff retry logic.

        Uses ``tenacity`` with additive jitter and Retry-After support.

        Args:
            method (str): The HTTP method (e.g., "GET", "POST").
            url (str): The request URL.
            **kwargs: Additional keyword arguments passed to the transport.

        Returns:
            The response from the request.

        Raises:
            ConnectionError: If a connection error persists after all retries.
            Timeout: If a timeout persists after all retries.
        """
        files = kwargs.get("files")

        def _before_sleep(retry_state: RetryCallState):
            attempt = retry_state.attempt_number
            delay = retry_state.next_action.sleep
            outcome = retry_state.outcome
            if outcome.failed:
                exc = outcome.exception()
                self.logger.warning(
                    "%s during request to %s. Retrying in %.1fs (attempt %d/%d).",
                    type(exc).__name__,
                    url,
                    delay,
                    attempt,
                    self.max_retries,
                )
            else:
                response = outcome.result()
                self.logger.warning(
                    "Request to %s returned %d. Retrying in %.1fs (attempt %d/%d).",
                    url,
                    response.status_code,
                    delay,
                    attempt,
                    self.max_retries,
                )
            # Rewind file objects before next attempt
            if files:
                self._rewind_files(files)

        def _retry_error_callback(retry_state: RetryCallState):
            outcome = retry_state.outcome
            if outcome.failed:
                exc = outcome.exception()
                self.logger.warning(
                    "%s during request to %s. Retries exhausted (%d/%d).",
                    type(exc).__name__,
                    url,
                    self.max_retries,
                    self.max_retries,
                )
                raise exc
            response = outcome.result()
            self.logger.warning(
                "Request to %s returned %d. Retries exhausted (%d/%d).",
                url,
                response.status_code,
                self.max_retries,
                self.max_retries,
            )
            return response

        retrier = Retrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=_WaitRetryAfterOrExponentialJitter(
                initial=self.initial_delay,
                max=self.max_delay,
                exp_base=self.backoff_factor,
                jitter=self.jitter,
            ),
            retry=(
                retry_if_result(lambda r: self._is_retryable_status(r.status_code))
                | retry_if_exception_type((ConnectionError, Timeout))
            ),
            before_sleep=_before_sleep,
            retry_error_callback=_retry_error_callback,
            sleep=time.sleep,
            reraise=False,
        )

        return retrier(self._send, method, url, **kwargs)

    def structure_file(
        self,
        file_paths: list[str],
        *,
        timeout: int | Unset = UNSET,
        include_metadata: bool | Unset = UNSET,
        include_metrics: bool | Unset = UNSET,
        include_extracted_text: bool | Unset = UNSET,
        use_file_history: bool | Unset = UNSET,
        tags: str | Unset = UNSET,
        llm_profile_id: str | None | Unset = UNSET,
        hitl_queue_name: str | None | Unset = UNSET,
        hitl_packet_id: str | None | Unset = UNSET,
        presigned_urls: list[str] | Unset = UNSET,
        custom_data: Any | Unset = UNSET,
    ) -> dict:
        """Invokes the API deployed on the Unstract platform.

        The keyword arguments are the request parameters the deployment accepts,
        named as the API names them. One left unset is not sent at all, so the
        server picks its own default; ``timeout`` and ``include_metadata`` fall
        back to the values given at construction.

        Args:
            file_paths (list[str]): The file path to the file to be uploaded.
            timeout (int): Execution mode — ``0`` or below runs asynchronously.
            include_metadata (bool): Include metadata in the result.
            include_metrics (bool): Include metrics in the result.
            include_extracted_text (bool): Include the extracted text.
            use_file_history (bool): Reuse a previous result for the same file.
            tags (str): Comma-separated tag names.
            llm_profile_id (str): LLM profile to override the deployment's.
            hitl_queue_name (str): Human-in-the-loop queue to route the file to.
            hitl_packet_id (str): Human-in-the-loop packet to attach the file to.
            presigned_urls (list[str]): URLs to fetch the inputs from.
            custom_data (Any): Arbitrary data echoed back with the result.

        Returns:
            dict: The response from the API.
        """
        self.logger.debug("Invoking API: " + self.api_url)
        self.logger.debug("File paths: " + str(file_paths))

        requested = {
            "timeout": timeout,
            "include_metadata": include_metadata,
            "include_metrics": include_metrics,
            "include_extracted_text": include_extracted_text,
            "use_file_history": use_file_history,
            "tags": tags,
            "llm_profile_id": llm_profile_id,
            "hitl_queue_name": hitl_queue_name,
            "hitl_packet_id": hitl_packet_id,
            "presigned_urls": presigned_urls,
            "custom_data": custom_data,
        }
        # ``None`` is dropped with ``UNSET``: these are optional overrides, and a
        # form field carries no null, so one would go out as the string "None"
        # for the service to look up.
        requested = {
            k: v
            for k, v in requested.items()
            if not isinstance(v, Unset) and v is not None
        }
        params = {
            "timeout": self.api_timeout,
            "include_metadata": self.include_metadata,
            **requested,
        }
        send_only = _EXECUTE_SEND_ONLY | requested.keys()

        handles = []
        try:
            for file_path in file_paths:
                handles.append(open(file_path, "rb"))
        except FileNotFoundError as e:
            for handle in handles:
                handle.close()
            raise APIDeploymentsClientException("File not found: " + str(e))

        body = ExecuteRequest(
            files=[
                File(
                    payload=handle,
                    file_name=ntpath.basename(file_path),
                    mime_type="application/octet-stream",
                )
                for file_path, handle in zip(file_paths, handles)
            ],
            **params,
        )
        # Only the fields this client sets are sent. Every other field carries the
        # spec's declared default, and sending a default is not the same as
        # omitting it: it pins a value the server would otherwise choose, and the
        # two diverge the moment the server's own default changes.
        for field in attrs.fields(ExecuteRequest):
            if field.name not in send_only:
                setattr(body, field.name, UNSET)

        org_name, api_name = self._deployment_route
        request_kwargs = execute._get_kwargs(org_name, api_name, body=body)
        # The generated builder pins a fixed multipart boundary in the header. An
        # uploaded file containing those bytes would break the encoding, so let
        # the transport pick a random boundary instead.
        request_kwargs.get("headers", {}).pop("Content-Type", None)
        method = request_kwargs.pop("method")
        request_kwargs.pop("url")
        # The deployment URL is the caller's, sent back verbatim. Rebuilding it
        # from the spec's path template drops any prefix the deployment is
        # served under, which no route template can express.
        url = self.api_url

        try:
            if params["timeout"] == 0:
                # The request only queues the execution, so a 5xx means queuing
                # failed and retrying cannot duplicate work.
                response = self._request_with_retry(method, url, **request_kwargs)
            else:
                # The request runs the execution, so a 5xx may mean it ran and
                # the response was lost: a retry would execute it twice.
                response = self._send(method, url, **request_kwargs)
        finally:
            for handle in handles:
                handle.close()
        self.logger.debug(response.status_code)
        self.logger.debug(response.text)
        # The returned object is wrapped in a "message" key.
        # Let's simplify the response.
        obj_to_return = {}

        response_data = self._read_body(response)
        if response_data is None:
            self.logger.error(
                "Failed to decode JSON response. Raw response: %s",
                response.text,
                exc_info=True,
            )
            obj_to_return = {
                "status_code": response.status_code,
                "pending": False,
                "execution_status": "",
                "error": "Invalid JSON response from API",
                "extraction_result": "",
            }
            return obj_to_return
        response_message = response_data.get("message", {})
        if response.status_code == 401:
            obj_to_return = {
                "status_code": response.status_code,
                "pending": False,
                "execution_status": "",
                "error": response_data.get("errors", [{}])[0].get(
                    "detail", "Unauthorized"
                ),
                "extraction_result": "",
            }
            return obj_to_return

        # If the execution status is pending, extract the execution ID from
        # the response and return it in the response.
        # Later, users can use the execution ID to check the status of the execution.
        # The returned object is wrapped in a "message" key.
        # Let's simplify the response.
        # Construct response object
        execution_status = response_message.get("execution_status", "")
        error_message = response_message.get("error", "")
        extraction_result = response_message.get("result", "")
        status_api_endpoint = response_message.get("status_api")

        obj_to_return = {
            "status_code": response.status_code,
            "pending": False,
            "execution_status": execution_status,
            "error": error_message,
            "extraction_result": extraction_result,
        }

        # Check if the status is pending or if it's successful but lacks a result.
        # The POST endpoint returns 200 for successful queuing (including
        # PENDING/EXECUTING) and 422 only on setup errors — guard against
        # incorrectly polling after an error response.
        if 200 <= response.status_code < 300:
            if execution_status in self.in_progress_statuses or (
                execution_status == "SUCCESS" and not extraction_result
            ):
                obj_to_return.update(
                    {
                        "status_check_api_endpoint": status_api_endpoint,
                        "pending": True,
                    }
                )

        return obj_to_return

    def check_execution_status(
        self,
        status_check_api_endpoint: str,
        *,
        include_metadata: bool | Unset = UNSET,
        include_metrics: bool | Unset = UNSET,
        include_extracted_text: bool | Unset = UNSET,
    ) -> dict:
        """Checks the status of the execution.

        The keyword arguments are the query parameters the endpoint accepts,
        named as the API names them. One left unset is not sent at all, so the
        server picks its own default; ``include_metadata`` falls back to the
        value given at construction.

        Args:
            status_check_api_endpoint (str):
                The API endpoint to check the status of the execution.
            include_metadata (bool): Include metadata in the result.
            include_metrics (bool): Include metrics in the result.
            include_extracted_text (bool): Include the extracted text.

        Returns:
            dict: The response from the API.
        """

        self.logger.debug(
            "Checking execution status via endpoint: " + status_check_api_endpoint
        )
        requested = {
            "include_metadata": include_metadata,
            "include_metrics": include_metrics,
            "include_extracted_text": include_extracted_text,
        }
        requested = {k: v for k, v in requested.items() if not isinstance(v, Unset)}
        params = {"include_metadata": self.include_metadata, **requested}

        org_name, api_name = self._deployment_route
        request_kwargs = status._get_kwargs(
            org_name,
            api_name,
            execution_id=_query_value(status_check_api_endpoint, "execution_id"),
            **params,
        )
        # The generated builder writes every declared query parameter, including
        # ones this client has never sent. Keep only what was asked for.
        send_only = _STATUS_SEND_ONLY | requested.keys()
        # Booleans are spelled the way urlencoding a Python bool spells them,
        # which is what the released client sent. The service reads either, but
        # traffic diffed against the previous release should show no change.
        request_kwargs["params"] = {
            **_forwarded_query(status_check_api_endpoint),
            **{
                k: str(v) if isinstance(v, bool) else v
                for k, v in request_kwargs["params"].items()
                if k in send_only
            },
        }
        response = self._request_with_retry(
            request_kwargs.pop("method"),
            self._status_url(status_check_api_endpoint, request_kwargs.pop("url")),
            **request_kwargs,
        )
        self.logger.debug(response.status_code)
        self.logger.debug(response.text)

        obj_to_return = {}

        response_data = self._read_body(response)
        if response_data is None:
            self.logger.error(
                "Failed to decode JSON response. Raw response: %s",
                response.text,
                exc_info=True,
            )
            obj_to_return = {
                "status_code": response.status_code,
                "pending": False,
                "execution_status": "",
                "error": "Invalid JSON response from API",
                "extraction_result": "",
            }
            return obj_to_return

        # Construct response object
        execution_status = response_data.get("status", "")
        error_message = response_data.get("error", "")
        extraction_result = response_data.get("message", "")

        obj_to_return = {
            "status_code": response.status_code,
            "pending": False,
            "execution_status": execution_status,
            "error": error_message,
            "extraction_result": extraction_result,
        }

        # If the execution status is pending, extract the execution ID from the response
        # and return it in the response.
        # Later, users can use the execution ID to check the status of the execution.
        if obj_to_return["execution_status"] in self.in_progress_statuses:
            obj_to_return["pending"] = True
        elif self._is_retryable_status(response.status_code):
            obj_to_return["pending"] = True
            self.logger.warning(
                "Status check returned %d after retries; "
                "marking as pending to continue polling.",
                response.status_code,
            )

        return obj_to_return
