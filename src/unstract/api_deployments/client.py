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
from urllib.parse import urlparse

import requests
from requests.exceptions import ConnectionError, JSONDecodeError, Timeout
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import wait_base

from unstract.api_deployments.utils import UnstractUtils


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
            if (
                response is not None
                and getattr(response, "status_code", None) == 429
            ):
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        return float(retry_after)
                    except (ValueError, TypeError):
                        pass
        return self._exp_jitter(retry_state)


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

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Makes an HTTP request with exponential backoff retry logic.

        Uses ``tenacity`` with additive jitter and Retry-After support.

        Args:
            method (str): The HTTP method (e.g., "GET", "POST").
            url (str): The request URL.
            **kwargs: Additional keyword arguments passed to requests.request().

        Returns:
            requests.Response: The response from the request.

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
                    "%s during request to %s. Retrying in %.1fs "
                    "(attempt %d/%d).",
                    type(exc).__name__,
                    url,
                    delay,
                    attempt,
                    self.max_retries,
                )
            else:
                response = outcome.result()
                self.logger.warning(
                    "Request to %s returned %d. Retrying in %.1fs "
                    "(attempt %d/%d).",
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
                retry_if_result(
                    lambda r: self._is_retryable_status(r.status_code)
                )
                | retry_if_exception_type((ConnectionError, Timeout))
            ),
            before_sleep=_before_sleep,
            retry_error_callback=_retry_error_callback,
            sleep=time.sleep,
            reraise=False,
        )

        return retrier(requests.request, method, url, **kwargs)

    def structure_file(self, file_paths: list[str]) -> dict:
        """Invokes the API deployed on the Unstract platform.

        Args:
            file_paths (list[str]): The file path to the file to be uploaded.

        Returns:
            dict: The response from the API.
        """
        self.logger.debug("Invoking API: " + self.api_url)
        self.logger.debug("File paths: " + str(file_paths))

        headers = {
            "Authorization": "Bearer " + self.api_key,
        }

        form_data = {
            "timeout": self.api_timeout,
            "include_metadata": self.include_metadata,
        }

        files = []

        try:
            for file_path in file_paths:
                record = (
                    "files",
                    (
                        ntpath.basename(file_path),
                        open(file_path, "rb"),
                        "application/octet-stream",
                    ),
                )
                files.append(record)
        except FileNotFoundError as e:
            raise APIDeploymentsClientException("File not found: " + str(e))

        if self.api_timeout == 0:
            # Async mode: server returns immediately after queuing.
            # A 5xx means queuing failed — safe to retry.
            response = self._request_with_retry(
                "POST",
                self.api_url,
                headers=headers,
                data=form_data,
                files=files,
                verify=self.verify,
            )
        else:
            # Sync mode: server blocks during processing.
            # A 5xx may mean it processed but response was lost — don't retry
            # to avoid duplicate executions.
            response = requests.post(
                self.api_url,
                headers=headers,
                data=form_data,
                files=files,
                verify=self.verify,
            )
        self.logger.debug(response.status_code)
        self.logger.debug(response.text)
        # The returned object is wrapped in a "message" key.
        # Let's simplify the response.
        obj_to_return = {}

        try:
            response_data = response.json()
            response_message = response_data.get("message", {})
        except JSONDecodeError:
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

    def check_execution_status(self, status_check_api_endpoint: str) -> dict:
        """Checks the status of the execution.

        Args:
            status_check_api_endpoint (str):
                The API endpoint to check the status of the execution.

        Returns:
            dict: The response from the API.
        """

        headers = {
            "Authorization": "Bearer " + self.api_key,
        }
        status_call_url = self.base_url + status_check_api_endpoint
        self.logger.debug("Checking execution status via endpoint: " + status_call_url)
        response = self._request_with_retry(
            "GET",
            status_call_url,
            headers=headers,
            params={"include_metadata": self.include_metadata},
            verify=self.verify,
        )
        self.logger.debug(response.status_code)
        self.logger.debug(response.text)

        obj_to_return = {}

        try:
            response_data = response.json()
        except JSONDecodeError:
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
