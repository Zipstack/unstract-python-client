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
from urllib.parse import urlparse

import requests
from requests.exceptions import JSONDecodeError

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
    ):
        """Initializes the APIClient class.

        Args:
            api_key (str): The API key to authenticate the API request.
            api_timeout (int): The timeout to wait for the API response.
            logging_level (str): The logging level to log messages.
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

    def __save_base_url(self, full_url: str):
        """Extracts the base URL from the full URL and saves it.

        Args:
            full_url (str): The full URL of the API.
        """
        parsed_url = urlparse(full_url)
        self.base_url = parsed_url.scheme + "://" + parsed_url.netloc
        self.logger.debug("Base URL: " + self.base_url)

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

        data = {"timeout": self.api_timeout, "include_metadata": self.include_metadata}

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

        response = requests.post(
            self.api_url,
            headers=headers,
            data=data,
            files=files,
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

        # Check if the status is pending or if it's successful but lacks a result
        if 200 <= response.status_code < 300:
            if execution_status in self.in_progress_statuses or (
                execution_status == "SUCCESS" and not extraction_result
            ):
                obj_to_return.update(
                    {"status_check_api_endpoint": status_api_endpoint, "pending": True}
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
        response = requests.get(
            status_call_url,
            headers=headers,
            params={"include_metadata": self.include_metadata},
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
        if (
            200 <= response.status_code < 500
            and obj_to_return["execution_status"] in self.in_progress_statuses
        ):
            obj_to_return["pending"] = True

        return obj_to_return
