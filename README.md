# unstract-python-client
[![PyPI - Downloads](https://img.shields.io/pypi/dm/unstract-client)](https://pypi.org/project/unstract-client/)
[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FZipstack%2Funstract-python-client%2Fmain%2Fpyproject.toml)
](https://pypi.org/project/unstract-client/)
[![PyPI - Version](https://img.shields.io/pypi/v/unstract-client)](https://pypi.org/project/unstract-client/)

Python client for the Unstract LLM-powered structured data extraction platform


## Installation

You can install the Unstract Python Client using pip:

```bash
pip install unstract-client
```

## Usage

First, import the `APIDeploymentsClient` from the `client` module:

```python
from unstract.api_deployments.client import APIDeploymentsClient
```

Then, create an instance of the `APIDeploymentsClient`:

```python
client = APIDeploymentsClient(api_url="url", api_key="your_api_key")
```

> **Note:** Pass the raw API key **without** the `"Bearer "` prefix — the client adds it automatically.

Now, you can use the client to interact with the Unstract API deployments API:

```python
try:
    adc = APIDeploymentsClient(
        api_url=os.getenv("UNSTRACT_API_URL"),
        api_key=os.getenv("UNSTRACT_API_DEPLOYMENT_KEY"),
        api_timeout=10,
        logging_level="DEBUG",
        include_metadata=False # optional
    )
    # Replace files with pdfs
    response = adc.structure_file(
        ["<files>"]
    )
    print(response)
    if response["pending"]:
        while True:
            p_response = adc.check_execution_status(
                response["status_check_api_endpoint"]
            )
            print(p_response)
            if not p_response["pending"]:
                break
            print("Sleeping and checking again in 5 seconds..")
            time.sleep(5)
except APIDeploymentsClientException as e:
    print(e)
```

## Parameter Details

`api_url`: The URL of the Unstract API deployment.
`api_key`: Your raw API key. **Do not** include the `"Bearer "` prefix — the client adds it automatically.
`api_timeout`: Set a timeout for API requests, e.g., `api_timeout=10`.
`logging_level`: Set logging verbosity (e.g., "`DEBUG`").
`include_metadata`: If set to `True`, the response will include additional metadata (cost, tokens consumed and context) for each call made by the Prompt Studio exported tool.

## Retry Configuration

The client includes built-in exponential backoff retry with the following behavior:

- **Async mode** (`api_timeout=0`): POST requests are retried on transient failures (5xx, 429) and connection errors, since the server returns immediately after queuing.
- **Sync mode** (`api_timeout > 0`, the default): POST requests are **not** retried, because the server blocks during processing — a failure may mean the request was processed but the response was lost.
- **Status polling** (`check_execution_status`): GET requests are always retried, as they are idempotent.

Retries are enabled by default and can be customized:

```python
client = APIDeploymentsClient(
    api_url="url",
    api_key="your_api_key",
    max_retries=4,       # Max retry attempts (default: 4, set to 0 to disable)
    initial_delay=2.0,   # Initial delay in seconds (default: 2.0)
    max_delay=60.0,      # Maximum delay cap in seconds (default: 60.0)
    backoff_factor=2.0,  # Multiplier per retry (default: 2.0)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_retries` | `4` | Maximum number of retry attempts. Set to `0` to disable retries. |
| `initial_delay` | `2.0` | Initial delay in seconds before the first retry. |
| `max_delay` | `60.0` | Maximum delay cap in seconds between retries. |
| `backoff_factor` | `2.0` | Multiplier applied to the delay for each subsequent retry. |

The retry logic uses exponential backoff with full jitter and respects the `Retry-After` header on 429 responses.


## Questions and Feedback

On Slack, [join great conversations](https://join-slack.unstract.com/) around LLMs, their ecosystem and leveraging them to automate the previously unautomatable!

[Unstract Cloud](https://unstract.com/): Signup and Try!

[Unstract developer documentation](https://docs.unstract.com/): Learn more about Unstract and its API.
