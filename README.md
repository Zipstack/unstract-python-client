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

`api_timeout`: Set a timeout for API requests, e.g., `api_timeout=10`.
`logging_level`: Set logging verbosity (e.g., "`DEBUG`").
`include_metadata`: If set to `True`, the response will include additional metadata (cost, tokens consumed and context) for each call made by the Prompt Studio exported tool.


## Questions and Feedback

On Slack, [join great conversations](https://join-slack.unstract.com/) around LLMs, their ecosystem and leveraging them to automate the previously unautomatable!

[Unstract Cloud](https://unstract.com/): Signup and Try!

[Unstract developer documentation](https://docs.unstract.com/): Learn more about Unstract and its API.
