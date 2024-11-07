import os
import time

import dotenv

from unstract.api_deployments.client import (
    APIDeploymentsClient,
    APIDeploymentsClientException,
)

dotenv.load_dotenv()


def main():
    try:
        adc = APIDeploymentsClient(
            api_url=os.getenv("API_URL"),
            api_key=os.getenv("UNSTRACT_API_DEPLOYMENT_KEY"),
            api_timeout=10,
            logging_level="DEBUG",
            include_metadata=False,
        )
        # Replace files with pdfs
        response = adc.structure_file(["<files>"])
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


if __name__ == "__main__":
    main()
