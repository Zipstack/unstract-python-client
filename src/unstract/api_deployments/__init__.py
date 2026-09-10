__version__ = "1.6.0"

from .client import APIDeploymentsClient as APIDeploymentsClient
from .client import PlatformAPIClient as PlatformAPIClient


def get_sdk_version():
    return __version__
