__version__ = "1.4.0"

from .client import APIDeploymentsClient as APIDeploymentsClient


def get_sdk_version():
    return __version__
