__version__ = "1.2.1"

from .client import APIDeploymentsClient as APIDeploymentsClient


def get_sdk_version():
    return __version__
