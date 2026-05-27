"""Exceptions raised by the clone subpackage."""


class CloneError(Exception):
    """Base class for all clone errors."""


class PlatformAPIError(CloneError):
    """Raised when the Platform API returns a non-2xx response we can't recover from."""

    def __init__(
        self, message: str, status_code: int | None = None, body: str | None = None
    ):
        full_message = f"{message}\n  body: {body}" if body else message
        super().__init__(full_message)
        self.status_code = status_code
        self.body = body


class NameConflictError(CloneError):
    """Raised on name collision when ``on_name_conflict='abort'``."""


class DependencyMissingError(CloneError):
    """Raised when a phase references a source UUID that no prior phase has mapped."""
