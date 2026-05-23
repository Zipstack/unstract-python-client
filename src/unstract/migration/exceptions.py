"""Exceptions raised by the migration subpackage."""


class MigrationError(Exception):
    """Base class for all migration errors."""


class PlatformAPIError(MigrationError):
    """Raised when the Platform API returns a non-2xx response we can't recover from."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class NameConflictError(MigrationError):
    """Raised when ``on_name_conflict='abort'`` and the target has a like-named entity."""


class DependencyMissingError(MigrationError):
    """Raised when a phase references a source UUID that no prior phase has mapped."""
