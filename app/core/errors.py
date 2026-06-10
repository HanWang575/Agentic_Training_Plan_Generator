"""Small shared exception types with user-readable messages."""


class ConfigurationError(RuntimeError):
    """Raised when a requested action needs configuration that is missing."""


class UnsupportedFileTypeError(ValueError):
    """Raised when a user uploads a file type the parser does not support."""


class NotFoundError(LookupError):
    """Raised when a requested source, chunk, or trace is not present."""
