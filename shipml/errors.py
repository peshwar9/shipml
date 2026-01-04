"""Custom exceptions for ShipML."""


class ShipMLError(Exception):
    """Base exception for all ShipML errors."""

    pass


class UnsupportedModelError(ShipMLError):
    """Raised when model framework cannot be detected or is not supported."""

    pass


class ModelLoadError(ShipMLError):
    """Raised when model file cannot be loaded."""

    pass


class ValidationError(ShipMLError):
    """Raised when input validation fails."""

    pass
