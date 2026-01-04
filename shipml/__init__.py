"""ShipML - Turn ML models into APIs with one command."""

__version__ = "0.1.0"
__author__ = "Prabhueshwar La"
__license__ = "MIT"

from shipml.errors import ShipMLError, UnsupportedModelError, ModelLoadError, ValidationError

__all__ = [
    "__version__",
    "ShipMLError",
    "UnsupportedModelError",
    "ModelLoadError",
    "ValidationError",
]
