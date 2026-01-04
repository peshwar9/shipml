"""Model loaders for different ML frameworks."""

from shipml.loaders.base import ModelLoader
from shipml.loaders.detector import detect_framework, get_loader

__all__ = ["ModelLoader", "detect_framework", "get_loader"]
