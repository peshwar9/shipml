"""
Fallback setup.py for older pip versions.
Modern installations should use pyproject.toml with:
    uv pip install .
"""

from setuptools import setup

# All configuration is in pyproject.toml
setup()
