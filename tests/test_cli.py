"""Tests for CLI."""

import pytest
from click.testing import CliRunner

from shipml.cli import cli


def test_version():
    """Test --version flag."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "ShipML version" in result.output


def test_help():
    """Test --help flag."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "ShipML" in result.output
    assert "serve" in result.output


def test_serve_help():
    """Test serve --help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--help"])

    assert result.exit_code == 0
    assert "MODEL_FILE" in result.output
    assert "--port" in result.output


def test_serve_nonexistent_file():
    """Test serve with non-existent file."""
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "nonexistent.pkl"])

    assert result.exit_code != 0
