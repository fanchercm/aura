"""
Tests for the Aura CLI.
"""

from click.testing import CliRunner

from aura.cli import main


def test_cli_help():
    """Test CLI --help runs successfully."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "Aura" in result.output


def test_cli_version():
    """Test CLI --version outputs version string."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output
