"""
Tests for the Aura CLI.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from aura.cli import main

DATA_DIR = Path(__file__).parent.parent / "testDataGsas"


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


def test_cli_subcommands_listed():
    result = CliRunner().invoke(main, ["--help"])
    assert "identify" in result.output
    assert "refine" in result.output


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="no test data")
@pytest.mark.integration
class TestCliOnRealData:

    def test_identify(self):
        result = CliRunner().invoke(
            main,
            [
                "identify",
                str(DATA_DIR / "PBSO4.CWN"),
                "--candidate",
                str(DATA_DIR / "PbSO4.cif"),
                "--candidate",
                str(DATA_DIR / "NAC.cif"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Candidates" in result.output

    def test_refine_pbso4(self, tmp_path):
        plot = tmp_path / "fit.png"
        ckpt = tmp_path / "state.pkl"
        result = CliRunner().invoke(
            main,
            [
                "refine",
                str(DATA_DIR / "PBSO4.CWN"),
                str(DATA_DIR / "PbSO4.cif"),
                "--background-order",
                "6",
                "--max-iter",
                "60",
                "--plot",
                str(plot),
                "--out",
                str(ckpt),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Rwp" in result.output
        assert plot.is_file() and plot.stat().st_size > 0
        assert ckpt.is_file()
