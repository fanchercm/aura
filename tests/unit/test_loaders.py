"""Tests for GSAS loaders using real SNAP data."""

from pathlib import Path

import numpy as np
import pytest

from aura.loaders import load_campaign_from_directory, load_gsa_file, load_instprm

# Path to the real test data shipped with the repo
DATA_DIR = Path(__file__).parent.parent / "testDataGsas"
SAMPLE_GSA = DATA_DIR / "SNAP067702_column.gsa"
INSTPRM = DATA_DIR / "SNAP066787_column.instprm"


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _check_data_exists():
    """Skip all tests in this module if test data is absent."""
    if not DATA_DIR.is_dir():
        pytest.skip("Test data directory not found")


# ---------------------------------------------------------------------------
# load_gsa_file
# ---------------------------------------------------------------------------

class TestLoadGsaFile:
    def test_loads_single_file(self):
        state = load_gsa_file(SAMPLE_GSA)
        assert state.id == "SNAP067702"
        assert state.n_slices == 6

    def test_run_number_in_metadata(self):
        state = load_gsa_file(SAMPLE_GSA)
        assert state.metadata["run_number"] == 67702

    def test_bank_metadata(self):
        state = load_gsa_file(SAMPLE_GSA)
        bank1 = state.get_slice("SNAP067702_bank1")
        assert bank1.metadata["bank_num"] == 1
        assert bank1.metadata["two_theta"] == pytest.approx(122.296, abs=0.01)
        assert bank1.metadata["flight_path"] == pytest.approx(15.568, abs=0.01)
        assert bank1.metadata["difc"] == pytest.approx(6890.56, abs=0.1)

    def test_bank_data_shape(self):
        state = load_gsa_file(SAMPLE_GSA)
        bank1 = state.get_slice("SNAP067702_bank1")
        # BANK 1 header says 3386 channels
        assert bank1.n_points == 3386
        assert bank1.x.shape == (3386,)
        assert bank1.y.shape == (3386,)
        assert bank1.e.shape == (3386,)

    def test_bank1_has_leading_nans(self):
        """Bank 1 of run 67702 has NaN values at the start."""
        state = load_gsa_file(SAMPLE_GSA)
        bank1 = state.get_slice("SNAP067702_bank1")
        assert bank1.n_valid < bank1.n_points
        # First values should be NaN
        assert np.isnan(bank1.y[0])

    def test_all_six_banks_present(self):
        state = load_gsa_file(SAMPLE_GSA)
        bank_nums = [s.metadata["bank_num"] for s in state.slices]
        assert bank_nums == [1, 2, 3, 4, 5, 6]

    def test_two_theta_decreases_with_bank(self):
        """Higher bank numbers have lower two-theta on SNAP."""
        state = load_gsa_file(SAMPLE_GSA)
        tth = [s.metadata["two_theta"] for s in state.slices]
        for a, b in zip(tth, tth[1:]):
            assert a > b

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_gsa_file("/nonexistent/file.gsa")

    def test_slice_ids_are_unique(self):
        state = load_gsa_file(SAMPLE_GSA)
        ids = [s.id for s in state.slices]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# load_instprm
# ---------------------------------------------------------------------------

class TestLoadInstprm:
    def test_loads_all_banks(self):
        params = load_instprm(INSTPRM)
        assert set(params.keys()) == {1, 2, 3, 4, 5, 6}

    def test_bank1_values(self):
        params = load_instprm(INSTPRM)
        b1 = params[1]
        assert b1["difC"] == pytest.approx(6877.054, abs=0.01)
        assert b1["2-theta"] == pytest.approx(121.3, abs=0.1)
        assert b1["fltPath"] == pytest.approx(15.5845, abs=0.01)

    def test_string_values_preserved(self):
        params = load_instprm(INSTPRM)
        assert params[1]["Type"] == "PNT"
        assert params[1]["Diff-type"] == "Debye-Scherrer"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_instprm("/nonexistent/file.instprm")


# ---------------------------------------------------------------------------
# load_campaign_from_directory
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestLoadCampaign:
    def test_loads_all_runs(self):
        campaign = load_campaign_from_directory(DATA_DIR)
        # 19 .gsa files in the directory
        assert campaign.n_states == 19

    def test_total_slices(self):
        campaign = load_campaign_from_directory(DATA_DIR)
        # 19 runs × 6 banks = 114 slices
        assert campaign.n_slices_total == 114

    def test_states_ordered_by_run_number(self):
        campaign = load_campaign_from_directory(DATA_DIR)
        run_nums = [s.metadata["run_number"] for s in campaign.states]
        assert run_nums == sorted(run_nums)

    def test_campaign_id_defaults_to_dirname(self):
        campaign = load_campaign_from_directory(DATA_DIR)
        assert campaign.id == "testDataGsas"

    def test_custom_campaign_id(self):
        campaign = load_campaign_from_directory(
            DATA_DIR, campaign_id="NaBr_Pb_pressure"
        )
        assert campaign.id == "NaBr_Pb_pressure"

    def test_directory_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_campaign_from_directory("/nonexistent/dir")

    def test_no_matching_files(self, tmp_path):
        with pytest.raises(ValueError, match="No files matching"):
            load_campaign_from_directory(tmp_path)
