"""
Tests for the native importer registry and the powder/phase/instrument readers,
plus the container→spec bridge, exercised on the real benchmark data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import aura.io as io
from aura.bridge import campaign_to_histograms, state_to_histograms
from aura.io.registry import UnsupportedFormatError
from aura.loaders import load_campaign_from_directory
from aura.spec import DataType

DATA_DIR = Path(__file__).parent.parent / "testDataGsas"


@pytest.fixture(autouse=True)
def _require_data():
    if not DATA_DIR.is_dir():
        pytest.skip("Test data directory not found")


# --- Registry behaviour --------------------------------------------------------


class TestRegistry:

    def test_readers_registered_in_every_core_domain(self):
        for domain in ("powder", "phase", "instrument"):
            assert io.registry.readers(domain), f"no readers for {domain}"

    def test_unknown_format_raises_clear_error(self, tmp_path):
        f = tmp_path / "mystery.unknownext"
        f.write_text("nothing parseable here\n")
        with pytest.raises(UnsupportedFormatError):
            io.find_reader(f, "powder")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            io.find_reader(DATA_DIR / "does_not_exist.gsa", "powder")

    def test_cif_resolves_to_phase_not_powder(self):
        reader = io.find_reader(DATA_DIR / "Phase_NaBr.cif", "phase")
        assert reader.domain == "phase"
        assert reader.name == "CIF"

    def test_content_sniffing_picks_const_over_column_for_xra(self):
        # .xra is claimed only by GSAS CONST; .gsa is shared (column vs CONST).
        reader = io.find_reader(DATA_DIR / "PBSO4.XRA", "powder")
        assert reader.name == "GSAS CONST"

    def test_stub_formats_are_registered_but_raise(self, tmp_path):
        f = tmp_path / "x.xrdml"
        f.write_text("<xrdMeasurements/>\n")
        reader = io.find_reader(f, "powder")
        with pytest.raises(NotImplementedError):
            reader.read(f)


# --- Powder readers ------------------------------------------------------------


class TestPowderReaders:

    @pytest.mark.parametrize(
        "fname,dtype,n,wl",
        [
            ("PBSO4.CWN", DataType.CW_NEUTRON, 2919, 1.909),
            ("PBSO4.XRA", DataType.CW_XRAY, 6001, 1.5406),
            ("FAP.XRA", DataType.CW_XRAY, 5753, None),
        ],
    )
    def test_gsas_const(self, fname, dtype, n, wl):
        st = io.read(DATA_DIR / fname, "powder")
        s = st.slices[0]
        assert s.n_points == n
        assert s.metadata["data_type"] is dtype
        assert s.metadata["wavelength"] == wl
        assert np.all(np.isfinite(s.y))
        assert np.all(s.y >= 0)

    def test_cwn_packed_decode_first_point(self):
        # CWN line 1 datum is ndet=1, counts=220 -> y=220, esd=sqrt(220).
        s = io.read(DATA_DIR / "PBSO4.CWN", "powder").slices[0]
        assert s.y[0] == 220.0
        assert abs(s.e[0] - np.sqrt(220.0)) < 1e-9

    def test_fxye_x_in_centidegrees(self):
        s = io.read(DATA_DIR / "11BM_NAC.fxye", "powder").slices[0]
        # First point 50 centideg -> 0.5 deg; ascending; high-res synchrotron range.
        assert abs(s.x[0] - 0.5) < 1e-6
        assert s.x[-1] < 60.0
        assert np.all(np.diff(s.x) > 0)

    def test_gsa_tof_reads_six_banks(self):
        st = io.read(DATA_DIR / "SNAP067702_column.gsa", "powder")
        assert len(st.slices) == 6
        assert all(s.metadata["data_type"] is DataType.TOF for s in st.slices)


# --- Phase reader --------------------------------------------------------------


class TestPhaseReader:

    @pytest.mark.parametrize(
        "fname,sg,a,natoms",
        [
            ("Phase_NaBr.cif", "F m -3 m", 5.9738, 2),
            ("Phase2_Pb.cif", "F m -3 m", 4.9500, 1),
            ("NAC.cif", "I 21 3", 10.2512, 6),
        ],
    )
    def test_cif_phase(self, fname, sg, a, natoms):
        ph = io.read(DATA_DIR / fname, "phase")
        assert ph.space_group == sg
        assert abs(ph.cell.a - a) < 1e-3
        assert ph.cell.is_physical()
        assert len(ph.atoms) == natoms
        assert all(at.occ > 0 and at.b_iso >= 0 for at in ph.atoms)


# --- Instrument readers --------------------------------------------------------


class TestInstrumentReaders:

    def test_instprm_tof_six_banks(self):
        inst = io.read(DATA_DIR / "SNAP066787_column.instprm", "instrument")
        assert sorted(inst) == [1, 2, 3, 4, 5, 6]
        assert inst[1]["data_type"] is DataType.TOF
        assert inst[1]["difc"] > 0

    def test_prm_d1a_cw_neutron_wavelength(self):
        inst = io.read(DATA_DIR / "inst_d1a.prm", "instrument")
        assert inst[1]["data_type"] is DataType.CW_NEUTRON
        assert abs(inst[1]["wavelength"] - 1.909) < 1e-6


# --- Bridge --------------------------------------------------------------------


class TestBridge:

    def test_snap_campaign_to_histograms(self):
        camp = load_campaign_from_directory(DATA_DIR)
        hs = campaign_to_histograms(camp)
        assert len(hs) == camp.n_slices_total == 114
        h = hs[0]
        assert h.data_type is DataType.TOF
        assert h.difc is not None and h.difc > 0
        # Leading NaN region dropped by the valid mask.
        assert len(h.x) == len(h.y_obs) == len(h.weights)
        assert np.all(np.isfinite(h.y_obs))
        assert "run_number" in h.driving

    def test_cw_bridge_sets_weights_and_wavelength(self):
        st = io.read(DATA_DIR / "PBSO4.CWN", "powder")
        h = state_to_histograms(st)[0]
        assert h.data_type is DataType.CW_NEUTRON
        assert h.wavelength == 1.909
        # Poisson weight on the first channel: 1/220.
        assert abs(float(h.weights[0]) - 1.0 / 220.0) < 1e-9

    def test_cw_without_wavelength_raises(self):
        st = io.read(DATA_DIR / "11BM_NAC.fxye", "powder")
        with pytest.raises(ValueError, match="wavelength"):
            state_to_histograms(st)
        # ...but succeeds when supplied.
        h = state_to_histograms(st, wavelength=0.4127)[0]
        assert h.wavelength == 0.4127
