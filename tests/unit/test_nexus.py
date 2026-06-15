"""
Tests for NeXus/HDF5 I/O (Phase 11).

Validates:
- RefinementState round-trip preserves all fields exactly.
- MeasurementState round-trip preserves all fields exactly.
- Unit-consistency: abscissa has the right units attribute per DataType.
- Parametric-model drop warning.
- Version attribute present on every file.
- Bad units raise ValueError on load.
"""

from __future__ import annotations

import numpy as np
import pytest

from aura.io.nexus import (
    AURA_FORMAT_VERSION,
    load_measurement_state,
    load_refinement_state,
    save_measurement_state,
    save_refinement_state,
)
from aura.models import DiffractionSlice, MeasurementState
from aura.spec import (
    AtomSite,
    DataType,
    Histogram,
    Parameter,
    ParametricModel,
    ParamKind,
    Phase,
    RefinementState,
    UnitCell,
)

pytest.importorskip("h5py")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CW_HIST = Histogram(
    id="cw1",
    data_type=DataType.CW_XRAY,
    x=np.linspace(10.0, 80.0, 50),
    y_obs=np.ones(50) * 100.0,
    weights=np.ones(50) * 0.01,
    driving={"T": 300.0},
    wavelength=1.5406,
)

TOF_HIST = Histogram(
    id="tof1",
    data_type=DataType.TOF,
    x=np.linspace(1000.0, 20000.0, 40),
    y_obs=np.ones(40) * 50.0,
    weights=np.ones(40) * 0.02,
    driving={"P": 1.0},
    difc=5400.0,
)

EDD_HIST = Histogram(
    id="edd1",
    data_type=DataType.EDD,
    x=np.linspace(20.0, 120.0, 30),
    y_obs=np.ones(30) * 20.0,
    weights=np.ones(30) * 0.05,
    driving={},
    two_theta_fixed=6.0,
)

NABR = Phase(
    "NaBr",
    "F m -3 m",
    UnitCell(5.9738, 5.9738, 5.9738),
    (AtomSite("Na", 0, 0, 0, 1.0, 0.5), AtomSite("Br", 0.5, 0.5, 0.5, 1.0, 0.6)),
)

PARAMS = (
    Parameter(
        "phase:NaBr:cell.a", ParamKind.PHASE, 5.97, vary=True, lower=5.8, upper=6.1
    ),
    Parameter(
        "hist:cw1:scale", ParamKind.HISTOGRAM, 1e-3, vary=True, lower=0.0, upper=1.0
    ),
    Parameter(
        "hist:cw1:bkg",
        ParamKind.HISTOGRAM,
        5.0,
        vary=False,
        sigma=0.2,
    ),
)


def _make_state() -> RefinementState:
    return RefinementState(
        phases=(NABR,),
        histograms=(CW_HIST, TOF_HIST),
        parameters=PARAMS,
        constraints=("sum_phase_fractions == 1",),
        provenance=("Phase 0 init",),
    )


# ---------------------------------------------------------------------------
# RefinementState round-trip
# ---------------------------------------------------------------------------


class TestRefinementStateRoundTrip:

    def test_phases_round_trip(self, tmp_path):
        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        rt = load_refinement_state(p)

        assert len(rt.phases) == 1
        ph = rt.phases[0]
        assert ph.name == "NaBr"
        assert ph.space_group == "F m -3 m"
        assert abs(ph.cell.a - 5.9738) < 1e-10
        assert len(ph.atoms) == 2
        assert ph.atoms[0].element == "Na"
        assert abs(ph.atoms[1].b_iso - 0.6) < 1e-10

    def test_histograms_round_trip(self, tmp_path):
        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        rt = load_refinement_state(p)

        assert len(rt.histograms) == 2
        cw = next(h for h in rt.histograms if h.id == "cw1")
        assert cw.data_type == DataType.CW_XRAY
        assert abs(cw.wavelength - 1.5406) < 1e-10
        assert cw.driving == {"T": 300.0}
        assert np.allclose(cw.x, CW_HIST.x)
        assert np.allclose(cw.y_obs, CW_HIST.y_obs)
        assert np.allclose(cw.weights, CW_HIST.weights)

        tof = next(h for h in rt.histograms if h.id == "tof1")
        assert tof.data_type == DataType.TOF
        assert abs(tof.difc - 5400.0) < 1e-10

    def test_parameters_round_trip(self, tmp_path):
        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        rt = load_refinement_state(p)

        assert len(rt.parameters) == 3
        a_param = next(q for q in rt.parameters if q.name == "phase:NaBr:cell.a")
        assert a_param.vary is True
        assert abs(a_param.lower - 5.8) < 1e-10
        bkg = next(q for q in rt.parameters if "bkg" in q.name)
        assert bkg.sigma is not None and abs(bkg.sigma - 0.2) < 1e-10

    def test_constraints_and_provenance_round_trip(self, tmp_path):
        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        rt = load_refinement_state(p)

        assert "sum_phase_fractions == 1" in rt.constraints
        assert "Phase 0 init" in rt.provenance

    def test_sigma_none_preserved(self, tmp_path):
        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        rt = load_refinement_state(p)
        a_param = next(q for q in rt.parameters if q.name == "phase:NaBr:cell.a")
        assert a_param.sigma is None

    def test_edd_histogram_round_trip(self, tmp_path):
        state = RefinementState(
            phases=(NABR,),
            histograms=(EDD_HIST,),
            parameters=PARAMS,
        )
        p = save_refinement_state(state, tmp_path / "state.h5")
        rt = load_refinement_state(p)
        edd = rt.histograms[0]
        assert edd.data_type == DataType.EDD
        assert abs(edd.two_theta_fixed - 6.0) < 1e-10


# ---------------------------------------------------------------------------
# Unit-consistency checks
# ---------------------------------------------------------------------------


class TestUnitConsistency:

    def test_version_attribute_present(self, tmp_path):
        import h5py

        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        with h5py.File(p, "r") as f:
            assert f.attrs["aura_version"] == AURA_FORMAT_VERSION

    def test_cw_abscissa_units_are_deg(self, tmp_path):
        import h5py

        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        with h5py.File(p, "r") as f:
            units = f["entry/histograms/cw1/x"].attrs["units"]
            assert units == "deg"

    def test_tof_abscissa_units_are_us(self, tmp_path):
        import h5py

        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        with h5py.File(p, "r") as f:
            units = f["entry/histograms/tof1/x"].attrs["units"]
            assert units == "us"

    def test_edd_abscissa_units_are_keV(self, tmp_path):
        import h5py

        state = RefinementState(
            phases=(NABR,),
            histograms=(EDD_HIST,),
            parameters=PARAMS,
        )
        p = save_refinement_state(state, tmp_path / "state.h5")
        with h5py.File(p, "r") as f:
            units = f["entry/histograms/edd1/x"].attrs["units"]
            assert units == "keV"

    def test_y_obs_labeled_counts(self, tmp_path):
        import h5py

        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        with h5py.File(p, "r") as f:
            units = f["entry/histograms/cw1/y_obs"].attrs["units"]
            assert units == "counts"

    def test_bad_units_raises_on_load(self, tmp_path):
        """Tampering with units attribute raises ValueError at load time."""
        import h5py

        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        with h5py.File(p, "a") as f:
            f["entry/histograms/cw1/x"].attrs["units"] = "furlongs"
        with pytest.raises(ValueError, match="unexpected abscissa units"):
            load_refinement_state(p)


# ---------------------------------------------------------------------------
# Parametric-model drop warning
# ---------------------------------------------------------------------------


def test_parametric_models_dropped_with_warning(tmp_path):
    from unittest.mock import MagicMock

    dummy_model = ParametricModel(
        target="phase:NaBr:cell.a",
        coeff_names=("param:alpha",),
        func=MagicMock(),
    )
    state = RefinementState(
        phases=(NABR,),
        histograms=(CW_HIST,),
        parameters=PARAMS,
        parametric_models=(dummy_model,),
    )
    with pytest.warns(UserWarning, match="Dropping 1 parametric model"):
        p = save_refinement_state(state, tmp_path / "state.h5")
    rt = load_refinement_state(p)
    assert rt.parametric_models == ()


# ---------------------------------------------------------------------------
# MeasurementState round-trip
# ---------------------------------------------------------------------------


class TestMeasurementStateRoundTrip:

    def _make_ms(self) -> MeasurementState:
        slc1 = DiffractionSlice(
            id="run1_bank1",
            x=np.linspace(1000.0, 5000.0, 20),
            y=np.ones(20) * 50.0,
            e=np.ones(20) * 7.1,
            metadata={"two_theta": 90.0, "difc": 5400.0, "data_type": "tof"},
        )
        slc2 = DiffractionSlice(
            id="run1_bank2",
            x=np.linspace(500.0, 4000.0, 25),
            y=np.ones(25) * 30.0,
            e=np.ones(25) * 5.5,
            metadata={"bank_num": 2},
        )
        return MeasurementState(id="run1", slices=[slc1, slc2])

    def test_id_round_trip(self, tmp_path):
        ms = self._make_ms()
        p = save_measurement_state(ms, tmp_path / "ms.h5")
        rt = load_measurement_state(p)
        assert rt.id == "run1"  # type: ignore[attr-defined]

    def test_slice_count_round_trip(self, tmp_path):
        ms = self._make_ms()
        p = save_measurement_state(ms, tmp_path / "ms.h5")
        rt = load_measurement_state(p)
        assert len(rt.slices) == 2  # type: ignore[attr-defined]

    def test_slice_arrays_round_trip(self, tmp_path):
        ms = self._make_ms()
        p = save_measurement_state(ms, tmp_path / "ms.h5")
        rt = load_measurement_state(p)
        slc = next(s for s in rt.slices if s.id == "run1_bank1")  # type: ignore[attr-defined]
        assert np.allclose(slc.x, np.linspace(1000.0, 5000.0, 20))
        assert np.allclose(slc.y, 50.0)
        assert np.allclose(slc.e, 7.1)

    def test_scalar_metadata_round_trip(self, tmp_path):
        ms = self._make_ms()
        p = save_measurement_state(ms, tmp_path / "ms.h5")
        rt = load_measurement_state(p)
        slc = next(s for s in rt.slices if s.id == "run1_bank1")  # type: ignore[attr-defined]
        assert abs(slc.metadata["two_theta"] - 90.0) < 1e-10

    def test_wrong_content_type_raises(self, tmp_path):
        state = _make_state()
        p = save_refinement_state(state, tmp_path / "state.h5")
        with pytest.raises(ValueError, match="MeasurementState"):
            load_measurement_state(p)

    def test_wrong_content_type_refinement_raises(self, tmp_path):
        ms = self._make_ms()
        p = save_measurement_state(ms, tmp_path / "ms.h5")
        with pytest.raises(ValueError, match="RefinementState"):
            load_refinement_state(p)
