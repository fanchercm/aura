"""
Production forward-model invariants (Phase 3).

Exercises :class:`aura.engine.forward.ProductionForward` directly (the shared
``engine`` fixture stays on the reference engine until the production minimizer
lands in Phase 4). Covers forward-model invariants (purity, non-negativity,
Jacobian sanity), rigorous peak-position correctness, and real-data positional
alignment on the SNAP TOF series.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import aura.io as io
from aura import spec
from aura.bridge import state_to_histograms
from aura.engine import positions, symmetry
from aura.engine.forward import ProductionEngine, ProductionForward
from aura.spec import (
    AtomSite,
    DataType,
    Histogram,
    Parameter,
    ParamKind,
    Phase,
    RefinementState,
    UnitCell,
)

DATA_DIR = Path(__file__).parent.parent / "testDataGsas"

# APS 11-BM calibrated wavelength for the NAC dataset (not stored in the .fxye).
NAC_11BM_WAVELENGTH = 0.413909

NABR = Phase(
    "NaBr",
    "F m -3 m",
    UnitCell(5.9738, 5.9738, 5.9738),
    (AtomSite("Na", 0, 0, 0), AtomSite("Br", 0.5, 0.5, 0.5)),
)


def _hist(dtype=DataType.CW_NEUTRON, n=2001, wavelength=1.909) -> Histogram:
    x = np.linspace(10.0, 120.0, n)
    return Histogram(
        "syn", dtype, x, np.zeros(n), np.ones(n), {}, wavelength=wavelength
    )


def _params(hid="syn", scale=1.0, bkg=0.0, fwhm=0.3) -> tuple[Parameter, ...]:
    return (
        Parameter(
            f"hist:{hid}:scale",
            ParamKind.HISTOGRAM,
            scale,
            vary=True,
            lower=0,
            upper=1e6,
        ),
        Parameter(
            f"hist:{hid}:bkg", ParamKind.HISTOGRAM, bkg, vary=True, lower=0, upper=1e6
        ),
        Parameter(
            f"hist:{hid}:fwhm",
            ParamKind.HISTOGRAM,
            fwhm,
            vary=True,
            lower=0.02,
            upper=5.0,
        ),
        Parameter(f"hist:{hid}:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
        Parameter(
            "phase:NaBr:cell.a",
            ParamKind.PHASE,
            5.9738,
            vary=True,
            lower=5.5,
            upper=6.5,
        ),
    )


# --- Forward-model invariants --------------------------------------------------


class TestForwardInvariants:

    def test_conforms_to_forward_model_protocol(self):
        assert isinstance(ProductionEngine(), spec.ForwardModel)

    def test_forward_is_pure(self):
        fwd = ProductionForward()
        h = _hist()
        st = RefinementState((NABR,), (h,), _params())
        assert np.array_equal(fwd.calculate(st, h), fwd.calculate(st, h))

    def test_forward_nonnegative(self):
        fwd = ProductionForward()
        h = _hist()
        st = RefinementState((NABR,), (h,), _params(scale=10.0, bkg=5.0))
        assert np.all(fwd.calculate(st, h) >= 0.0)

    def test_jacobian_shape_and_finiteness(self):
        fwd = ProductionForward()
        h = _hist(n=801)
        st = RefinementState((NABR,), (h,), _params())
        J = fwd.jacobian(st, h)
        n_varied = sum(1 for p in st.parameters if p.vary)
        assert J.shape == (len(h.x), n_varied)
        assert np.all(np.isfinite(J))

    def test_jacobian_scale_column_matches_pattern_shape(self):
        # d(pattern)/d(scale) of a linear-in-scale model == pattern with scale=1, bkg=0.
        fwd = ProductionForward()
        h = _hist(n=801)
        params = (
            Parameter(
                "hist:syn:scale",
                ParamKind.HISTOGRAM,
                1.0,
                vary=True,
                lower=0,
                upper=1e6,
            ),
            Parameter("hist:syn:bkg", ParamKind.HISTOGRAM, 0.0, vary=False),
            Parameter("hist:syn:fwhm", ParamKind.HISTOGRAM, 0.3, vary=False),
            Parameter("hist:syn:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
        )
        st = RefinementState((NABR,), (h,), params)
        J = fwd.jacobian(st, h)  # only scale varies -> column 0
        unit = fwd.calculate(
            RefinementState(
                (NABR,),
                (h,),
                (
                    Parameter("hist:syn:scale", ParamKind.HISTOGRAM, 1.0),
                    Parameter("hist:syn:bkg", ParamKind.HISTOGRAM, 0.0),
                    Parameter("hist:syn:fwhm", ParamKind.HISTOGRAM, 0.3),
                    Parameter("hist:syn:eta", ParamKind.HISTOGRAM, 0.5),
                ),
            ),
            h,
        )
        assert np.allclose(J[:, 0], unit, rtol=1e-4, atol=1e-6)


# --- Peak-position correctness -------------------------------------------------


class TestPeakPositions:

    @pytest.mark.parametrize(
        "dtype,wl",
        [
            (DataType.CW_NEUTRON, 1.909),
            (DataType.CW_XRAY, 1.5406),
        ],
    )
    def test_calc_peaks_land_at_predicted_positions(self, dtype, wl):
        fwd = ProductionForward()
        h = _hist(dtype=dtype, n=5501, wavelength=wl)
        st = RefinementState((NABR,), (h,), _params(fwhm=0.3))
        y = fwd.calculate(st, h)
        d_min, d_max = positions.d_range_for_histogram(h)
        refl = symmetry.generate_reflections(NABR, round(d_min, 4), round(d_max, 4))
        checked = 0
        for r in refl:
            pos = positions.position(r.d, h)
            if pos <= h.x[3] or pos >= h.x[-4]:
                continue
            checked += 1
            i = int(np.argmin(np.abs(h.x - pos)))
            window = y[max(0, i - 3) : i + 4]
            assert window.max() > 1e-9
            assert y[i] >= 0.5 * window.max()  # local maximum at the predicted angle
        assert checked >= 5

    def test_predicted_position_equals_analytic_bragg(self):
        h = _hist(dtype=DataType.CW_NEUTRON, wavelength=1.909)
        refl = symmetry.generate_reflections(NABR, 1.5, 3.5)
        for r in refl:
            analytic = math.degrees(2 * math.asin(1.909 / (2 * r.d)))
            assert abs(positions.position(r.d, h) - analytic) < 1e-9


# --- Real-data positional alignment -------------------------------------------


class TestRealDataAlignment:

    @pytest.fixture(autouse=True)
    def _require_data(self):
        if not DATA_DIR.is_dir():
            pytest.skip("Test data directory not found")

    def test_snap_tof_nabr_pb_positive_correlation(self):
        """A NaBr+Pb forward pattern correlates with the observed SNAP TOF data:
        predicted Bragg peaks fall where the measured pattern has intensity."""
        nabr = io.read(DATA_DIR / "Phase_NaBr.cif", "phase")
        pb = io.read(DATA_DIR / "Phase2_Pb.cif", "phase")
        h = state_to_histograms(io.read(DATA_DIR / "SNAP067702_column.gsa", "powder"))[
            3
        ]
        params = (
            Parameter(f"hist:{h.id}:scale", ParamKind.HISTOGRAM, 5e-4),
            Parameter(
                f"hist:{h.id}:bkg", ParamKind.HISTOGRAM, float(np.median(h.y_obs))
            ),
            Parameter(f"hist:{h.id}:fwhm", ParamKind.HISTOGRAM, 25.0),
            Parameter(f"hist:{h.id}:eta", ParamKind.HISTOGRAM, 0.5),
        )
        yc = ProductionForward().calculate(RefinementState((nabr, pb), (h,), params), h)
        a = yc - yc.mean()
        b = h.y_obs - h.y_obs.mean()
        corr = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        assert (
            corr > 0.2
        ), f"NaBr+Pb forward should align with observed peaks, corr={corr:.3f}"

    def test_nac_synchrotron_cw_xray_positive_correlation(self):
        """An NAC forward pattern aligns with the observed APS 11-BM data using
        the beamline's calibrated wavelength — closing the CW X-ray real-data
        validation. (11-BM .fxye carries no wavelength; 0.413909 A is the
        calibrated value for this dataset.)"""
        from dataclasses import replace

        nac = io.read(DATA_DIR / "NAC.cif", "phase")
        h = state_to_histograms(
            io.read(DATA_DIR / "11BM_NAC.fxye", "powder"),
            wavelength=NAC_11BM_WAVELENGTH,
        )[0]
        # Low-angle window: the strong, well-separated reflections; keeps it fast.
        m = h.x <= 12.0
        h = replace(h, x=h.x[m], y_obs=h.y_obs[m], weights=h.weights[m])
        params = (
            Parameter(f"hist:{h.id}:scale", ParamKind.HISTOGRAM, 5e-3),
            Parameter(
                f"hist:{h.id}:bkg", ParamKind.HISTOGRAM, float(np.median(h.y_obs))
            ),
            Parameter(f"hist:{h.id}:fwhm", ParamKind.HISTOGRAM, 0.008),
            Parameter(f"hist:{h.id}:eta", ParamKind.HISTOGRAM, 0.5),
        )
        yc = ProductionForward().calculate(RefinementState((nac,), (h,), params), h)
        a = yc - yc.mean()
        b = h.y_obs - h.y_obs.mean()
        corr = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        assert corr > 0.3, f"NAC forward should align with 11-BM peaks, corr={corr:.3f}"
