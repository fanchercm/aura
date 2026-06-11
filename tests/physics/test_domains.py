"""
Domain-module invariants (Phase 7): compact background, preferred-orientation
texture, and quantitative phase-fraction closure.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import aura.io as io
from aura import spec
from aura.bridge import state_to_histograms
from aura.domains.background import ChebyshevBackground
from aura.domains.texture import MarchDollase
from aura.engine.forward import ProductionEngine, ProductionForward
from aura.quantify import phase_weight_fractions
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

NABR = Phase(
    "NaBr",
    "F m -3 m",
    UnitCell(5.9738, 5.9738, 5.9738),
    (AtomSite("Na", 0, 0, 0), AtomSite("Br", 0.5, 0.5, 0.5)),
)


def _cw_hist(n=1500):
    x = np.linspace(20.0, 120.0, n)
    return Histogram(
        "h", DataType.CW_NEUTRON, x, np.zeros(n), np.ones(n), {}, wavelength=1.909
    )


# --- Background DomainModule ---------------------------------------------------


class TestChebyshevBackground:

    def test_zero_coeffs_no_change(self):
        bg = ChebyshevBackground(order=4)
        h = _cw_hist()
        st = RefinementState((NABR,), (h,), ())  # no bkg coeffs
        y = np.ones(len(h.x)) * 7.0
        assert np.array_equal(bg.contribute(st, h, y), y)

    def test_constant_term_adds_offset(self):
        bg = ChebyshevBackground(order=3)
        h = _cw_hist()
        st = RefinementState(
            (NABR,), (h,), (Parameter("hist:h:bkg_c0", ParamKind.HISTOGRAM, 12.0),)
        )
        y = np.zeros(len(h.x))
        out = bg.contribute(st, h, y)
        assert np.allclose(out, 12.0)  # T0 = 1 -> constant offset

    def test_linear_term_is_monotonic(self):
        bg = ChebyshevBackground(order=3)
        h = _cw_hist()
        st = RefinementState(
            (NABR,), (h,), (Parameter("hist:h:bkg_c1", ParamKind.HISTOGRAM, 5.0),)
        )
        out = bg.contribute(st, h, np.zeros(len(h.x)))
        assert np.all(np.diff(out) > 0)  # T1 = t, increasing across the range

    def test_conforms_to_domain_module(self):
        assert isinstance(ChebyshevBackground(), spec.DomainModule)

    @pytest.mark.skipif(not DATA_DIR.is_dir(), reason="no test data")
    def test_refined_background_improves_pbso4_rwp(self):
        """A refined compact background substantially lowers Rwp on real PbSO4
        neutron data vs the flat-background model (the MOD-3 compact-param win)."""
        ph = io.read(DATA_DIR / "PbSO4.cif", "phase")
        c = ph.cell
        h = state_to_histograms(io.read(DATA_DIR / "PBSO4.CWN", "powder"))[0]
        eng = ProductionEngine(
            backend="numpy", domain_modules=[ChebyshevBackground(order=6)]
        )
        med = float(np.median(h.y_obs))
        seed = [
            Parameter(
                f"phase:{ph.name}:cell.a",
                ParamKind.PHASE,
                c.a,
                vary=True,
                lower=c.a - 0.1,
                upper=c.a + 0.1,
            ),
            Parameter(
                f"phase:{ph.name}:cell.b",
                ParamKind.PHASE,
                c.b,
                vary=True,
                lower=c.b - 0.1,
                upper=c.b + 0.1,
            ),
            Parameter(
                f"phase:{ph.name}:cell.c",
                ParamKind.PHASE,
                c.c,
                vary=True,
                lower=c.c - 0.1,
                upper=c.c + 0.1,
            ),
            Parameter(
                f"hist:{h.id}:scale",
                ParamKind.HISTOGRAM,
                5e-4,
                vary=True,
                lower=1e-8,
                upper=1e2,
            ),
            Parameter(
                f"hist:{h.id}:fwhm",
                ParamKind.HISTOGRAM,
                0.35,
                vary=True,
                lower=0.1,
                upper=1.0,
            ),
            Parameter(f"hist:{h.id}:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
        ]
        for k in range(7):
            seed.append(
                Parameter(
                    f"hist:{h.id}:bkg_c{k}",
                    ParamKind.HISTOGRAM,
                    med if k == 0 else 0.0,
                    vary=True,
                    lower=-1e4,
                    upper=1e4,
                )
            )
        st = RefinementState((ph,), (h,), tuple(seed))
        res = eng.refine(st, eng, eng, max_iter=80, seed=1)
        assert (
            res.rwp < 0.45
        ), f"refined background should beat flat-bkg ~0.52, got {res.rwp:.3f}"
        # Lattice still recovered.
        a = next(
            p.value for p in res.state.parameters if p.name == f"phase:{ph.name}:cell.a"
        )
        assert abs(a - c.a) < 0.05


# --- Texture (March–Dollase) ---------------------------------------------------


class TestMarchDollase:

    def test_random_texture_limit_is_identity(self):
        """ratio = 1 leaves every reflection unchanged (texture-free limit)."""
        md = MarchDollase("NaBr", axis=(0, 0, 1))
        for hkl in [(1, 1, 1), (2, 0, 0), (2, 2, 0), (0, 0, 2)]:
            assert md.factor(hkl, NABR.cell, 1.0) == 1.0

    def test_axis_vs_perpendicular_diverge(self):
        """For ratio != 1, reflections along the PO axis differ from perpendicular."""
        md = MarchDollase("NaBr", axis=(0, 0, 1))
        along = md.factor((0, 0, 2), NABR.cell, 0.7)
        perp = md.factor((2, 0, 0), NABR.cell, 0.7)
        assert not math.isclose(along, perp, rel_tol=1e-3)
        # March–Dollase r<1: along-axis (cosα=1) enhanced (P=r^-3), perpendicular
        # (cosα=0) suppressed (P=r^1.5) — platy habit with the axis as plate normal.
        assert along > perp
        assert math.isclose(along, 0.7**-3, rel_tol=1e-6)
        assert math.isclose(perp, 0.7**1.5, rel_tol=1e-6)

    def test_texture_changes_pattern_but_ratio1_does_not(self):
        h = _cw_hist()
        params = (
            Parameter("hist:h:scale", ParamKind.HISTOGRAM, 100.0),
            Parameter("hist:h:fwhm", ParamKind.HISTOGRAM, 0.3),
            Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5),
            Parameter("phase:NaBr:march.ratio", ParamKind.PHASE_DATA, 1.0),
        )
        fwd_tex = ProductionForward(texture=MarchDollase("NaBr", (0, 0, 1)))
        st1 = RefinementState((NABR,), (h,), params)
        plain = ProductionForward()
        y_ratio1 = fwd_tex.calculate(st1, h)
        y_plain = plain.calculate(st1, h)
        assert np.allclose(y_ratio1, y_plain)  # ratio=1 == no texture

        params2 = params[:-1] + (
            Parameter("phase:NaBr:march.ratio", ParamKind.PHASE_DATA, 0.6),
        )
        y_tex = fwd_tex.calculate(RefinementState((NABR,), (h,), params2), h)
        assert not np.allclose(y_tex, y_plain)  # texture changes the pattern


# --- Quantitative phase fractions (closure) -----------------------------------


class TestPhaseFractionClosure:

    @pytest.mark.skipif(not DATA_DIR.is_dir(), reason="no test data")
    def test_nabr_pb_fractions_sum_to_one(self):
        nabr = io.read(DATA_DIR / "Phase_NaBr.cif", "phase")
        pb = io.read(DATA_DIR / "Phase2_Pb.cif", "phase")
        fr = phase_weight_fractions((nabr, pb), np.array([2.0, 1.0]))
        assert math.isclose(
            fr.sum(), 1.0, abs_tol=spec.ACCEPTANCE["phase_fraction_sum_atol"]
        )
        assert np.all(fr > 0) and np.all(fr < 1)

    def test_zero_total_weight_raises(self):
        with pytest.raises(ValueError, match="Non-positive"):
            phase_weight_fractions((NABR,), np.array([0.0]))
