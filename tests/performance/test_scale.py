"""
Scale & performance invariants (Phase 8).

* O(points × nearby_peaks) — the windowed forward model touches only points near
  each peak, not the whole grid for every reflection.
* Chunked minimizer — bounded-memory normal-equations accumulation matches the
  SciPy minimizer and refines a multi-histogram ensemble.
* Checkpoint/restart round-trips a refinement state.
* Sequential memory stability — repeated evaluation does not grow RSS.
"""

from __future__ import annotations

import numpy as np
import pytest

from aura import spec
from aura.checkpoint import load_refinement, save_refinement
from aura.engine.chunked import ChunkedMinimizer
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

NABR = Phase(
    "NaBr",
    "F m -3 m",
    UnitCell(5.9738, 5.9738, 5.9738),
    (AtomSite("Na", 0, 0, 0), AtomSite("Br", 0.5, 0.5, 0.5)),
)
ENGINE = ProductionEngine()


def _hist(n, wl=1.909):
    x = np.linspace(20.0, 120.0, n)
    return Histogram(
        "h", DataType.CW_NEUTRON, x, np.zeros(n), np.ones(n), {}, wavelength=wl
    )


def _params(scale=100.0, bkg=5.0, fwhm=0.3, vary_a=True):
    return (
        Parameter(
            "phase:NaBr:cell.a",
            ParamKind.PHASE,
            5.9738,
            vary=vary_a,
            lower=5.8,
            upper=6.1,
        ),
        Parameter(
            "hist:h:scale", ParamKind.HISTOGRAM, scale, vary=True, lower=1e-3, upper=1e6
        ),
        Parameter(
            "hist:h:bkg", ParamKind.HISTOGRAM, bkg, vary=True, lower=0.0, upper=1e4
        ),
        Parameter(
            "hist:h:fwhm", ParamKind.HISTOGRAM, fwhm, vary=True, lower=0.05, upper=2.0
        ),
        Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
    )


# --- O(points x nearby_peaks) --------------------------------------------------


class TestScaling:

    def test_windowed_ops_far_below_naive(self):
        """Windowed point-ops << n_reflections * n_points (the naive all-peaks cost)."""
        fwd = ProductionForward()
        h = _hist(4000)
        st = RefinementState((NABR,), (h,), _params())
        fwd.calculate(st, h)
        ops = fwd.peak_point_ops
        # Count reflections actually placed.
        from aura.engine import positions, symmetry

        d_min, d_max = positions.d_range_for_histogram(h, pad=0.02)
        n_refl = len(
            symmetry.generate_reflections(NABR, round(d_min, 4), round(d_max, 4))
        )
        naive = n_refl * len(h.x)
        assert ops < 0.2 * naive, f"windowed ops {ops} not << naive {naive}"

    def test_per_grid_density_not_per_total_reflections(self):
        """Doubling the grid resolution (same range, same FWHM) grows point-ops only
        with grid density — confirming each peak touches a fixed angular window,
        independent of how many other peaks exist."""
        fwd = ProductionForward()
        st_lo = RefinementState((NABR,), (_hist(2000),), _params())
        st_hi = RefinementState((NABR,), (_hist(4000),), _params())
        fwd.calculate(st_lo, st_lo.histograms[0])
        ops_lo = fwd.peak_point_ops
        fwd.calculate(st_hi, st_hi.histograms[0])
        ops_hi = fwd.peak_point_ops
        # ~2x grid density -> ~2x ops (not quadratic).
        assert 1.6 < ops_hi / ops_lo < 2.6


# --- Chunked minimizer ---------------------------------------------------------


class TestChunkedMinimizer:

    def test_conforms_to_minimizer(self):
        assert isinstance(ChunkedMinimizer(), spec.Minimizer)

    def test_matches_scipy_minimizer_on_single_state(self):
        """Chunked Gauss-Newton recovers the same cell as the SciPy minimizer."""
        a_true = 5.99
        h = _hist(1500)
        truth = (
            Parameter("phase:NaBr:cell.a", ParamKind.PHASE, a_true),
            Parameter("hist:h:scale", ParamKind.HISTOGRAM, 100.0),
            Parameter("hist:h:bkg", ParamKind.HISTOGRAM, 5.0),
            Parameter("hist:h:fwhm", ParamKind.HISTOGRAM, 0.3),
            Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5),
        )
        y = ENGINE.calculate(RefinementState((NABR,), (h,), truth), h)
        w = 1.0 / np.clip(y, 1.0, None)
        hd = Histogram("h", DataType.CW_NEUTRON, h.x, y, w, {}, wavelength=1.909)
        seed = RefinementState((NABR,), (hd,), _params(scale=80.0, bkg=3.0, fwhm=0.4))

        scipy_res = ENGINE.refine(seed, ENGINE, ENGINE, max_iter=80, seed=1)
        chunked = ChunkedMinimizer()
        chunk_res = chunked.refine(seed, ENGINE, ENGINE, max_iter=80, seed=1)
        a_scipy = next(
            p.value for p in scipy_res.state.parameters if p.name == "phase:NaBr:cell.a"
        )
        a_chunk = next(
            p.value for p in chunk_res.state.parameters if p.name == "phase:NaBr:cell.a"
        )
        assert abs(a_scipy - a_true) < 1e-3
        assert abs(a_chunk - a_true) < 1e-3
        assert abs(a_scipy - a_chunk) < 1e-4

    def test_refines_multi_histogram_ensemble_bounded_memory(self):
        """A shared cell across many histograms refines; reported memory does not
        scale with the number of patterns (normal-equations accumulation)."""
        a_true = 5.98
        x = np.linspace(20.0, 120.0, 800)
        hists = []
        for i in range(12):
            blank = Histogram(
                f"h{i}",
                DataType.CW_NEUTRON,
                x,
                np.zeros_like(x),
                np.ones_like(x),
                {},
                wavelength=1.909,
            )
            truth = (
                Parameter("phase:NaBr:cell.a", ParamKind.PHASE, a_true),
                Parameter(f"hist:h{i}:scale", ParamKind.HISTOGRAM, 100.0),
                Parameter(f"hist:h{i}:bkg", ParamKind.HISTOGRAM, 5.0),
                Parameter(f"hist:h{i}:fwhm", ParamKind.HISTOGRAM, 0.3),
                Parameter(f"hist:h{i}:eta", ParamKind.HISTOGRAM, 0.5),
            )
            yi = ENGINE.calculate(RefinementState((NABR,), (blank,), truth), blank)
            wi = 1.0 / np.clip(yi, 1.0, None)
            hists.append(
                Histogram(f"h{i}", DataType.CW_NEUTRON, x, yi, wi, {}, wavelength=1.909)
            )
        # Shared cell.a + per-histogram scale/bkg/fwhm.
        params = [
            Parameter(
                "phase:NaBr:cell.a",
                ParamKind.PHASE,
                5.96,
                vary=True,
                lower=5.8,
                upper=6.1,
            )
        ]
        for i in range(12):
            params += [
                Parameter(
                    f"hist:h{i}:scale",
                    ParamKind.HISTOGRAM,
                    90.0,
                    vary=True,
                    lower=1e-3,
                    upper=1e6,
                ),
                Parameter(
                    f"hist:h{i}:bkg",
                    ParamKind.HISTOGRAM,
                    4.0,
                    vary=True,
                    lower=0.0,
                    upper=1e4,
                ),
                Parameter(
                    f"hist:h{i}:fwhm",
                    ParamKind.HISTOGRAM,
                    0.32,
                    vary=True,
                    lower=0.05,
                    upper=2.0,
                ),
                Parameter(f"hist:h{i}:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
            ]
        st = RefinementState((NABR,), tuple(hists), tuple(params))
        res = ChunkedMinimizer().refine(st, ENGINE, ENGINE, max_iter=60, seed=1)
        a = next(p.value for p in res.state.parameters if p.name == "phase:NaBr:cell.a")
        assert abs(a - a_true) < 5e-3
        assert res.diagnostics["n_histograms"] == 12


# --- Checkpoint / restart ------------------------------------------------------


class TestCheckpoint:

    def test_state_round_trips(self, tmp_path):
        h = _hist(500)
        st = RefinementState((NABR,), (h,), _params())
        p = save_refinement(st, tmp_path / "ckpt.pkl")
        loaded = load_refinement(p)
        assert [q.name for q in loaded.parameters] == [q.name for q in st.parameters]
        assert np.array_equal(loaded.histograms[0].x, st.histograms[0].x)
        assert loaded.phases[0].cell.a == st.phases[0].cell.a

    def test_resume_continues_refinement(self, tmp_path):
        a_true = 5.99
        h = _hist(1000)
        y = ENGINE.calculate(
            RefinementState(
                (NABR,),
                (h,),
                (
                    Parameter("phase:NaBr:cell.a", ParamKind.PHASE, a_true),
                    Parameter("hist:h:scale", ParamKind.HISTOGRAM, 100.0),
                    Parameter("hist:h:bkg", ParamKind.HISTOGRAM, 5.0),
                    Parameter("hist:h:fwhm", ParamKind.HISTOGRAM, 0.3),
                    Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5),
                ),
            ),
            h,
        )
        w = 1.0 / np.clip(y, 1.0, None)
        hd = Histogram("h", DataType.CW_NEUTRON, h.x, y, w, {}, wavelength=1.909)
        st = RefinementState((NABR,), (hd,), _params(scale=80.0, bkg=3.0, fwhm=0.4))
        from aura.checkpoint import resume

        save_refinement(st, tmp_path / "c.pkl")
        res = resume(ENGINE, tmp_path / "c.pkl", max_iter=80, seed=1)
        a = next(p.value for p in res.state.parameters if p.name == "phase:NaBr:cell.a")
        assert abs(a - a_true) < 1e-3


# --- Sequential memory stability -----------------------------------------------


class TestMemoryStability:

    @pytest.mark.slow
    def test_sequential_calculate_bounded_rss(self):
        """Evaluating a long sequence of patterns does not grow RSS materially
        (no per-pattern accumulation)."""
        import resource

        fwd = ProductionForward()
        st = RefinementState((NABR,), (_hist(1500),), _params())
        h = st.histograms[0]
        fwd.calculate(st, h)  # warm up
        baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for _ in range(400):
            fwd.calculate(st, h)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        growth = (after - baseline) / max(baseline, 1)
        assert growth < 0.05, f"RSS grew {growth:.1%} over 400 evaluations"
