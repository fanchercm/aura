"""
JAX/autodiff backend invariants (Phase 6).

The differentiable backend's defining contract: its Jacobian comes from automatic
differentiation and must agree with an independent finite difference of its own
forward model (the invariant that protects against finite-difference fragility).
We also check it reproduces the numpy forward physics (full-grid) and that the
cell parameter — the GSAS-II metric-tensor failure class — differentiates cleanly
through the direct reciprocal metric.

JAX is optional; these tests skip if it is not installed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from aura import spec  # noqa: E402
from aura.engine.forward import ProductionEngine, ProductionForward  # noqa: E402
from aura.engine.forward_jax import JaxForward  # noqa: E402
from aura.spec import (  # noqa: E402
    AtomSite,
    DataType,
    Histogram,
    Parameter,
    ParamKind,
    Phase,
    RefinementState,
    UnitCell,
)

SI = (
    AtomSite("Si", 0, 0, 0),
    AtomSite("Si", 0.5, 0.5, 0.5),
    AtomSite("Si", 0.5, 0.0, 0.5),
    AtomSite("Si", 0.0, 0.5, 0.5),
)


def _setup(dtype=DataType.CW_XRAY, wl=1.5406, vary_atoms=False):
    phase = Phase("Si", "Fd-3m", UnitCell(5.431, 5.431, 5.431), SI)
    x = np.linspace(20.0, 120.0, 1000)
    h = Histogram("h", dtype, x, np.zeros_like(x), np.ones_like(x), {}, wavelength=wl)
    params = [
        Parameter(
            "phase:Si:cell.a", ParamKind.PHASE, 5.431, vary=True, lower=5.0, upper=6.0
        ),
        Parameter(
            "hist:h:scale", ParamKind.HISTOGRAM, 1.0, vary=True, lower=1e-6, upper=1e6
        ),
        Parameter(
            "hist:h:bkg", ParamKind.HISTOGRAM, 5.0, vary=True, lower=0.0, upper=1e4
        ),
        Parameter(
            "hist:h:fwhm", ParamKind.HISTOGRAM, 0.3, vary=True, lower=0.05, upper=2.0
        ),
        Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
    ]
    if vary_atoms:
        params.append(
            Parameter(
                "phase:Si:atom0.b_iso",
                ParamKind.PHASE,
                0.5,
                vary=True,
                lower=0.0,
                upper=5.0,
            )
        )
    return RefinementState((phase,), (h,), tuple(params)), h


def _fd_jacobian(engine, st, h):
    varied = [p for p in st.parameters if p.vary]
    base = engine.calculate(st, h)
    J = np.zeros((len(base), len(varied)))
    for j, p in enumerate(varied):
        step = 1e-6 * max(abs(p.value), 1.0)
        up = replace(
            st,
            parameters=tuple(
                replace(q, value=q.value + step) if q.name == p.name else q
                for q in st.parameters
            ),
        )
        dn = replace(
            st,
            parameters=tuple(
                replace(q, value=q.value - step) if q.name == p.name else q
                for q in st.parameters
            ),
        )
        J[:, j] = (engine.calculate(up, h) - engine.calculate(dn, h)) / (2 * step)
    return J


class TestJaxBackend:

    def test_conforms_to_forward_model(self):
        assert isinstance(ProductionEngine(backend="jax"), spec.ForwardModel)

    @pytest.mark.parametrize(
        "dtype,wl",
        [
            (DataType.CW_XRAY, 1.5406),
            (DataType.CW_NEUTRON, 1.909),
        ],
    )
    def test_ad_jacobian_matches_finite_difference(self, dtype, wl):
        """THE invariant: the AD Jacobian agrees with an independent FD of the
        same JAX forward model — including cell.a (no A-tensor layer)."""
        jf = JaxForward()
        st, h = _setup(dtype, wl, vary_atoms=True)
        J_ad = jf.jacobian(st, h)
        J_fd = _fd_jacobian(jf, st, h)
        for j in range(J_ad.shape[1]):
            num = np.linalg.norm(J_ad[:, j] - J_fd[:, j])
            den = np.linalg.norm(J_fd[:, j]) + 1e-30
            assert (
                num / den < spec.ACCEPTANCE["jacobian_rtol"]
            ), f"AD vs FD column {j}: {num/den:.2e}"

    def test_jax_matches_numpy_forward(self):
        """The JAX forward reproduces the numpy forward physics (full-grid)."""
        st, h = _setup()
        y_jax = JaxForward().calculate(st, h)
        pf = ProductionForward()
        pf.peak_window = 1e6  # full grid, comparable to JAX (no windowing)
        y_np = pf.calculate(st, h)
        rel = np.linalg.norm(y_jax - y_np) / (np.linalg.norm(y_np) + 1e-30)
        assert rel < 1e-5, f"JAX vs numpy forward rel L2 = {rel:.2e}"

    def test_cell_parameter_differentiates_cleanly(self):
        """cell.a has a finite, nonzero AD derivative (the metric-tensor class
        that breaks legacy A-tensor parameterizations)."""
        jf = JaxForward()
        st, h = _setup()
        J = jf.jacobian(st, h)
        assert np.all(np.isfinite(J))
        assert np.linalg.norm(J[:, 0]) > 0  # column 0 = cell.a

    def test_tof_raises_on_jax_backend(self):
        jf = JaxForward()
        x = np.linspace(2000.0, 14000.0, 500)
        h = Histogram(
            "t", DataType.TOF, x, np.zeros_like(x), np.ones_like(x), {}, difc=5000.0
        )
        phase = Phase("Si", "Fd-3m", UnitCell(5.431, 5.431, 5.431), SI)
        st = RefinementState(
            (phase,),
            (h,),
            (Parameter("hist:t:scale", ParamKind.HISTOGRAM, 1.0, vary=True),),
        )
        with pytest.raises(NotImplementedError, match="CW"):
            jf.calculate(st, h)
