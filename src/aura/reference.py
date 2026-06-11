"""
Reference (oracle) engine for Aura.

A minimal, correct numpy implementation of the :mod:`aura.spec` Protocols
(:class:`~aura.spec.ForwardModel`, :class:`~aura.spec.ParametricEngine`,
:class:`~aura.spec.Minimizer`). It is deliberately simple — a single cubic
phase, scale + flat background, one pseudo-Voigt per reflection — but it is
*correct*, so it serves two roles:

1. It is the **oracle** the physics-invariant suite
   (``tests/physics/test_invariants.py``) runs green against today, before any
   production engine exists.
2. It is the reference a production (numpy/JAX/GPU) engine is checked against:
   the production engine must pass the SAME invariants with richer physics.

The suite swaps the production engine in by re-pointing a single pytest
fixture (see ``tests/conftest.py``).
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np

from aura import spec
from aura.spec import (
    Parameter,
    RefinementResult,
    RefinementState,
    UnitCell,
)


@functools.lru_cache(maxsize=256)
def _reflections_cubic_cached(a: float, wavelength: float, tt_max: float = 120.0):
    return tuple(_reflections_cubic(UnitCell(a, a, a), wavelength, tt_max))


def _reflections_cubic(cell: UnitCell, wavelength: float, tt_max: float = 120.0):
    """Enumerate low-index reflections observable for a CW experiment."""
    refl = []
    for h in range(0, 5):
        for k in range(0, 5):
            for l in range(0, 5):
                if h == k == l == 0:
                    continue
                try:
                    d = spec.d_spacing((h, k, l), cell)
                    tt = spec.two_theta_from_d(d, wavelength)
                except ValueError:
                    continue
                if 5.0 < tt < tt_max:
                    refl.append(((h, k, l), d, tt))
    return refl


class RefEngine:
    """Reference ForwardModel + ParametricEngine + Minimizer (numpy, oracle)."""

    name = "reference-numpy"

    # ---- parameter access helpers -------------------------------------------
    @staticmethod
    def _pmap(state: RefinementState) -> dict[str, Parameter]:
        return {p.name: p for p in state.parameters}

    @staticmethod
    def _varied(state: RefinementState) -> list[Parameter]:
        return [p for p in state.parameters if p.vary]

    # ---- ParametricEngine.expand --------------------------------------------
    def expand(self, state: RefinementState, histogram: spec.Histogram) -> RefinementState:
        """Resolve every ParametricModel to a concrete value for this histogram."""
        pmap = self._pmap(state)
        new_params = list(state.parameters)
        for model in state.parametric_models:
            coeffs = {cn: pmap[cn].value for cn in model.coeff_names}
            val = model.func(coeffs, dict(histogram.driving))
            # write resolved value into the target parameter
            for i, p in enumerate(new_params):
                if p.name == model.target:
                    new_params[i] = replace(p, value=val)
        return replace(state, parameters=tuple(new_params))

    # ---- ForwardModel.calculate ---------------------------------------------
    def calculate(self, state: RefinementState, histogram: spec.Histogram) -> np.ndarray:
        st = self.expand(state, histogram)
        pmap = self._pmap(st)
        phase = st.phases[0]
        # effective cell parameter "a" may be driven; read from params if present
        a = pmap.get(f"phase:{phase.name}:cell.a")
        cell = phase.cell if a is None else replace(phase.cell, a=a.value, b=a.value, c=a.value)
        scale = pmap[f"hist:{histogram.id}:scale"].value
        bkg = pmap[f"hist:{histogram.id}:bkg"].value
        fwhm = pmap[f"hist:{histogram.id}:fwhm"].value
        eta = pmap[f"hist:{histogram.id}:eta"].value
        f = {at.element: 14.0 for at in phase.atoms}

        y = np.full_like(histogram.x, bkg)
        for hkl, d, _tt in _reflections_cubic_cached(round(cell.a, 6), histogram.wavelength):
            tt = spec.two_theta_from_d(d, histogram.wavelength)
            Fsq = abs(spec.structure_factor(hkl, phase.atoms, cell, f)) ** 2
            # crude Lorentz-polarization + multiplicity proxy for the reference
            lp = 1.0 / (math.sin(math.radians(tt / 2.0)) ** 2 * math.cos(math.radians(tt / 2.0)))
            y = y + scale * Fsq * lp * spec.pseudo_voigt(histogram.x, tt, fwhm, eta)
        return y

    # ---- ForwardModel.jacobian (finite-difference REFERENCE only) ------------
    def jacobian(self, state: RefinementState, histogram: spec.Histogram) -> np.ndarray:
        varied = self._varied(state)
        base = self.calculate(state, histogram)  # noqa: F841 (kept for parity/readability)
        J = np.zeros((len(histogram.x), len(varied)))
        for j, p in enumerate(varied):
            h = 1e-6 * max(abs(p.value), 1.0)
            up = self._perturb(state, p.name, p.value + h)
            dn = self._perturb(state, p.name, p.value - h)
            J[:, j] = (self.calculate(up, histogram) - self.calculate(dn, histogram)) / (2 * h)
        return J

    def _perturb(self, state: RefinementState, name: str, value: float) -> RefinementState:
        params = tuple(replace(p, value=value) if p.name == name else p
                       for p in state.parameters)
        return replace(state, parameters=params)

    # ---- Minimizer.refine (Gauss–Newton over the FULL ensemble) -------------
    def refine(self, state, forward, parametric, max_iter=100, tol=1e-8) -> RefinementResult:
        varied = self._varied(state)
        names = [p.name for p in varied]
        x = np.array([p.value for p in varied], dtype=float)
        lower = np.array([p.lower for p in varied])
        upper = np.array([p.upper for p in varied])

        def set_x(xv: np.ndarray) -> RefinementState:
            pm = {n: v for n, v in zip(names, xv)}
            params = tuple(replace(p, value=pm.get(p.name, p.value)) for p in state.parameters)
            return replace(state, parameters=params)

        converged = False
        it = 0
        cov = None
        for it in range(1, max_iter + 1):
            cur = set_x(x)
            # stack residuals & jacobians across ALL histograms (parametric objective)
            res_blocks, jac_blocks, wsum = [], [], []
            for hist in state.histograms:
                yc = forward.calculate(cur, hist)
                r = spec.weighted_residual(hist.y_obs, yc, hist.weights)
                Jh = forward.jacobian(cur, hist)
                sw = np.sqrt(hist.weights)[:, None]
                res_blocks.append(r)
                jac_blocks.append(sw * Jh)
                wsum.append(hist.weights)
            r = np.concatenate(res_blocks)
            J = np.concatenate(jac_blocks, axis=0)
            JTJ = J.T @ J
            # Levenberg damping for conditioning (guards the GSAS-II failure class)
            lam = 1e-6 * np.trace(JTJ) / max(JTJ.shape[0], 1)
            step = np.linalg.solve(JTJ + lam * np.eye(JTJ.shape[0]), J.T @ r)
            x_new = np.clip(x + step, lower, upper)
            if np.max(np.abs(x_new - x)) < tol:
                x = x_new
                converged = True
                # covariance ~ chi2_red * inv(JTJ)
                cur = set_x(x)
                break
            x = x_new

        # final stats
        cur = set_x(x)
        yo = np.concatenate([h.y_obs for h in state.histograms])
        w = np.concatenate([h.weights for h in state.histograms])
        yc = np.concatenate([forward.calculate(cur, h) for h in state.histograms])
        gof = spec.reduced_chi_square(yo, yc, w, len(varied))
        try:
            cov = gof * np.linalg.inv(JTJ + lam * np.eye(JTJ.shape[0]))
            sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            sig = np.full(len(varied), np.nan)
        params = []
        sig_map = {n: s for n, s in zip(names, sig)}
        for p in cur.parameters:
            params.append(replace(p, sigma=sig_map.get(p.name)) if p.name in sig_map else p)
        result_state = replace(cur, parameters=tuple(params)).with_log(
            f"refined: {len(varied)} params, {it} iters, converged={converged}")
        return RefinementResult(
            state=result_state, rwp=spec.rwp(yo, yc, w), reduced_chi2=gof,
            converged=converged, n_iterations=it, covariance=cov,
            diagnostics={"condition_number": float(np.linalg.cond(JTJ))},
        )
