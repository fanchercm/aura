"""
Production forward model: params -> calculated pattern.

A real, multi-phase Rietveld forward model covering all four data types
(CW X-ray / CW neutron / TOF / EDD), built on real symmetry
(:mod:`~aura.engine.symmetry`), real scattering factors
(:mod:`~aura.engine.scattering`), and the data-type position laws
(:mod:`~aura.engine.positions`).

Key properties (set up later phases):

* **Windowed peak evaluation** — each reflection's profile is added only over
  the ``±peak_window·FWHM`` neighborhood, giving O(points × nearby_peaks)
  rather than O(points × all_peaks) (the Phase-8 scaling invariant).
* **Vectorized / branch-light** numpy, so the Phase-6 JAX port is mechanical.

Parameter naming convention (opaque keys, resolved here):
``phase:<name>:cell.{a,b,c,alpha,beta,gamma}``, ``phase:<name>:atom<i>.{x,y,z,occ,b_iso}``,
``phase_scale:<phase>:<hist>``, ``hist:<id>:{scale,bkg,fwhm,eta}``. Any key not
present falls back to the phase/default value, so a refinement varies only the
handful of parameters it declares.
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np

from aura import spec
from aura.engine import positions, scattering, symmetry
from aura.spec import DataType, Histogram, Phase, RefinementState, UnitCell


@functools.lru_cache(maxsize=256)
def _cached_reflections(phase: Phase, d_min: float, d_max: float):
    """Reflections for a (hashable) phase + rounded d-range; cached across calls."""
    return tuple(symmetry.generate_reflections(phase, d_min, d_max))


def _lorentz_polarization(d: float, pos: float, hist: Histogram) -> float:
    """Approximate Lorentz(-polarization) intensity factor by data type.

    Affects relative intensities only (positions are exact); the refined scale
    absorbs any overall constant. Refined per-modality forms come later.
    """
    dt = hist.data_type
    if dt in (DataType.CW_XRAY, DataType.CW_NEUTRON):
        theta = math.radians(pos) / 2.0
        s, c = math.sin(theta), math.cos(theta)
        if s <= 0 or c <= 0:
            return 0.0
        lorentz = 1.0 / (s * s * c)
        if dt is DataType.CW_XRAY:
            pol = (1.0 + math.cos(math.radians(pos)) ** 2) / 2.0
            return lorentz * pol
        return lorentz
    if dt is DataType.TOF:
        return d**4  # standard TOF Lorentz factor
    return d * d  # EDD (rough)


class ProductionForward:
    """Real forward model implementing :class:`aura.spec.ForwardModel`."""

    name = "production-numpy"
    peak_window = 12.0  # half-width of the evaluation window, in FWHM units

    # ---- parameter resolution -----------------------------------------------
    @staticmethod
    def _pmap(state: RefinementState) -> dict[str, float]:
        return {p.name: p.value for p in state.parameters}

    @staticmethod
    def _effective_phase(phase: Phase, pmap: dict[str, float]) -> Phase:
        c = phase.cell

        def g(key: str, default: float) -> float:
            return pmap.get(f"phase:{phase.name}:cell.{key}", default)

        cell = UnitCell(
            g("a", c.a),
            g("b", c.b),
            g("c", c.c),
            g("alpha", c.alpha),
            g("beta", c.beta),
            g("gamma", c.gamma),
        )
        atoms = []
        for i, at in enumerate(phase.atoms):
            pre = f"phase:{phase.name}:atom{i}"
            atoms.append(
                replace(
                    at,
                    x=pmap.get(f"{pre}.x", at.x),
                    y=pmap.get(f"{pre}.y", at.y),
                    z=pmap.get(f"{pre}.z", at.z),
                    occ=pmap.get(f"{pre}.occ", at.occ),
                    b_iso=pmap.get(f"{pre}.b_iso", at.b_iso),
                )
            )
        return replace(phase, cell=cell, atoms=tuple(atoms))

    # ---- ForwardModel.calculate ---------------------------------------------
    def calculate(self, state: RefinementState, histogram: Histogram) -> np.ndarray:
        pmap = self._pmap(state)
        x = np.asarray(histogram.x, dtype=float)
        hid = histogram.id

        scale = pmap.get(f"hist:{hid}:scale", 1.0)
        bkg = pmap.get(f"hist:{hid}:bkg", 0.0)
        fwhm = pmap.get(f"hist:{hid}:fwhm", 0.1)
        eta = pmap.get(f"hist:{hid}:eta", 0.5)

        y = np.full_like(x, bkg)
        if fwhm <= 0 or x.size == 0:
            return y

        d_min, d_max = positions.d_range_for_histogram(histogram, pad=0.02)
        for phase in state.phases:
            eff = self._effective_phase(phase, pmap)
            pscale = pmap.get(f"phase_scale:{phase.name}:{hid}", 1.0)
            refl = _cached_reflections(eff, round(d_min, 4), round(d_max, 4))
            for r in refl:
                try:
                    pos = positions.position(r.d, histogram)
                except ValueError:
                    continue
                lp = _lorentz_polarization(r.d, pos, histogram)
                if lp <= 0:
                    continue
                fsq = scattering.f_squared(r.hkl, r.d, eff, histogram.data_type)
                amp = scale * pscale * r.multiplicity * fsq * lp
                if amp != 0.0:
                    _add_peak(y, x, pos, fwhm, eta, amp, self.peak_window)
        return y

    # ---- ForwardModel.jacobian (central finite difference) ------------------
    def jacobian(self, state: RefinementState, histogram: Histogram) -> np.ndarray:
        varied = [p for p in state.parameters if p.vary]
        J = np.zeros((len(histogram.x), len(varied)))
        for j, p in enumerate(varied):
            step = 1e-6 * max(abs(p.value), 1.0)
            up = self._perturb(state, p.name, p.value + step)
            dn = self._perturb(state, p.name, p.value - step)
            J[:, j] = (
                self.calculate(up, histogram) - self.calculate(dn, histogram)
            ) / (2 * step)
        return J

    @staticmethod
    def _perturb(state: RefinementState, name: str, value: float) -> RefinementState:
        params = tuple(
            replace(p, value=value) if p.name == name else p for p in state.parameters
        )
        return replace(state, parameters=params)


def _add_peak(
    y: np.ndarray,
    x: np.ndarray,
    center: float,
    fwhm: float,
    eta: float,
    amplitude: float,
    window: float,
) -> None:
    """Add a pseudo-Voigt peak to *y* over the ``±window·fwhm`` neighborhood only.

    Assumes *x* is sorted ascending (true for all reader outputs). A peak whose
    center is far outside the measured range contributes nothing; one near an
    edge contributes only its in-range tail.
    """
    half = window * fwhm
    lo = int(np.searchsorted(x, center - half, side="left"))
    hi = int(np.searchsorted(x, center + half, side="right"))
    if hi <= lo:
        return
    seg = x[lo:hi]
    y[lo:hi] += amplitude * spec.pseudo_voigt(seg, center, fwhm, eta)


class ProductionEngine:
    """Composes the production forward model (and, later, minimizer/parametric).

    For Phase 3 it implements :class:`aura.spec.ForwardModel` (``calculate`` +
    ``jacobian``) by delegation. The scipy minimizer (Phase 4), real parametric
    engine (Phase 5), and JAX backend (Phase 6) attach here.
    """

    name = "production-numpy"

    def __init__(self) -> None:
        self.forward = ProductionForward()

    def calculate(self, state: RefinementState, histogram: Histogram) -> np.ndarray:
        return self.forward.calculate(state, histogram)

    def jacobian(self, state: RefinementState, histogram: Histogram) -> np.ndarray:
        return self.forward.jacobian(state, histogram)
