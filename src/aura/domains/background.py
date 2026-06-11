"""
Compact background as a DomainModule.

Background is one of the dominant sources of parameter explosion (5–500 terms
per pattern if done per-channel). A Chebyshev polynomial basis replaces that with
a handful of refinable coefficients — the compact-parameterization requirement
(MOD-3). It implements the spec :class:`aura.spec.DomainModule` contract,
``contribute(state, histogram, y_calc) -> y_calc + background``, reading its
coefficients ``hist:<id>:bkg_c{k}`` from the refinement state's parameters so they
refine like any other parameter (and can be tied across slices/states via the
parametric engine).
"""

from __future__ import annotations

import numpy as np

from aura.spec import Histogram, RefinementState


class ChebyshevBackground:
    """Chebyshev-polynomial background of a given order.

    Coefficients are read from parameters named ``hist:<id>:bkg_c0 .. bkg_c{order}``;
    missing coefficients default to 0, so a fresh refinement starts from a flat
    (or zero) background and adds curvature only as coefficients are refined.
    """

    name = "chebyshev-background"

    def __init__(self, order: int = 5) -> None:
        if order < 0:
            raise ValueError(f"Chebyshev order must be >= 0, got {order}")
        self.order = order

    def coeff_names(self, hist_id: str) -> list[str]:
        """Parameter names this module reads for *hist_id* (for building states)."""
        return [f"hist:{hist_id}:bkg_c{k}" for k in range(self.order + 1)]

    def contribute(
        self, state: RefinementState, histogram: Histogram, y_calc: np.ndarray
    ) -> np.ndarray:
        pmap = {p.name: p.value for p in state.parameters}
        coeffs = np.array(
            [
                pmap.get(f"hist:{histogram.id}:bkg_c{k}", 0.0)
                for k in range(self.order + 1)
            ],
            dtype=float,
        )
        if not np.any(coeffs):
            return y_calc
        x = np.asarray(histogram.x, dtype=float)
        t = _normalize(x)
        return y_calc + np.polynomial.chebyshev.chebval(t, coeffs)


def _normalize(x: np.ndarray) -> np.ndarray:
    """Map the abscissa onto [-1, 1] (the Chebyshev domain)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x)
    return 2.0 * (x - lo) / (hi - lo) - 1.0
