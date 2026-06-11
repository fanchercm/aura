"""
Production parametric engine: resolve evolving models into per-histogram values.

This is the heart of the parametric-first design. For a given histogram, it
evaluates every :class:`aura.spec.ParametricModel` against that histogram's
*driving* coordinate (T, P, t, …) and writes the resulting value into the
model's target parameter. The minimizer therefore varies a small set of shared
coefficients while each histogram sees its own effective parameter values.

``expand`` is a pure ``RefinementState -> RefinementState`` transform (no
mutation), so it composes cleanly inside the optimizer loop and is safe to call
repeatedly. With no parametric models it is the identity, so ordinary
(independent/sequential) refinement is the degenerate case.
"""

from __future__ import annotations

from dataclasses import replace

from aura.spec import Histogram, RefinementState


class ProductionParametric:
    """Resolves parametric models to concrete per-histogram parameter values."""

    name = "production-parametric"

    def expand(self, state: RefinementState, histogram: Histogram) -> RefinementState:
        if not state.parametric_models:
            return state
        values = {p.name: p.value for p in state.parameters}
        driving = dict(histogram.driving)
        resolved: dict[str, float] = {}
        for model in state.parametric_models:
            coeffs = {cn: values[cn] for cn in model.coeff_names}
            resolved[model.target] = model.func(coeffs, driving)
        if not resolved:
            return state
        new_params = tuple(
            replace(p, value=resolved[p.name]) if p.name in resolved else p
            for p in state.parameters
        )
        return replace(state, parameters=new_params)
