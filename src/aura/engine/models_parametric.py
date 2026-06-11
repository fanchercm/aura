"""
Parametric-model library.

Factory functions that build :class:`aura.spec.ParametricModel` instances — the
maps from refinable *coefficients* + a histogram's *driving* variables to the
value of a target parameter. The parametric engine (:mod:`aura.engine.parametric`)
resolves these per histogram so the minimizer refines a few shared coefficients
instead of one duplicated parameter per pattern (Stinton & Evans 2007).

The **degenerate** model (one free coefficient per histogram, identity map)
reproduces independent/sequential refinement exactly — so independent refinement
is a special case of the parametric path, not a separate code path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from aura.spec import ParametricModel


def identity_model(target: str, coeff: str) -> ParametricModel:
    """Degenerate model: ``target = coeff`` (one free value, no driving dependence)."""
    return ParametricModel(
        target=target, coeff_names=(coeff,), func=lambda c, _d: c[coeff]
    )


def linear_model(
    target: str, intercept: str, slope: str, variable: str
) -> ParametricModel:
    """``target = intercept + slope * driving[variable]`` (e.g. thermal expansion a(T))."""
    return ParametricModel(
        target=target,
        coeff_names=(intercept, slope),
        func=lambda c, d: c[intercept] + c[slope] * d[variable],
    )


def polynomial_model(
    target: str, coeffs: Sequence[str], variable: str
) -> ParametricModel:
    """``target = Σ_i coeffs[i] * driving[variable]**i`` (order = len(coeffs)-1)."""
    names = tuple(coeffs)

    def func(c: Mapping[str, float], d: Mapping[str, float]) -> float:
        v = d[variable]
        return sum(c[name] * v**i for i, name in enumerate(names))

    return ParametricModel(target=target, coeff_names=names, func=func)


def from_callable(target: str, coeff_names: Sequence[str], func) -> ParametricModel:
    """Wrap an arbitrary ``func(coeffs, driving) -> value`` as a ParametricModel."""
    return ParametricModel(target=target, coeff_names=tuple(coeff_names), func=func)
