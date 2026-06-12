"""
Refinement diagnostics at multiple aggregation levels (UX-3 / UX-4).

A campaign can hold ~10⁵ patterns and thousands of parameters — far too many to
inspect individually. These helpers summarize a result, classify every parameter
as converged / pinned-at-bound / ill-determined / fixed (the "which parameters
converged, which diverged, which are not converging" view from the project
brief), rank histograms by misfit for drill-down, and roll up a campaign of
states. Everything returns plain data so it can be printed, tested, or plotted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aura import spec
from aura.spec import Parameter, RefinementResult, RefinementState


def summarize(result: RefinementResult) -> dict:
    """Top-level scalar summary of a refinement."""
    diag = result.diagnostics
    return {
        "rwp": result.rwp,
        "reduced_chi2": result.reduced_chi2,
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "n_varied": result.state.n_varied,
        "condition_number": diag.get("condition_number", float("nan")),
        "n_points": int(diag.get("n_points", 0)),
        "seed_quality_ok": result.seed_quality_ok,
    }


def classify_parameter(
    p: Parameter, bound_rtol: float = 1e-3, sigma_rtol: float = 1.0
) -> str:
    """Classify one parameter: 'fixed' | 'at_bound' | 'ill_determined' | 'converged'.

    * fixed — not varied.
    * at_bound — refined value sits on a bound (the fit wants to leave the
      physical range; usually a model problem).
    * ill_determined — no/!finite σ, or σ larger than ``sigma_rtol`` × |value|
      (the data don't constrain it — "not converging").
    * converged — finite, well-determined σ inside the bounds.
    """
    if not p.vary:
        return "fixed"
    span = max(abs(p.upper - p.lower), 1e-12)
    if math.isfinite(p.lower) and abs(p.value - p.lower) <= bound_rtol * span:
        return "at_bound"
    if math.isfinite(p.upper) and abs(p.value - p.upper) <= bound_rtol * span:
        return "at_bound"
    if p.sigma is None or not math.isfinite(p.sigma):
        return "ill_determined"
    if p.sigma > sigma_rtol * max(abs(p.value), 1e-12):
        return "ill_determined"
    return "converged"


def classify_parameters(result: RefinementResult) -> dict[str, str]:
    """Per-parameter status for every parameter in the result."""
    return {p.name: classify_parameter(p) for p in result.state.parameters}


def convergence_counts(result: RefinementResult) -> dict[str, int]:
    """Tally of parameter statuses (for a campaign-level health glance)."""
    counts = {"converged": 0, "at_bound": 0, "ill_determined": 0, "fixed": 0}
    for status in classify_parameters(result).values():
        counts[status] += 1
    return counts


@dataclass(frozen=True)
class HistogramMisfit:
    hist_id: str
    rwp: float
    n_points: int


def histogram_misfits(state: RefinementState, forward) -> list[HistogramMisfit]:
    """Per-histogram Rwp, sorted worst-first — the drill-down entry point (UX-4)."""
    out: list[HistogramMisfit] = []
    for h in state.histograms:
        yc = forward.calculate(state, h)
        out.append(HistogramMisfit(h.id, spec.rwp(h.y_obs, yc, h.weights), len(h.x)))
    out.sort(key=lambda m: m.rwp, reverse=True)
    return out


def campaign_summary(results: Sequence[RefinementResult]) -> dict:
    """Roll up a campaign of per-state results: fit-quality spread + health."""
    if not results:
        return {"n_states": 0}
    rwps = np.array([r.rwp for r in results])
    gofs = np.array([r.reduced_chi2 for r in results])
    totals = {"converged": 0, "at_bound": 0, "ill_determined": 0, "fixed": 0}
    for r in results:
        for k, v in convergence_counts(r).items():
            totals[k] += v
    return {
        "n_states": len(results),
        "rwp_mean": float(rwps.mean()),
        "rwp_max": float(rwps.max()),
        "rwp_worst_index": int(np.argmax(rwps)),
        "gof_mean": float(gofs.mean()),
        "n_converged_states": int(sum(r.converged for r in results)),
        "parameter_status_totals": totals,
    }
