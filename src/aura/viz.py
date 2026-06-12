"""
Visualization for high-volume refinement (project-brief UX goal).

Three views that make a large refinement comprehensible:

* :func:`plot_fit` — the classic observed / calculated / difference plot for one
  histogram.
* :func:`plot_parameter_convergence` — every varied parameter's relative
  uncertainty, colored by status (converged / at-bound / ill-determined), so a
  user can see at a glance which parameters converged and which are not.
* :func:`plot_campaign_trajectory` — a refined parameter (with error bars) versus
  the driving variable across a campaign of states (the stimulus trajectory).

Uses the non-interactive Agg backend, so it works headless / in CI. Each function
returns a Matplotlib ``Figure`` and optionally saves it to ``path``.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aura.diagnostics import classify_parameter  # noqa: E402
from aura.spec import RefinementResult, RefinementState  # noqa: E402

_STATUS_COLOR = {
    "converged": "#2ca02c",
    "at_bound": "#ff7f0e",
    "ill_determined": "#d62728",
    "fixed": "#7f7f7f",
}


def plot_fit(state: RefinementState, forward, hist_id: str | None = None, path=None):
    """Observed / calculated / difference plot for one histogram."""
    hist = _pick_histogram(state, hist_id)
    yc = forward.calculate(state, hist)
    diff = hist.y_obs - yc
    offset = float(np.min(hist.y_obs)) - 0.15 * float(np.ptp(hist.y_obs) or 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist.x, hist.y_obs, ".", ms=2, color="#1f77b4", label="observed")
    ax.plot(hist.x, yc, "-", lw=1.0, color="#d62728", label="calculated")
    ax.plot(hist.x, diff + offset, "-", lw=0.8, color="#7f7f7f", label="difference")
    ax.axhline(offset, color="k", lw=0.4)
    ax.set_xlabel(_abscissa_label(hist))
    ax.set_ylabel("intensity")
    ax.set_title(f"Fit: {hist.id}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_parameter_convergence(result: RefinementResult, path=None):
    """Relative uncertainty (σ/|value|) per varied parameter, colored by status."""
    rows = []
    for p in result.state.parameters:
        if not p.vary:
            continue
        status = classify_parameter(p)
        rel = (p.sigma / abs(p.value)) if (p.sigma is not None and p.value) else np.nan
        rows.append((p.name, rel, status))

    fig, ax = plt.subplots(figsize=(8, max(2.0, 0.3 * len(rows) + 1)))
    if rows:
        names = [r[0] for r in rows]
        # ill-determined / nan relative error shown at the right edge for visibility.
        finite = [r[1] for r in rows if np.isfinite(r[1])]
        cap = (max(finite) * 1.2) if finite else 1.0
        vals = [r[1] if np.isfinite(r[1]) else cap for r in rows]
        colors = [_STATUS_COLOR[r[2]] for r in rows]
        ax.barh(range(len(rows)), vals, color=colors)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("relative uncertainty  σ/|value|")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _STATUS_COLOR.values()]
    ax.legend(handles, list(_STATUS_COLOR), fontsize=7, loc="lower right")
    ax.set_title("Parameter convergence")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_campaign_trajectory(
    results: Sequence[RefinementResult],
    param_name: str,
    driving_key: str,
    path=None,
):
    """A refined parameter (± σ) versus the driving variable across states."""
    xs, ys, es = [], [], []
    for r in results:
        p = next((q for q in r.state.parameters if q.name == param_name), None)
        if p is None or not r.state.histograms:
            continue
        drive = r.state.histograms[0].driving.get(driving_key)
        if drive is None:
            continue
        xs.append(float(drive))
        ys.append(p.value)
        es.append(p.sigma if (p.sigma is not None and np.isfinite(p.sigma)) else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = np.argsort(xs) if xs else []
    if len(xs):
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        es = np.array(es)[order]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=4, lw=1.0, capsize=3, color="#1f77b4")
    ax.set_xlabel(driving_key)
    ax.set_ylabel(param_name)
    ax.set_title(f"{param_name} vs {driving_key}")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def _pick_histogram(state: RefinementState, hist_id: str | None):
    if hist_id is None:
        return state.histograms[0]
    for h in state.histograms:
        if h.id == hist_id:
            return h
    raise KeyError(f"No histogram {hist_id!r} in state")


def _abscissa_label(hist) -> str:
    from aura.spec import DataType

    return {
        DataType.CW_XRAY: "2θ (deg)",
        DataType.CW_NEUTRON: "2θ (deg)",
        DataType.TOF: "TOF (µs)",
        DataType.EDD: "energy (keV)",
    }.get(hist.data_type, "x")
