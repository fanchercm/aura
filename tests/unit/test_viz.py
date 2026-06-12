"""
Smoke tests for visualization: each function produces a Matplotlib figure and a
non-empty file (headless Agg backend).
"""

from __future__ import annotations

import numpy as np
import pytest

from aura import viz
from aura.engine.forward import ProductionEngine
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


@pytest.fixture(scope="module")
def result():
    x = np.linspace(20.0, 120.0, 700)
    blank = Histogram(
        "h",
        DataType.CW_NEUTRON,
        x,
        np.zeros_like(x),
        np.ones_like(x),
        {"P": 1.0},
        wavelength=1.909,
    )
    truth = (
        Parameter("phase:NaBr:cell.a", ParamKind.PHASE, 5.99),
        Parameter("hist:h:scale", ParamKind.HISTOGRAM, 200.0),
        Parameter("hist:h:bkg", ParamKind.HISTOGRAM, 10.0),
        Parameter("hist:h:fwhm", ParamKind.HISTOGRAM, 0.4),
        Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5),
    )
    y = ENGINE.calculate(RefinementState((NABR,), (blank,), truth), blank)
    h = Histogram(
        "h",
        DataType.CW_NEUTRON,
        x,
        y,
        1.0 / np.clip(y, 1.0, None),
        {"P": 1.0},
        wavelength=1.909,
    )
    seed = (
        Parameter(
            "phase:NaBr:cell.a", ParamKind.PHASE, 5.97, vary=True, lower=5.8, upper=6.1
        ),
        Parameter(
            "hist:h:scale", ParamKind.HISTOGRAM, 150.0, vary=True, lower=1e-3, upper=1e6
        ),
        Parameter(
            "hist:h:bkg", ParamKind.HISTOGRAM, 8.0, vary=True, lower=0.0, upper=1e4
        ),
        Parameter(
            "hist:h:fwhm", ParamKind.HISTOGRAM, 0.45, vary=True, lower=0.05, upper=2.0
        ),
        Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
    )
    return ENGINE.refine(
        RefinementState((NABR,), (h,), seed), ENGINE, ENGINE, max_iter=50, seed=1
    )


def test_plot_fit_saves_file(result, tmp_path):
    import matplotlib.figure

    p = tmp_path / "fit.png"
    fig = viz.plot_fit(result.state, ENGINE, path=p)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert p.is_file() and p.stat().st_size > 0


def test_plot_parameter_convergence_saves_file(result, tmp_path):
    p = tmp_path / "conv.png"
    viz.plot_parameter_convergence(result, path=p)
    assert p.is_file() and p.stat().st_size > 0


def test_plot_campaign_trajectory_saves_file(result, tmp_path):
    p = tmp_path / "traj.png"
    viz.plot_campaign_trajectory([result], "phase:NaBr:cell.a", "P", path=p)
    assert p.is_file() and p.stat().st_size > 0
