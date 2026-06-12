"""
Tests for refinement diagnostics (summaries, parameter convergence, misfits).
"""

from __future__ import annotations

import numpy as np
import pytest

from aura import diagnostics
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
    x = np.linspace(20.0, 120.0, 900)
    blank = Histogram(
        "h",
        DataType.CW_NEUTRON,
        x,
        np.zeros_like(x),
        np.ones_like(x),
        {},
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
        {},
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
        RefinementState((NABR,), (h,), seed), ENGINE, ENGINE, max_iter=60, seed=1
    )


def test_summarize_keys(result):
    s = diagnostics.summarize(result)
    for key in (
        "rwp",
        "reduced_chi2",
        "converged",
        "n_varied",
        "condition_number",
        "n_points",
    ):
        assert key in s
    assert s["n_varied"] == 4
    assert s["n_points"] == 900


def test_classify_parameters_distinguishes_converged_and_fixed(result):
    statuses = diagnostics.classify_parameters(result)
    assert statuses["hist:h:eta"] == "fixed"
    # On clean data the cell refines to a well-determined value.
    assert statuses["phase:NaBr:cell.a"] == "converged"


def test_classify_at_bound():
    p = Parameter(
        "x", ParamKind.PHASE, 6.1, vary=True, lower=5.8, upper=6.1, sigma=1e-4
    )
    assert diagnostics.classify_parameter(p) == "at_bound"


def test_classify_ill_determined():
    p = Parameter(
        "x", ParamKind.PHASE, 5.0, vary=True, lower=0.0, upper=10.0, sigma=None
    )
    assert diagnostics.classify_parameter(p) == "ill_determined"
    p2 = Parameter(
        "x", ParamKind.PHASE, 1.0, vary=True, lower=-10.0, upper=10.0, sigma=50.0
    )
    assert diagnostics.classify_parameter(p2) == "ill_determined"


def test_convergence_counts_sum(result):
    counts = diagnostics.convergence_counts(result)
    assert sum(counts.values()) == len(result.state.parameters)


def test_histogram_misfits_sorted(result):
    misfits = diagnostics.histogram_misfits(result.state, ENGINE)
    assert len(misfits) == 1
    assert misfits[0].rwp >= 0.0
    assert misfits[0].n_points == 900


def test_campaign_summary(result):
    summ = diagnostics.campaign_summary([result, result])
    assert summ["n_states"] == 2
    assert "rwp_mean" in summ and "parameter_status_totals" in summ
    assert summ["n_converged_states"] in (0, 1, 2)
