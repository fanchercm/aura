"""
Parametric-engine equivalence & benefit (oracle category D) — the core contribution.

On production-self-generated CW-neutron data across a driving variable (pressure),
this proves:

* a degenerate per-histogram identity model reproduces independent refinement
  (the parametric path is a strict generalization, not a divergent code path);
* a physical a(P) law is recovered across the ensemble; and
* the parametric fit reduces per-state scatter vs independent fits
  (the Stinton–Evans precision benefit).
"""

from __future__ import annotations

import numpy as np

from aura.engine.forward import ProductionEngine
from aura.engine.models_parametric import identity_model, linear_model
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

RNG = np.random.default_rng(20260612)
WL = 1.909
A0_TRUE = 5.9700
ALPHA_TRUE = 0.02  # Angstrom per unit pressure (exaggerated for a clean signal)
PRESSURES = [0.0, 1.0, 2.0, 3.0, 4.0]

ENGINE = ProductionEngine()


def _phase(a):
    return Phase(
        "NaBr",
        "F m -3 m",
        UnitCell(a, a, a),
        (AtomSite("Na", 0, 0, 0), AtomSite("Br", 0.5, 0.5, 0.5)),
    )


def _truth(a, hid):
    return (
        Parameter("phase:NaBr:cell.a", ParamKind.PHASE, a),
        Parameter(f"hist:{hid}:scale", ParamKind.HISTOGRAM, 2000.0),
        Parameter(f"hist:{hid}:bkg", ParamKind.HISTOGRAM, 50.0),
        Parameter(f"hist:{hid}:fwhm", ParamKind.HISTOGRAM, 0.4),
        Parameter(f"hist:{hid}:eta", ParamKind.HISTOGRAM, 0.5),
    )


def _make_hist(hid, a_true, pressure, noise=True):
    x = np.linspace(20.0, 110.0, 1600)
    blank = Histogram(
        hid,
        DataType.CW_NEUTRON,
        x,
        np.zeros_like(x),
        np.ones_like(x),
        {"P": pressure},
        wavelength=WL,
    )
    y = ENGINE.calculate(
        RefinementState((_phase(a_true),), (blank,), _truth(a_true, hid)), blank
    )
    if noise:
        y = y + RNG.normal(0.0, np.sqrt(np.clip(y, 1.0, None)))
    w = 1.0 / np.clip(np.abs(y), 1.0, None)
    return Histogram(hid, DataType.CW_NEUTRON, x, y, w, {"P": pressure}, wavelength=WL)


def _hist_params(hid, scale=1500.0, bkg=40.0, fwhm=0.5):
    return [
        Parameter(
            f"hist:{hid}:scale",
            ParamKind.HISTOGRAM,
            scale,
            vary=True,
            lower=1e-3,
            upper=1e7,
        ),
        Parameter(
            f"hist:{hid}:bkg", ParamKind.HISTOGRAM, bkg, vary=True, lower=0.0, upper=1e5
        ),
        Parameter(
            f"hist:{hid}:fwhm",
            ParamKind.HISTOGRAM,
            fwhm,
            vary=True,
            lower=0.05,
            upper=2.0,
        ),
        Parameter(f"hist:{hid}:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
    ]


def _refine(state, **kw):
    return ENGINE.refine(state, ENGINE, ENGINE, seed=1, **kw)


class TestParametricEquivalence:

    def test_degenerate_parametric_equals_independent(self):
        """Per-histogram identity models reproduce independent per-pattern fits."""
        a_trues = [5.96, 5.98, 6.00]
        hists = [
            _make_hist(f"h{i}", a, P, noise=True)
            for i, (a, P) in enumerate(zip(a_trues, PRESSURES[:3]))
        ]

        # (1) Independent: each histogram refined on its own with its own cell.a.
        indep = []
        for h in hists:
            params = [
                Parameter(
                    "phase:NaBr:cell.a",
                    ParamKind.PHASE,
                    5.97,
                    vary=True,
                    lower=5.8,
                    upper=6.1,
                )
            ] + _hist_params(h.id)
            st = RefinementState((_phase(5.97),), (h,), tuple(params))
            r = _refine(st, max_iter=120)
            indep.append(
                next(
                    p.value for p in r.state.parameters if p.name == "phase:NaBr:cell.a"
                )
            )

        # (2) Degenerate parametric: each histogram gets its own identity coefficient.
        for h, a_ind in zip(hists, indep):
            coeff = f"param:a_{h.id}"
            params = _hist_params(h.id) + [
                Parameter(
                    coeff, ParamKind.PARAMETRIC, 5.97, vary=True, lower=5.8, upper=6.1
                ),
                Parameter("phase:NaBr:cell.a", ParamKind.PHASE, 5.97, vary=False),
            ]
            st = RefinementState(
                (_phase(5.97),),
                (h,),
                tuple(params),
                parametric_models=(identity_model("phase:NaBr:cell.a", coeff),),
            )
            r = _refine(st, max_iter=120)
            a_param = next(p.value for p in r.state.parameters if p.name == coeff)
            assert (
                abs(a_param - a_ind) < 1e-4
            ), f"parametric {a_param} != independent {a_ind}"

    def test_parametric_recovers_linear_law(self):
        """a(P) = a0 + alpha*P refined across the ensemble recovers the true law."""
        hists = [
            _make_hist(f"h{i}", A0_TRUE + ALPHA_TRUE * P, P, noise=True)
            for i, P in enumerate(PRESSURES)
        ]
        params: list[Parameter] = []
        for h in hists:
            params += _hist_params(h.id)
        params += [
            Parameter(
                "param:a0", ParamKind.PARAMETRIC, 5.95, vary=True, lower=5.8, upper=6.1
            ),
            Parameter(
                "param:alpha",
                ParamKind.PARAMETRIC,
                0.0,
                vary=True,
                lower=-0.2,
                upper=0.2,
            ),
            Parameter("phase:NaBr:cell.a", ParamKind.PHASE, A0_TRUE, vary=False),
        ]
        st = RefinementState(
            (_phase(A0_TRUE),),
            tuple(hists),
            tuple(params),
            parametric_models=(
                linear_model("phase:NaBr:cell.a", "param:a0", "param:alpha", "P"),
            ),
        )
        res = _refine(st, max_iter=200)
        a0 = next(p for p in res.state.parameters if p.name == "param:a0")
        al = next(p for p in res.state.parameters if p.name == "param:alpha")
        assert abs(a0.value - A0_TRUE) < 0.01, f"a0={a0.value}"
        assert abs(al.value - ALPHA_TRUE) < 0.005, f"alpha={al.value}"
        assert al.sigma is not None and al.sigma > 0

    def test_parametric_reduces_scatter_vs_independent(self):
        """The a(P) law constrains the trend, so its per-state cell values scatter
        less around the true line than independent per-state fits (Stinton–Evans)."""
        truths = {P: A0_TRUE + ALPHA_TRUE * P for P in PRESSURES}
        hists = [
            _make_hist(f"h{i}", truths[P], P, noise=True)
            for i, P in enumerate(PRESSURES)
        ]

        # Independent per-state cell.a.
        indep_resid = []
        for h, P in zip(hists, PRESSURES):
            params = [
                Parameter(
                    "phase:NaBr:cell.a",
                    ParamKind.PHASE,
                    5.97,
                    vary=True,
                    lower=5.8,
                    upper=6.1,
                )
            ] + _hist_params(h.id)
            r = _refine(
                RefinementState((_phase(5.97),), (h,), tuple(params)), max_iter=120
            )
            a = next(
                p.value for p in r.state.parameters if p.name == "phase:NaBr:cell.a"
            )
            indep_resid.append(a - truths[P])

        # Parametric a(P): evaluate the fitted law at each pressure.
        params = []
        for h in hists:
            params += _hist_params(h.id)
        params += [
            Parameter(
                "param:a0", ParamKind.PARAMETRIC, 5.95, vary=True, lower=5.8, upper=6.1
            ),
            Parameter(
                "param:alpha",
                ParamKind.PARAMETRIC,
                0.0,
                vary=True,
                lower=-0.2,
                upper=0.2,
            ),
            Parameter("phase:NaBr:cell.a", ParamKind.PHASE, A0_TRUE, vary=False),
        ]
        res = _refine(
            RefinementState(
                (_phase(A0_TRUE),),
                tuple(hists),
                tuple(params),
                parametric_models=(
                    linear_model("phase:NaBr:cell.a", "param:a0", "param:alpha", "P"),
                ),
            ),
            max_iter=200,
        )
        a0 = next(p.value for p in res.state.parameters if p.name == "param:a0")
        al = next(p.value for p in res.state.parameters if p.name == "param:alpha")
        param_resid = [a0 + al * P - truths[P] for P in PRESSURES]

        rms_indep = float(np.sqrt(np.mean(np.square(indep_resid))))
        rms_param = float(np.sqrt(np.mean(np.square(param_resid))))
        assert (
            rms_param < rms_indep
        ), f"parametric rms {rms_param:.2e} !< independent {rms_indep:.2e}"

    def test_parametric_refinement_is_deterministic(self):
        h = _make_hist("h0", 5.98, 1.0, noise=False)
        coeff = "param:a_h0"
        params = _hist_params("h0") + [
            Parameter(
                coeff, ParamKind.PARAMETRIC, 5.97, vary=True, lower=5.8, upper=6.1
            ),
            Parameter("phase:NaBr:cell.a", ParamKind.PHASE, 5.97, vary=False),
        ]
        st = RefinementState(
            (_phase(5.97),),
            (h,),
            tuple(params),
            parametric_models=(identity_model("phase:NaBr:cell.a", coeff),),
        )
        r1 = _refine(st, max_iter=60)
        r2 = _refine(st, max_iter=60)
        v1 = next(p.value for p in r1.state.parameters if p.name == coeff)
        v2 = next(p.value for p in r2.state.parameters if p.name == coeff)
        assert v1 == v2


def test_expand_is_identity_without_models():
    """With no parametric models, expand returns the state unchanged."""
    h = _make_hist("h0", 5.97, 0.0, noise=False)
    st = RefinementState((_phase(5.97),), (h,), _truth(5.97, "h0"))
    assert ENGINE.expand(st, h) is st


def test_expand_resolves_driving_variable():
    """expand writes the model value (evaluated at the histogram's driving P)
    into the target parameter."""
    h = _make_hist("h0", 5.97, 3.0, noise=False)
    params = (
        Parameter("param:a0", ParamKind.PARAMETRIC, 5.90),
        Parameter("param:alpha", ParamKind.PARAMETRIC, 0.02),
        Parameter("phase:NaBr:cell.a", ParamKind.PHASE, 0.0),
    )
    st = RefinementState(
        (_phase(5.90),),
        (h,),
        params,
        parametric_models=(
            linear_model("phase:NaBr:cell.a", "param:a0", "param:alpha", "P"),
        ),
    )
    expanded = ENGINE.expand(st, h)
    a = next(p.value for p in expanded.parameters if p.name == "phase:NaBr:cell.a")
    assert abs(a - (5.90 + 0.02 * 3.0)) < 1e-12
    # Original state unchanged (immutability).
    a_orig = next(p.value for p in st.parameters if p.name == "phase:NaBr:cell.a")
    assert a_orig == 0.0
