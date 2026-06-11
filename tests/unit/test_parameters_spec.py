"""
Parameter / refinement-state semantics against the spec model.

Re-expresses the still-valuable ideas from the retired ``aura.parameters``
tests (parameter value handling, free-parameter vector round-trip, constraint
propagation) against the authoritative immutable :mod:`aura.spec` types. The
full parametric-equivalence physics lives in ``tests/physics/test_invariants.py``;
these are the fast unit-level checks.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from aura.spec import (
    AtomSite,
    Parameter,
    ParametricModel,
    ParamKind,
    Phase,
    RefinementState,
    UnitCell,
)


def _params() -> tuple[Parameter, ...]:
    return (
        Parameter(
            "phase:X:cell.a", ParamKind.PHASE, 5.0, vary=True, lower=4.0, upper=6.0
        ),
        Parameter(
            "hist:h0:scale", ParamKind.HISTOGRAM, 1.0, vary=True, lower=0.0, upper=10.0
        ),
        Parameter("hist:h0:bkg", ParamKind.HISTOGRAM, 2.0, vary=False),
    )


def _state(params=None, **kw) -> RefinementState:
    phase = Phase("X", "P 1", UnitCell(5, 5, 5), (AtomSite("Si", 0, 0, 0),))
    return RefinementState(
        phases=(phase,),
        histograms=(),
        parameters=params if params is not None else _params(),
        **kw,
    )


# --- Parameter value semantics -------------------------------------------------


class TestParameterSemantics:

    def test_parameter_is_immutable(self):
        p = Parameter("x", ParamKind.PHASE, 1.0)
        try:
            p.value = 2.0  # type: ignore[misc]
        except AttributeError:
            pass
        else:
            raise AssertionError("Parameter must be frozen/immutable")

    def test_replace_makes_a_new_value(self):
        p = Parameter("x", ParamKind.PHASE, 1.0)
        p2 = replace(p, value=2.0)
        assert p.value == 1.0 and p2.value == 2.0

    def test_n_varied_counts_only_varying(self):
        st = _state()
        assert st.n_varied == 2  # cell.a + scale vary; bkg fixed


# --- Free-parameter vector round-trip -----------------------------------------


def _get_free_vector(state: RefinementState) -> np.ndarray:
    return np.array([p.value for p in state.parameters if p.vary], dtype=float)


def _set_free_vector(state: RefinementState, x: np.ndarray) -> RefinementState:
    it = iter(x)
    new = tuple(
        replace(p, value=float(next(it))) if p.vary else p for p in state.parameters
    )
    return replace(state, parameters=new)


class TestFreeVectorRoundTrip:

    def test_get_set_roundtrip_identity(self):
        st = _state()
        x = _get_free_vector(st)
        assert np.allclose(x, [5.0, 1.0])
        st2 = _set_free_vector(st, x)
        assert np.allclose(_get_free_vector(st2), x)

    def test_set_updates_only_free_params(self):
        st = _state()
        st2 = _set_free_vector(st, np.array([5.5, 2.5]))
        vals = {p.name: p.value for p in st2.parameters}
        assert vals["phase:X:cell.a"] == 5.5
        assert vals["hist:h0:scale"] == 2.5
        assert vals["hist:h0:bkg"] == 2.0  # fixed, untouched


# --- Constraint / parametric propagation --------------------------------------


class TestParametricPropagation:

    def test_identity_model_is_an_equality_tie(self):
        """A degenerate identity ParametricModel ties a target to one coefficient."""
        model = ParametricModel(
            target="phase:X:cell.a",
            coeff_names=("param:a",),
            func=lambda c, d: c["param:a"],
        )
        coeffs = {"param:a": 5.43}
        assert model.func(coeffs, {}) == 5.43

    def test_linear_model_propagates_driving(self):
        """a(T) = a0 + alpha*T evaluates against the histogram's driving coord."""
        model = ParametricModel(
            target="phase:X:cell.a",
            coeff_names=("param:a0", "param:alpha"),
            func=lambda c, d: c["param:a0"] + c["param:alpha"] * d["T"],
        )
        val = model.func({"param:a0": 5.0, "param:alpha": 1e-3}, {"T": 100.0})
        assert abs(val - 5.1) < 1e-12

    def test_provenance_is_appended_immutably(self):
        st = _state()
        st2 = st.with_log("did a thing")
        assert st.provenance == ()
        assert st2.provenance == ("did a thing",)
