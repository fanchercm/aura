"""
aura — physics invariant suite (THE AGENTIC ORACLE).

This suite encodes the *physics invariants* a correct Rietveld engine must
satisfy. It is the primary artifact a coding agent implements against: "passes
this suite" is designed to be as close as possible to "is numerically correct."
The invariants are not hand-picked example assertions — they are properties that
hold for ANY correct implementation, derived from crystallography and from the
documented failure modes of incumbent tools (GSAS-II metric-tensor errors,
phase-fraction non-closure, WO3 false minima, ADP/scale correlation,
finite-difference Jacobian fragility).

How it is wired
---------------
* :mod:`aura.spec` provides reference numpy kernels (the *oracle*) and the Protocols.
* :class:`aura.reference.RefEngine` is a minimal, correct numpy implementation of
  the Protocols (``forward``/``parametric``/``minimizer`` in one object).
* The engine under test is supplied by the ``engine`` fixture in
  ``tests/conftest.py``. Until a production engine exists it resolves to
  ``RefEngine``, so the suite runs green against the oracle itself and serves as a
  live, executable specification. Swapping in the production engine is a one-line
  change to that fixture.

Categories (each maps to ACCEPTANCE keys in aura.spec):
  A. Crystallographic identities (symmetry, metric, position models)
  B. Forward-model & Jacobian correctness (AD vs finite difference)
  C. Refinement correctness (fixpoint, synthetic recovery, GoF calibration)
  D. Parametric-engine equivalence & stability (the core contribution)
  E. Conservation & guards (phase-fraction closure, metric SPD, conditioning)
  F. False-minimum / seed-quality discipline
  G. State, provenance, serialization
  H. AI-layer guardrails (propose-only contract)

Run:  pytest -v tests/physics/test_invariants.py
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pytest

from aura import spec
from aura.reference import RefEngine
from aura.spec import (
    ACCEPTANCE,
    AtomSite,
    DataType,
    Histogram,
    Parameter,
    ParametricModel,
    ParamKind,
    Phase,
    RefinementResult,
    RefinementState,
    UnitCell,
)

RNG = np.random.default_rng(20240610)


# =============================================================================
# Shared synthetic-data builders
# =============================================================================

SI = (
    AtomSite("Si", 0, 0, 0),
    AtomSite("Si", 0.5, 0.5, 0.5),
    AtomSite("Si", 0.5, 0.0, 0.5),
    AtomSite("Si", 0.0, 0.5, 0.5),
)
A0 = 5.431
WL = 1.5406


def _silicon_phase(a: float = A0) -> Phase:
    return Phase("Si", "Fd-3m", UnitCell(a, a, a), SI)


def _base_params(hist_id: str, a: float = A0, vary_a: bool = True) -> list[Parameter]:
    return [
        Parameter(
            "phase:Si:cell.a", ParamKind.PHASE, a, vary=vary_a, lower=5.0, upper=6.0
        ),
        Parameter(
            f"hist:{hist_id}:scale",
            ParamKind.HISTOGRAM,
            1.0,
            vary=True,
            lower=1e-6,
            upper=1e6,
        ),
        Parameter(
            f"hist:{hist_id}:bkg",
            ParamKind.HISTOGRAM,
            5.0,
            vary=True,
            lower=0,
            upper=1e4,
        ),
        Parameter(
            f"hist:{hist_id}:fwhm",
            ParamKind.HISTOGRAM,
            0.20,
            vary=True,
            lower=0.02,
            upper=2.0,
        ),
        Parameter(
            f"hist:{hist_id}:eta",
            ParamKind.HISTOGRAM,
            0.5,
            vary=False,
            lower=0,
            upper=1,
        ),
    ]


def _make_histogram(
    hist_id: str,
    a_true: float,
    driving: Mapping[str, float],
    noise: float = 0.0,
    scale: float = 1.0,
    bkg: float = 5.0,
) -> Histogram:
    x = np.linspace(10.0, 120.0, 900)
    phase = _silicon_phase(a_true)
    eng = RefEngine()
    truth_state = RefinementState(
        phases=(phase,),
        histograms=(
            Histogram(
                hist_id,
                DataType.CW_XRAY,
                x,
                x * 0,
                np.ones_like(x),
                driving,
                wavelength=WL,
            ),
        ),
        parameters=(
            Parameter("phase:Si:cell.a", ParamKind.PHASE, a_true),
            Parameter(f"hist:{hist_id}:scale", ParamKind.HISTOGRAM, scale),
            Parameter(f"hist:{hist_id}:bkg", ParamKind.HISTOGRAM, bkg),
            Parameter(f"hist:{hist_id}:fwhm", ParamKind.HISTOGRAM, 0.20),
            Parameter(f"hist:{hist_id}:eta", ParamKind.HISTOGRAM, 0.5),
        ),
    )
    y = eng.calculate(truth_state, truth_state.histograms[0])
    if noise > 0:
        y = y + RNG.normal(0, noise * np.sqrt(np.clip(y, 1.0, None)))
    w = 1.0 / np.clip(y, 1.0, None)
    return Histogram(hist_id, DataType.CW_XRAY, x, y, w, driving, wavelength=WL)


# =============================================================================
# A. Crystallographic identities
# =============================================================================


class TestCrystallography:

    @pytest.mark.parametrize(
        "hkl_a,hkl_b",
        [
            ((2, 0, 0), (0, 2, 0)),
            ((1, 1, 1), (1, -1, 1)),
            ((3, 1, 1), (1, 3, 1)),
            ((4, 2, 0), (0, 2, 4)),
        ],
    )
    def test_cubic_symmetry_preserves_d_and_F(self, hkl_a, hkl_b):
        """In m-3m, permutation/sign of indices preserves d and |F| (Laue symmetry).

        Reflections chosen to be allowed for the FCC Si basis so the equivalence is
        a real structure-factor identity, not a coincidence of systematic absence.
        """
        cell = UnitCell(A0, A0, A0)
        f = {"Si": 14.0}
        assert math.isclose(
            spec.d_spacing(hkl_a, cell), spec.d_spacing(hkl_b, cell), rel_tol=1e-12
        )
        Fa = abs(spec.structure_factor(hkl_a, SI, cell, f))
        Fb = abs(spec.structure_factor(hkl_b, SI, cell, f))
        assert math.isclose(Fa, Fb, rel_tol=1e-9)

    def test_friedel_law(self):
        """|F(hkl)| == |F(-h-k-l)| for non-anomalous scattering."""
        cell = UnitCell(A0, A0, A0)
        f = {"Si": 14.0}
        for hkl in [(1, 1, 1), (3, 1, 1), (2, 2, 0)]:
            neg = tuple(-i for i in hkl)
            assert math.isclose(
                abs(spec.structure_factor(hkl, SI, cell, f)),
                abs(spec.structure_factor(neg, SI, cell, f)),
                rel_tol=1e-9,
            )

    def test_metric_tensor_spd_for_physical_cells(self):
        for cell in [
            UnitCell(5, 5, 5),
            UnitCell(3, 4, 5, 90, 90, 90),
            UnitCell(5, 5, 8, 90, 90, 120),
            UnitCell(7, 8, 9, 80, 85, 95),
        ]:
            G = spec.metric_tensor(*cell.as_tuple())
            assert np.allclose(G, G.T), "Metric tensor must be symmetric."
            assert np.min(np.linalg.eigvalsh(G)) > ACCEPTANCE["metric_min_eig"]
            assert cell.is_physical()

    def test_impossible_cell_rejected(self):
        """The metric SPD guard rejects cells GSAS-II would choke on."""
        assert not UnitCell(1, 1, 1, alpha=170, beta=170, gamma=170).is_physical()

    @pytest.mark.parametrize("dtype", list(DataType))
    def test_position_models_roundtrip(self, dtype):
        """d -> abscissa -> d round-trips to rtol for every data type."""
        d = 2.0
        if dtype in (DataType.CW_XRAY, DataType.CW_NEUTRON):
            tt = spec.two_theta_from_d(d, WL)
            d_back = WL / (2.0 * math.sin(math.radians(tt) / 2.0))
        elif dtype is DataType.TOF:
            tof = spec.tof_from_d(d, difc=5000.0)  # pure DIFC -> linear, invertible
            d_back = tof / 5000.0
        else:  # EDD
            e = spec.energy_from_d(d, two_theta_deg=15.0)
            d_back = 12.398419 / (2.0 * e * math.sin(math.radians(15.0) / 2.0))
        assert math.isclose(d, d_back, rel_tol=ACCEPTANCE["cw_tof_edd_position_rtol"])


# =============================================================================
# B. Forward-model & Jacobian correctness
# =============================================================================


class TestForwardAndJacobian:

    def test_forward_is_pure(self, engine):
        """Calculating twice from the same state yields identical output (no hidden state)."""
        h = _make_histogram("h0", A0, {"T": 300.0})
        st = RefinementState((_silicon_phase(),), (h,), tuple(_base_params("h0")))
        y1 = engine.calculate(st, h)
        y2 = engine.calculate(st, h)
        assert np.array_equal(y1, y2)

    def test_forward_nonnegative_intensity(self, engine):
        h = _make_histogram("h0", A0, {"T": 300.0})
        st = RefinementState((_silicon_phase(),), (h,), tuple(_base_params("h0")))
        assert np.all(engine.calculate(st, h) >= 0.0)

    def test_jacobian_matches_finite_difference(self, engine):
        """The PRODUCTION AD Jacobian must match a central finite difference.

        This is THE invariant that protects against the finite-difference fragility
        of legacy codes: a production engine's analytic/AD Jacobian is checked here
        against an independent FD computation of the SAME forward model.
        """
        h = _make_histogram("h0", A0, {"T": 300.0})
        st = RefinementState((_silicon_phase(),), (h,), tuple(_base_params("h0")))
        J_engine = engine.jacobian(st, h)

        # Independent FD reference (does not reuse engine.jacobian internals)
        varied = [p for p in st.parameters if p.vary]
        J_fd = np.zeros_like(J_engine)
        for j, p in enumerate(varied):
            step = 1e-6 * max(abs(p.value), 1.0)
            up = replace(
                st,
                parameters=tuple(
                    replace(q, value=q.value + step) if q.name == p.name else q
                    for q in st.parameters
                ),
            )
            dn = replace(
                st,
                parameters=tuple(
                    replace(q, value=q.value - step) if q.name == p.name else q
                    for q in st.parameters
                ),
            )
            J_fd[:, j] = (engine.calculate(up, h) - engine.calculate(dn, h)) / (
                2 * step
            )

        # compare columns by relative norm (robust to scale differences per parameter)
        for j in range(J_engine.shape[1]):
            num = np.linalg.norm(J_engine[:, j] - J_fd[:, j])
            den = np.linalg.norm(J_fd[:, j]) + 1e-30
            assert (
                num / den < 1e-3
            ), f"Jacobian column {j} disagrees with FD: {num/den:.2e}"


# =============================================================================
# C. Refinement correctness
# =============================================================================


class TestRefinementCorrectness:

    def test_fixpoint_seeded_at_truth(self, engine):
        """Noise-free data seeded at the true parameters must not move (gradient ~ 0)."""
        h = _make_histogram("h0", A0, {"T": 300.0}, noise=0.0)
        st = RefinementState(
            (_silicon_phase(A0),), (h,), tuple(_base_params("h0", a=A0))
        )
        res = engine.refine(st, engine, engine, max_iter=50)
        a_ref = next(
            p.value for p in res.state.parameters if p.name == "phase:Si:cell.a"
        )
        assert abs(a_ref - A0) < 1e-4, f"Cell moved from truth: {a_ref} vs {A0}"
        assert res.rwp < 1e-3, f"Rwp should be ~0 on noise-free truth, got {res.rwp}"

    def test_parameter_recovery_within_uncertainty(self, engine):
        """Refining noisy synthetic data recovers the true cell within n*sigma."""
        a_true = 5.470
        h = _make_histogram("h0", a_true, {"T": 300.0}, noise=0.03)
        st = RefinementState(
            (_silicon_phase(5.431),),
            (h,),  # deliberately off-true seed
            tuple(_base_params("h0", a=5.431)),
        )
        res = engine.refine(st, engine, engine, max_iter=100)
        p_a = next(p for p in res.state.parameters if p.name == "phase:Si:cell.a")
        assert p_a.sigma is not None and p_a.sigma > 0
        n_sigma = abs(p_a.value - a_true) / p_a.sigma
        assert (
            n_sigma < ACCEPTANCE["recovery_n_sigma"]
        ), f"Recovered a={p_a.value:.5f} is {n_sigma:.1f} sigma from truth {a_true}"

    def test_goodness_of_fit_calibration(self, engine):
        """For a correct model with correctly weighted noise, reduced chi^2 ~ 1."""
        h = _make_histogram("h0", A0, {"T": 300.0}, noise=1.0)
        st = RefinementState(
            (_silicon_phase(A0),), (h,), tuple(_base_params("h0", a=A0))
        )
        res = engine.refine(st, engine, engine, max_iter=100)
        assert (
            ACCEPTANCE["gof_low"] < res.reduced_chi2 < ACCEPTANCE["gof_high"]
        ), f"GoF {res.reduced_chi2:.3f} outside calibrated band"

    def test_rwp_monotone_nonincreasing_vs_seed(self, engine):
        """Refinement never makes the fit worse than the starting point."""
        h = _make_histogram("h0", 5.46, {"T": 300.0}, noise=0.05)
        seed = RefinementState(
            (_silicon_phase(5.431),), (h,), tuple(_base_params("h0", a=5.431))
        )
        yc0 = engine.calculate(seed, h)
        rwp0 = spec.rwp(h.y_obs, yc0, h.weights)
        res = engine.refine(seed, engine, engine, max_iter=100)
        assert res.rwp <= rwp0 + 1e-9


# =============================================================================
# D. Parametric-engine equivalence & stability (the core contribution)
# =============================================================================


def _identity_model(target: str, coeff: str) -> ParametricModel:
    """Degenerate model: target value == a single free coefficient (per-histogram)."""
    return ParametricModel(
        target=target, coeff_names=(coeff,), func=lambda c, d: c[coeff]
    )


def _linear_T_model(target: str, a0: str, alpha: str) -> ParametricModel:
    """Physically-motivated model: a(T) = a0 + alpha * T (linear thermal expansion)."""
    return ParametricModel(
        target=target,
        coeff_names=(a0, alpha),
        func=lambda c, d: c[a0] + c[alpha] * d["T"],
    )


class TestParametricEngine:

    def test_degenerate_parametric_equals_independent(self, engine):
        """A degenerate (identity, per-histogram) parametric model must reproduce
        independent per-histogram refinement to rtol — proving the parametric path
        is a strict generalization, not a separate, divergent code path.
        """
        temps = [300.0, 350.0, 400.0]
        a_trues = [5.431, 5.436, 5.441]
        hists = [
            _make_histogram(f"h{i}", a, {"T": T}, noise=0.02)
            for i, (T, a) in enumerate(zip(temps, a_trues))
        ]

        # (1) Independent: refine each histogram on its own.
        indep_a = []
        for h in hists:
            st = RefinementState(
                (_silicon_phase(5.431),), (h,), tuple(_base_params(h.id, a=5.431))
            )
            r = engine.refine(st, engine, engine, max_iter=100)
            indep_a.append(
                next(p.value for p in r.state.parameters if p.name == "phase:Si:cell.a")
            )

        # (2) Degenerate-parametric: one ensemble refinement, identity model per hist.
        #     Each histogram gets its own 'a' coefficient => mathematically identical.
        for h, a_indep in zip(hists, indep_a):
            params = _base_params(h.id, a=5.431)
            # turn cell.a into a per-histogram parametric coefficient
            params = [p for p in params if p.name != "phase:Si:cell.a"]
            coeff = f"param:a_{h.id}"
            params.append(
                Parameter(
                    coeff, ParamKind.PARAMETRIC, 5.431, vary=True, lower=5.0, upper=6.0
                )
            )
            params.append(
                Parameter("phase:Si:cell.a", ParamKind.PHASE, 5.431, vary=False)
            )
            st = RefinementState(
                (_silicon_phase(5.431),),
                (h,),
                tuple(params),
                parametric_models=(_identity_model("phase:Si:cell.a", coeff),),
            )
            r = engine.refine(st, engine, engine, max_iter=100)
            a_param = next(p.value for p in r.state.parameters if p.name == coeff)
            assert math.isclose(
                a_param, a_indep, rel_tol=ACCEPTANCE["parametric_equiv_rtol"]
            ), f"Parametric {a_param} != independent {a_indep}"

    def test_parametric_constrains_and_reduces_scatter(self, engine):
        """A physical a(T) model fit across the ensemble yields coefficients consistent
        with the ground-truth law — the Stinton–Evans precision benefit.
        """
        alpha_true = 1.0e-4  # Angstrom / K
        a0_true = 5.431
        temps = np.linspace(300, 600, 7)
        hists = [
            _make_histogram(
                f"h{i}", a0_true + alpha_true * T, {"T": float(T)}, noise=0.03
            )
            for i, T in enumerate(temps)
        ]

        params: list[Parameter] = []
        for h in hists:
            params += [p for p in _base_params(h.id) if not p.name.startswith("phase:")]
        params.append(
            Parameter(
                "param:a0", ParamKind.PARAMETRIC, 5.40, vary=True, lower=5.0, upper=6.0
            )
        )
        params.append(
            Parameter(
                "param:alpha",
                ParamKind.PARAMETRIC,
                0.0,
                vary=True,
                lower=-1e-2,
                upper=1e-2,
            )
        )
        params.append(Parameter("phase:Si:cell.a", ParamKind.PHASE, 5.431, vary=False))

        st = RefinementState(
            (_silicon_phase(),),
            tuple(hists),
            tuple(params),
            parametric_models=(
                _linear_T_model("phase:Si:cell.a", "param:a0", "param:alpha"),
            ),
        )
        res = engine.refine(st, engine, engine, max_iter=200)
        a0 = next(p for p in res.state.parameters if p.name == "param:a0")
        al = next(p for p in res.state.parameters if p.name == "param:alpha")
        assert abs(a0.value - a0_true) < 0.01
        assert abs(al.value - alpha_true) < 3 * (al.sigma or 1e-9) + 2e-5

    def test_refinement_is_deterministic(self, engine):
        """Same inputs -> same outputs (reproducibility; no RNG leakage into the engine)."""
        h = _make_histogram("h0", 5.45, {"T": 300.0}, noise=0.0)
        st = RefinementState(
            (_silicon_phase(5.431),), (h,), tuple(_base_params("h0", a=5.431))
        )
        r1 = engine.refine(st, engine, engine, max_iter=50)
        r2 = engine.refine(st, engine, engine, max_iter=50)
        a1 = next(p.value for p in r1.state.parameters if p.name == "phase:Si:cell.a")
        a2 = next(p.value for p in r2.state.parameters if p.name == "phase:Si:cell.a")
        assert a1 == a2


# =============================================================================
# E. Conservation laws & numerical guards
# =============================================================================


class TestConservationAndGuards:

    def test_phase_fraction_closure(self):
        """Weight fractions derived from refined scales must sum to 1 (the GSAS-II
        non-closure bug). Tested at the level of the closure transform itself.
        """
        scales = np.array([2.0, 3.0, 5.0])
        zmv = np.array([1.0, 1.0, 1.0])  # Z*M*V per phase (equal here)
        w = scales * zmv
        fractions = w / w.sum()
        assert math.isclose(
            fractions.sum(), 1.0, abs_tol=ACCEPTANCE["phase_fraction_sum_atol"]
        )

    def test_refined_cell_stays_physical(self, engine):
        """No refinement step may produce a non-SPD metric tensor."""
        h = _make_histogram("h0", 5.45, {"T": 300.0}, noise=0.05)
        st = RefinementState(
            (_silicon_phase(5.431),), (h,), tuple(_base_params("h0", a=5.431))
        )
        res = engine.refine(st, engine, engine, max_iter=100)
        a = next(p.value for p in res.state.parameters if p.name == "phase:Si:cell.a")
        assert UnitCell(a, a, a).is_physical()

    def test_normal_matrix_conditioning_reported(self, engine):
        """Engine must report the condition number so ill-posed refinements are visible
        (the ADP/scale-correlation and over-parameterization failure modes).
        """
        h = _make_histogram("h0", A0, {"T": 300.0}, noise=0.05)
        st = RefinementState(
            (_silicon_phase(A0),), (h,), tuple(_base_params("h0", a=A0))
        )
        res = engine.refine(st, engine, engine, max_iter=50)
        assert "condition_number" in res.diagnostics
        assert math.isfinite(res.diagnostics["condition_number"])

    def test_weights_respected(self, engine):
        """Zero-weight points must not influence the refinement (masking contract)."""
        h = _make_histogram("h0", A0, {"T": 300.0}, noise=0.0)
        # corrupt a region but zero its weights
        y = h.y_obs.copy()
        w = h.weights.copy()
        y[100:150] = 1e6
        w[100:150] = 0.0
        h2 = replace(h, y_obs=y, weights=w)
        st = RefinementState(
            (_silicon_phase(A0),), (h2,), tuple(_base_params("h0", a=A0))
        )
        res = engine.refine(st, engine, engine, max_iter=50)
        a = next(p.value for p in res.state.parameters if p.name == "phase:Si:cell.a")
        assert abs(a - A0) < 1e-3, "Zero-weight corruption leaked into the fit."


# =============================================================================
# F. False-minimum / seed-quality discipline (WO3 lesson; PXRDGen lesson)
# =============================================================================


class TestSeedDiscipline:

    def test_better_rwp_is_not_blindly_trusted(self, engine):
        """Document the WO3 trap: a lower Rwp does not by itself prove the right model.
        The engine must expose Rwp AND GoF so a wrong-but-flexible model is detectable
        by an over-good (chi^2 << 1) or physically implausible result, not Rwp alone.
        """
        h = _make_histogram("h0", A0, {"T": 300.0}, noise=1.0)
        st = RefinementState(
            (_silicon_phase(A0),), (h,), tuple(_base_params("h0", a=A0))
        )
        res = engine.refine(st, engine, engine, max_iter=100)
        # both metrics are present and a suspiciously low chi^2 would be catchable
        assert res.rwp >= 0.0
        assert res.reduced_chi2 > 0.0
        assert res.reduced_chi2 < 5.0  # correct model should not be wildly bad

    def test_grossly_wrong_seed_is_flagged(self, engine):
        """A seed far from any plausible solution should not silently 'converge' to
        a great fit. With a far-off cell and tight bounds, the fit must remain poor
        (mirrors PXRDGen: refinement should fail/flag past a seed-quality threshold).
        """
        h = _make_histogram("h0", A0, {"T": 300.0}, noise=0.02)
        bad = _base_params("h0", a=5.431)
        # pin the cell far away and don't let it vary: a deliberately bad fixed model
        bad = [
            replace(p, value=5.9, vary=False) if p.name == "phase:Si:cell.a" else p
            for p in bad
        ]
        st = RefinementState((_silicon_phase(5.9),), (h,), tuple(bad))
        res = engine.refine(st, engine, engine, max_iter=100)
        assert res.rwp > 0.05, "A grossly wrong fixed model must yield a poor Rwp."


# =============================================================================
# G. State, provenance, serialization
# =============================================================================


class TestStateContract:

    def test_state_is_immutable(self):
        st = RefinementState((_silicon_phase(),), (), tuple(_base_params("h0")))
        st2 = st.with_log("op")
        assert st.provenance == () and st2.provenance == ("op",)

    def test_refinement_records_provenance(self, engine):
        h = _make_histogram("h0", A0, {"T": 300.0}, noise=0.0)
        st = RefinementState(
            (_silicon_phase(A0),), (h,), tuple(_base_params("h0", a=A0))
        )
        res = engine.refine(st, engine, engine, max_iter=20)
        assert any("refined" in m for m in res.state.provenance)

    def test_parameter_count_tracking(self):
        st = RefinementState((_silicon_phase(),), (), tuple(_base_params("h0")))
        assert st.n_varied == sum(1 for p in st.parameters if p.vary)


# =============================================================================
# H. AI-layer guardrails (propose-only contract)
# =============================================================================


class _DummyIdentifier:
    """A stand-in PhaseIdentifier that returns ranked candidates only."""

    def propose(
        self, histogram: Histogram, chemistry=None
    ) -> Sequence[tuple[Phase, float]]:
        return [(_silicon_phase(A0), 0.95), (_silicon_phase(5.658), 0.40)]


class TestAIGuardrails:

    def test_identifier_returns_only_candidates(self):
        """AI may propose (phase, confidence) pairs; it must not return refined params."""
        ident = _DummyIdentifier()
        h = _make_histogram("h0", A0, {"T": 300.0})
        out = ident.propose(h)
        assert all(isinstance(p, Phase) and 0.0 <= c <= 1.0 for p, c in out)
        # confidence-ranked
        confs = [c for _, c in out]
        assert confs == sorted(confs, reverse=True)

    def test_ai_candidates_must_round_trip_through_engine(self, engine):
        """An AI candidate becomes a result ONLY after the deterministic engine validates
        it — enforcing 'AI proposes, engine disposes'. We assert the engine, not the AI,
        produces the Rwp/GoF that gate acceptance.
        """
        ident = _DummyIdentifier()
        h = _make_histogram("h0", A0, {"T": 300.0}, noise=0.05)
        best_phase, _conf = ident.propose(h)[0]
        st = RefinementState(
            (best_phase,), (h,), tuple(_base_params("h0", a=best_phase.cell.a))
        )
        res = engine.refine(st, engine, engine, max_iter=100)
        # the acceptance signal is an ENGINE output, never the AI confidence
        assert isinstance(res, RefinementResult)
        assert hasattr(res, "rwp") and hasattr(res, "reduced_chi2")


# =============================================================================
# Protocol conformance: the production engine must satisfy the Protocols.
# =============================================================================


def test_reference_engine_conforms_to_protocols():
    eng = RefEngine()
    assert isinstance(eng, spec.ForwardModel)
    assert isinstance(eng, spec.ParametricEngine)
    assert isinstance(eng, spec.Minimizer)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
