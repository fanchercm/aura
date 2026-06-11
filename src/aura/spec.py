"""
nextgen_rietveld.spec
=====================

EXECUTABLE SPECIFICATION for a parametric-first, differentiable Rietveld engine
for dynamic (parametric) X-ray (CW, EDD) and neutron (CW, TOF) diffraction
experiments.

This file is the Phase 0/1 deliverable. It is *executable* in three senses:

  1. It imports and type-checks (run `python -m mypy spec.py` or just import it).
  2. The Protocols are @runtime_checkable, so an implementation can be validated
     against them with isinstance(impl, ForwardModel) etc.
  3. Running `python spec.py` executes a self-consistency check of the contract:
     it instantiates the data model, exercises the reference (numpy) implementations
     of the pure-math helpers, and asserts the invariants those helpers must satisfy.

Design commitments (the architecture the agent must NOT relitigate):

  * PARAMETRIC-FIRST. A refined quantity is, in general, a function p(v) of one or
    more external driving variables v (temperature, time, pressure, field, load).
    "Independent per-histogram" and "sequential" refinement are *degenerate
    parametric models*, not separate code paths. (Stinton & Evans, 2007.)
  * DIRECT, AUTODIFF-FRIENDLY CELL PARAMETERIZATION. The reciprocal metric tensor
    is derived from (a,b,c,alpha,beta,gamma); there is no separate "A-tensor" or
    "Dij offset" abstraction layer for sequential mode. (Avoids the GSAS-II
    metric-tensor failure class.)
  * DIFFERENTIABLE FORWARD MODEL. The map params -> calculated pattern is a single
    differentiable function; Jacobians come from automatic differentiation, not
    finite differences. (JAXFit; JAX-COSMO.)
  * AI PROPOSES, ENGINE DISPOSES. AI services may emit candidate phases, seeds, and
    bounds; they round-trip through the same deterministic engine and provenance log.
  * IMMUTABLE, SERIALIZABLE STATE. A refinement is a pure transformation
    RefinementState -> RefinementResult with full provenance.

The numeric kernels below are written in numpy for clarity and to serve as the
*reference oracle* implementations the test suite checks a production
(JAX/PyTorch/GPU) engine against. A production implementation MUST agree with
these to tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Protocol,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


# =============================================================================
# 1. Data-type / experiment taxonomy
# =============================================================================


class DataType(str, Enum):
    """Diffraction data type. Determines the abscissa and the position model."""

    CW_XRAY = "cw_xray"  # constant-wavelength X-ray; abscissa = 2theta (deg)
    CW_NEUTRON = "cw_neutron"  # constant-wavelength neutron; abscissa = 2theta (deg)
    TOF = "tof"  # time-of-flight neutron; abscissa = TOF (microseconds)
    EDD = "edd"  # energy-dispersive X-ray; abscissa = energy (keV), fixed 2theta


class ParamKind(str, Enum):
    """Taxonomy of refinable parameters (the four classes the engine must track)."""

    PHASE = "phase"  # one set per phase: cell, fractional xyz, occ, ADP
    HISTOGRAM = "histogram"  # one set per histogram: scale, background, instrument profile, displacement
    PHASE_DATA = "phase_data"  # per (phase, histogram): phase fraction, size, microstrain, texture
    PARAMETRIC = "parametric"  # coefficients of a p(v) model that drives any of the above across the ensemble


# =============================================================================
# 2. Pure-math reference kernels (the ORACLE implementations)
#    A production engine reimplements these on GPU/autodiff and must AGREE.
# =============================================================================


def metric_tensor(
    a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float
) -> Array:
    """Real-space metric tensor G. d^2 between fractional vectors uses G.

    G is symmetric positive-definite for any physically valid cell.
    """
    al, be, ga = map(math.radians, (alpha_deg, beta_deg, gamma_deg))
    return np.array(
        [
            [a * a, a * b * math.cos(ga), a * c * math.cos(be)],
            [a * b * math.cos(ga), b * b, b * c * math.cos(al)],
            [a * c * math.cos(be), b * c * math.cos(al), c * c],
        ],
        dtype=np.float64,
    )


def reciprocal_metric_tensor(
    a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float
) -> Array:
    """Reciprocal metric tensor G* = inv(G). d*^2(hkl) = h^T G* h."""
    return np.linalg.inv(metric_tensor(a, b, c, alpha_deg, beta_deg, gamma_deg))


def d_spacing(hkl: Sequence[int], cell: UnitCell) -> float:
    """Interplanar spacing d for a reflection, from the reciprocal metric tensor."""
    h = np.asarray(hkl, dtype=np.float64)
    gstar = reciprocal_metric_tensor(*cell.as_tuple())
    dstar2 = float(h @ gstar @ h)
    if dstar2 <= 0:
        raise ValueError("Non-positive d*^2; invalid cell or reflection.")
    return 1.0 / math.sqrt(dstar2)


def two_theta_from_d(d: float, wavelength: float) -> float:
    """Bragg 2theta (deg) for CW data. Raises if reflection is unobservable."""
    s = wavelength / (2.0 * d)
    if not -1.0 <= s <= 1.0:
        raise ValueError("Reflection beyond Ewald limit for this wavelength.")
    return math.degrees(2.0 * math.asin(s))


def tof_from_d(d: float, difc: float, difa: float = 0.0, zero: float = 0.0) -> float:
    """TOF (us) from d-spacing. TOF = DIFC*d + DIFA*d^2 + ZERO (GSAS convention)."""
    return difc * d + difa * d * d + zero


def energy_from_d(d: float, two_theta_deg: float) -> float:
    """EDD: photon energy (keV) for a reflection at the fixed detector angle.

    E = h c / (2 d sin(theta)); using hc = 12.398419 keV*Angstrom.
    """
    HC = 12.398419  # keV * Angstrom
    theta = math.radians(two_theta_deg) / 2.0
    return HC / (2.0 * d * math.sin(theta))


def structure_factor(
    hkl: Sequence[int],
    atoms: Sequence[AtomSite],
    cell: UnitCell,
    scattering: Mapping[str, float],
) -> complex:
    """Structure factor F(hkl) with isotropic ADP (reference, scalar form factors).

    F = sum_j occ_j * f_j * exp(2 pi i (h x + k y + l z)) * exp(-B_j (sin th/lambda)^2)
    'scattering' maps element symbol -> f (treated Q-independent here for the oracle;
    a production model uses Q-dependent form factors / neutron b).
    """
    h, k, l = hkl
    s2 = 1.0 / (4.0 * d_spacing(hkl, cell) ** 2)  # (sin theta / lambda)^2 = 1/(4 d^2)
    F = 0.0 + 0.0j
    for at in atoms:
        f = scattering[at.element]
        dw = math.exp(-at.b_iso * s2)
        phase = 2.0 * math.pi * (h * at.x + k * at.y + l * at.z)
        F += at.occ * f * dw * complex(math.cos(phase), math.sin(phase))
    return F


def pseudo_voigt(x: Array, center: float, fwhm: float, eta: float) -> Array:
    """Normalized pseudo-Voigt profile (eta in [0,1]: 1=Lorentzian, 0=Gaussian).

    Integral over x is ~1 (used as the peak shape in the reference forward model).
    """
    if fwhm <= 0:
        raise ValueError("FWHM must be positive.")
    eta = min(max(eta, 0.0), 1.0)
    dx = x - center
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gauss = np.exp(-0.5 * (dx / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
    gamma = fwhm / 2.0
    lorentz = (gamma / math.pi) / (dx * dx + gamma * gamma)
    return eta * lorentz + (1.0 - eta) * gauss


def weighted_residual(y_obs: Array, y_calc: Array, weights: Array) -> Array:
    """Per-point weighted residual sqrt(w)*(y_obs - y_calc). Sum of squares = chi^2 numerator."""
    return np.sqrt(weights) * (y_obs - y_calc)


def rwp(y_obs: Array, y_calc: Array, weights: Array) -> float:
    """Weighted profile R-factor (fraction, not percent)."""
    num = float(np.sum(weights * (y_obs - y_calc) ** 2))
    den = float(np.sum(weights * y_obs**2))
    return math.sqrt(num / den)


def reduced_chi_square(
    y_obs: Array, y_calc: Array, weights: Array, n_params: int
) -> float:
    """Goodness of fit chi^2 / (N_obs - N_params). ~1 for a correct model + correct noise."""
    chi2 = float(np.sum(weights * (y_obs - y_calc) ** 2))
    dof = len(y_obs) - n_params
    if dof <= 0:
        raise ValueError("Non-positive degrees of freedom.")
    return chi2 / dof


# =============================================================================
# 3. Immutable state model
# =============================================================================


@dataclass(frozen=True)
class UnitCell:
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def is_physical(self) -> bool:
        """Cell is valid iff its metric tensor is symmetric positive-definite."""
        if min(self.a, self.b, self.c) <= 0:
            return False
        try:
            eig = np.linalg.eigvalsh(metric_tensor(*self.as_tuple()))
        except np.linalg.LinAlgError:
            return False
        return bool(np.all(eig > 0))


@dataclass(frozen=True)
class AtomSite:
    element: str
    x: float
    y: float
    z: float
    occ: float = 1.0
    b_iso: float = 0.5  # Angstrom^2


@dataclass(frozen=True)
class Phase:
    name: str
    space_group: str  # Hermann–Mauguin; resolved via cctbx in production
    cell: UnitCell
    atoms: tuple[AtomSite, ...]


@dataclass(frozen=True)
class Histogram:
    """One measured pattern plus its driving-variable coordinate(s)."""

    id: str
    data_type: DataType
    x: Array  # abscissa (2theta deg | TOF us | energy keV)
    y_obs: Array
    weights: Array  # typically 1/sigma^2
    driving: Mapping[str, float]  # e.g. {"T": 300.0} or {"t": 12.0, "P": 1.0}
    wavelength: float | None = None  # CW only
    two_theta_fixed: float | None = None  # EDD only
    difc: float | None = None  # TOF only
    difa: float = 0.0
    zero: float = 0.0

    def __post_init__(self) -> None:
        if not (len(self.x) == len(self.y_obs) == len(self.weights)):
            raise ValueError("x, y_obs, weights length mismatch.")
        if (
            self.data_type in (DataType.CW_XRAY, DataType.CW_NEUTRON)
            and self.wavelength is None
        ):
            raise ValueError("CW data requires a wavelength.")
        if self.data_type is DataType.TOF and self.difc is None:
            raise ValueError("TOF data requires DIFC.")
        if self.data_type is DataType.EDD and self.two_theta_fixed is None:
            raise ValueError("EDD data requires a fixed 2theta.")


@dataclass(frozen=True)
class Parameter:
    """A single scalar handle the minimizer may vary."""

    name: str  # unique, e.g. "phase:alumina:cell.a" or "param:therm.alpha"
    kind: ParamKind
    value: float
    vary: bool = False
    lower: float = -math.inf
    upper: float = math.inf
    sigma: float | None = None  # populated post-refinement


@dataclass(frozen=True)
class ParametricModel:
    """Maps a driving-variable vector to the value of a target quantity.

    `func(coeffs, driving) -> value`. The DEGENERATE model (one free coeff per
    histogram, identity map) reproduces independent/sequential refinement and is
    what the equivalence invariant tests against.
    """

    target: str  # which parameter this model drives, e.g. "phase:Si:cell.a"
    coeff_names: tuple[str, ...]
    func: Callable[[Mapping[str, float], Mapping[str, float]], float]


@dataclass(frozen=True)
class RefinementState:
    """The complete, serializable description of a refinement problem."""

    phases: tuple[Phase, ...]
    histograms: tuple[Histogram, ...]
    parameters: tuple[Parameter, ...]
    parametric_models: tuple[ParametricModel, ...] = ()
    constraints: tuple[str, ...] = ()  # e.g. "sum(phase_fraction) == 1"
    provenance: tuple[str, ...] = ()  # ordered log of operations applied

    def with_log(self, msg: str) -> RefinementState:
        return replace(self, provenance=self.provenance + (msg,))

    @property
    def n_varied(self) -> int:
        return sum(1 for p in self.parameters if p.vary)


@dataclass(frozen=True)
class ProvenanceManifest:
    """Immutable record making one refinement reproducible (oracle §13).

    Every refinement must be traceable to the exact inputs, code, and
    configuration that produced it. Missing required fields are a failure: a
    result without provenance is not a trustworthy scientific result. Built by
    :mod:`aura.provenance`; serialized via its ``to_yaml``/``from_yaml``.
    """

    input_data_hash: str  # sha256 over all histogram (x, y_obs, weights)
    software_version: str  # aura.__version__
    git_commit: str  # repo HEAD, or "unknown"
    kernel_backend: str  # engine.name (e.g. "reference-numpy")
    optimizer: Mapping[str, object]  # max_iter, tol, damping, method, ...
    random_seed: int | None  # seed threaded into refine(), for determinism
    parameter_graph_hash: str  # topology: (name, kind, vary, bounds) + models
    phase_model_hash: str  # phases: cells + atom sites
    instrument_model_hash: str  # per-histogram instrument terms
    environment_lock: str  # sha256 of pixi.lock (or "unknown")
    agent_patch_id: str | None = None  # AI patch identifier, if any
    human_review_state: str | None = None  # e.g. "approved", "pending"


@dataclass(frozen=True)
class RefinementResult:
    state: RefinementState  # updated parameters (with sigmas)
    rwp: float
    reduced_chi2: float
    converged: bool
    n_iterations: int
    covariance: Array | None = None  # parameter covariance matrix
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    seed_quality_ok: bool = True  # False => seed too poor to refine (PXRDGen lesson)
    provenance: ProvenanceManifest | None = None  # populated at refine() time


# =============================================================================
# 4. Behavioral Protocols (the contracts the agent implements)
# =============================================================================


@runtime_checkable
class ForwardModel(Protocol):
    """Computes a calculated pattern for one histogram given the current state.

    MUST be a pure function of (state, histogram). MUST be differentiable in a
    production implementation so that `jacobian` returns AD (not finite-difference)
    derivatives. The reference numpy oracle may use finite differences only inside
    tests that *check* a production AD Jacobian.
    """

    def calculate(self, state: RefinementState, histogram: Histogram) -> Array:
        """Return y_calc on the histogram's abscissa."""
        ...

    def jacobian(self, state: RefinementState, histogram: Histogram) -> Array:
        """Return d y_calc / d theta for all VARIED parameters.

        Shape (len(histogram.x), state.n_varied), columns ordered as the varied
        parameters appear in state.parameters.
        """
        ...


@runtime_checkable
class ParametricEngine(Protocol):
    """Resolves parametric models into per-histogram effective parameters.

    This is the heart of the design: it expands a single evolving model across the
    whole ensemble so the minimizer sees shared coefficients, not duplicated
    per-histogram parameters.
    """

    def expand(self, state: RefinementState, histogram: Histogram) -> RefinementState:
        """Return a per-histogram state with parametric targets resolved to values."""
        ...


@runtime_checkable
class Minimizer(Protocol):
    """Trust-region / Levenberg–Marquardt least-squares solver.

    Operates on the stacked weighted residual across ALL histograms simultaneously
    (the parametric/surface objective), using AD Jacobians on GPU in production.
    """

    def refine(
        self,
        state: RefinementState,
        forward: ForwardModel,
        parametric: ParametricEngine,
        max_iter: int = 100,
        tol: float = 1e-8,
        seed: int | None = None,
    ) -> RefinementResult:
        """Refine *state*. ``seed`` is recorded in the result's provenance so a
        run is reproducible; an implementation that uses no randomness still
        records it.
        """
        ...


@runtime_checkable
class DomainModule(Protocol):
    """A pluggable physics contribution (texture, lattice strain, size-strain, ...).

    Each module modifies the calculated intensities/positions for a (phase, histogram)
    through the same differentiable interface, so the engine never special-cases them.
    """

    name: str

    def contribute(
        self, state: RefinementState, histogram: Histogram, y_calc: Array
    ) -> Array:
        """Return the modified y_calc."""
        ...


@runtime_checkable
class PhaseIdentifier(Protocol):
    """AI triage service. PROPOSES candidates only; never returns final parameters."""

    def propose(
        self, histogram: Histogram, chemistry: Sequence[str] | None = None
    ) -> Sequence[tuple[Phase, float]]:
        """Return ranked (candidate phase, confidence) pairs for engine validation."""
        ...


# =============================================================================
# 5. Acceptance criteria as machine-checkable predicates
#    (referenced by the invariant test suite; see test_invariants.py)
# =============================================================================

ACCEPTANCE = {
    "jacobian_rtol": 1e-4,  # AD vs finite-difference Jacobian agreement
    "fixpoint_step_atol": 1e-6,  # refine-from-truth must not move (noise-free)
    "recovery_n_sigma": 3.0,  # synthetic recovery within 3 sigma of truth
    "gof_low": 0.8,
    "gof_high": 1.5,  # reduced chi^2 band for correct model+noise
    "phase_fraction_sum_atol": 1e-6,
    "metric_min_eig": 1e-9,  # metric tensor positive-definite guard
    "parametric_equiv_rtol": 1e-5,  # degenerate-parametric == independent refinement
    "cw_tof_edd_position_rtol": 1e-9,  # position-model round-trip accuracy
    "seed_rmse_refinable_max": 0.3,  # PXRDGen: refine should flag seeds worse than this
}


# =============================================================================
# 6. Self-consistency check (makes this file "executable")
# =============================================================================


def _self_check() -> None:
    # 6a. Metric tensor is SPD for a physical cell, rejected for an impossible one.
    good = UnitCell(4.05, 4.05, 4.05)  # cubic Al-like
    assert good.is_physical()
    bad = UnitCell(1.0, 1.0, 1.0, alpha=170, beta=170, gamma=170)  # non-SPD metric
    assert not bad.is_physical(), "Impossible cell must be rejected by metric SPD test."

    # 6b. Symmetry-equivalent reflections in a cubic cell share |F| and d.
    si = [AtomSite("Si", 0, 0, 0), AtomSite("Si", 0.5, 0.5, 0.5)]
    cell = UnitCell(5.431, 5.431, 5.431)
    f = {"Si": 14.0}
    d1, d2 = d_spacing((2, 0, 0), cell), d_spacing((0, 2, 0), cell)
    assert math.isclose(d1, d2, rel_tol=1e-12), "Cubic permutation must preserve d."
    F1 = abs(structure_factor((2, 0, 0), si, cell, f))
    F2 = abs(structure_factor((0, 2, 0), si, cell, f))
    assert math.isclose(F1, F2, rel_tol=1e-9), "Cubic permutation must preserve |F|."

    # 6c. Position models round-trip: d -> abscissa -> back is internally consistent.
    d = 2.0
    tt = two_theta_from_d(d, 1.5406)
    d_back = 1.5406 / (2.0 * math.sin(math.radians(tt) / 2.0))
    assert math.isclose(d, d_back, rel_tol=1e-12)
    tof = tof_from_d(d, difc=5000.0, difa=1.0, zero=-3.0)
    assert tof > 0
    e = energy_from_d(d, two_theta_deg=15.0)
    assert e > 0

    # 6d. Pseudo-Voigt is positive, peaks at center, ~normalized.
    grid = np.linspace(-10, 10, 20001)
    pv = pseudo_voigt(grid, center=0.0, fwhm=1.0, eta=0.5)
    assert np.all(pv >= 0)
    assert abs(grid[int(np.argmax(pv))]) < 1e-2
    area = float(np.trapezoid(pv, grid))
    assert 0.98 < area < 1.02, f"Pseudo-Voigt area {area} not ~1."

    # 6e. R-factors / GoF behave: identical patterns => Rwp 0; correct-noise GoF ~ 1.
    rng = np.random.default_rng(0)
    truth = 100.0 + 50.0 * np.exp(-0.5 * ((grid - 1.0) / 0.3) ** 2)
    w = 1.0 / np.clip(truth, 1.0, None)
    assert math.isclose(rwp(truth, truth, w), 0.0, abs_tol=1e-12)
    noisy = truth + rng.normal(0, np.sqrt(1.0 / w))
    gof = reduced_chi_square(noisy, truth, w, n_params=3)
    assert 0.8 < gof < 1.2, f"GoF {gof} outside expected band for correct model+noise."

    # 6f. State model: immutability and provenance logging.
    state = RefinementState(
        phases=(Phase("Si", "Fd-3m", cell, tuple(si)),), histograms=(), parameters=()
    )
    logged = state.with_log("created")
    assert logged.provenance == ("created",)
    assert state.provenance == (), "Original state must be unchanged (immutable)."

    print("spec self-check: PASS — contracts are internally consistent.")


if __name__ == "__main__":
    _self_check()
