"""
AI phase-identification triage: the propose-only contract (oracle category H),
hallucination containment (§11.3), and provenance (§11.1).

The heuristic identifier is a stand-in for a trained model; tests assert the
*contract* (which any model must satisfy) plus correct ranking on controlled
single-phase synthetic data. Real multi-phase / pressure-shifted data is
intentionally disambiguated by the engine, not the AI ("AI proposes, engine
disposes").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aura.ai.identify import AIProposal, PhaseIdentifier
from aura.engine.forward import ProductionEngine
from aura.spec import (
    AtomSite,
    DataType,
    Histogram,
    Parameter,
    ParamKind,
    Phase,
    RefinementResult,
    RefinementState,
    UnitCell,
)

DATA_DIR = Path(__file__).parent.parent / "testDataGsas"
ENGINE = ProductionEngine()

NABR = Phase(
    "NaBr",
    "F m -3 m",
    UnitCell(5.9738, 5.9738, 5.9738),
    (AtomSite("Na", 0, 0, 0), AtomSite("Br", 0.5, 0.5, 0.5)),
)
FAP = Phase(
    "FAP",
    "P 63/m",
    UnitCell(9.37, 9.37, 6.88, 90, 90, 120),
    (AtomSite("Ca", 0.25, 0.0, 0.25), AtomSite("P", 0.4, 0.37, 0.25)),
)
NAC = Phase(
    "NAC",
    "I 21 3",
    UnitCell(10.2512, 10.2512, 10.2512),
    (AtomSite("Ca", 0.4665, 0.0, 0.25),),
)


def _synthetic(phase, scale=500.0, bkg=20.0, fwhm=0.4):
    x = np.linspace(20.0, 120.0, 3000)
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
        Parameter(f"phase:{phase.name}:cell.a", ParamKind.PHASE, phase.cell.a),
        Parameter("hist:h:scale", ParamKind.HISTOGRAM, scale),
        Parameter("hist:h:bkg", ParamKind.HISTOGRAM, bkg),
        Parameter("hist:h:fwhm", ParamKind.HISTOGRAM, fwhm),
        Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5),
    )
    y = ENGINE.calculate(RefinementState((phase,), (blank,), truth), blank)
    return Histogram(
        "h",
        DataType.CW_NEUTRON,
        x,
        y,
        1.0 / np.clip(y, 1.0, None),
        {},
        wavelength=1.909,
    )


# --- Category H: propose-only contract -----------------------------------------


class TestProposeContract:

    def test_returns_only_ranked_bounded_candidates(self):
        ident = PhaseIdentifier([NABR, FAP, NAC])
        out = ident.propose(_synthetic(NABR))
        assert all(isinstance(ph, Phase) and 0.0 <= c <= 1.0 for ph, c in out)
        confs = [c for _, c in out]
        assert confs == sorted(confs, reverse=True)  # ranked descending
        # No refined parameters are returned — only (phase, confidence) pairs.
        assert all(len(item) == 2 for item in out)

    def test_ranks_correct_phase_first_on_clean_data(self):
        ident = PhaseIdentifier([NABR, FAP, NAC])
        ranked = ident.propose(_synthetic(NABR))
        assert ranked[0][0].name == "NaBr"
        assert ranked[0][1] > 0.0

    def test_candidate_must_round_trip_through_engine(self):
        """Acceptance is an ENGINE output (Rwp/GoF), never the AI confidence."""
        ident = PhaseIdentifier([NABR, FAP])
        h = _synthetic(NABR)
        best, _conf = ident.propose(h)[0]
        params = (
            Parameter(
                f"phase:{best.name}:cell.a",
                ParamKind.PHASE,
                best.cell.a,
                vary=True,
                lower=5.8,
                upper=6.1,
            ),
            Parameter(
                "hist:h:scale",
                ParamKind.HISTOGRAM,
                400.0,
                vary=True,
                lower=1e-3,
                upper=1e6,
            ),
            Parameter(
                "hist:h:bkg", ParamKind.HISTOGRAM, 15.0, vary=True, lower=0.0, upper=1e4
            ),
            Parameter(
                "hist:h:fwhm",
                ParamKind.HISTOGRAM,
                0.45,
                vary=True,
                lower=0.05,
                upper=2.0,
            ),
            Parameter("hist:h:eta", ParamKind.HISTOGRAM, 0.5, vary=False),
        )
        res = ENGINE.refine(
            RefinementState((best,), (h,), params), ENGINE, ENGINE, max_iter=60, seed=1
        )
        assert isinstance(res, RefinementResult)
        assert np.isfinite(res.rwp) and np.isfinite(res.reduced_chi2)
        # The correct candidate refines to a good fit on its own data.
        assert res.rwp < 0.05

    def test_chemistry_filter_restricts_candidates(self):
        ident = PhaseIdentifier([NABR, FAP, NAC])
        out = ident.propose(_synthetic(NABR), chemistry=["Na", "Br"])
        names = {ph.name for ph, _ in out}
        assert names == {"NaBr"}  # FAP (Ca,P) and NAC (Ca) filtered out


# --- §11.3 Hallucination containment -------------------------------------------


class TestHallucinationContainment:

    def test_negative_counts_rejected(self):
        ident = PhaseIdentifier([NABR])
        x = np.linspace(20.0, 120.0, 200)
        h = Histogram(
            "h",
            DataType.CW_NEUTRON,
            x,
            -np.ones_like(x),
            np.ones_like(x),
            {},
            wavelength=1.909,
        )
        with pytest.raises(ValueError, match="[Nn]egative"):
            ident.propose(h)

    def test_invalid_space_group_candidate_not_fabricated(self):
        """A candidate the engine can't enumerate (bad space group) scores 0 — it
        is not fabricated into a confident match. (CW-without-wavelength is caught
        upstream by Histogram construction, so it cannot reach the identifier.)"""
        bad = Phase(
            "Bogus", "not a space group", UnitCell(5, 5, 5), (AtomSite("Na", 0, 0, 0),)
        )
        ident = PhaseIdentifier([NABR, bad])
        out = {ph.name: c for ph, c in ident.propose(_synthetic(NABR))}
        assert out["Bogus"] == 0.0
        assert out["NaBr"] > 0.0


# --- §11.1 AI provenance -------------------------------------------------------


class TestAIProvenance:

    def test_proposals_carry_provenance(self):
        ident = PhaseIdentifier([NABR, FAP])
        proposals = ident.propose_with_provenance(_synthetic(NABR))
        assert proposals and all(isinstance(p, AIProposal) for p in proposals)
        top = proposals[0]
        assert top.source == "ai_initialization"
        assert top.model_version == ident.model_version
        assert top.human_acceptance is None  # AI never self-accepts
        assert 0.0 <= top.confidence <= 1.0
