"""
Phase-identification triage (propose-only).

Given an observed pattern and a candidate-phase library, :class:`PhaseIdentifier`
proposes a confidence-ranked list of candidate phases — and nothing else. It does
**not** refine: ranking is a fast, forward-only peak-position match score. The
returned candidates are seeds the deterministic engine then validates by
refinement; the acceptance signal is the engine's Rwp/GoF, never the proposal
confidence. This mirrors the "AI proposes, engine disposes" layering of CPICANN
and Dara.

The scorer here is a transparent heuristic (a stand-in for a trained model such
as CPICANN): it predicts each candidate's strong peak positions and measures how
many coincide with observed peaks. The *contract* — propose-only, ranked, bounded
confidence, provenance, hallucination containment — is what matters and is what a
learned model would also satisfy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aura.engine import positions, scattering, symmetry
from aura.engine.forward import _lorentz_polarization
from aura.spec import DataType, Histogram, Phase

_CW = (DataType.CW_XRAY, DataType.CW_NEUTRON)


@dataclass(frozen=True)
class AIProposal:
    """An AI recommendation with provenance (oracle §11.1).

    Carries the candidate, its confidence, and the metadata required to audit an
    AI-sourced recommendation. ``human_acceptance`` starts None — a human (or the
    engine's validation) decides acceptance; the AI never self-accepts.
    """

    phase: Phase
    confidence: float
    source: str = "ai_initialization"
    model_version: str = "heuristic-peakmatch-1"
    human_acceptance: str | None = None


class PhaseIdentifier:
    """Propose-only candidate-phase triage from a fixed library."""

    model_version = "heuristic-peakmatch-1"

    def __init__(self, library: Sequence[Phase], peak_tol_channels: int = 3) -> None:
        self.library = tuple(library)
        self.peak_tol_channels = peak_tol_channels

    # ---- the protocol method ------------------------------------------------
    def propose(
        self, histogram: Histogram, chemistry: Sequence[str] | None = None
    ) -> list[tuple[Phase, float]]:
        """Return candidates ranked by confidence (descending), confidence ∈ [0, 1].

        Never returns refined parameters. ``chemistry`` (if given) restricts to
        candidates whose elements are a subset of the allowed set.

        Raises:
            ValueError: On impossible input — negative counts, or CW data with no
                wavelength (hallucination containment: refuse, do not fabricate).
        """
        self._validate(histogram)
        allowed = set(chemistry) if chemistry is not None else None
        obs = self._observed_peaks(histogram)
        scored: list[tuple[Phase, float]] = []
        for phase in self.library:
            if allowed is not None and not self._elements(phase) <= allowed:
                continue
            scored.append((phase, self._match_score(phase, histogram, obs)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def propose_with_provenance(
        self, histogram: Histogram, chemistry: Sequence[str] | None = None
    ) -> list[AIProposal]:
        """Same ranking, wrapped as :class:`AIProposal`s carrying provenance."""
        return [
            AIProposal(phase=ph, confidence=conf, model_version=self.model_version)
            for ph, conf in self.propose(histogram, chemistry)
        ]

    # ---- internals ----------------------------------------------------------
    @staticmethod
    def _validate(histogram: Histogram) -> None:
        if np.any(np.asarray(histogram.y_obs) < 0):
            raise ValueError(
                "Negative counts in observed data; cannot identify phases "
                "(check data reduction). No candidate will be fabricated."
            )
        if histogram.data_type in _CW and not histogram.wavelength:
            raise ValueError(
                "CW data has no wavelength; supply one before phase identification."
            )

    @staticmethod
    def _elements(phase: Phase) -> set[str]:
        return {at.element for at in phase.atoms}

    def _observed_peaks(self, histogram: Histogram) -> np.ndarray:
        from scipy.signal import find_peaks

        y = np.asarray(histogram.y_obs, dtype=float)
        if y.size == 0:
            return np.empty(0)
        thresh = float(np.median(y) + 1.0 * np.std(y))
        idx, _ = find_peaks(y, height=thresh, distance=5)
        return np.asarray(histogram.x, dtype=float)[idx]

    def _match_score(
        self, phase: Phase, histogram: Histogram, obs_peaks: np.ndarray, top_k: int = 12
    ) -> float:
        """Fraction of the candidate's strongest predicted peaks that coincide
        with an observed peak — a confidence in [0, 1]."""
        if obs_peaks.size == 0:
            return 0.0
        try:
            d_min, d_max = positions.d_range_for_histogram(histogram, pad=0.0)
            refl = symmetry.generate_reflections(
                phase, round(d_min, 4), round(d_max, 4)
            )
        except (ValueError, RuntimeError):
            return 0.0
        if not refl:
            return 0.0
        # Rank predicted reflections by an intensity proxy and keep the strongest.
        scored = []
        for r in refl:
            try:
                pos = positions.position(r.d, histogram)
            except ValueError:
                continue
            fsq = scattering.f_squared(r.hkl, r.d, phase, histogram.data_type)
            lp = _lorentz_polarization(r.d, pos, histogram)
            scored.append((pos, r.multiplicity * fsq * lp))
        if not scored:
            return 0.0
        scored.sort(key=lambda t: t[1], reverse=True)
        predicted = np.array([p for p, _ in scored[:top_k]])

        x = np.asarray(histogram.x, dtype=float)
        tol = self.peak_tol_channels * float(np.median(np.abs(np.diff(x))))
        matched = sum(1 for p in predicted if np.any(np.abs(obs_peaks - p) <= tol))
        return matched / len(predicted)
