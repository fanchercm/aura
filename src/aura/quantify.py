"""
Quantitative phase analysis: weight fractions from refined scales.

The Rietveld weight fraction of phase p is ``W_p ∝ s_p · (Z M V)_p`` — the
refined scale times the cell's mass-times-volume — normalized so the fractions
sum to one. The **closure** property (Σ W_p = 1, all positive) is the invariant
the oracle guards against (the GSAS-II phase-fraction non-closure bug); it holds
for any positive Z·M·V weights.

Note: ``cell_zmv`` uses the asymmetric-unit mass × cell volume as the Z·M·V
proxy. Exact site multiplicities (full-cell stoichiometry) are deferred; they
rescale every phase's weight by a constant and so do not affect closure, only the
absolute fractions.
"""

from __future__ import annotations

import gemmi
import numpy as np

from aura.spec import Phase


def cell_zmv(phase: Phase) -> float:
    """Z·M·V proxy: cell volume (Å³) × asymmetric-unit mass (amu)."""
    volume = gemmi.UnitCell(*phase.cell.as_tuple()).volume
    mass = sum(at.occ * gemmi.Element(at.element).weight for at in phase.atoms)
    return float(volume * mass)


def phase_weight_fractions(phases: tuple[Phase, ...], scales: np.ndarray) -> np.ndarray:
    """Weight fractions for *phases* given their refined *scales*.

    Returns fractions that sum to 1 (closure). Raises if the total weight is
    non-positive (all-zero scales / degenerate cells).
    """
    scales = np.asarray(scales, dtype=float)
    if scales.shape[0] != len(phases):
        raise ValueError("scales and phases length mismatch")
    weights = scales * np.array([cell_zmv(p) for p in phases])
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Non-positive total phase weight; cannot form fractions.")
    return weights / total
