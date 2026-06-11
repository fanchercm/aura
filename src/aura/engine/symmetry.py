"""
Reflection generation with real crystallographic symmetry (gemmi-backed).

Replaces the reference engine's brute ``0..4`` cubic loop: enumerates Miller
indices for a phase over an observable d-range, drops systematically absent
reflections, and groups symmetry-equivalents into one representative per orbit
carrying the powder **multiplicity** (orbit size, including Friedel pairs).

This is the foundation of correct peak *positions* for any space group — the
part of the physics that must be right before refinement is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np

from aura import spec
from aura.spec import Phase


@dataclass(frozen=True)
class Reflection:
    """One symmetry-unique reflection."""

    hkl: tuple[int, int, int]
    d: float  # d-spacing in Angstrom
    multiplicity: int


def _group_ops(phase: Phase) -> gemmi.GroupOps:
    sg = gemmi.SpaceGroup(phase.space_group)
    if sg is None:
        raise ValueError(f"Unrecognized space group: {phase.space_group!r}")
    return sg.operations()


def _rotations(ops: gemmi.GroupOps) -> list[np.ndarray]:
    """Integer reciprocal-space rotation matrices (h' = h @ R)."""
    mats = []
    for op in ops.sym_ops:
        mats.append(np.array(op.rot, dtype=float).reshape(3, 3) / gemmi.Op.DEN)
    return mats


def _orbit(
    hkl: tuple[int, int, int], rotations: list[np.ndarray]
) -> set[tuple[int, int, int]]:
    """All symmetry+Friedel equivalents of *hkl* (powder multiplicity = size)."""
    h = np.array(hkl, dtype=float)
    seen: set[tuple[int, int, int]] = set()
    for r in rotations:
        hp = tuple(int(round(v)) for v in (h @ r))
        seen.add(hp)
        seen.add((-hp[0], -hp[1], -hp[2]))
    return seen


def generate_reflections(phase: Phase, d_min: float, d_max: float) -> list[Reflection]:
    """Generate symmetry-unique reflections for *phase* with ``d_min <= d <= d_max``.

    The d-spacing of every index in the bounding box is computed at once via the
    reciprocal metric tensor (vectorized numpy); only the few that fall in range
    are passed to gemmi for the per-reflection absence test and orbit/multiplicity
    — so cost scales with the number of *observable* reflections, not the cube of
    the index range.

    Args:
        phase: The crystal phase (cell + Hermann–Mauguin space group).
        d_min: Smallest d-spacing to include (Angstrom). Sets the index range.
        d_max: Largest d-spacing to include.

    Returns:
        Reflections sorted by descending d (low-angle / high-d first), one per
        symmetry orbit, each with its powder multiplicity.

    Raises:
        ValueError: If ``d_min <= 0`` or the space group is unrecognized.
    """
    if d_min <= 0:
        raise ValueError(f"d_min must be positive, got {d_min}")

    ops = _group_ops(phase)
    rotations = _rotations(ops)

    a, b, c = phase.cell.a, phase.cell.b, phase.cell.c
    n_max = int(max(a, b, c) / d_min) + 1

    # Vectorized d-spacing over the whole index box via the reciprocal metric.
    gstar = spec.reciprocal_metric_tensor(*phase.cell.as_tuple())
    rng = np.arange(-n_max, n_max + 1)
    H, K, L = np.meshgrid(rng, rng, rng, indexing="ij")
    hkl_all = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=1).astype(np.int64)
    nonzero = np.any(hkl_all != 0, axis=1)
    hkl_all = hkl_all[nonzero]
    h = hkl_all.astype(np.float64)
    dstar2 = np.einsum("ij,jk,ik->i", h, gstar, h)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / np.sqrt(dstar2)
    keep = np.isfinite(d) & (d >= d_min) & (d <= d_max)
    survivors = hkl_all[keep]
    d_surv = d[keep]
    order = np.argsort(-d_surv)  # high-d first

    reflections: list[Reflection] = []
    seen_orbits: set[tuple[int, int, int]] = set()
    for idx in order:
        hkl = (int(survivors[idx, 0]), int(survivors[idx, 1]), int(survivors[idx, 2]))
        if hkl in seen_orbits:
            continue
        if ops.is_systematically_absent(hkl):
            seen_orbits.add(hkl)
            continue
        orbit = _orbit(hkl, rotations)
        seen_orbits.update(orbit)
        rep = max(orbit)
        reflections.append(
            Reflection(hkl=rep, d=float(d_surv[idx]), multiplicity=len(orbit))
        )
    return reflections
