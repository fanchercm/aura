"""
Atomic scattering factors and structure factors (gemmi-backed).

Generalizes the reference oracle's scalar form factors to real, element- and
radiation-specific values:

* **X-ray / EDD** — Q-dependent form factor f(Q) from the International Tables
  IT92 (Cromer–Mann-style) coefficients, ``f(stol2)`` with
  ``stol2 = (sin θ / λ)² = 1/(4 d²)``.
* **Neutron (CW or TOF)** — the bound coherent scattering length b (fm),
  Q-independent; may be negative (e.g. H, Ti, Mn).

The structure factor keeps the isotropic-ADP form of :func:`aura.spec.structure_factor`:
``F(hkl) = Σ_j occ_j · f_j · exp(2πi (h·x+k·y+l·z)) · exp(-B_j · stol2)``.
"""

from __future__ import annotations

import functools
import math

import gemmi

from aura.spec import AtomSite, DataType, Phase

_NEUTRON_TYPES = (DataType.CW_NEUTRON, DataType.TOF)


@functools.lru_cache(maxsize=512)
def _xray_coefs(element: str):
    return gemmi.Element(element).it92


@functools.lru_cache(maxsize=512)
def _neutron_coefs(element: str):
    return gemmi.Element(element).neutron92


def form_factor(element: str, stol2: float, data_type: DataType) -> float:
    """Scattering factor for *element* at ``stol2 = (sinθ/λ)²``.

    X-ray returns the Q-dependent electron form factor; neutron returns the
    (constant) bound coherent scattering length.
    """
    if data_type in _NEUTRON_TYPES:
        return float(_neutron_coefs(element).calculate_sf(stol2))
    return float(_xray_coefs(element).calculate_sf(stol2))


def stol2_from_d(d: float) -> float:
    """``(sin θ / λ)² = 1/(4 d²)`` — the argument of every form factor / DW term."""
    return 1.0 / (4.0 * d * d)


def structure_factor(
    hkl: tuple[int, int, int],
    d: float,
    atoms: tuple[AtomSite, ...],
    data_type: DataType,
) -> complex:
    """Structure factor F(hkl) with isotropic ADPs and real scattering factors."""
    h, k, l = hkl  # noqa: E741 (Miller index)
    stol2 = stol2_from_d(d)
    f_real = 0.0
    f_imag = 0.0
    for at in atoms:
        f = form_factor(at.element, stol2, data_type)
        dw = math.exp(-at.b_iso * stol2)
        phase = 2.0 * math.pi * (h * at.x + k * at.y + l * at.z)
        amp = at.occ * f * dw
        f_real += amp * math.cos(phase)
        f_imag += amp * math.sin(phase)
    return complex(f_real, f_imag)


def f_squared(
    hkl: tuple[int, int, int],
    d: float,
    phase: Phase,
    data_type: DataType,
) -> float:
    """|F(hkl)|² for one phase — the intensity-bearing quantity."""
    fc = structure_factor(hkl, d, phase.atoms, data_type)
    return fc.real * fc.real + fc.imag * fc.imag
