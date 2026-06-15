"""
Peak-position laws and observable d-range, dispatched by data type.

``position(d, hist)`` maps a d-spacing to the histogram's abscissa using the
spec kernels (CW Bragg, TOF ``DIFC·d+DIFA·d²+ZERO``, EDD ``E=hc/(2d sinθ)``).
``d_range_for_histogram(hist)`` inverts that over the measured x-range so the
reflection generator only enumerates observable reflections.
"""

from __future__ import annotations

import math

import numpy as np

from aura import spec
from aura.spec import DataType, Histogram

_HC = 12.398419  # keV * Angstrom


def position(d: float, hist: Histogram) -> float:
    """Abscissa value (2θ deg | TOF µs | E keV) of a reflection at d-spacing *d*."""
    dt = hist.data_type
    if dt in (DataType.CW_XRAY, DataType.CW_NEUTRON):
        return spec.two_theta_from_d(d, hist.wavelength)
    if dt is DataType.TOF:
        return spec.tof_from_d(d, hist.difc, hist.difa, hist.zero)
    if dt is DataType.EDD:
        return spec.energy_from_d(d, hist.two_theta_fixed)
    raise ValueError(f"Unsupported data type: {dt}")


def _d_from_abscissa(x: float, hist: Histogram) -> float:
    """Inverse of :func:`position` (linearized for TOF: ignores DIFA)."""
    dt = hist.data_type
    if dt in (DataType.CW_XRAY, DataType.CW_NEUTRON):
        theta = math.radians(x) / 2.0
        s = math.sin(theta)
        if s <= 0:
            return math.inf
        return hist.wavelength / (2.0 * s)
    if dt is DataType.TOF:
        difc = hist.difc or 1.0
        return max((x - hist.zero) / difc, 0.0)
    if dt is DataType.EDD:
        theta = math.radians(hist.two_theta_fixed) / 2.0
        if x <= 0:
            return math.inf
        return _HC / (2.0 * x * math.sin(theta))
    raise ValueError(f"Unsupported data type: {dt}")


def d_range_for_histogram(hist: Histogram, pad: float = 0.0) -> tuple[float, float]:
    """Observable ``(d_min, d_max)`` for *hist* from its x-range.

    A reflection outside this range cannot contribute a peak to the pattern.
    *pad* (fractional) widens the range so peaks just off the edge whose tails
    intrude are still generated.

    Raises:
        ValueError: If the histogram has no positive x samples.
    """
    x = np.asarray(hist.x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError(f"Histogram {hist.id!r} has no finite x samples.")
    d_lo = _d_from_abscissa(float(np.max(x)), hist)
    d_hi = _d_from_abscissa(float(np.min(x)), hist)
    d_min, d_max = sorted((d_lo, d_hi))
    if not math.isfinite(d_max):
        # Low-angle edge → enormous d; cap so enumeration stays bounded.
        d_max = d_min * 50.0
    # Pad each bound relative to itself, not to the (possibly huge) span: a
    # near-beam-center bin makes d_max enormous, and a span-relative pad would
    # push d_min to its floor and explode reflection enumeration (n_max ∝ 1/d_min).
    return max(d_min * (1.0 - pad), 1e-3), d_max * (1.0 + pad)
