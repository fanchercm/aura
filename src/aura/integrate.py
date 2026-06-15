"""
Azimuthal integration: 2D detector image → 1D angular slices.

This is the bridge from a raw detector frame to the *partially-integrated* data
the engine consumes. Two modes:

* :func:`integrate_full` — collapse the whole Debye ring into one 1D pattern
  (the conventional reduction; the degrade-to-1D path, SCI-4).
* :func:`integrate_sectors` — split the azimuth into N sectors and integrate each
  separately, yielding N directionally-resolved 1D patterns as a
  ``MeasurementState`` (SCI-1/SCI-3 — the 50–100 linked patterns per state).

Integration is a weighted histogram of pixel intensities over 2θ bins (restricted
to an azimuth range for sectors). Each bin's value is the **summed** intensity, so
the sectors *partition* the pixels and recombining them (summing bin values and
counts) reproduces the full integration exactly — the SCI-4 consistency the tests
check. A windowed numpy implementation; pyFAI can be cross-checked but is not
required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aura.io.image.geometry import DetectorGeometry
from aura.io.image.readers import Image
from aura.models import DiffractionSlice, MeasurementState
from aura.spec import DataType


@dataclass(frozen=True)
class IntegrationResult:
    """Raw integrated bin sums + counts (so sectors recombine exactly)."""

    two_theta: np.ndarray  # bin centers (deg)
    intensity_sum: np.ndarray  # summed pixel intensity per bin
    counts: np.ndarray  # pixel count per bin

    @property
    def mean_intensity(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.intensity_sum / self.counts, 0.0)


def _integrate(
    image: Image, geometry: DetectorGeometry, tth_bins: np.ndarray, azimuth_mask=None
) -> IntegrationResult:
    tth = geometry.two_theta_array().ravel()
    inten = np.asarray(image.data, dtype=float).ravel()
    if azimuth_mask is not None:
        m = azimuth_mask.ravel()
        tth, inten = tth[m], inten[m]
    intensity_sum, _ = np.histogram(tth, bins=tth_bins, weights=inten)
    counts, _ = np.histogram(tth, bins=tth_bins)
    centers = 0.5 * (tth_bins[:-1] + tth_bins[1:])
    return IntegrationResult(centers, intensity_sum, counts.astype(float))


def _tth_bins(geometry: DetectorGeometry, n_bins: int, tth_range=None) -> np.ndarray:
    if tth_range is None:
        tth = geometry.two_theta_array()
        tth_range = (float(tth.min()), float(tth.max()))
    return np.linspace(tth_range[0], tth_range[1], n_bins + 1)


def integrate_full(
    image: Image, geometry: DetectorGeometry, n_bins: int = 1000, tth_range=None
) -> IntegrationResult:
    """Integrate the full azimuth into one 1D pattern (degrade-to-1D, SCI-4)."""
    return _integrate(image, geometry, _tth_bins(geometry, n_bins, tth_range))


def integrate_sectors(
    image: Image,
    geometry: DetectorGeometry,
    n_sectors: int = 8,
    n_bins: int = 1000,
    tth_range=None,
    state_id: str = "image",
) -> MeasurementState:
    """Integrate into *n_sectors* azimuthal sectors → a MeasurementState of N
    directionally-resolved 1D patterns (one DiffractionSlice per sector)."""
    if n_sectors < 1:
        raise ValueError("n_sectors must be >= 1")
    bins = _tth_bins(geometry, n_bins, tth_range)
    az = geometry.azimuth_array()
    edges = np.linspace(0.0, 360.0, n_sectors + 1)
    slices = []
    for s in range(n_sectors):
        lo, hi = edges[s], edges[s + 1]
        mask = (az >= lo) & (az < hi)
        res = _integrate(image, geometry, bins, azimuth_mask=mask)
        y = res.intensity_sum
        e = np.sqrt(np.clip(y, 1.0, None))
        slices.append(
            DiffractionSlice(
                id=f"{state_id}_sector{s}",
                x=res.two_theta,
                y=y,
                e=e,
                metadata={
                    # Area detectors here are X-ray; integrated slices are CW X-ray.
                    "data_type": DataType.CW_XRAY,
                    "wavelength": geometry.wavelength,
                    "azimuth_min": float(lo),
                    "azimuth_max": float(hi),
                    "counts": res.counts,
                },
            )
        )
    return MeasurementState(
        id=state_id,
        slices=slices,
        metadata={"n_sectors": n_sectors, "source": image.source},
    )
