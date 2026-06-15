"""
Flat-detector geometry: pixel ↔ (2θ, azimuth, d).

A pyFAI-style PONI parameterization for a planar detector normal to the beam:
the sample sits ``distance`` metres from the detector, and the beam pierces it at
the point of normal incidence (``poni1``, ``poni2``) in metres along the slow
(dim-1, rows) and fast (dim-2, columns) axes. Pixel centres are at
``(index + 0.5) * pixel_size``. Detector tilt/rotation is omitted (a refinement);
forward and inverse maps are mutually consistent, so geometry round-trips.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DetectorGeometry:
    distance: float  # sample-detector distance (m)
    poni1: float  # point of normal incidence, slow axis / rows (m)
    poni2: float  # point of normal incidence, fast axis / cols (m)
    pixel1: float  # pixel size, slow axis (m)
    pixel2: float  # pixel size, fast axis (m)
    wavelength: float  # Angstrom
    shape: tuple[int, int]  # (n_rows, n_cols)

    # ---- per-pixel arrays ---------------------------------------------------
    def _radial_components(self):
        n1, n2 = self.shape
        r1 = (np.arange(n1) + 0.5) * self.pixel1 - self.poni1
        r2 = (np.arange(n2) + 0.5) * self.pixel2 - self.poni2
        x1, x2 = np.meshgrid(r1, r2, indexing="ij")
        return x1, x2

    def two_theta_array(self) -> np.ndarray:
        """2θ (degrees) for every pixel."""
        x1, x2 = self._radial_components()
        rr = np.hypot(x1, x2)
        return np.degrees(np.arctan2(rr, self.distance))

    def azimuth_array(self) -> np.ndarray:
        """Azimuth (degrees, [0, 360)) for every pixel."""
        x1, x2 = self._radial_components()
        az = np.degrees(np.arctan2(x1, x2))
        return np.mod(az, 360.0)

    def d_array(self) -> np.ndarray:
        """d-spacing (Å) for every pixel (∞ at the beam centre)."""
        tt = np.radians(self.two_theta_array())
        s = np.sin(tt / 2.0)
        with np.errstate(divide="ignore"):
            return self.wavelength / (2.0 * s)

    # ---- single-pixel round trip --------------------------------------------
    def pixel_to_angles(self, r1: float, r2: float) -> tuple[float, float]:
        """(row, col) → (2θ deg, azimuth deg)."""
        x1 = (r1 + 0.5) * self.pixel1 - self.poni1
        x2 = (r2 + 0.5) * self.pixel2 - self.poni2
        rr = math.hypot(x1, x2)
        tt = math.degrees(math.atan2(rr, self.distance))
        az = math.degrees(math.atan2(x1, x2)) % 360.0
        return tt, az

    def angles_to_pixel(
        self, two_theta_deg: float, azimuth_deg: float
    ) -> tuple[float, float]:
        """(2θ deg, azimuth deg) → (row, col) (inverse of :meth:`pixel_to_angles`)."""
        rr = self.distance * math.tan(math.radians(two_theta_deg))
        az = math.radians(azimuth_deg)
        x1 = rr * math.sin(az)
        x2 = rr * math.cos(az)
        r1 = (x1 + self.poni1) / self.pixel1 - 0.5
        r2 = (x2 + self.poni2) / self.pixel2 - 0.5
        return r1, r2

    # ---- (de)serialization --------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        data = asdict(self)
        data["shape"] = list(self.shape)
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> DetectorGeometry:
        data = json.loads(Path(path).read_text())
        data["shape"] = tuple(data["shape"])
        return cls(**data)
