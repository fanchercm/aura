"""
2D detector image ingest + geometry (the partially-integrated data pipeline).

The scientific motivation for the whole project is retaining angular resolution
(SCI-1/SCI-3): a measurement state is ~50–100 directionally-resolved 1D patterns,
which come from azimuthally integrating a 2D detector image in sectors rather
than collapsing the full ring. This subpackage provides:

* :class:`~aura.io.image.geometry.DetectorGeometry` — the flat-detector PONI
  geometry mapping each pixel to (2θ, azimuth, d).
* image readers (native ``.npy`` + vendor formats via fabio) registered in the
  importer registry under ``domain="image"``.

Azimuthal integration to angular slices lives in :mod:`aura.integrate`.
"""

from __future__ import annotations

from aura.io.image.geometry import DetectorGeometry
from aura.io.image.readers import Image

__all__ = ["DetectorGeometry", "Image"]
