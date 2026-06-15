"""
2D detector-image readers (domain="image").

* :class:`NpyImageReader` — native NumPy ``.npy`` frames (dependency-free; the
  format the synthetic-image tests use).
* :class:`FabioImageReader` — vendor formats (TIFF, CBF, EDF, GE, Mar, …) via
  fabio, the pyFAI-ecosystem reader. Used when fabio is installed.

Both return an :class:`Image` (data + header). Detector geometry is separate
(:class:`aura.io.image.geometry.DetectorGeometry`) — an image is just counts; the
geometry that turns pixels into angles is supplied at integration time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aura.io.registry import registry

_FABIO_EXT = (
    ".tif",
    ".tiff",
    ".cbf",
    ".edf",
    ".ge",
    ".ge2",
    ".ge3",
    ".mar3450",
    ".img",
)


@dataclass(frozen=True)
class Image:
    """One 2D detector frame."""

    data: np.ndarray  # 2D intensity array
    header: dict[str, Any] = field(default_factory=dict)
    source: str = ""


class NpyImageReader:
    name = "NumPy image"
    domain = "image"
    extensions = (".npy",)

    def contents_validator(self, path: Path) -> bool:
        with path.open("rb") as f:
            return f.read(6) == b"\x93NUMPY"

    def read(self, path: Path) -> Image:
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path.name}: expected a 2D image, got shape {arr.shape}")
        return Image(data=np.asarray(arr, dtype=float), source=path.name)


class FabioImageReader:
    name = "fabio image"
    domain = "image"
    extensions = _FABIO_EXT

    def contents_validator(self, path: Path) -> bool:
        if path.suffix.lower() not in self.extensions:
            return False
        try:
            import fabio  # noqa: F401
        except ImportError:
            return False
        return True

    def read(self, path: Path) -> Image:
        try:
            import fabio
        except ImportError as exc:  # pragma: no cover
            raise NotImplementedError(
                f"Reading {path.suffix} needs fabio (pip install fabio)."
            ) from exc
        frame = fabio.open(str(path))
        return Image(
            data=np.asarray(frame.data, dtype=float),
            header=dict(frame.header),
            source=path.name,
        )


registry.register(NpyImageReader())
registry.register(FabioImageReader())
