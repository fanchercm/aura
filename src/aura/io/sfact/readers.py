"""
Single-crystal Fobs readers: SHELX ``.hkl`` (HKLF4) and CIF ``_refln`` loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aura.io.registry import registry


@dataclass(frozen=True)
class StructureFactors:
    """A measured single-crystal reflection table."""

    hkl: np.ndarray  # (N, 3) int
    f_squared: np.ndarray  # (N,) measured |F|²
    sigma: np.ndarray  # (N,) esd on |F|²

    def __post_init__(self) -> None:
        n = self.hkl.shape[0]
        if self.hkl.shape != (n, 3):
            raise ValueError("hkl must be (N, 3)")
        if self.f_squared.shape != (n,) or self.sigma.shape != (n,):
            raise ValueError("f_squared and sigma must be length N")
        if not (
            np.all(np.isfinite(self.f_squared)) and np.all(np.isfinite(self.sigma))
        ):
            raise ValueError("Non-finite F² or sigma in reflection table.")

    @property
    def n_reflections(self) -> int:
        return self.hkl.shape[0]


class ShelxHklReader:
    name = "SHELX hkl"
    domain = "sfact"
    extensions = (".hkl",)

    def contents_validator(self, path: Path) -> bool:
        for line in path.read_text(errors="ignore").splitlines():
            if line.strip():
                return _parse_hkl_line(line) is not None
        return False

    def read(self, path: Path) -> StructureFactors:
        hkls, fsq, sig = [], [], []
        for line in path.read_text(errors="ignore").splitlines():
            parsed = _parse_hkl_line(line)
            if parsed is None:
                continue
            h, k, l, f, s = parsed  # noqa: E741 (Miller index)
            if (h, k, l) == (0, 0, 0):
                break  # HKLF4 terminator
            hkls.append((h, k, l))
            fsq.append(f)
            sig.append(s)
        if not hkls:
            raise ValueError(f"No reflections parsed from {path}")
        return StructureFactors(np.array(hkls, dtype=int), np.array(fsq), np.array(sig))


class CifReflnReader:
    name = "CIF refln"
    domain = "sfact"
    extensions = (".cif", ".fcf")

    def contents_validator(self, path: Path) -> bool:
        head = path.read_text(errors="ignore")[:8192]
        return "_refln_index_h" in head or "_refln.index_h" in head

    def read(self, path: Path) -> StructureFactors:
        import gemmi

        block = gemmi.cif.read(str(path)).sole_block()
        h = _cif_ints(block, "_refln_index_h", "_refln.index_h")
        k = _cif_ints(block, "_refln_index_k", "_refln.index_k")
        l = _cif_ints(block, "_refln_index_l", "_refln.index_l")  # noqa: E741
        fsq, sig = _cif_intensities(block)
        hkl = np.stack([h, k, l], axis=1)
        return StructureFactors(hkl, fsq, sig)


def _parse_hkl_line(line: str):
    """SHELX HKLF4: h k l F² σ in (3I4,2F8.2) — parsed leniently by whitespace."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h, k, l = (int(parts[i]) for i in range(3))  # noqa: E741
        f = float(parts[3])
        s = float(parts[4])
    except ValueError:
        return None
    return h, k, l, f, s


def _cif_ints(block, *tags) -> np.ndarray:
    import gemmi

    for tag in tags:
        col = block.find_loop(tag)
        if col:
            return np.array([int(gemmi.cif.as_int(v)) for v in col])
    raise ValueError(f"CIF reflection loop missing any of {tags}")


def _cif_intensities(block):
    import gemmi

    candidates = [
        ("_refln_F_squared_meas", "_refln_F_squared_sigma"),
        ("_refln.F_squared_meas", "_refln.F_squared_sigma"),
        ("_refln_intensity_meas", "_refln_intensity_sigma"),
    ]
    for meas_tag, sig_tag in candidates:
        meas = block.find_loop(meas_tag)
        if meas:
            fsq = np.array([gemmi.cif.as_number(v) for v in meas])
            sig_col = block.find_loop(sig_tag)
            sig = (
                np.array([gemmi.cif.as_number(v) for v in sig_col])
                if sig_col
                else np.sqrt(np.clip(fsq, 0, None))
            )
            return fsq, sig
    raise ValueError("CIF has no recognized _refln intensity column.")


registry.register(ShelxHklReader())
registry.register(CifReflnReader())
