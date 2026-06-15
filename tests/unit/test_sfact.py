"""
Single-crystal Fobs readers: SHELX .hkl and CIF _refln → validated tables.
"""

from __future__ import annotations

import numpy as np
import pytest

import aura.io as io
from aura.io.sfact.readers import StructureFactors

_HKL = """\
   1   1   1   123.45     2.10
   2   0   0   456.78     3.40
   2   2   0    98.10     1.90
  -1   1   1   120.00     2.00
   0   0   0     0.00     0.00
   3   1   1    55.20     1.50
"""

_CIF_REFLN = """\
data_xtal
loop_
_refln_index_h
_refln_index_k
_refln_index_l
_refln_F_squared_meas
_refln_F_squared_sigma
1 1 1 123.45 2.10
2 0 0 456.78 3.40
2 2 0  98.10 1.90
"""


def test_shelx_hkl_reads_and_stops_at_terminator(tmp_path):
    p = tmp_path / "xtal.hkl"
    p.write_text(_HKL)
    sf = io.read(p, "sfact")
    assert isinstance(sf, StructureFactors)
    # Stops at the 0 0 0 terminator: 4 reflections before it, the 3 1 1 after is dropped.
    assert sf.n_reflections == 4
    assert list(sf.hkl[1]) == [2, 0, 0]
    assert abs(sf.f_squared[1] - 456.78) < 1e-6
    assert abs(sf.sigma[0] - 2.10) < 1e-6


def test_cif_refln_reads(tmp_path):
    p = tmp_path / "xtal.cif"
    p.write_text(_CIF_REFLN)
    sf = io.read(p, "sfact")
    assert sf.n_reflections == 3
    assert list(sf.hkl[0]) == [1, 1, 1]
    assert abs(sf.f_squared[2] - 98.10) < 1e-6


def test_sfact_readers_registered():
    names = {r.name for r in io.registry.readers("sfact")}
    assert {"SHELX hkl", "CIF refln"} <= names


def test_structure_factors_validation():
    with pytest.raises(ValueError, match="length N"):
        StructureFactors(np.zeros((3, 3), int), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        StructureFactors(np.zeros((1, 3), int), np.array([np.nan]), np.array([1.0]))


def test_cif_routes_to_phase_not_sfact_by_domain(tmp_path):
    """A structure CIF asked for as 'sfact' has no _refln loop → unsupported,
    confirming domain separation (the same .cif serves different domains)."""
    from aura.io.registry import UnsupportedFormatError

    p = tmp_path / "structure.cif"
    p.write_text("data_x\n_cell_length_a 5.0\n_atom_site_fract_x\n0.0\n")
    with pytest.raises((UnsupportedFormatError, ValueError)):
        io.read(p, "sfact")
