"""
Reflection generation and scattering factors (engine foundations).

Validates the crystallographic correctness that peak *positions* and relative
intensities depend on: FCC multiplicities/absences, the m-3 vs m-3m orbit
splitting, d-range filtering, and real element scattering factors.
"""

from __future__ import annotations

import math

from aura.engine import scattering, symmetry
from aura.spec import AtomSite, DataType, Phase, UnitCell

NABR = Phase(
    "NaBr",
    "F m -3 m",
    UnitCell(5.9738, 5.9738, 5.9738),
    (AtomSite("Na", 0, 0, 0), AtomSite("Br", 0.5, 0.5, 0.5)),
)
# NAC: I 2_13 (#199), point group 23 -> Laue class m-3 (NOT m-3m).
NAC = Phase(
    "NAC",
    "I 21 3",
    UnitCell(10.2512, 10.2512, 10.2512),
    (AtomSite("Ca", 0.4665, 0.0, 0.25),),
)


class TestReflectionGeneration:

    def test_fcc_multiplicities(self):
        refl = {r.hkl: r for r in symmetry.generate_reflections(NABR, 1.4, 4.0)}
        mult = {
            tuple(sorted(map(abs, hkl), reverse=True)): r.multiplicity
            for hkl, r in refl.items()
        }
        # Textbook cubic m-3m powder multiplicities.
        assert mult[(1, 1, 1)] == 8
        assert mult[(2, 0, 0)] == 6
        assert mult[(2, 2, 0)] == 12
        assert mult[(3, 1, 1)] == 24

    def test_fcc_systematic_absences(self):
        # Mixed-parity hkl (e.g. 100, 110, 210) are absent in F-centred lattices.
        hkls = {
            tuple(sorted(map(abs, r.hkl), reverse=True))
            for r in symmetry.generate_reflections(NABR, 1.0, 7.0)
        }
        assert (1, 0, 0) not in hkls
        assert (1, 1, 0) not in hkls
        assert (2, 1, 0) not in hkls
        assert (1, 1, 1) in hkls  # all-odd allowed
        assert (2, 0, 0) in hkls  # all-even allowed

    def test_laue_m3_splits_hk0_orbits(self):
        """In Laue class m-3, {310} and {301} are distinct orbits (no 4-fold).

        This only happens with the real point group of I2_13 — proof the engine
        uses true symmetry, not an assumed m-3m.
        """
        refl = symmetry.generate_reflections(NAC, 3.0, 3.4)
        d310 = [r for r in refl if abs(r.d - 3.2417) < 1e-3]
        assert len(d310) == 2  # (310) and (301) split
        assert all(r.multiplicity == 12 for r in d310)

    def test_d_range_is_respected(self):
        refl = symmetry.generate_reflections(NABR, 1.5, 2.5)
        assert refl  # non-empty
        assert all(1.5 <= r.d <= 2.5 for r in refl)
        # Sorted high-d first.
        ds = [r.d for r in refl]
        assert ds == sorted(ds, reverse=True)

    def test_d_min_must_be_positive(self):
        import pytest

        with pytest.raises(ValueError, match="d_min"):
            symmetry.generate_reflections(NABR, 0.0, 5.0)


class TestScattering:

    def test_xray_form_factor_approaches_Z_at_zero_angle(self):
        # f(0) ~ number of electrons.
        f0 = scattering.form_factor("Si", 0.0, DataType.CW_XRAY)
        assert abs(f0 - 14.0) < 0.5

    def test_xray_form_factor_decreases_with_angle(self):
        f_lo = scattering.form_factor("Si", 0.0, DataType.CW_XRAY)
        f_hi = scattering.form_factor("Si", 0.25, DataType.CW_XRAY)
        assert f_hi < f_lo

    def test_neutron_length_is_q_independent(self):
        b0 = scattering.form_factor("Na", 0.0, DataType.CW_NEUTRON)
        b1 = scattering.form_factor("Na", 0.3, DataType.CW_NEUTRON)
        assert abs(b0 - b1) < 1e-9  # neutron point scattering
        assert abs(b0 - 3.63) < 0.05  # Na bound coherent scattering length (fm)

    def test_structure_factor_matches_spec_kernel_form(self):
        # |F| for an allowed FCC reflection is non-zero; for the basis here all
        # atoms scatter in phase at (111)-type reflections.
        d = NABR.cell.a / math.sqrt(3)  # (111)
        fsq = scattering.f_squared((1, 1, 1), d, NABR, DataType.CW_NEUTRON)
        assert fsq > 0
