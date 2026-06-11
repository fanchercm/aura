"""Tests for the parameter management system."""

from pathlib import Path

import numpy as np
import pytest

from aura.loaders import load_campaign_from_directory, load_instprm
from aura.models import ParameterScope
from aura.parameters import (
    Constraint,
    Parameter,
    ParameterSet,
    RefinementModel,
    equality_constraint,
    linear_constraint,
)

GLOBAL = ParameterScope.GLOBAL
STATE = ParameterScope.STATE
SLICE = ParameterScope.SLICE

DATA_DIR = Path(__file__).parent.parent / "testDataGsas"


# ================================================================== #
# Parameter
# ================================================================== #


class TestParameter:
    def test_creation(self):
        p = Parameter("NaBr:a", "a", 5.9738, GLOBAL)
        assert p.param_id == "NaBr:a"
        assert p.name == "a"
        assert p.value == pytest.approx(5.9738)
        assert p.scope is GLOBAL
        assert p.fixed is False
        assert p.bounds is None

    def test_initial_value_snapshot(self):
        p = Parameter("x", "x", 10.0, GLOBAL)
        assert p.initial_value == 10.0
        p.value = 20.0
        assert p.initial_value == 10.0  # unchanged

    def test_reset(self):
        p = Parameter("x", "x", 10.0, GLOBAL)
        p.value = 99.0
        p.reset()
        assert p.value == 10.0

    def test_fixed_flag(self):
        p = Parameter("x", "x", 1.0, GLOBAL, fixed=True)
        assert p.fixed is True

    def test_bounds_valid(self):
        p = Parameter("x", "x", 5.0, GLOBAL, bounds=(0.0, 10.0))
        assert p.bounds == (0.0, 10.0)

    def test_bounds_inverted_raises(self):
        with pytest.raises(ValueError, match="Lower bound"):
            Parameter("x", "x", 5.0, GLOBAL, bounds=(10.0, 0.0))

    def test_value_outside_bounds_raises(self):
        with pytest.raises(ValueError, match="outside bounds"):
            Parameter("x", "x", 20.0, GLOBAL, bounds=(0.0, 10.0))

    def test_repr_free(self):
        p = Parameter("NaBr:a", "a", 5.97, GLOBAL)
        r = repr(p)
        assert "NaBr:a" in r
        assert "free" in r

    def test_repr_fixed(self):
        p = Parameter("x", "x", 1.0, GLOBAL, fixed=True)
        assert "fixed" in repr(p)


# ================================================================== #
# ParameterSet
# ================================================================== #


class TestParameterSet:
    def _make_set(self) -> ParameterSet:
        ps = ParameterSet()
        ps.add(Parameter("a", "a", 5.97, GLOBAL))
        ps.add(Parameter("scale", "scale", 1.0, STATE))
        ps.add(Parameter("bkg_c0", "bkg_c0", 0.0, SLICE))
        ps.add(Parameter("bkg_c1", "bkg_c1", 0.0, SLICE))
        return ps

    def test_add_and_len(self):
        ps = self._make_set()
        assert len(ps) == 4

    def test_getitem(self):
        ps = self._make_set()
        assert ps["a"].value == pytest.approx(5.97)

    def test_getitem_missing_raises(self):
        ps = self._make_set()
        with pytest.raises(KeyError, match="no_such"):
            ps["no_such"]

    def test_contains(self):
        ps = self._make_set()
        assert "a" in ps
        assert "zzz" not in ps

    def test_duplicate_raises(self):
        ps = ParameterSet()
        ps.add(Parameter("a", "a", 1.0, GLOBAL))
        with pytest.raises(ValueError, match="Duplicate"):
            ps.add(Parameter("a", "a", 2.0, GLOBAL))

    def test_iter(self):
        ps = self._make_set()
        ids = [p.param_id for p in ps]
        assert ids == ["a", "scale", "bkg_c0", "bkg_c1"]

    def test_free_params_all_free(self):
        ps = self._make_set()
        assert len(ps.free_params) == 4

    def test_free_params_with_fixed(self):
        ps = self._make_set()
        ps.fix("a", "bkg_c1")
        assert len(ps.free_params) == 2
        free_ids = [p.param_id for p in ps.free_params]
        assert free_ids == ["scale", "bkg_c0"]

    def test_fix_and_free(self):
        ps = self._make_set()
        ps.fix("a")
        assert ps["a"].fixed is True
        ps.free("a")
        assert ps["a"].fixed is False

    def test_fix_all_and_free_all(self):
        ps = self._make_set()
        ps.fix_all()
        assert all(p.fixed for p in ps)
        ps.free_all()
        assert all(not p.fixed for p in ps)

    def test_reset_all(self):
        ps = self._make_set()
        ps["a"].value = 99.0
        ps["scale"].value = 42.0
        ps.reset_all()
        assert ps["a"].value == pytest.approx(5.97)
        assert ps["scale"].value == pytest.approx(1.0)

    def test_by_scope(self):
        ps = self._make_set()
        assert len(ps.by_scope(GLOBAL)) == 1
        assert len(ps.by_scope(STATE)) == 1
        assert len(ps.by_scope(SLICE)) == 2

    def test_by_name(self):
        ps = ParameterSet()
        ps.add(Parameter("bank1:sig1", "sig-1", 130.0, GLOBAL))
        ps.add(Parameter("bank2:sig1", "sig-1", 278.0, GLOBAL))
        ps.add(Parameter("bank1:sig0", "sig-0", 43.0, GLOBAL))
        result = ps.by_name("sig-1")
        assert len(result) == 2

    def test_free_values_getter(self):
        ps = self._make_set()
        ps.fix("bkg_c1")
        vals = ps.free_values
        assert len(vals) == 3
        np.testing.assert_allclose(vals, [5.97, 1.0, 0.0])

    def test_free_values_setter(self):
        ps = self._make_set()
        ps.fix("bkg_c1")
        ps.free_values = np.array([6.0, 2.0, 0.5])
        assert ps["a"].value == pytest.approx(6.0)
        assert ps["scale"].value == pytest.approx(2.0)
        assert ps["bkg_c0"].value == pytest.approx(0.5)
        assert ps["bkg_c1"].value == pytest.approx(0.0)  # unchanged

    def test_free_values_wrong_length_raises(self):
        ps = self._make_set()
        with pytest.raises(ValueError, match="Expected 4"):
            ps.free_values = np.array([1.0, 2.0])

    def test_all_ids(self):
        ps = self._make_set()
        assert ps.all_ids == ["a", "scale", "bkg_c0", "bkg_c1"]

    def test_summary(self):
        ps = self._make_set()
        ps.fix("a")
        s = ps.summary()
        assert "3 free" in s
        assert "1 fixed" in s


# ================================================================== #
# Constraints
# ================================================================== #


class TestConstraints:
    def _make_pair(self) -> ParameterSet:
        ps = ParameterSet()
        ps.add(Parameter("master", "scale", 2.0, GLOBAL))
        ps.add(Parameter("slave", "scale", 0.0, GLOBAL))
        return ps

    def test_equality_constraint(self):
        ps = self._make_pair()
        c = equality_constraint("slave", "master")
        val = c.evaluate(ps)
        assert val == pytest.approx(2.0)

    def test_linear_constraint(self):
        ps = self._make_pair()
        c = linear_constraint("slave", "master", factor=0.5, offset=1.0)
        val = c.evaluate(ps)
        assert val == pytest.approx(2.0)  # 0.5 * 2.0 + 1.0

    def test_custom_callable_constraint(self):
        ps = ParameterSet()
        ps.add(Parameter("x", "x", 3.0, GLOBAL))
        ps.add(Parameter("y", "y", 4.0, GLOBAL))
        ps.add(Parameter("r", "r", 0.0, GLOBAL))
        c = Constraint(
            dependent_id="r",
            independent_ids=["x", "y"],
            expression=lambda x, y: np.sqrt(x**2 + y**2),
            description="r = sqrt(x² + y²)",
        )
        assert c.evaluate(ps) == pytest.approx(5.0)

    def test_equality_description(self):
        c = equality_constraint("b", "a")
        assert "b = a" in c.description

    def test_linear_description(self):
        c = linear_constraint("b", "a", factor=2.0, offset=0.5)
        assert "2.0" in c.description
        assert "0.5" in c.description


# ================================================================== #
# RefinementModel
# ================================================================== #


class TestRefinementModel:
    def _make_model(self) -> RefinementModel:
        """Minimal model with a small synthetic campaign."""
        from aura.models import Campaign, DiffractionSlice, MeasurementState

        slices = [
            DiffractionSlice(
                id=f"s1_bank{i}",
                x=np.linspace(3000, 14000, 50),
                y=np.ones(50),
                e=np.ones(50),
            )
            for i in range(1, 4)
        ]
        state = MeasurementState(id="s1", slices=slices)
        campaign = Campaign(id="test", states=[state])

        ps = ParameterSet()
        ps.add(Parameter("a", "a", 5.97, GLOBAL))
        ps.add(Parameter("scale", "scale", 1.0, STATE))
        ps.add(Parameter("bkg0", "bkg_c0", 0.0, SLICE))
        ps.add(Parameter("fixed_x", "x", 0.5, GLOBAL, fixed=True))

        return RefinementModel(campaign, ps)

    def test_creation(self):
        model = self._make_model()
        assert len(model.parameters) == 4
        assert model.campaign.id == "test"

    def test_independent_params_no_constraints(self):
        model = self._make_model()
        # 4 params, 1 fixed → 3 free, 0 constrained → 3 independent
        assert model.n_independent == 3

    def test_add_constraint_reduces_independent(self):
        model = self._make_model()
        model.add_constraint(equality_constraint("bkg0", "scale"))
        # bkg0 is now dependent → 2 independent
        assert model.n_independent == 2
        assert "bkg0" in model.dependent_ids

    def test_apply_constraints(self):
        model = self._make_model()
        model.add_constraint(
            linear_constraint("bkg0", "scale", factor=10.0, offset=0.0)
        )
        model.parameters["scale"].value = 3.0
        model.apply_constraints()
        assert model.parameters["bkg0"].value == pytest.approx(30.0)

    def test_get_free_vector(self):
        model = self._make_model()
        v = model.get_free_vector()
        assert len(v) == 3  # a, scale, bkg0 (fixed_x excluded)

    def test_set_free_vector(self):
        model = self._make_model()
        model.set_free_vector(np.array([6.0, 2.0, 0.5]))
        assert model.parameters["a"].value == pytest.approx(6.0)
        assert model.parameters["scale"].value == pytest.approx(2.0)
        assert model.parameters["bkg0"].value == pytest.approx(0.5)
        # fixed_x unchanged
        assert model.parameters["fixed_x"].value == pytest.approx(0.5)

    def test_set_free_vector_with_constraint(self):
        model = self._make_model()
        model.add_constraint(
            linear_constraint("bkg0", "scale", factor=5.0)
        )
        # Only a and scale are independent now
        model.set_free_vector(np.array([6.0, 3.0]))
        assert model.parameters["a"].value == pytest.approx(6.0)
        assert model.parameters["scale"].value == pytest.approx(3.0)
        assert model.parameters["bkg0"].value == pytest.approx(15.0)

    def test_set_free_vector_wrong_length_raises(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="Expected 3"):
            model.set_free_vector(np.array([1.0]))

    def test_add_constraint_bad_id_raises(self):
        model = self._make_model()
        with pytest.raises(KeyError):
            model.add_constraint(equality_constraint("nonexistent", "a"))

    def test_constraints_readonly_copy(self):
        model = self._make_model()
        model.add_constraint(equality_constraint("bkg0", "a"))
        c_list = model.constraints
        assert len(c_list) == 1
        c_list.clear()  # modifying the copy
        assert len(model.constraints) == 1  # original unchanged

    def test_repr(self):
        model = self._make_model()
        r = repr(model)
        assert "test" in r
        assert "n_independent=3" in r

    def test_summary(self):
        model = self._make_model()
        model.add_constraint(equality_constraint("bkg0", "scale"))
        s = model.summary()
        assert "independent:" in s
        assert "constrained:" in s


# ================================================================== #
# Realistic SNAP NaBr+Pb parameter setup
# ================================================================== #


@pytest.fixture()
def snap_campaign():
    """Load the real SNAP data if available."""
    if not DATA_DIR.is_dir():
        pytest.skip("Test data directory not found")
    return load_campaign_from_directory(DATA_DIR)


@pytest.fixture()
def snap_instprm():
    """Load the instrument parameters if available."""
    instprm_file = DATA_DIR / "SNAP066787_column.instprm"
    if not instprm_file.is_file():
        pytest.skip("Instrument parameter file not found")
    return load_instprm(instprm_file)


class TestSnapParameterSetup:
    """Build a realistic parameter setup for the SNAP NaBr+Pb data.

    This test class demonstrates how the parameter system works at
    realistic scale: 19 states × 6 banks = 114 slices, two phases.
    """

    def _build_params(self, campaign, instprm) -> ParameterSet:
        """Construct a realistic ParameterSet for NaBr + Pb."""
        ps = ParameterSet()

        # -- Global crystal structure (2 phases) ------------------- #
        # NaBr: Fm-3m, a=5.9738 — 1 lattice param
        ps.add(Parameter("NaBr:a", "a", 5.9738, GLOBAL))
        # Pb: Fm-3m, a=4.950 — 1 lattice param
        ps.add(Parameter("Pb:a", "a", 4.950, GLOBAL))

        # -- Per-bank profile params (from instprm) ---------------- #
        for bank_num, bank_params in instprm.items():
            for pname in ("sig-0", "sig-1", "sig-2", "alpha", "beta-0", "beta-1"):
                pid = f"bank{bank_num}:{pname}"
                val = float(bank_params.get(pname, 0.0))
                ps.add(Parameter(pid, pname, val, GLOBAL, fixed=True))

        # -- Per-bank instrument calibration (from instprm) -------- #
        for bank_num, bank_params in instprm.items():
            for pname in ("difC", "difA", "Zero"):
                pid = f"bank{bank_num}:{pname}"
                val = float(bank_params.get(pname, 0.0))
                ps.add(Parameter(pid, pname, val, GLOBAL, fixed=True))

        # -- Per-state phase scales -------------------------------- #
        for state in campaign.states:
            for phase in ("NaBr", "Pb"):
                pid = f"{state.id}:{phase}:scale"
                ps.add(Parameter(pid, "scale", 1.0, STATE))

        # -- Per-slice background (3-term polynomial) -------------- #
        for state in campaign.states:
            for slc in state.slices:
                for i in range(3):
                    pid = f"{slc.id}:bkg_c{i}"
                    ps.add(Parameter(pid, f"bkg_c{i}", 0.0, SLICE))

        return ps

    def test_parameter_count(self, snap_campaign, snap_instprm):
        ps = self._build_params(snap_campaign, snap_instprm)
        # 2 lattice + 6×6 profile + 6×3 calibration + 19×2 scales +
        # 19×6×3 backgrounds
        # = 2 + 36 + 18 + 38 + 342 = 436
        assert len(ps) == 436

    def test_scope_breakdown(self, snap_campaign, snap_instprm):
        ps = self._build_params(snap_campaign, snap_instprm)
        n_global = len(ps.by_scope(GLOBAL))
        n_state = len(ps.by_scope(STATE))
        n_slice = len(ps.by_scope(SLICE))
        # 2 lattice + 36 profile + 18 calibration = 56 global
        assert n_global == 56
        # 19 states × 2 phases = 38 state
        assert n_state == 38
        # 19 × 6 × 3 = 342 slice
        assert n_slice == 342

    def test_profile_fixed_by_default(self, snap_campaign, snap_instprm):
        ps = self._build_params(snap_campaign, snap_instprm)
        profile = ps.by_name("sig-1")
        assert all(p.fixed for p in profile)

    def test_free_count_default(self, snap_campaign, snap_instprm):
        """Only lattice params, scales, and backgrounds are free."""
        ps = self._build_params(snap_campaign, snap_instprm)
        # 2 lattice + 38 scales + 342 backgrounds = 382
        assert len(ps.free_params) == 382

    def test_free_vector_roundtrip(self, snap_campaign, snap_instprm):
        ps = self._build_params(snap_campaign, snap_instprm)
        v = ps.free_values
        assert len(v) == 382
        # Perturb and set back
        v_new = v + 0.01
        ps.free_values = v_new
        np.testing.assert_allclose(ps.free_values, v_new)

    def test_refinement_model_independent(self, snap_campaign, snap_instprm):
        ps = self._build_params(snap_campaign, snap_instprm)
        model = RefinementModel(snap_campaign, ps)
        assert model.n_independent == 382
        # Add a constraint — tie Pb scale to NaBr scale in first state
        s0 = snap_campaign.states[0].id
        model.add_constraint(
            linear_constraint(
                f"{s0}:Pb:scale", f"{s0}:NaBr:scale",
                factor=0.3,
            )
        )
        assert model.n_independent == 381

    def test_summary_output(self, snap_campaign, snap_instprm):
        ps = self._build_params(snap_campaign, snap_instprm)
        model = RefinementModel(snap_campaign, ps)
        s = model.summary()
        assert "19" in s  # states
        assert "114" in s  # slices
        assert "436" in s  # params
