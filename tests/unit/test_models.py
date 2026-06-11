"""Tests for the core data model classes."""

import numpy as np
import pytest

from aura.models import (
    Campaign,
    DiffractionSlice,
    MeasurementState,
    ParameterScope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slice(
    slice_id: str = "test_bank1",
    n: int = 100,
    with_nans: bool = False,
) -> DiffractionSlice:
    """Create a simple DiffractionSlice for testing."""
    x = np.linspace(3000.0, 14000.0, n)
    y = np.random.default_rng(42).poisson(1000, size=n).astype(float)
    e = np.sqrt(y)
    if with_nans:
        y[:10] = np.nan
        e[:10] = 0.0
    return DiffractionSlice(
        id=slice_id,
        x=x,
        y=y,
        e=e,
        metadata={"two_theta": 90.0, "bank_num": 1, "difc": 5400.0},
    )


def _make_state(
    state_id: str = "SNAP067702",
    n_banks: int = 3,
) -> MeasurementState:
    """Create a MeasurementState with several slices."""
    slices = [
        _make_slice(slice_id=f"{state_id}_bank{i+1}")
        for i in range(n_banks)
    ]
    return MeasurementState(
        id=state_id,
        slices=slices,
        metadata={"run_number": 67702},
    )


# ---------------------------------------------------------------------------
# ParameterScope
# ---------------------------------------------------------------------------

class TestParameterScope:
    def test_scope_values(self):
        assert ParameterScope.GLOBAL.value == "global"
        assert ParameterScope.STATE.value == "state"
        assert ParameterScope.SLICE.value == "slice"

    def test_scope_members(self):
        assert set(ParameterScope.__members__) == {"GLOBAL", "STATE", "SLICE"}


# ---------------------------------------------------------------------------
# DiffractionSlice
# ---------------------------------------------------------------------------

class TestDiffractionSlice:
    def test_creation(self):
        s = _make_slice()
        assert s.id == "test_bank1"
        assert s.n_points == 100
        assert s.n_valid == 100

    def test_with_nans(self):
        s = _make_slice(with_nans=True)
        assert s.n_points == 100
        assert s.n_valid == 90
        mask = s.valid_mask
        assert mask.sum() == 90
        assert not mask[:10].any()

    def test_metadata_access(self):
        s = _make_slice()
        assert s.metadata["two_theta"] == 90.0
        assert s.metadata["bank_num"] == 1

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Array length mismatch"):
            DiffractionSlice(
                id="bad",
                x=np.array([1.0, 2.0]),
                y=np.array([1.0]),
                e=np.array([1.0, 2.0]),
            )

    def test_repr(self):
        s = _make_slice()
        r = repr(s)
        assert "test_bank1" in r
        assert "n_points=100" in r

    def test_empty_slice(self):
        s = DiffractionSlice(
            id="empty",
            x=np.array([]),
            y=np.array([]),
            e=np.array([]),
        )
        assert s.n_points == 0
        assert s.n_valid == 0


# ---------------------------------------------------------------------------
# MeasurementState
# ---------------------------------------------------------------------------

class TestMeasurementState:
    def test_creation(self):
        st = _make_state()
        assert st.id == "SNAP067702"
        assert st.n_slices == 3

    def test_get_slice(self):
        st = _make_state()
        s = st.get_slice("SNAP067702_bank2")
        assert s.id == "SNAP067702_bank2"

    def test_get_slice_missing_raises(self):
        st = _make_state()
        with pytest.raises(KeyError, match="no_such_slice"):
            st.get_slice("no_such_slice")

    def test_empty_state_raises(self):
        with pytest.raises(ValueError, match="at least one slice"):
            MeasurementState(id="empty", slices=[])

    def test_repr(self):
        st = _make_state(n_banks=6)
        assert "n_slices=6" in repr(st)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

class TestCampaign:
    def test_creation(self):
        st = _make_state()
        c = Campaign(id="test_campaign", states=[st])
        assert c.n_states == 1
        assert c.n_slices_total == 3

    def test_multiple_states(self):
        states = [_make_state(state_id=f"run_{i}") for i in range(5)]
        c = Campaign(id="multi", states=states)
        assert c.n_states == 5
        assert c.n_slices_total == 15

    def test_get_state(self):
        states = [_make_state(state_id=f"run_{i}") for i in range(3)]
        c = Campaign(id="test", states=states)
        assert c.get_state("run_1").id == "run_1"

    def test_get_state_missing_raises(self):
        c = Campaign(id="test", states=[_make_state()])
        with pytest.raises(KeyError, match="missing"):
            c.get_state("missing")

    def test_empty_campaign_raises(self):
        with pytest.raises(ValueError, match="at least one state"):
            Campaign(id="empty", states=[])

    def test_repr(self):
        c = Campaign(id="test", states=[_make_state()])
        r = repr(c)
        assert "n_states=1" in r
        assert "n_slices_total=3" in r
