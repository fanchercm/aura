"""
Core data model for Aura.

Defines the hierarchical container classes that represent a diffraction
campaign:

    Campaign → MeasurementState → DiffractionSlice

Each level holds metadata and references to its children. Parameters
can be scoped to any level via the ParameterScope enum.

Example:
    >>> from aura.models import Campaign, MeasurementState, DiffractionSlice
    >>> import numpy as np
    >>> slc = DiffractionSlice(
    ...     id="run1_bank1",
    ...     x=np.array([1.0, 2.0, 3.0]),
    ...     y=np.array([100.0, 200.0, 150.0]),
    ...     e=np.array([10.0, 14.1, 12.2]),
    ...     metadata={"two_theta": 90.0, "flight_path": 15.5, "difc": 5400.0},
    ... )
    >>> state = MeasurementState(id="run1", slices=[slc])
    >>> campaign = Campaign(id="my_campaign", states=[state])
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class ParameterScope(enum.Enum):
    """Scope at which a refinement parameter is shared.

    Attributes:
        GLOBAL: Shared across the entire campaign (e.g. crystal structure).
        STATE: Shared across all slices in one measurement state
            (e.g. phase fractions at a given pressure).
        SLICE: Local to a single diffraction histogram
            (e.g. background polynomial, local scale).
    """

    GLOBAL = "global"
    STATE = "state"
    SLICE = "slice"


@dataclass
class DiffractionSlice:
    """One 1-D diffraction histogram with directional metadata.

    Attributes:
        id: Unique identifier (e.g. ``"SNAP067702_bank1"``).
        x: Independent variable array (TOF in µs, or d-spacing, or 2θ).
        y: Intensity values.
        e: Uncertainty (standard deviation) on *y*.
        metadata: Bank-level metadata. Expected keys include
            ``"two_theta"``, ``"flight_path"``, ``"difc"``,
            ``"bank_num"``, ``"spectrum_index"``.

    Raises:
        ValueError: If *x*, *y*, *e* have mismatched lengths.
    """

    id: str
    x: np.ndarray
    y: np.ndarray
    e: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (len(self.x) == len(self.y) == len(self.e)):
            raise ValueError(
                f"Array length mismatch: x={len(self.x)}, "
                f"y={len(self.y)}, e={len(self.e)}"
            )

    @property
    def n_points(self) -> int:
        """Number of data points in this histogram."""
        return len(self.x)

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask: True where data is finite (not NaN/Inf)."""
        return np.isfinite(self.y) & np.isfinite(self.e)

    @property
    def n_valid(self) -> int:
        """Number of valid (non-NaN) data points."""
        return int(np.sum(self.valid_mask))

    def __repr__(self) -> str:
        return (
            f"DiffractionSlice(id={self.id!r}, n_points={self.n_points}, "
            f"n_valid={self.n_valid})"
        )


@dataclass
class MeasurementState:
    """A collection of slices measured at one sample condition.

    All slices in a state were acquired simultaneously (same pressure,
    temperature, etc.) but from different detector banks / angular
    ranges.

    Attributes:
        id: Unique identifier (e.g. ``"SNAP067702"``).
        slices: Ordered list of :class:`DiffractionSlice` objects.
        metadata: State-level metadata.  Expected keys include
            ``"run_number"``, ``"temperature"``, ``"pressure"``
            (or a proxy like run number).

    Raises:
        ValueError: If *slices* is empty.
    """

    id: str
    slices: list[DiffractionSlice] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("MeasurementState must contain at least one slice")

    @property
    def n_slices(self) -> int:
        """Number of diffraction slices in this state."""
        return len(self.slices)

    def get_slice(self, slice_id: str) -> DiffractionSlice:
        """Look up a slice by its *id*.

        Args:
            slice_id: The id of the slice to retrieve.

        Returns:
            The matching :class:`DiffractionSlice`.

        Raises:
            KeyError: If no slice with that id exists.
        """
        for s in self.slices:
            if s.id == slice_id:
                return s
        raise KeyError(f"No slice with id={slice_id!r} in state {self.id!r}")

    def __repr__(self) -> str:
        return (
            f"MeasurementState(id={self.id!r}, n_slices={self.n_slices})"
        )


@dataclass
class Campaign:
    """Top-level container for an entire diffraction experiment.

    A campaign holds multiple measurement states (e.g. a pressure
    series) and campaign-wide metadata.

    Attributes:
        id: Descriptive identifier for the campaign.
        states: Ordered list of :class:`MeasurementState` objects.
        metadata: Campaign-level metadata (instrument, sample info, etc.).

    Raises:
        ValueError: If *states* is empty.
    """

    id: str
    states: list[MeasurementState] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("Campaign must contain at least one state")

    @property
    def n_states(self) -> int:
        """Number of measurement states."""
        return len(self.states)

    @property
    def n_slices_total(self) -> int:
        """Total number of slices across all states."""
        return sum(s.n_slices for s in self.states)

    def get_state(self, state_id: str) -> MeasurementState:
        """Look up a measurement state by its *id*.

        Args:
            state_id: The id of the state to retrieve.

        Returns:
            The matching :class:`MeasurementState`.

        Raises:
            KeyError: If no state with that id exists.
        """
        for st in self.states:
            if st.id == state_id:
                return st
        raise KeyError(f"No state with id={state_id!r} in campaign {self.id!r}")

    def __repr__(self) -> str:
        return (
            f"Campaign(id={self.id!r}, n_states={self.n_states}, "
            f"n_slices_total={self.n_slices_total})"
        )
