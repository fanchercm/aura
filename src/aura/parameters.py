"""
Parameter management system for Aura.

Provides the machinery for declaring, scoping, and constraining
refinement parameters.  The key classes are:

- :class:`Parameter` — a single refinable value with scope and bounds.
- :class:`ParameterSet` — an ordered collection of parameters with
  bulk operations (fix / free / reset / vector I/O).
- :class:`Constraint` — a functional relationship between parameters
  (equality ties, linear ties, or arbitrary callables).
- :class:`RefinementModel` — ties a :class:`~aura.models.Campaign`
  to its parameters and constraints, and provides the independent
  free-parameter vector that an optimiser works with.

Design decisions
~~~~~~~~~~~~~~~~
* **Trajectory / history** is stored externally (in the refinement
  engine), not on the Parameter.  Parameters keep only their current
  ``value`` and an ``initial_value`` for reset.
* **Crystallographic constraints** (bond-length, symmetry-aware) are
  deferred.  Phase 1 supports equality ties, linear ties, and
  arbitrary callable ties.
* **Scope** is declared on each Parameter via :class:`ParameterScope`.
  The ``param_id`` string encodes the scope qualifiers
  (phase / bank / state / slice) by convention but is not parsed —
  it is an opaque unique key.

Example:
    >>> from aura.parameters import (
    ...     Parameter, ParameterSet, RefinementModel,
    ...     equality_constraint, linear_constraint,
    ... )
    >>> from aura.models import ParameterScope
    >>> ps = ParameterSet()
    >>> ps.add(Parameter("NaBr:a", "a", 5.9738, ParameterScope.GLOBAL))
    >>> ps.add(Parameter("bank1:sig1", "sig-1", 130.0, ParameterScope.GLOBAL))
    >>> ps["NaBr:a"].value
    5.9738
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from aura.models import Campaign, ParameterScope


# ------------------------------------------------------------------ #
# Parameter
# ------------------------------------------------------------------ #


@dataclass
class Parameter:
    """A single refinable parameter.

    Attributes:
        param_id: Unique key (e.g. ``"NaBr:a"``, ``"SNAP067702:bank1:bkg_c0"``).
        name: Short human-readable name (e.g. ``"a"``, ``"bkg_c0"``).
        value: Current numerical value.
        scope: The hierarchy level at which this parameter is shared.
        fixed: If ``True`` the parameter is held constant during
            refinement.
        bounds: Optional ``(lower, upper)`` box constraint.

    The ``initial_value`` is captured automatically at construction
    and can be restored with :meth:`reset`.

    Raises:
        ValueError: If *bounds* are given and ``lower > upper``, or
            if *value* falls outside *bounds*.
    """

    param_id: str
    name: str
    value: float
    scope: ParameterScope
    fixed: bool = False
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo > hi:
                raise ValueError(
                    f"Lower bound ({lo}) > upper bound ({hi}) "
                    f"for parameter {self.param_id!r}"
                )
            if not (lo <= self.value <= hi):
                raise ValueError(
                    f"Value {self.value} outside bounds [{lo}, {hi}] "
                    f"for parameter {self.param_id!r}"
                )
        # Snapshot for reset — not a dataclass field so it stays
        # out of __repr__ / __eq__.
        object.__setattr__(self, "_initial_value", self.value)

    @property
    def initial_value(self) -> float:
        """Value at construction time (or last :meth:`reset` origin)."""
        return self._initial_value  # type: ignore[attr-defined]

    def reset(self) -> None:
        """Restore *value* to *initial_value*."""
        self.value = self._initial_value  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        state = "fixed" if self.fixed else "free"
        return (
            f"Parameter({self.param_id!r}, value={self.value}, "
            f"scope={self.scope.value}, {state})"
        )


# ------------------------------------------------------------------ #
# ParameterSet
# ------------------------------------------------------------------ #


class ParameterSet:
    """Ordered collection of :class:`Parameter` objects.

    Parameters are stored in insertion order, which determines the
    layout of the free-parameter vector.

    Example:
        >>> ps = ParameterSet()
        >>> ps.add(Parameter("a", "a", 5.0, ParameterScope.GLOBAL))
        >>> ps.add(Parameter("b", "b", 3.0, ParameterScope.STATE))
        >>> len(ps)
        2
        >>> ps["a"].value
        5.0
    """

    def __init__(self) -> None:
        self._params: dict[str, Parameter] = {}

    # -- mutators -------------------------------------------------- #

    def add(self, param: Parameter) -> None:
        """Add a parameter.  Raises if *param_id* already exists."""
        if param.param_id in self._params:
            raise ValueError(f"Duplicate param_id: {param.param_id!r}")
        self._params[param.param_id] = param

    def fix(self, *param_ids: str) -> None:
        """Mark one or more parameters as fixed."""
        for pid in param_ids:
            self[pid].fixed = True

    def free(self, *param_ids: str) -> None:
        """Mark one or more parameters as free (refinable)."""
        for pid in param_ids:
            self[pid].fixed = False

    def fix_all(self) -> None:
        """Fix every parameter."""
        for p in self._params.values():
            p.fixed = True

    def free_all(self) -> None:
        """Free every parameter."""
        for p in self._params.values():
            p.fixed = False

    def reset_all(self) -> None:
        """Reset every parameter to its initial value."""
        for p in self._params.values():
            p.reset()

    # -- lookup ---------------------------------------------------- #

    def __getitem__(self, param_id: str) -> Parameter:
        try:
            return self._params[param_id]
        except KeyError:
            raise KeyError(f"No parameter with id={param_id!r}") from None

    def __contains__(self, param_id: str) -> bool:
        return param_id in self._params

    def __len__(self) -> int:
        return len(self._params)

    def __iter__(self):
        return iter(self._params.values())

    # -- filtered views -------------------------------------------- #

    @property
    def free_params(self) -> list[Parameter]:
        """All parameters where ``fixed is False``, in insertion order."""
        return [p for p in self._params.values() if not p.fixed]

    def by_scope(self, scope: ParameterScope) -> list[Parameter]:
        """Return parameters matching *scope*."""
        return [p for p in self._params.values() if p.scope == scope]

    def by_name(self, name: str) -> list[Parameter]:
        """Return parameters whose *name* matches."""
        return [p for p in self._params.values() if p.name == name]

    # -- vector I/O (for optimisers) ------------------------------- #

    @property
    def free_values(self) -> np.ndarray:
        """Current values of free parameters as a 1-D array."""
        return np.array([p.value for p in self.free_params])

    @free_values.setter
    def free_values(self, values: np.ndarray) -> None:
        """Set free-parameter values from a 1-D array.

        Raises:
            ValueError: If length does not match.
        """
        free = self.free_params
        if len(values) != len(free):
            raise ValueError(
                f"Expected {len(free)} values, got {len(values)}"
            )
        for param, val in zip(free, values):
            param.value = float(val)

    # -- display --------------------------------------------------- #

    @property
    def all_ids(self) -> list[str]:
        """All parameter IDs in insertion order."""
        return list(self._params.keys())

    def summary(self) -> str:
        """One-line summary string."""
        n_free = len(self.free_params)
        n_fixed = len(self) - n_free
        return f"ParameterSet({len(self)} params: {n_free} free, {n_fixed} fixed)"

    def __repr__(self) -> str:
        return self.summary()


# ------------------------------------------------------------------ #
# Constraints
# ------------------------------------------------------------------ #


@dataclass
class Constraint:
    """Functional relationship: dependent = f(independent_1, ...).

    During refinement the *dependent* parameter is not free — its
    value is recomputed from the *independents* after each step.

    Attributes:
        dependent_id: ``param_id`` of the derived parameter.
        independent_ids: ``param_id``\\s of the driving parameters.
        expression: Callable mapping independent values → dependent
            value.
        description: Human-readable description of the relationship.
    """

    dependent_id: str
    independent_ids: list[str]
    expression: Callable[..., float]
    description: str = ""

    def evaluate(self, params: ParameterSet) -> float:
        """Compute the dependent value from current independent values.

        Args:
            params: The :class:`ParameterSet` containing all
                referenced parameters.

        Returns:
            The computed dependent value.
        """
        indep_vals = [params[pid].value for pid in self.independent_ids]
        return float(self.expression(*indep_vals))


def equality_constraint(
    dependent_id: str,
    independent_id: str,
) -> Constraint:
    """Create ``dependent = independent``.

    Args:
        dependent_id: The parameter whose value is derived.
        independent_id: The parameter whose value is copied.

    Returns:
        A :class:`Constraint` enforcing equality.
    """
    return Constraint(
        dependent_id=dependent_id,
        independent_ids=[independent_id],
        expression=lambda x: x,
        description=f"{dependent_id} = {independent_id}",
    )


def linear_constraint(
    dependent_id: str,
    independent_id: str,
    factor: float = 1.0,
    offset: float = 0.0,
) -> Constraint:
    """Create ``dependent = factor * independent + offset``.

    Args:
        dependent_id: The parameter whose value is derived.
        independent_id: The driving parameter.
        factor: Multiplicative factor.
        offset: Additive offset.

    Returns:
        A :class:`Constraint` enforcing the linear relationship.
    """
    return Constraint(
        dependent_id=dependent_id,
        independent_ids=[independent_id],
        expression=lambda x, _f=factor, _o=offset: _f * x + _o,
        description=f"{dependent_id} = {factor}*{independent_id} + {offset}",
    )


# ------------------------------------------------------------------ #
# RefinementModel
# ------------------------------------------------------------------ #


class RefinementModel:
    """Ties a :class:`~aura.models.Campaign` to parameters and constraints.

    The model distinguishes three kinds of parameter:

    * **fixed** — value does not change (``Parameter.fixed is True``).
    * **constrained dependent** — value is derived from other
      parameters via a :class:`Constraint`.
    * **independent free** — value is adjusted by the optimiser.

    The optimiser works only with the independent-free vector.
    After each optimiser step, :meth:`apply_constraints` propagates
    values to all constrained dependents.

    Attributes:
        campaign: The data this model describes.
        parameters: The full :class:`ParameterSet`.
    """

    def __init__(
        self,
        campaign: Campaign,
        parameters: ParameterSet,
    ) -> None:
        self.campaign = campaign
        self.parameters = parameters
        self._constraints: list[Constraint] = []

    # -- constraint management ------------------------------------- #

    def add_constraint(self, constraint: Constraint) -> None:
        """Register a constraint.

        The dependent parameter is validated to exist in the
        parameter set, as are all independents.

        Args:
            constraint: The constraint to add.

        Raises:
            KeyError: If any referenced parameter ID is missing.
        """
        # Validate all IDs exist
        _ = self.parameters[constraint.dependent_id]
        for pid in constraint.independent_ids:
            _ = self.parameters[pid]
        self._constraints.append(constraint)

    @property
    def constraints(self) -> list[Constraint]:
        """All registered constraints (read-only view)."""
        return list(self._constraints)

    @property
    def dependent_ids(self) -> set[str]:
        """Parameter IDs that are determined by constraints."""
        return {c.dependent_id for c in self._constraints}

    def apply_constraints(self) -> None:
        """Recompute all constrained-dependent parameter values.

        Constraints are applied in registration order.  If there are
        dependency chains (A→B→C), register them in dependency order.
        """
        for c in self._constraints:
            self.parameters[c.dependent_id].value = c.evaluate(
                self.parameters
            )

    # -- independent free vector ----------------------------------- #

    @property
    def independent_params(self) -> list[Parameter]:
        """Free parameters that are not constrained dependents."""
        dep = self.dependent_ids
        return [
            p
            for p in self.parameters.free_params
            if p.param_id not in dep
        ]

    @property
    def n_independent(self) -> int:
        """Number of independent free parameters."""
        return len(self.independent_params)

    def get_free_vector(self) -> np.ndarray:
        """Return values of independent free parameters as a 1-D array."""
        return np.array([p.value for p in self.independent_params])

    def set_free_vector(self, values: np.ndarray) -> None:
        """Set independent free parameters and apply constraints.

        This is the main interface for an optimiser:

        1. Writes *values* into the independent free parameters.
        2. Calls :meth:`apply_constraints` to update dependents.

        Args:
            values: 1-D array of length :attr:`n_independent`.

        Raises:
            ValueError: If length does not match.
        """
        indep = self.independent_params
        if len(values) != len(indep):
            raise ValueError(
                f"Expected {len(indep)} values, got {len(values)}"
            )
        for param, val in zip(indep, values):
            param.value = float(val)
        self.apply_constraints()

    # -- display --------------------------------------------------- #

    def summary(self) -> str:
        """Multi-line summary of the model."""
        lines = [
            f"RefinementModel for campaign {self.campaign.id!r}",
            f"  States:       {self.campaign.n_states}",
            f"  Total slices: {self.campaign.n_slices_total}",
            f"  Parameters:   {len(self.parameters)}",
            f"    fixed:      {len(self.parameters) - len(self.parameters.free_params)}",
            f"    constrained:{len(self.dependent_ids)}",
            f"    independent:{self.n_independent}",
            f"  Constraints:  {len(self._constraints)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"RefinementModel(campaign={self.campaign.id!r}, "
            f"n_params={len(self.parameters)}, "
            f"n_independent={self.n_independent})"
        )
