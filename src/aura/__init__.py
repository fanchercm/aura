"""
Aura - Prototype framework for large-volume Rietveld analysis.

Explores solutions for high-throughput Rietveld refinement of
partially integrated diffraction data at scale.
"""

__version__ = "0.1.0"

from aura.models import (
    Campaign,
    DiffractionSlice,
    MeasurementState,
    ParameterScope,
)
from aura.parameters import (
    Constraint,
    Parameter,
    ParameterSet,
    RefinementModel,
    equality_constraint,
    linear_constraint,
)

__all__ = [
    "__version__",
    "Campaign",
    "Constraint",
    "DiffractionSlice",
    "MeasurementState",
    "Parameter",
    "ParameterScope",
    "ParameterSet",
    "RefinementModel",
    "equality_constraint",
    "linear_constraint",
]
