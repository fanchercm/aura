"""
Aura - Prototype framework for large-volume Rietveld analysis.

Explores solutions for high-throughput Rietveld refinement of
partially integrated diffraction data at scale.

The authoritative refinement model lives in :mod:`aura.spec` (immutable
``RefinementState``/``Phase``/``Histogram``/``Parameter`` + the engine
Protocols). The container/ingest layer (:mod:`aura.models`) and the importer
registry (:mod:`aura.io`) feed it via the one-way :mod:`aura.bridge`.
"""

__version__ = "0.1.0"

from aura.models import (
    Campaign,
    DiffractionSlice,
    MeasurementState,
    ParameterScope,
)

__all__ = [
    "__version__",
    "Campaign",
    "DiffractionSlice",
    "MeasurementState",
    "ParameterScope",
]
