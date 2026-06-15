"""
Aura versioned plugin SDK.

This module generalizes the Phase-1 importer registry to cover *all* Aura
extension points, not just file readers. The full list of officially stable
extension points and their entry-point group names:

    aura.readers          — file readers (Reader Protocol; see aura.io.registry)
    aura.exporters        — data exporters (Exporter Protocol)
    aura.profile_models   — peak-profile functions (ProfileModel Protocol)
    aura.background_models — background functions (BackgroundModel Protocol)
    aura.scattering_models — scattering-factor calculators (ScatteringModel Protocol)
    aura.optimizers       — minimizer back-ends (Minimizer Protocol from aura.spec)
    aura.instrument_kernels — instrument-resolution kernels (InstrumentKernel Protocol)

SDK versioning
--------------
The SDK follows *semantic versioning*: breaking changes to any Protocol
increment the major version; new optional methods increment the minor version.
Third-party plugins that target ``AURA_SDK_VERSION`` can assert compatibility:

    from aura.plugins import AURA_SDK_VERSION, assert_compatible
    assert_compatible("1.0")   # raises if major version differs

Entry-point registration
------------------------
A third-party package registers, e.g., a new profile model by declaring in its
``pyproject.toml``::

    [project.entry-points."aura.profile_models"]
    lorentzian = "mypkg.models:LorentzianFactory"

Aura discovers them at runtime via :func:`load_plugins`. Built-in extensions
do **not** use entry points — they call :func:`register` directly at import.

Example::

    from aura.plugins import register, AURA_SDK_VERSION
    from aura.spec import ForwardModel

    class MyOptimizer:
        name = "cobyla-1.0"
        ...

    register("aura.optimizers", MyOptimizer())
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Bumped on any breaking Protocol change.
AURA_SDK_VERSION = "1.0"
_AURA_SDK_MAJOR = int(AURA_SDK_VERSION.split(".")[0])


def assert_compatible(required_version: str) -> None:
    """Raise if the running SDK's major version differs from *required_version*.

    Args:
        required_version: Version string the plugin was built against (e.g. ``"1.0"``).

    Raises:
        RuntimeError: If major versions differ (breaking change).
    """
    major = int(required_version.split(".")[0])
    if major != _AURA_SDK_MAJOR:
        raise RuntimeError(
            f"Plugin requires aura SDK {required_version!r} "
            f"(major={major}), but running SDK is {AURA_SDK_VERSION!r} "
            f"(major={_AURA_SDK_MAJOR}). Cannot load."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps entry-point group name → list of registered extensions
_registry: dict[str, list[object]] = {}


def register(group: str, extension: object) -> object:
    """Register *extension* under *group*.

    Returns *extension* so this can be used as a decorator factory.

    Args:
        group: Entry-point group name (e.g. ``"aura.profile_models"``).
        extension: Any object that satisfies the Protocol for that group.

    Returns:
        The registered extension.
    """
    _registry.setdefault(group, []).append(extension)
    return extension


def extensions(group: str) -> list[object]:
    """Return all registered extensions for *group* (insertion order).

    Args:
        group: Entry-point group name.

    Returns:
        List of registered extension objects. Empty if nothing registered.
    """
    return list(_registry.get(group, []))


def load_plugins(group: str | None = None) -> int:
    """Discover and register extensions from installed packages' entry points.

    If *group* is given, only that group is loaded; otherwise all known
    ``aura.*`` entry-point groups are scanned.

    Returns the total number of new extensions registered.
    """
    from importlib.metadata import entry_points

    _KNOWN_GROUPS = [
        "aura.readers",
        "aura.exporters",
        "aura.profile_models",
        "aura.background_models",
        "aura.scattering_models",
        "aura.optimizers",
        "aura.instrument_kernels",
    ]
    groups_to_load = [group] if group else _KNOWN_GROUPS
    count = 0
    for grp in groups_to_load:
        for ep in entry_points(group=grp):
            factory = ep.load()
            obj = factory() if callable(factory) else factory
            if grp == "aura.readers":
                # Readers go through the io.registry for content-sniff dispatch.
                from aura.io.registry import registry as io_registry

                io_registry.register(obj)
            register(grp, obj)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Extension-point Protocols (stable SDK surface)
# ---------------------------------------------------------------------------


@runtime_checkable
class Exporter(Protocol):
    """Writes Aura objects to an external format.

    Attributes:
        name: Short human-readable name (e.g. ``"GSAS-II .gpx"``).
        format_id: Stable machine-readable identifier (e.g. ``"gsasii-gpx"``).
    """

    name: str
    format_id: str

    def export(self, obj: object, path: str) -> None:
        """Write *obj* to *path*."""
        ...


@runtime_checkable
class ProfileModel(Protocol):
    """A peak-profile function usable in the forward model.

    Attributes:
        name: Short name (e.g. ``"pseudo-voigt"``).
    """

    name: str

    def profile(
        self,
        x: object,
        center: float,
        fwhm: float,
        eta: float,
    ) -> object:
        """Evaluate the profile at abscissa *x* around *center*."""
        ...


@runtime_checkable
class BackgroundModel(Protocol):
    """An alternative background-function implementation.

    Attributes:
        name: Short name (e.g. ``"chebyshev-5"``).
        n_params: Number of free coefficients.
    """

    name: str
    n_params: int

    def evaluate(self, x: object, coeffs: object) -> object:
        """Evaluate background at *x* using *coeffs*."""
        ...


@runtime_checkable
class ScatteringModel(Protocol):
    """A scattering-factor calculator.

    Attributes:
        name: Short name (e.g. ``"IT92"``).
    """

    name: str

    def f_squared(self, element: str, q: float) -> float:
        """Return |f(Q)|² for *element* at momentum transfer *q* (Å⁻¹)."""
        ...


@runtime_checkable
class InstrumentKernel(Protocol):
    """A resolution / instrument-response kernel.

    Attributes:
        name: Short name (e.g. ``"gaussian-voigt"``).
        data_types: Which DataTypes this kernel supports.
    """

    name: str
    data_types: tuple[str, ...]

    def fwhm(self, x: float, params: object) -> float:
        """Return FWHM at abscissa *x* given instrument *params*."""
        ...
