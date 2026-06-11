"""
One-way bridge: container model → immutable refinement state.

The container hierarchy (:class:`aura.models.Campaign` →
:class:`~aura.models.MeasurementState` → :class:`~aura.models.DiffractionSlice`)
is the *ingest* layer produced by the I/O readers. The refinement engine works
on the immutable :mod:`aura.spec` model. This module maps the former onto the
latter and **only** in that direction — nothing flows back, so the two models
never become a leaky bidirectional abstraction.

Each ``DiffractionSlice`` becomes one :class:`aura.spec.Histogram`:

* invalid (NaN/Inf) points are dropped via ``DiffractionSlice.valid_mask``;
* weights are ``1/σ²`` from the per-point esd (variance floored at 1 count, the
  standard Poisson treatment, so zero-count channels don't get infinite weight);
* the data-type-specific fields (``wavelength`` for CW, ``difc``/``difa``/
  ``zero`` for TOF, ``two_theta_fixed`` for EDD) are pulled from slice metadata
  or supplied by the caller (e.g. after :func:`aura.io.instrument.apply_instprm`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from aura.models import Campaign, MeasurementState
from aura.spec import DataType, Histogram

# Metadata keys treated as driving (stimulus) coordinates when numeric.
_DRIVING_KEYS = ("temperature", "pressure", "time", "field", "load", "run_number")


def _driving_from_metadata(metadata: Mapping) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in _DRIVING_KEYS:
        if key in metadata:
            try:
                out[key] = float(metadata[key])
            except (TypeError, ValueError):
                continue
    return out


def slice_to_histogram(
    slc,
    *,
    driving: Mapping[str, float] | None = None,
    data_type: DataType | None = None,
    wavelength: float | None = None,
    difc: float | None = None,
    difa: float = 0.0,
    zero: float = 0.0,
    two_theta_fixed: float | None = None,
) -> Histogram:
    """Convert one :class:`~aura.models.DiffractionSlice` to a ``Histogram``.

    Explicit keyword arguments override the corresponding slice metadata, so a
    caller can supply a wavelength/DIFC the raw file did not carry.

    Raises:
        ValueError: If the resolved data type lacks its required field
            (CW→wavelength, TOF→difc, EDD→two_theta_fixed) — surfaced via
            ``Histogram.__post_init__``.
    """
    meta = slc.metadata
    dtype = data_type or meta.get("data_type") or DataType.TOF

    mask = slc.valid_mask
    x = np.asarray(slc.x, dtype=float)[mask]
    y = np.asarray(slc.y, dtype=float)[mask]
    e = np.asarray(slc.e, dtype=float)[mask]
    weights = 1.0 / np.clip(e**2, 1.0, None)

    drive = dict(driving) if driving is not None else _driving_from_metadata(meta)

    kwargs: dict = {}
    if dtype in (DataType.CW_XRAY, DataType.CW_NEUTRON):
        wl = wavelength if wavelength is not None else meta.get("wavelength")
        if wl is None:
            raise ValueError(
                f"CW slice {slc.id!r} has no wavelength; supply one via "
                "apply_instprm() or the wavelength= argument."
            )
        kwargs["wavelength"] = float(wl)
    elif dtype is DataType.TOF:
        d = difc if difc is not None else meta.get("difc")
        if d is None:
            raise ValueError(f"TOF slice {slc.id!r} has no DIFC.")
        kwargs["difc"] = float(d)
        kwargs["difa"] = float(difa if difa else meta.get("difa", 0.0))
        kwargs["zero"] = float(zero if zero else meta.get("zero", 0.0))
    elif dtype is DataType.EDD:
        tt = (
            two_theta_fixed
            if two_theta_fixed is not None
            else meta.get("two_theta_fixed")
        )
        if tt is None:
            raise ValueError(f"EDD slice {slc.id!r} has no fixed 2theta.")
        kwargs["two_theta_fixed"] = float(tt)

    return Histogram(
        id=slc.id,
        data_type=dtype,
        x=x,
        y_obs=y,
        weights=weights,
        driving=drive,
        **kwargs,
    )


def state_to_histograms(
    state: MeasurementState,
    *,
    driving: Mapping[str, float] | None = None,
    wavelength: float | None = None,
) -> tuple[Histogram, ...]:
    """Convert every slice of a ``MeasurementState`` to histograms.

    All slices share one ``driving`` dict (the state's stimulus coordinate);
    if not given it is derived from ``state.metadata``.
    """
    drive = (
        dict(driving) if driving is not None else _driving_from_metadata(state.metadata)
    )
    return tuple(
        slice_to_histogram(slc, driving=drive, wavelength=wavelength)
        for slc in state.slices
    )


def campaign_to_histograms(
    campaign: Campaign,
    *,
    wavelength: float | None = None,
) -> tuple[Histogram, ...]:
    """Flatten a whole ``Campaign`` to histograms, one per slice.

    Each state contributes its slices with that state's driving coordinate, so
    the resulting ensemble carries the stimulus axis the parametric engine needs.
    """
    out: list[Histogram] = []
    for state in campaign.states:
        out.extend(state_to_histograms(state, wavelength=wavelength))
    return tuple(out)
