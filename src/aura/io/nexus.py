"""
NeXus/HDF5 I/O for Aura refinement state and measurement data.

Writes to a lightweight HDF5 layout inspired by NeXus NXentry conventions:

    /aura_version     (attr)
    /entry/
        histograms/<id>/
            x            (deg | us | keV)  — unit stored as HDF5 attribute
            y_obs        (counts or mean_intensity)
            weights      (1/sigma^2)
            data_type    (str attr)
            wavelength   (float attr, CW only)
            difc         (float attr, TOF only)
            two_theta_fixed (float attr, EDD only)
            difa, zero   (float attrs)
            driving/     (one scalar dataset per driving variable)
        phases/<name>/
            space_group  (str attr)
            cell/        {a, b, c, alpha, beta, gamma}
            atoms/       {element, x, y, z, occ, b_iso} columnar
        parameters/
            names        (string dataset)
            kinds        (string dataset)
            values       (float64 dataset)
            vary         (bool dataset)
            lower        (float64 dataset)
            upper        (float64 dataset)
            sigma        (float64 dataset, NaN = absent)

Unit consistency is enforced by the ``units`` attribute on every abscissa
dataset — readers reject data whose units are unexpected (see :func:`load_refinement_state`).

Parametric models use callables which are not serializable to HDF5; they are
dropped on save with a warning (same policy as :mod:`aura.checkpoint`).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from aura.spec import (
    AtomSite,
    DataType,
    Histogram,
    Parameter,
    ParamKind,
    Phase,
    RefinementState,
    UnitCell,
)

try:
    import h5py

    _H5PY = True
except ImportError:  # pragma: no cover
    _H5PY = False

# Maps DataType → abscissa units string
_ABSCISSA_UNITS: dict[DataType, str] = {
    DataType.CW_XRAY: "deg",
    DataType.CW_NEUTRON: "deg",
    DataType.TOF: "us",
    DataType.EDD: "keV",
}

AURA_FORMAT_VERSION = "1.0"


def _require_h5py() -> None:
    if not _H5PY:
        raise ImportError("h5py is required for NeXus/HDF5 I/O: pip install h5py")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_refinement_state(state: RefinementState, path: str | Path) -> Path:
    """Serialize *state* to an HDF5 file at *path*.

    Parametric models (callables) are dropped with a warning — they cannot
    be encoded in HDF5. All other fields round-trip exactly.

    Args:
        state: Immutable refinement state to save.
        path: Destination file path (created or overwritten).

    Returns:
        Resolved path of the written file.
    """
    _require_h5py()
    path = Path(path)
    if state.parametric_models:
        warnings.warn(
            f"Dropping {len(state.parametric_models)} parametric model(s): "
            "callables cannot be serialized to HDF5.",
            stacklevel=2,
        )
    with h5py.File(path, "w") as f:
        f.attrs["aura_version"] = AURA_FORMAT_VERSION
        f.attrs["aura_content"] = "RefinementState"
        entry = f.require_group("entry")
        _write_histograms(entry, state.histograms)
        _write_phases(entry, state.phases)
        _write_parameters(entry, state.parameters)
        if state.constraints:
            entry.create_dataset(
                "constraints",
                data=np.array(
                    list(state.constraints), dtype=h5py.special_dtype(vlen=str)
                ),
            )
        if state.provenance:
            entry.create_dataset(
                "provenance",
                data=np.array(
                    list(state.provenance), dtype=h5py.special_dtype(vlen=str)
                ),
            )
    return path


def load_refinement_state(path: str | Path) -> RefinementState:
    """Load a :class:`~aura.spec.RefinementState` from an HDF5 file.

    Args:
        path: Path to an HDF5 file written by :func:`save_refinement_state`.

    Returns:
        Reconstructed immutable state (no parametric models).

    Raises:
        ValueError: If the file is not an aura RefinementState or units are inconsistent.
    """
    _require_h5py()
    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_version(f)
        if f.attrs.get("aura_content") != "RefinementState":
            raise ValueError(f"{path} does not contain a RefinementState")
        entry = f["entry"]
        histograms = _read_histograms(entry)
        phases = _read_phases(entry)
        parameters = _read_parameters(entry)
        constraints: tuple[str, ...] = ()
        if "constraints" in entry:
            constraints = tuple(_decode(s) for s in entry["constraints"][:])
        provenance: tuple[str, ...] = ()
        if "provenance" in entry:
            provenance = tuple(_decode(s) for s in entry["provenance"][:])
    return RefinementState(
        phases=phases,
        histograms=histograms,
        parameters=parameters,
        constraints=constraints,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Measurement state (MeasurementState from aura.models)
# ---------------------------------------------------------------------------


def save_measurement_state(state: object, path: str | Path) -> Path:
    """Serialize a :class:`~aura.models.MeasurementState` to HDF5.

    Args:
        state: MeasurementState instance.
        path: Destination file path.

    Returns:
        Resolved path of the written file.
    """
    _require_h5py()
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["aura_version"] = AURA_FORMAT_VERSION
        f.attrs["aura_content"] = "MeasurementState"
        f.attrs["state_id"] = state.id  # type: ignore[attr-defined]
        slices_grp = f.require_group("slices")
        for slc in state.slices:  # type: ignore[attr-defined]
            sg = slices_grp.require_group(slc.id)
            sg.create_dataset("x", data=slc.x)
            sg.create_dataset("y", data=slc.y)
            sg.create_dataset("e", data=slc.e)
            for k, v in (slc.metadata or {}).items():
                if isinstance(v, (int, float, str, bool)):
                    sg.attrs[k] = v
                elif isinstance(v, np.ndarray):
                    sg.create_dataset(f"meta_{k}", data=v)
    return path


def load_measurement_state(path: str | Path) -> object:
    """Load a :class:`~aura.models.MeasurementState` from HDF5.

    Args:
        path: Path to an HDF5 file written by :func:`save_measurement_state`.

    Returns:
        Reconstructed MeasurementState.

    Raises:
        ValueError: If the file does not contain a MeasurementState.
    """
    _require_h5py()
    from aura.models import DiffractionSlice, MeasurementState

    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_version(f)
        if f.attrs.get("aura_content") != "MeasurementState":
            raise ValueError(f"{path} does not contain a MeasurementState")
        state_id = _decode(f.attrs["state_id"])
        slices = []
        for slc_id, sg in f["slices"].items():
            x = sg["x"][:]
            y = sg["y"][:]
            e = sg["e"][:]
            meta = dict(sg.attrs.items())
            # Restore array metadata saved under meta_ prefix
            for key in sg:
                if key.startswith("meta_"):
                    meta[key[5:]] = sg[key][:]
            slices.append(DiffractionSlice(id=slc_id, x=x, y=y, e=e, metadata=meta))
    return MeasurementState(id=state_id, slices=slices)


# ---------------------------------------------------------------------------
# Private write helpers
# ---------------------------------------------------------------------------


def _write_histograms(entry: h5py.Group, histograms: tuple[Histogram, ...]) -> None:
    hgrp = entry.require_group("histograms")
    for h in histograms:
        hg = hgrp.require_group(h.id)
        units = _ABSCISSA_UNITS[h.data_type]
        x_ds = hg.create_dataset("x", data=h.x)
        x_ds.attrs["units"] = units
        x_ds.attrs["long_name"] = _abscissa_label(h.data_type)
        hg.create_dataset("y_obs", data=h.y_obs).attrs["units"] = "counts"
        hg.create_dataset("weights", data=h.weights).attrs["units"] = "1/counts^2"
        hg.attrs["data_type"] = h.data_type.value
        if h.wavelength is not None:
            hg.attrs["wavelength_A"] = h.wavelength
        if h.difc is not None:
            hg.attrs["difc"] = h.difc
        if h.two_theta_fixed is not None:
            hg.attrs["two_theta_fixed_deg"] = h.two_theta_fixed
        hg.attrs["difa"] = h.difa
        hg.attrs["zero"] = h.zero
        dg = hg.require_group("driving")
        for k, v in h.driving.items():
            dg.create_dataset(k, data=np.float64(v))


def _write_phases(entry: h5py.Group, phases: tuple[Phase, ...]) -> None:
    pgrp = entry.require_group("phases")
    for ph in phases:
        pg = pgrp.require_group(ph.name)
        pg.attrs["space_group"] = ph.space_group
        cg = pg.require_group("cell")
        cell = ph.cell
        for attr in ("a", "b", "c", "alpha", "beta", "gamma"):
            cg.attrs[attr] = getattr(cell, attr)
        ag = pg.require_group("atoms")
        if ph.atoms:
            ag.create_dataset(
                "element",
                data=np.array(
                    [a.element for a in ph.atoms], dtype=h5py.special_dtype(vlen=str)
                ),
            )
            ag.create_dataset("x", data=np.array([a.x for a in ph.atoms]))
            ag.create_dataset("y", data=np.array([a.y for a in ph.atoms]))
            ag.create_dataset("z", data=np.array([a.z for a in ph.atoms]))
            ag.create_dataset("occ", data=np.array([a.occ for a in ph.atoms]))
            ag.create_dataset("b_iso", data=np.array([a.b_iso for a in ph.atoms]))


def _write_parameters(entry: h5py.Group, parameters: tuple[Parameter, ...]) -> None:
    pg = entry.require_group("parameters")
    names = [p.name for p in parameters]
    kinds = [p.kind.value for p in parameters]
    values = np.array([p.value for p in parameters], dtype=np.float64)
    vary = np.array([p.vary for p in parameters], dtype=bool)
    lower = np.array([p.lower for p in parameters], dtype=np.float64)
    upper = np.array([p.upper for p in parameters], dtype=np.float64)
    sigma = np.array(
        [p.sigma if p.sigma is not None else np.nan for p in parameters],
        dtype=np.float64,
    )
    str_dtype = h5py.special_dtype(vlen=str)
    pg.create_dataset("names", data=np.array(names, dtype=str_dtype))
    pg.create_dataset("kinds", data=np.array(kinds, dtype=str_dtype))
    pg.create_dataset("values", data=values)
    pg.create_dataset("vary", data=vary)
    pg.create_dataset("lower", data=lower)
    pg.create_dataset("upper", data=upper)
    pg.create_dataset("sigma", data=sigma)


# ---------------------------------------------------------------------------
# Private read helpers
# ---------------------------------------------------------------------------


def _check_version(f: h5py.File) -> None:
    ver = f.attrs.get("aura_version", "unknown")
    if ver != AURA_FORMAT_VERSION:
        warnings.warn(
            f"HDF5 file was written by aura format {ver!r}; "
            f"current is {AURA_FORMAT_VERSION!r}. Loading may succeed.",
            stacklevel=3,
        )


def _read_histograms(entry: h5py.Group) -> tuple[Histogram, ...]:
    hists = []
    for hid, hg in entry["histograms"].items():
        _check_units(hg["x"], hid)
        data_type = DataType(_decode(hg.attrs["data_type"]))
        x = hg["x"][:]
        y_obs = hg["y_obs"][:]
        weights = hg["weights"][:]
        wavelength = (
            float(hg.attrs["wavelength_A"]) if "wavelength_A" in hg.attrs else None
        )
        difc = float(hg.attrs["difc"]) if "difc" in hg.attrs else None
        two_theta_fixed = (
            float(hg.attrs["two_theta_fixed_deg"])
            if "two_theta_fixed_deg" in hg.attrs
            else None
        )
        difa = float(hg.attrs.get("difa", 0.0))
        zero = float(hg.attrs.get("zero", 0.0))
        driving: dict[str, float] = {}
        if "driving" in hg:
            for k, ds in hg["driving"].items():
                driving[k] = float(ds[()])
        hists.append(
            Histogram(
                id=hid,
                data_type=data_type,
                x=x,
                y_obs=y_obs,
                weights=weights,
                driving=driving,
                wavelength=wavelength,
                difc=difc,
                two_theta_fixed=two_theta_fixed,
                difa=difa,
                zero=zero,
            )
        )
    return tuple(hists)


def _read_phases(entry: h5py.Group) -> tuple[Phase, ...]:
    phases = []
    for pname, pg in entry["phases"].items():
        space_group = _decode(pg.attrs["space_group"])
        cg = pg["cell"]
        cell = UnitCell(
            a=float(cg.attrs["a"]),
            b=float(cg.attrs["b"]),
            c=float(cg.attrs["c"]),
            alpha=float(cg.attrs["alpha"]),
            beta=float(cg.attrs["beta"]),
            gamma=float(cg.attrs["gamma"]),
        )
        atoms: tuple[AtomSite, ...] = ()
        if "atoms" in pg:
            ag = pg["atoms"]
            if "element" in ag:
                elements = [_decode(e) for e in ag["element"][:]]
                xs = ag["x"][:]
                ys = ag["y"][:]
                zs = ag["z"][:]
                occs = ag["occ"][:]
                bisos = ag["b_iso"][:]
                atoms = tuple(
                    AtomSite(
                        element=elements[i],
                        x=float(xs[i]),
                        y=float(ys[i]),
                        z=float(zs[i]),
                        occ=float(occs[i]),
                        b_iso=float(bisos[i]),
                    )
                    for i in range(len(elements))
                )
        phases.append(
            Phase(name=pname, space_group=space_group, cell=cell, atoms=atoms)
        )
    return tuple(phases)


def _read_parameters(entry: h5py.Group) -> tuple[Parameter, ...]:
    pg = entry["parameters"]
    names = [_decode(n) for n in pg["names"][:]]
    kinds = [ParamKind(_decode(k)) for k in pg["kinds"][:]]
    values = pg["values"][:]
    vary = pg["vary"][:].astype(bool)
    lower = pg["lower"][:]
    upper = pg["upper"][:]
    sigma_arr = pg["sigma"][:]
    params = []
    for i, name in enumerate(names):
        sig = None if np.isnan(sigma_arr[i]) else float(sigma_arr[i])
        params.append(
            Parameter(
                name=name,
                kind=kinds[i],
                value=float(values[i]),
                vary=bool(vary[i]),
                lower=float(lower[i]),
                upper=float(upper[i]),
                sigma=sig,
            )
        )
    return tuple(params)


def _decode(val: object) -> str:
    """Decode h5py variable-length string (bytes or str) to str."""
    if isinstance(val, bytes):
        return val.decode()
    return str(val)


def _check_units(ds: h5py.Dataset, hist_id: str) -> None:
    """Raise if the abscissa units attribute is present but unrecognized."""
    allowed = set(_ABSCISSA_UNITS.values())
    units_raw = ds.attrs.get("units", None)
    units = _decode(units_raw) if units_raw is not None else None
    if units is not None and units not in allowed:
        raise ValueError(
            f"Histogram '{hist_id}': unexpected abscissa units {units!r}; "
            f"expected one of {sorted(allowed)}."
        )


def _abscissa_label(dt: DataType) -> str:
    return {
        DataType.CW_XRAY: "2theta (deg)",
        DataType.CW_NEUTRON: "2theta (deg)",
        DataType.TOF: "time-of-flight (us)",
        DataType.EDD: "energy (keV)",
    }[dt]
