"""
Powder-pattern readers.

Native readers for the powder formats with real test data, plus registered
stubs for the GSAS-II long tail (so coverage gaps are explicit, not silent).

Live readers
------------
* **GSAS column (TOF)** — ``.gsa``/``.gss`` with ``SLOG``/``RALF`` banks
  (e.g. SNAP). Wraps the original :func:`aura.loaders.load_gsa_file`.
* **GSAS CONST** — constant-step raw (``.xra`` STD lab X-ray, ``.cwn`` D1A
  multidetector neutron). Points are packed ``(I2,I6)``: ``I2`` = number of
  detectors contributing, ``I6`` = summed counts ⇒ ``y = counts/ndet``,
  ``esd = sqrt(counts)/ndet``. Abscissa from the BANK record's start/step,
  given in centidegrees.
* **FXYE** — free-format ``X Y E`` with ``X`` in centidegrees (e.g. APS 11-BM).
* **XYE** — generic two/three-column ascii (``.xye``/``.xy``/``.chi``/``.dat``),
  ``X`` in degrees.

Each returns a :class:`aura.models.MeasurementState` (one slice per bank), with
``metadata`` carrying ``data_type`` and (for CW) ``wavelength`` so the bridge
can build the right :class:`aura.spec.Histogram`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from aura.io.registry import registry
from aura.loaders import load_gsa_file
from aura.models import DiffractionSlice, MeasurementState
from aura.spec import DataType

_BANK_RE = re.compile(
    r"^BANK\s+(?P<bank>\d+)\s+(?P<nchan>\d+)\s+(?P<nrec>\d+)\s+(?P<rest>.*)", re.M
)
_WAVELEN_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*A\b", re.I)


def _detect_cw_type(text: str) -> DataType:
    low = text.lower()
    if "neutron" in low:
        return DataType.CW_NEUTRON
    return DataType.CW_XRAY


def _detect_wavelength(text: str, data_type: DataType) -> float | None:
    m = _WAVELEN_RE.search(text)
    if m:
        return float(m.group(1))
    if data_type is DataType.CW_XRAY and re.search(r"cu\s*k", text, re.I):
        return 1.5406  # Cu Kα1
    return None


# ---------------------------------------------------------------------------
# GSAS column (TOF) — wraps the existing tested loader
# ---------------------------------------------------------------------------


class GSASColumnReader:
    """GSAS column TOF files (``.gsa``/``.gss`` with SLOG/RALF banks)."""

    name = "GSAS column (TOF)"
    domain = "powder"
    extensions = (".gsa", ".gss", ".gda")

    def contents_validator(self, path: Path) -> bool:
        head = path.read_text(errors="ignore")[:8192]
        if "BANK" not in head:
            return False
        return (
            "SLOG" in head
            or "RALF" in head
            or "Total flight path" in head
            or "DIFC" in head
        )

    def read(self, path: Path) -> MeasurementState:
        state = load_gsa_file(path)
        for slc in state.slices:
            slc.metadata.setdefault("data_type", DataType.TOF)
        return state


# ---------------------------------------------------------------------------
# GSAS CONST (CW) — (I2,I6) packed points
# ---------------------------------------------------------------------------


class GSASConstReader:
    """GSAS constant-wavelength raw files (``.xra``, ``.cwn``, ``.gsa`` CONST)."""

    name = "GSAS CONST"
    domain = "powder"
    extensions = (".xra", ".cwn", ".gsas", ".raw", ".gsa", ".gss")

    def contents_validator(self, path: Path) -> bool:
        head = path.read_text(errors="ignore")[:8192]
        m = _BANK_RE.search(head)
        return bool(m) and "CONST" in m.group("rest")

    def read(self, path: Path) -> MeasurementState:
        text = path.read_text(errors="ignore")
        lines = text.splitlines()
        title = lines[0] if lines else path.stem
        data_type = _detect_cw_type(title)
        wavelength = _detect_wavelength(title, data_type)

        slices: list[DiffractionSlice] = []
        for m in _BANK_RE.finditer(text):
            bank = int(m.group("bank"))
            nchan = int(m.group("nchan"))
            tokens = m.group("rest").split()
            # tokens: BINTYPE start step ... [FORMAT]
            if not tokens or tokens[0] != "CONST":
                continue
            start_cdeg = float(tokens[1])
            step_cdeg = float(tokens[2])
            x = (start_cdeg + np.arange(nchan) * step_cdeg) / 100.0  # centideg -> deg

            # Data lines follow the BANK line until nchan points are read.
            bank_line_end = text.index("\n", m.start())
            body = text[bank_line_end + 1 :]
            y, e = _parse_packed_i2i6(body, nchan)

            slices.append(
                DiffractionSlice(
                    id=f"{path.stem}_bank{bank}",
                    x=x,
                    y=y,
                    e=e,
                    metadata={
                        "bank_num": bank,
                        "data_type": data_type,
                        "wavelength": wavelength,
                        "source_file": path.name,
                        "title": title.strip(),
                    },
                )
            )
        if not slices:
            raise ValueError(f"No CONST BANK data found in {path}")
        return MeasurementState(
            id=path.stem,
            slices=slices,
            metadata={"source_file": path.name, "title": title.strip()},
        )


def _parse_packed_i2i6(body: str, nchan: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode GSAS (I2,I6) packed points: (ndet, counts) -> (y, esd).

    Each point occupies 8 columns: a 2-char detector count (blank ⇒ 1) and a
    6-char integer count. Intensity is normalized per detector and the esd is
    Poisson on the raw counts: ``y = counts/ndet``, ``e = sqrt(max(counts,1))/ndet``.
    """
    y = np.empty(nchan, dtype=np.float64)
    e = np.empty(nchan, dtype=np.float64)
    got = 0
    for raw_line in body.splitlines():
        line = raw_line.rstrip("\r")
        if got >= nchan:
            break
        if not line.strip() or line.startswith(("BANK", "#")):
            # A blank line or the next BANK ends this bank's data.
            if line.startswith("BANK"):
                break
            continue
        # Walk fixed 8-char fields.
        for i in range(0, len(line), 8):
            if got >= nchan:
                break
            field = line[i : i + 8]
            if len(field) < 8 or not field.strip():
                continue
            ndet_s, counts_s = field[:2].strip(), field[2:].strip()
            try:
                ndet = int(ndet_s) if ndet_s else 1
                counts = float(counts_s) if counts_s else 0.0
            except ValueError:
                continue
            ndet = ndet if ndet > 0 else 1
            y[got] = counts / ndet
            e[got] = math_sqrt(counts) / ndet
            got += 1
    if got != nchan:
        # Trim to what was actually read (robust to truncated files).
        y, e = y[:got], e[:got]
    return y, e


def math_sqrt(v: float) -> float:
    return float(np.sqrt(v if v > 1.0 else 1.0))


# ---------------------------------------------------------------------------
# FXYE — free format, X in centidegrees
# ---------------------------------------------------------------------------


class FXYEReader:
    """GSAS FXYE free-format files (``X`` in centidegrees, e.g. APS 11-BM)."""

    name = "FXYE"
    domain = "powder"
    extensions = (".fxye", ".fxy")

    def contents_validator(self, path: Path) -> bool:
        for line in _data_lines(path, limit=200):
            cols = line.split()
            if len(cols) in (2, 3) and _all_floats(cols):
                return True
        return False

    def read(self, path: Path) -> MeasurementState:
        text = path.read_text(errors="ignore")
        title = text.splitlines()[0].strip() if text else path.stem
        xs, ys, es = [], [], []
        for line in _data_lines(path):
            cols = line.split()
            if len(cols) < 2 or not _all_floats(cols[:3]):
                continue
            xs.append(float(cols[0]) / 100.0)  # centideg -> deg
            ys.append(float(cols[1]))
            es.append(float(cols[2]) if len(cols) >= 3 else math_sqrt(float(cols[1])))
        if not xs:
            raise ValueError(f"No FXYE data rows found in {path}")
        slc = DiffractionSlice(
            id=f"{path.stem}_bank1",
            x=np.array(xs),
            y=np.array(ys),
            e=np.array(es),
            metadata={
                "bank_num": 1,
                "data_type": DataType.CW_XRAY,
                "wavelength": None,
                "source_file": path.name,
                "title": title,
            },
        )
        return MeasurementState(
            id=path.stem,
            slices=[slc],
            metadata={"source_file": path.name, "title": title},
        )


# ---------------------------------------------------------------------------
# Generic XYE — two/three-column ascii, X in degrees
# ---------------------------------------------------------------------------


class XYEReader:
    """Generic two/three-column ascii (``.xye``/``.xy``/``.chi``/``.dat``)."""

    name = "XY(E) ascii"
    domain = "powder"
    extensions = (".xye", ".xy", ".chi", ".qchi", ".dat", ".txt")

    def contents_validator(self, path: Path) -> bool:
        for line in _data_lines(path, limit=50):
            cols = line.split()
            if len(cols) in (2, 3) and _all_floats(cols):
                return True
        return False

    def read(self, path: Path) -> MeasurementState:
        xs, ys, es = [], [], []
        for line in _data_lines(path):
            cols = line.split()
            if len(cols) < 2 or not _all_floats(cols[:3]):
                continue
            xs.append(float(cols[0]))  # degrees as-is
            ys.append(float(cols[1]))
            es.append(float(cols[2]) if len(cols) >= 3 else math_sqrt(float(cols[1])))
        if not xs:
            raise ValueError(f"No XY data rows found in {path}")
        slc = DiffractionSlice(
            id=f"{path.stem}_bank1",
            x=np.array(xs),
            y=np.array(ys),
            e=np.array(es),
            metadata={
                "bank_num": 1,
                "data_type": DataType.CW_XRAY,
                "wavelength": None,
                "source_file": path.name,
            },
        )
        return MeasurementState(
            id=path.stem, slices=[slc], metadata={"source_file": path.name}
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_lines(path: Path, limit: int | None = None):
    """Yield content lines, skipping comments (#, !) and obvious headers."""
    n = 0
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.rstrip("\r").strip()
        if not line or line.startswith(("#", "!", "'")) or line.startswith("BANK"):
            continue
        yield line
        n += 1
        if limit is not None and n >= limit:
            return


def _all_floats(cols) -> bool:
    if not cols:
        return False
    try:
        for c in cols:
            float(c)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Long-tail stubs (registered, explicit NotImplementedError)
# ---------------------------------------------------------------------------


class _StubReader:
    """A registered placeholder so coverage gaps are explicit and testable."""

    domain = "powder"

    def __init__(self, name: str, extensions: tuple[str, ...]):
        self.name = name
        self.extensions = extensions

    def contents_validator(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def read(self, path: Path):
        raise NotImplementedError(
            f"{self.name} reader is not yet implemented "
            f"(extensions {', '.join(self.extensions)}). "
            "Aura targets GSAS-II format parity; this format is on the roadmap."
        )


_STUBS = [
    _StubReader("Bruker RAW", (".raw1", ".brml")),
    _StubReader("PANalytical XRDML", (".xrdml",)),
    _StubReader("Rigaku", (".ras", ".rasx")),
    _StubReader("TOPAS xye", (".xye_topas",)),
    _StubReader("FullProf", (".prf",)),
    _StubReader("pdCIF powder", (".pdcif",)),
]


for _reader in (
    GSASColumnReader(),
    GSASConstReader(),
    FXYEReader(),
    XYEReader(),
    *_STUBS,
):
    registry.register(_reader)
