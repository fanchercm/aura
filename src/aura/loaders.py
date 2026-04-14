"""
Loaders for GSAS-format diffraction data and instrument parameters.

Provides functions to read GSAS column files (``.gsa``) and
instrument parameter files (``.instprm``) into Aura data-model
objects.

Example:
    >>> from aura.loaders import load_gsa_file, load_campaign_from_directory
    >>> state = load_gsa_file("SNAP067702_column.gsa")
    >>> campaign = load_campaign_from_directory("tests/testDataGsas")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from aura.models import Campaign, DiffractionSlice, MeasurementState


# ---------------------------------------------------------------------------
# .gsa (GSAS column) loader
# ---------------------------------------------------------------------------

# Regex for the comment line with metadata:
#   # Total flight path 15.5679m, tth 122.296deg, DIFC 6890.56
_META_RE = re.compile(
    r"Total flight path\s+(?P<fltpath>[\d.]+)m,\s+"
    r"tth\s+(?P<tth>[\d.]+)deg,\s+"
    r"DIFC\s+(?P<difc>[\d.]+)"
)

# Regex for the BANK header line:
#   BANK 1 3386 3386 SLOG  3075  14254  0.0004531 0 FXYE
_BANK_RE = re.compile(r"^BANK\s+(?P<bank>\d+)\s+(?P<nchan>\d+)\s+")

# Regex for the spectrum comment line:
#   # Data for spectrum :0
_SPECTRUM_RE = re.compile(r"Data for spectrum\s*:(?P<idx>\d+)")


def _parse_gsa_header(lines: list[str]) -> dict[str, Any]:
    """Parse the JSON-ish header block at the top of a .gsa file.

    The header spans from the first line to just before the first
    ``#`` comment line.  It is a brace-delimited block of key-value
    pairs (one per line, colon-separated), but isn't strictly valid
    JSON because individual lines aren't comma-separated.  We fix
    that up before parsing.

    Args:
        lines: All lines of the file.

    Returns:
        Dictionary of header fields.
    """
    header_lines: list[str] = []
    for line in lines:
        if line.startswith("#") or line.startswith("BANK"):
            break
        header_lines.append(line)

    if not header_lines:
        return {}

    # Join and attempt to make it valid JSON by adding commas between
    # lines that don't already have them.
    raw = "\n".join(header_lines)
    # Add commas between `"value"\n"key"` pairs that lack them.
    raw = re.sub(r'([}\]"\w])\s*\n\s*"', r'\1,\n"', raw)
    try:
        return dict(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        return {"_raw_header": raw}


def load_gsa_file(path: str | Path) -> MeasurementState:
    """Load a single GSAS column file into a :class:`MeasurementState`.

    Each bank in the file becomes a :class:`DiffractionSlice`.

    Args:
        path: Path to the ``.gsa`` file.

    Returns:
        A :class:`MeasurementState` containing one slice per bank.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file contains no BANK data.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GSA file not found: {path}")

    text = path.read_text()
    lines = text.splitlines()

    file_header = _parse_gsa_header(lines)

    # Extract run number from filename  e.g. SNAP067702_column.gsa → 67702
    run_match = re.search(r"(\d{5,})", path.stem)
    run_number = int(run_match.group(1)) if run_match else 0
    state_id = f"SNAP{run_number:06d}"

    # Walk through the file collecting banks
    slices: list[DiffractionSlice] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for the metadata comment line preceding each bank
        meta_match = _META_RE.search(line)
        if meta_match:
            flight_path = float(meta_match.group("fltpath"))
            two_theta = float(meta_match.group("tth"))
            difc = float(meta_match.group("difc"))

            # Next line should be the spectrum comment
            i += 1
            spectrum_idx = 0
            if i < len(lines):
                spec_match = _SPECTRUM_RE.search(lines[i])
                if spec_match:
                    spectrum_idx = int(spec_match.group("idx"))

            # Next line should be the BANK header
            i += 1
            if i >= len(lines):
                break
            bank_match = _BANK_RE.match(lines[i])
            if not bank_match:
                continue
            bank_num = int(bank_match.group("bank"))
            n_chan = int(bank_match.group("nchan"))

            # Read n_chan data lines
            x_vals: list[float] = []
            y_vals: list[float] = []
            e_vals: list[float] = []
            for _ in range(n_chan):
                i += 1
                if i >= len(lines):
                    break
                parts = lines[i].split()
                if len(parts) >= 3:
                    x_vals.append(float(parts[0]))
                    y_vals.append(float(parts[1]))
                    e_vals.append(float(parts[2]))

            slice_id = f"{state_id}_bank{bank_num}"
            slices.append(
                DiffractionSlice(
                    id=slice_id,
                    x=np.array(x_vals),
                    y=np.array(y_vals),
                    e=np.array(e_vals),
                    metadata={
                        "bank_num": bank_num,
                        "spectrum_index": spectrum_idx,
                        "two_theta": two_theta,
                        "flight_path": flight_path,
                        "difc": difc,
                    },
                )
            )
        i += 1

    if not slices:
        raise ValueError(f"No BANK data found in {path}")

    return MeasurementState(
        id=state_id,
        slices=slices,
        metadata={
            "run_number": run_number,
            "source_file": str(path.name),
            **file_header,
        },
    )


# ---------------------------------------------------------------------------
# .instprm loader
# ---------------------------------------------------------------------------

def load_instprm(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load a GSAS-II instrument parameter file.

    Args:
        path: Path to the ``.instprm`` file.

    Returns:
        Dictionary keyed by bank number (int), where each value is a
        dict of parameter name → value.  Numeric strings are
        converted to ``float``.

    Raises:
        FileNotFoundError: If *path* does not exist.

    Example:
        >>> params = load_instprm("SNAP066787_column.instprm")
        >>> params[1]["difC"]
        6877.054
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Instrument parameter file not found: {path}")

    banks: dict[int, dict[str, Any]] = {}
    current_bank: dict[str, Any] | None = None

    for line in path.read_text().splitlines():
        line = line.strip()

        # Bank header comment: #Bank 1: ...
        if line.startswith("#Bank") or line.startswith("# Bank"):
            bank_match = re.search(r"Bank\s+(\d+)", line)
            if bank_match:
                current_bank = {}
                banks[int(bank_match.group(1))] = current_bank
            continue

        if current_bank is None:
            continue

        # Key:value pair
        if ":" in line:
            key, _, raw_val = line.partition(":")
            key = key.strip()
            raw_val = raw_val.strip()
            try:
                value: Any = float(raw_val)
            except ValueError:
                value = raw_val
            current_bank[key] = value

    return banks


# ---------------------------------------------------------------------------
# Directory-level loader → Campaign
# ---------------------------------------------------------------------------

def load_campaign_from_directory(
    directory: str | Path,
    campaign_id: str | None = None,
    gsa_glob: str = "*.gsa",
) -> Campaign:
    """Load all ``.gsa`` files in a directory into a :class:`Campaign`.

    Files are sorted by name so that run-number order is preserved.

    Args:
        directory: Path to the folder containing ``.gsa`` files.
        campaign_id: Optional campaign identifier.  Defaults to the
            directory name.
        gsa_glob: Glob pattern for matching GSA files.

    Returns:
        A fully populated :class:`Campaign`.

    Raises:
        FileNotFoundError: If *directory* does not exist.
        ValueError: If no ``.gsa`` files are found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    gsa_files = sorted(directory.glob(gsa_glob))
    if not gsa_files:
        raise ValueError(f"No files matching {gsa_glob!r} in {directory}")

    if campaign_id is None:
        campaign_id = directory.name

    states = [load_gsa_file(f) for f in gsa_files]

    return Campaign(id=campaign_id, states=states)
