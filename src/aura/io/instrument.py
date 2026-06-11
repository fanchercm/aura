"""
Instrument-parameter readers.

Two formats:

* **GSAS-II ``.instprm``** — ``#Bank N:`` comment headers then ``key:value``
  pairs (``difC``, ``2-theta``, profile coeffs ``sig-0/1/2``, ``alpha``,
  ``beta-0/1``, …). Ported from the original :func:`aura.loaders.load_instprm`.
* **Old GSAS ``.prm``/``.inst``** — fixed-column ``INS`` records: ``HTYPE``
  (e.g. ``PNCR`` = powder neutron constant-wavelength reactor), ``ICONS``
  (CW wavelength, or TOF DIFC/DIFA/ZERO), ``PRCF`` profile coefficients.

Both return a dict keyed by bank number (int). Each bank dict carries the raw
parameters plus normalized keys the bridge consumes: ``wavelength`` (CW) or
``difc``/``difa``/``zero`` (TOF), and ``data_type`` (a :class:`aura.spec.DataType`)
when it can be inferred.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aura.io.registry import registry
from aura.spec import DataType

# ---------------------------------------------------------------------------
# GSAS-II .instprm
# ---------------------------------------------------------------------------


class InstprmReader:
    """Reader for GSAS-II ``.instprm`` instrument-parameter files."""

    name = "GSAS-II instprm"
    domain = "instrument"
    extensions = (".instprm",)

    def contents_validator(self, path: Path) -> bool:
        head = path.read_text(errors="ignore")[:4096]
        return (
            "#Bank" in head or "# Bank" in head or head.lstrip().startswith("#GSAS-II")
        )

    def read(self, path: Path) -> dict[int, dict[str, Any]]:
        banks: dict[int, dict[str, Any]] = {}
        current: dict[str, Any] | None = None
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#Bank") or line.startswith("# Bank"):
                m = re.search(r"Bank\s+(\d+)", line)
                if m:
                    current = {}
                    banks[int(m.group(1))] = current
                continue
            if current is None:
                continue
            if ":" in line:
                key, _, raw = line.partition(":")
                key, raw = key.strip(), raw.strip()
                try:
                    current[key] = float(raw)
                except ValueError:
                    current[key] = raw
        for bank in banks.values():
            _normalize_instprm(bank)
        return banks


def _normalize_instprm(bank: dict[str, Any]) -> None:
    """Add normalized ``difc``/``wavelength``/``data_type`` keys to a bank."""
    if "difC" in bank:
        bank.setdefault("difc", bank["difC"])
        bank.setdefault("difa", bank.get("difA", 0.0))
        bank.setdefault("zero", bank.get("Zero", 0.0))
        bank.setdefault("data_type", DataType.TOF)
    elif "Lam" in bank or "Lam1" in bank:
        bank.setdefault("wavelength", bank.get("Lam", bank.get("Lam1")))
        bank.setdefault("data_type", DataType.CW_XRAY)


# ---------------------------------------------------------------------------
# Old GSAS .prm / .inst
# ---------------------------------------------------------------------------


class PrmReader:
    """Reader for old-style GSAS ``.prm``/``.inst`` instrument files."""

    name = "GSAS prm"
    domain = "instrument"
    extensions = (".prm", ".inst")

    def contents_validator(self, path: Path) -> bool:
        head = path.read_text(errors="ignore")[:2048]
        return bool(re.search(r"^INS\s", head, re.MULTILINE))

    def read(self, path: Path) -> dict[int, dict[str, Any]]:
        banks: dict[int, dict[str, Any]] = {}

        def bank(n: int) -> dict[str, Any]:
            return banks.setdefault(n, {})

        for line in path.read_text().splitlines():
            if not line.startswith("INS"):
                continue
            # Fixed columns: 1-3 "INS", 4 blank, 5-6 bank number (blank for a
            # bank-independent header), 7+ the parameter key and payload.
            bank_field = line[4:6].strip()
            rest = line[6:]
            m = re.match(r"\s*([A-Za-z0-9]+)\s+(.*)", rest)
            if not m:
                continue
            key, payload = m.group(1), m.group(2).strip()
            bnum = int(bank_field) if bank_field.isdigit() else 0

            if key == "HTYPE":
                bank(bnum or 1)["htype"] = payload.split()[0] if payload else ""
            elif key == "ICONS":
                vals = _floats(payload)
                b = bank(bnum or 1)
                b["icons"] = vals
                # CW: ICONS[0] = wavelength. TOF: ICONS = DIFC, DIFA, ZERO.
                b["wavelength"] = vals[0] if vals else None
            elif key.startswith("PRCF"):
                bank(bnum or 1).setdefault("prcf", []).extend(_floats(payload))

        # Drop the bank-0 header bucket into real banks; infer data_type.
        header = banks.pop(0, {})
        for b in banks.values():
            for k, v in header.items():
                b.setdefault(k, v)
            _infer_prm_data_type(b)
        if not banks and header:
            _infer_prm_data_type(header)
            banks[1] = header
        return banks


def _infer_prm_data_type(bank: dict[str, Any]) -> None:
    htype = str(bank.get("htype", "")).upper()
    if (
        htype.startswith("PNC")
        or htype.startswith("RNC")
        or "XC" in htype
        or htype.startswith("PXC")
    ):
        # *NC* = constant wavelength; neutron (PNC) vs X-ray (PXC).
        bank["data_type"] = (
            DataType.CW_NEUTRON if "N" in htype[:3] else DataType.CW_XRAY
        )
    elif htype.startswith("PNT") or htype.startswith("RNT") or "T" in htype[2:3]:
        bank["data_type"] = DataType.TOF


def _floats(text: str) -> list[float]:
    out: list[float] = []
    for tok in text.split():
        try:
            out.append(float(tok))
        except ValueError:
            break
    return out


# ---------------------------------------------------------------------------
# Applying instrument parameters to histograms
# ---------------------------------------------------------------------------


def apply_instprm(state, instprm: dict[int, dict[str, Any]]):
    """Attach instrument parameters to each slice of a ``MeasurementState``.

    Matches each slice to a bank (by ``metadata["bank_num"]`` if present, else
    by order) and merges the bank's parameters into ``slice.metadata`` under an
    ``instprm`` key plus normalized top-level keys (``difc``/``wavelength``/
    ``data_type``) the bridge reads. Returns *state* (mutated in place — the
    container layer is mutable; the spec layer downstream is not).
    """
    bank_ids = sorted(instprm)
    for i, slc in enumerate(state.slices):
        bnum = slc.metadata.get("bank_num")
        if bnum not in instprm:
            bnum = (
                bank_ids[i]
                if i < len(bank_ids)
                else (bank_ids[0] if bank_ids else None)
            )
        if bnum is None:
            continue
        bank = instprm[bnum]
        slc.metadata["instprm"] = bank
        for key in ("difc", "difa", "zero", "wavelength", "data_type"):
            if key in bank and key not in slc.metadata:
                slc.metadata[key] = bank[key]
    return state


registry.register(InstprmReader())
registry.register(PrmReader())
