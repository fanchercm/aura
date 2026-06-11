"""
Phase (crystal-structure) readers.

The live reader is :class:`CIFReader`, built on **gemmi**. It handles the
GSAS-II-flavored CIFs shipped as test data (``_space_group_name_H-M_alt`` plus
an explicit symop loop, cell values with esd notation like ``5.9738(7)``) and
returns an immutable :class:`aura.spec.Phase` (``UnitCell``, ``AtomSite`` tuple,
Hermann–Mauguin space-group string). Symmetry *expansion* (generating equivalent
reflections/positions) is deferred to the Phase-3 engine — this reader only
parses the asymmetric unit and the declared symmetry.

Stubs for GSAS ``.EXP``, SHELX ``.ins/.res``, PDB, and JANA are registered so
coverage gaps are explicit.
"""

from __future__ import annotations

from pathlib import Path

from aura.io.registry import registry
from aura.spec import AtomSite, Phase, UnitCell

_U_TO_B = 8.0 * 3.141592653589793**2  # B_iso = 8 pi^2 U_iso


class CIFReader:
    """Crystal-structure reader for CIF files (gemmi-backed)."""

    name = "CIF"
    domain = "phase"
    extensions = (".cif", ".mcif")

    def contents_validator(self, path: Path) -> bool:
        head = path.read_text(errors="ignore")[:4096]
        # A *phase* CIF carries cell + atom-site or symmetry tags. (Distinguish
        # from a pdCIF powder block, which carries _pd_meas/_pd_proc data.)
        has_struct = "_cell_length_a" in head and (
            "_atom_site" in head or "_space_group" in head or "_symmetry_space" in head
        )
        is_powder_only = "_pd_meas_" in head and "_atom_site_fract" not in head
        return has_struct and not is_powder_only

    def read(self, path: Path) -> Phase:
        import gemmi

        doc = gemmi.cif.read(str(path))
        block = _structure_block(doc)

        a = _num(block, "_cell_length_a")
        b = _num(block, "_cell_length_b")
        c = _num(block, "_cell_length_c")
        alpha = _num(block, "_cell_angle_alpha", 90.0)
        beta = _num(block, "_cell_angle_beta", 90.0)
        gamma = _num(block, "_cell_angle_gamma", 90.0)
        cell = UnitCell(a, b, c, alpha, beta, gamma)

        sg = (
            block.find_value("_space_group_name_H-M_alt")
            or block.find_value("_symmetry_space_group_name_H-M")
            or block.find_value("_space_group_name_Hall")
            or "P 1"
        )
        sg = sg.strip().strip("'\"")

        name = (
            (
                block.find_value("_pd_phase_name")
                or block.find_value("_chemical_name_mineral")
                or block.name
                or path.stem
            )
            .strip()
            .strip("'\"")
        )

        atoms = _read_atoms(block)
        return Phase(name=name, space_group=sg, cell=cell, atoms=tuple(atoms))


def _structure_block(doc):
    """Pick the block that actually holds a crystal structure."""
    for block in doc:
        if block.find_value("_cell_length_a") is not None:
            return block
    return doc.sole_block()


def _read_atoms(block) -> list[AtomSite]:
    import gemmi

    table = block.find(
        "_atom_site_",
        [
            "label",
            "type_symbol",
            "fract_x",
            "fract_y",
            "fract_z",
            "?occupancy",
            "?B_iso_or_equiv",
            "?U_iso_or_equiv",
        ],
    )
    atoms: list[AtomSite] = []
    for row in table:
        label = row.str(0)
        type_symbol = row.str(1) if row.has(1) and row[1] not in (".", "?") else label
        element = _element(type_symbol or label)
        x = gemmi.cif.as_number(row[2])
        y = gemmi.cif.as_number(row[3])
        z = gemmi.cif.as_number(row[4])
        occ = (
            gemmi.cif.as_number(row[5])
            if row.has(5) and row[5] not in (".", "?")
            else 1.0
        )
        b_iso = 0.5
        if row.has(6) and row[6] not in (".", "?"):
            b_iso = gemmi.cif.as_number(row[6])
        elif row.has(7) and row[7] not in (".", "?"):
            b_iso = gemmi.cif.as_number(row[7]) * _U_TO_B
        atoms.append(AtomSite(element=element, x=x, y=y, z=z, occ=occ, b_iso=b_iso))
    return atoms


def _num(block, tag: str, default: float | None = None) -> float:
    import gemmi

    val = block.find_value(tag)
    if val is None:
        if default is None:
            raise ValueError(f"CIF missing required tag {tag}")
        return default
    return gemmi.cif.as_number(val)


def _element(symbol: str) -> str:
    """Strip charge/oxidation and trailing digits: 'Na1+' -> 'Na', 'O2-' -> 'O'."""
    out = []
    for ch in symbol.strip():
        if ch.isalpha():
            out.append(ch)
        else:
            break
    return "".join(out) or symbol.strip()


class _StubPhaseReader:
    domain = "phase"

    def __init__(self, name: str, extensions: tuple[str, ...]):
        self.name = name
        self.extensions = extensions

    def contents_validator(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def read(self, path: Path):
        raise NotImplementedError(
            f"{self.name} phase reader is not yet implemented "
            f"(extensions {', '.join(self.extensions)}). On the GSAS-II-parity roadmap."
        )


_STUBS = [
    _StubPhaseReader("GSAS EXP", (".exp",)),
    _StubPhaseReader("SHELX", (".ins", ".res")),
    _StubPhaseReader("PDB", (".pdb", ".ent")),
    _StubPhaseReader("JANA", (".m40",)),
]


registry.register(CIFReader())
for _reader in _STUBS:
    registry.register(_reader)
