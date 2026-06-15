"""
Single-crystal structure-factor (Fobs) ingest — domain="sfact".

Reads measured reflection intensities (SHELX ``.hkl`` HKLF4, or a CIF ``_refln``
loop) into a validated :class:`~aura.io.sfact.readers.StructureFactors` table.
This is **represent + validate only**: aura's engine is a powder-profile model,
so it does not refine against single-crystal data. The readers exist for
completeness of the GSAS-II importer parity and for cross-checking.
"""

from __future__ import annotations

from aura.io.sfact.readers import StructureFactors

__all__ = ["StructureFactors"]
