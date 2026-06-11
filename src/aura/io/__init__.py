"""
Aura I/O subsystem — a native, pluggable importer registry.

Modeled on GSAS-II's reader-class design (without any GSAS-II dependency): each
file format is a :class:`~aura.io.registry.Reader` that self-registers, and
:func:`~aura.io.registry.find_reader` selects one by sniffing file *content*,
not just its extension. The registry is the GSAS-II-parity importer floor and
the seed of the future plugin SDK.

Domains: ``powder`` (1D patterns), ``phase`` (crystal structures),
``instrument`` (instrument-parameter files), ``image`` and ``sfact``
(Phase 10).

Importing this package registers the built-in readers.
"""

from __future__ import annotations

# Importing the reader subpackages triggers their self-registration.
from aura.io import instrument as _instrument  # noqa: F401
from aura.io import phase as _phase  # noqa: F401
from aura.io import powder as _powder  # noqa: F401
from aura.io.registry import (
    Reader,
    ReaderRegistry,
    UnsupportedFormatError,
    find_reader,
    read,
    registry,
)

__all__ = [
    "Reader",
    "ReaderRegistry",
    "UnsupportedFormatError",
    "find_reader",
    "read",
    "registry",
]
