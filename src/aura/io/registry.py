"""
Reader Protocol and the central reader registry.

A :class:`Reader` knows how to turn one file format in one *domain* into an
Aura object. Readers self-register with the module-level :data:`registry`; the
registry selects a reader for a path by extension *and* a content check
(``contents_validator``), so ambiguous extensions (e.g. ``.cif`` is both a
powder and a phase format) resolve correctly.

This mirrors GSAS-II's reader-class selection without depending on GSAS-II.
Third-party packages can extend it two ways:

1. Call :meth:`ReaderRegistry.register` at import time.
2. Advertise an entry point in the ``aura.readers`` group; call
   :func:`load_entry_point_readers` to load them (the future plugin SDK).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

Domain = Literal["powder", "phase", "instrument", "image", "sfact"]


class UnsupportedFormatError(RuntimeError):
    """Raised when no registered reader can handle a path for a domain."""


@runtime_checkable
class Reader(Protocol):
    """Contract every format reader implements.

    Attributes:
        name: Short human-readable format name (e.g. ``"GSAS CONST"``).
        domain: Which importer domain this reader serves.
        extensions: Lower-case file extensions (with leading dot) this reader
            claims, e.g. ``(".gsa", ".gss")``. Used as a fast pre-filter only;
            final selection is by :meth:`contents_validator`.
    """

    name: str
    domain: Domain
    extensions: tuple[str, ...]

    def contents_validator(self, path: Path) -> bool:
        """Return True if this reader can parse *path* (sniff the content).

        Must be cheap and side-effect-free: read only as much as needed to
        decide. Returning False lets the registry try the next candidate.
        """
        ...

    def read(self, path: Path):
        """Parse *path* and return the domain object (e.g. a MeasurementState,
        Phase, or instrument dict). May raise on malformed but matching files.
        """
        ...


class ReaderRegistry:
    """Holds readers and selects one for a (path, domain) pair."""

    def __init__(self) -> None:
        self._readers: list[Reader] = []

    def register(self, reader: Reader) -> Reader:
        """Register *reader*. Returns it, so it can be used as a decorator."""
        if not isinstance(reader, Reader):
            raise TypeError(
                f"{reader!r} does not satisfy the Reader protocol "
                "(needs name, domain, extensions, contents_validator, read)."
            )
        self._readers.append(reader)
        return reader

    def readers(self, domain: Domain | None = None) -> list[Reader]:
        """All registered readers, optionally filtered to one domain."""
        if domain is None:
            return list(self._readers)
        return [r for r in self._readers if r.domain == domain]

    def candidates(self, path: Path, domain: Domain) -> list[Reader]:
        """Readers in *domain* whose extension matches *path* (pre-filter)."""
        ext = path.suffix.lower()
        return [r for r in self.readers(domain) if ext in r.extensions]

    def find(self, path: str | Path, domain: Domain) -> Reader:
        """Select the reader for *path* in *domain*.

        Tries extension-matched candidates first (validating content); falls
        back to validating every reader in the domain for extension-less or
        misnamed files.

        Raises:
            FileNotFoundError: If *path* does not exist.
            UnsupportedFormatError: If no reader validates the file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        tried: list[Reader] = []
        for reader in self.candidates(path, domain):
            tried.append(reader)
            if _safe_validate(reader, path):
                return reader
        # Fall back to content sniffing across all domain readers.
        for reader in self.readers(domain):
            if reader in tried:
                continue
            if _safe_validate(reader, path):
                return reader

        ext = path.suffix.lower() or "(none)"
        known = sorted({e for r in self.readers(domain) for e in r.extensions})
        raise UnsupportedFormatError(
            f"No {domain} reader can handle {path.name!r} (extension {ext}). "
            f"Known {domain} extensions: {', '.join(known) or '—'}."
        )

    def read(self, path: str | Path, domain: Domain):
        """Find the reader for *path* in *domain* and parse it."""
        return self.find(path, domain).read(Path(path))


def _safe_validate(reader: Reader, path: Path) -> bool:
    """Run a reader's content validator, treating read errors as 'no match'."""
    try:
        return bool(reader.contents_validator(path))
    except (OSError, ValueError, UnicodeDecodeError):
        return False


# Module-level singleton the built-in readers register with.
registry = ReaderRegistry()


def find_reader(path: str | Path, domain: Domain) -> Reader:
    """Select a reader for *path* in *domain* using the default registry."""
    return registry.find(path, domain)


def read(path: str | Path, domain: Domain):
    """Read *path* in *domain* using the default registry."""
    return registry.read(path, domain)


def load_entry_point_readers(group: str = "aura.readers") -> int:
    """Load and register readers advertised via the *group* entry point.

    Each entry point must resolve to a zero-argument factory returning a
    :class:`Reader`. Returns the number of readers registered. This is the
    hook the Phase 11 plugin SDK builds on; built-in readers do not rely on it.
    """
    from importlib.metadata import entry_points

    count = 0
    for ep in entry_points(group=group):
        factory = ep.load()
        registry.register(factory())
        count += 1
    return count
