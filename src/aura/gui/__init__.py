"""
Aura desktop workbench (PySide6).

The workbench is structured around the *stimulus trajectory*: the ordered
sequence of experiment states (pressure steps, temperature ramp, etc.) that
the parametric engine treats as a single evolving model.

Entry points
------------
``aura gui``  — launch the workbench from the CLI.
:func:`launch` — Python API, returns when the window closes.

Graceful degradation
--------------------
All PySide6 imports are guarded. When PySide6 is not installed:

* ``from aura.gui import CampaignView`` raises :class:`GUINotAvailable` with a
  clear installation message.
* ``aura gui`` prints the installation message and exits non-zero.
* The module itself imports cleanly (no ImportError at the ``aura`` package level).

To install PySide6::

    pip install PySide6

Or add it to your environment::

    pixi add PySide6
"""

from __future__ import annotations

PYSIDE6_AVAILABLE: bool

try:
    from PySide6 import QtCore  # noqa: F401

    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

_INSTALL_MSG = (
    "The Aura GUI workbench requires PySide6, which is not installed.\n"
    "Install it with:\n"
    "    pip install PySide6\n"
    "or:\n"
    "    pixi add PySide6"
)


class GUINotAvailable(ImportError):
    """Raised when PySide6 is not installed and a GUI component is requested."""


def _require_pyside6() -> None:
    if not PYSIDE6_AVAILABLE:
        raise GUINotAvailable(_INSTALL_MSG)


def launch(campaign=None) -> int:
    """Launch the Aura workbench window.

    Args:
        campaign: Optional Campaign to open on startup.

    Returns:
        Application exit code (0 = clean exit).

    Raises:
        GUINotAvailable: If PySide6 is not installed.
    """
    _require_pyside6()
    import sys

    from PySide6.QtWidgets import QApplication  # type: ignore[import]

    from aura.gui.workbench import AuraWorkbench

    app = QApplication.instance() or QApplication(sys.argv)
    w = AuraWorkbench()
    if campaign is not None:
        w.load_campaign(campaign)
    w.show()
    return app.exec()


__all__ = [
    "PYSIDE6_AVAILABLE",
    "GUINotAvailable",
    "launch",
]
