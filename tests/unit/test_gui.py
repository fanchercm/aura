"""
Tests for the GUI workbench module (Phase 11).

Since PySide6 is an optional dep, we only test the import-guard behaviour when
it is absent, and smoke-test the module's public API in all cases.
"""

from __future__ import annotations

import pytest


class TestGUIImportGuard:

    def test_module_imports_without_pyside6(self):
        """aura.gui must be importable even when PySide6 is missing."""
        import aura.gui as gui  # must not raise

        assert hasattr(gui, "PYSIDE6_AVAILABLE")
        assert hasattr(gui, "GUINotAvailable")
        assert hasattr(gui, "launch")

    def test_pyside6_available_is_bool(self):
        import aura.gui as gui

        assert isinstance(gui.PYSIDE6_AVAILABLE, bool)

    def test_launch_raises_when_pyside6_absent(self):
        import aura.gui as gui

        if gui.PYSIDE6_AVAILABLE:
            pytest.skip("PySide6 is installed; cannot test absent-guard path.")
        with pytest.raises(gui.GUINotAvailable, match="PySide6"):
            gui.launch()

    def test_gui_not_available_is_import_error_subclass(self):
        from aura.gui import GUINotAvailable

        assert issubclass(GUINotAvailable, ImportError)

    def test_cli_gui_command_registered(self):
        """The 'gui' sub-command must be registered in the CLI."""
        from click.testing import CliRunner

        from aura.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["gui", "--help"])
        # With PySide6 absent the command still shows help.
        assert "workbench" in result.output.lower() or result.exit_code == 0
