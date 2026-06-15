"""
Aura desktop workbench — PySide6 implementation.

The workbench is organized around three panels:

  CampaignView   — tree listing states in the stimulus trajectory.
                   Selecting a state updates PatternView and ProvenancePanel.
  PatternView    — matplotlib canvas showing the observed vs calculated pattern
                   and difference curve for the selected state.
  ProvenancePanel — key/value table showing the ProvenanceManifest for the
                   selected refinement result.

All of this is guarded behind the ``PYSIDE6_AVAILABLE`` flag in
``aura.gui.__init__``. Import this module only when PySide6 is confirmed
available; the import will fail cleanly if not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.models import Campaign
    from aura.spec import RefinementResult

# PySide6 imports — only reached when PYSIDE6_AVAILABLE is True.
from PySide6.QtCore import Qt  # type: ignore[import]
from PySide6.QtWidgets import (  # type: ignore[import]
    QDockWidget,
    QLabel,
    QMainWindow,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class CampaignView(QTreeWidget):
    """Tree listing measurement states in stimulus order.

    Emits ``itemSelectionChanged`` when the user selects a state.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(["State", "Driving variable"])
        self.setColumnWidth(0, 220)

    def load_campaign(self, campaign: Campaign) -> None:
        """Populate the tree from *campaign*."""
        self.clear()
        for state in campaign.states:
            item = QTreeWidgetItem(
                [state.id, _driving_label(state.metadata.get("driving", {}))]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, state)
            self.addTopLevelItem(item)


class PatternView(QWidget):
    """Matplotlib canvas showing observed vs calculated pattern + difference."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        try:
            from matplotlib.backends.backend_qtagg import (  # type: ignore[import]
                FigureCanvasQTAgg,
            )
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(8, 4), tight_layout=True)
            self._ax_fit = self._fig.add_subplot(2, 1, 1)
            self._ax_diff = self._fig.add_subplot(2, 1, 2)
            self._canvas = FigureCanvasQTAgg(self._fig)
            layout.addWidget(self._canvas)
            self._has_mpl = True
        except ImportError:
            self._has_mpl = False
            layout.addWidget(QLabel("matplotlib required for pattern view."))

    def show_result(self, result: RefinementResult, hist_id: str | None = None) -> None:
        """Display the fit for one histogram from *result*."""
        if not self._has_mpl:
            return
        import numpy as np

        hist = (
            result.state.histograms[0]
            if hist_id is None
            else next(
                (h for h in result.state.histograms if h.id == hist_id),
                result.state.histograms[0],
            )
        )
        # y_calc stored in diagnostics["y_calc"] dict by the engine.
        y_calc = (
            (result.diagnostics or {})
            .get("y_calc", {})
            .get(hist.id, np.zeros_like(hist.y_obs))
        )
        diff = hist.y_obs - y_calc

        self._ax_fit.clear()
        self._ax_diff.clear()
        self._ax_fit.plot(hist.x, hist.y_obs, "k.", ms=1.5, label="obs")
        self._ax_fit.plot(hist.x, y_calc, "r-", lw=1, label="calc")
        self._ax_fit.legend(fontsize=7)
        self._ax_fit.set_ylabel("Intensity")
        self._ax_diff.plot(hist.x, diff, "b-", lw=0.8)
        self._ax_diff.axhline(0, color="k", lw=0.5)
        self._ax_diff.set_ylabel("Difference")
        self._ax_diff.set_xlabel(_abscissa_label(hist.data_type))
        self._canvas.draw()


class ProvenancePanel(QTableWidget):
    """Key/value table showing the ProvenanceManifest."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Field", "Value"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def show_provenance(self, manifest) -> None:
        """Populate from a ProvenanceManifest or plain dict."""
        if manifest is None:
            self.setRowCount(0)
            return
        fields = (
            vars(manifest).items()
            if hasattr(manifest, "__dataclass_fields__")
            else manifest.items()
        )
        rows = list(fields)
        self.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.setItem(i, 0, QTableWidgetItem(str(k)))
            self.setItem(i, 1, QTableWidgetItem(str(v)))
        self.resizeColumnsToContents()


class AuraWorkbench(QMainWindow):
    """Main workbench window.

    Layout::

        ┌────────────────────────────────────────────────┐
        │  Campaign tree │   Pattern view (matplotlib)   │
        │                │                               │
        │                ├───────────────────────────────┤
        │                │   Provenance panel            │
        └────────────────────────────────────────────────┘
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Aura — Parametric Rietveld Workbench")
        self.resize(1200, 700)

        # Central splitter: tree on left, fit+provenance on right.
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        self._campaign_view = CampaignView()
        self._campaign_view.itemSelectionChanged.connect(self._on_state_selected)
        splitter.addWidget(self._campaign_view)

        right = QSplitter(Qt.Orientation.Vertical)
        self._pattern_view = PatternView()
        right.addWidget(self._pattern_view)

        prov_dock = QDockWidget("Provenance", self)
        self._provenance_panel = ProvenancePanel()
        prov_dock.setWidget(self._provenance_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, prov_dock)

        splitter.addWidget(right)
        splitter.setSizes([260, 940])

        self._results: dict[str, object] = {}  # state_id → RefinementResult

    def load_campaign(self, campaign: Campaign) -> None:
        """Populate the workbench from *campaign*."""
        self._campaign_view.load_campaign(campaign)

    def set_result(self, state_id: str, result: object) -> None:
        """Associate a RefinementResult with a state ID for display."""
        self._results[state_id] = result

    def _on_state_selected(self) -> None:
        items = self._campaign_view.selectedItems()
        if not items:
            return
        state = items[0].data(0, Qt.ItemDataRole.UserRole)
        if state is None:
            return
        result = self._results.get(state.id)
        if result is not None:
            self._pattern_view.show_result(result)
            self._provenance_panel.show_provenance(getattr(result, "provenance", None))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _driving_label(driving: dict) -> str:
    if not driving:
        return ""
    return ", ".join(f"{k}={v}" for k, v in driving.items())


def _abscissa_label(data_type) -> str:
    from aura.spec import DataType

    return {
        DataType.CW_XRAY: "2θ (°)",
        DataType.CW_NEUTRON: "2θ (°)",
        DataType.TOF: "TOF (μs)",
        DataType.EDD: "Energy (keV)",
    }.get(data_type, "x")
