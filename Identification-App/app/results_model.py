"""Table model backing the results panel: one row per identified bout."""
from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

COLUMNS = ["File", "Bout", "Animal", "Start (s)", "End (s)"]


class ResultsModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []

    def add_bout_result(self, result):
        row = len(self._rows)
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.append(result)
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self._rows = []
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return COLUMNS[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or not index.isValid():
            return None
        r = self._rows[index.row()]
        col = index.column()
        if col == 0:
            return r["file"]
        if col == 1:
            return r["bout_id"]
        if col == 2:
            return r["animal"]
        if col == 3:
            return f"{r['start']:.2f}"
        if col == 4:
            return f"{r['end']:.2f}"
        return None
