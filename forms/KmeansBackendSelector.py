# -*- coding: utf-8 -*-
"""Custom backend selector widget for the K-Means processing dialog."""

from processing.gui.wrappers import WidgetWrapper, DIALOG_STANDARD
from qgis.PyQt.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
)
from qgis.PyQt.QtCore import Qt

from .sklearn_utils import has_sklearn


class BackendSelectorWidget(QWidget):
    """Widget containing radio buttons to pick the clustering backend."""

    def __init__(self):
        super().__init__(None)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        self.button_group = QButtonGroup(self)

        self.scipy_radio = QRadioButton(self.tr('SciPy'))
        self.sklearn_radio = QRadioButton(self.tr('scikit-learn'))

        self.button_group.addButton(self.scipy_radio, 0)
        self.button_group.addButton(self.sklearn_radio, 1)

        self.scipy_radio.setChecked(True)

        if not has_sklearn():
            self.sklearn_radio.setEnabled(False)
            self.sklearn_radio.setToolTip(self.tr('scikit-learn is not installed.'))

        row.addWidget(self.scipy_radio)
        row.addWidget(self.sklearn_radio)
        row.addStretch()

        layout.addLayout(row)
        self.setLayout(layout)

    def value(self):
        checked_id = self.button_group.checkedId()
        return 0 if checked_id < 0 else checked_id

    def setValue(self, value):
        try:
            idx = int(value)
        except (TypeError, ValueError):
            idx = 0
        button = self.button_group.button(idx)
        if not button or not button.isEnabled():
            self.scipy_radio.setChecked(True)
        else:
            button.setChecked(True)
        return True


class BackendSelectorWrapper(WidgetWrapper):
    """Processing wrapper that exposes the backend selector widget."""

    def __init__(self, param, dialog, row=0, col=0, **kwargs):
        super().__init__(param, dialog, row, col, **kwargs)

    def createWidget(self):
        if self.dialogType == DIALOG_STANDARD:
            widget = BackendSelectorWidget()
            widget.button_group.buttonClicked.connect(
                lambda _btn: self.widgetValueHasChanged.emit(self)
            )
            return widget

    def value(self):
        return self.widget.value()

    def setValue(self, value):
        self.widget.setValue(value)
