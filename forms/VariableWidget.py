# -*- coding: utf-8 -*-
"""
/***************************************************************************
                                 A QGIS plugin
 SpatialAnalyzer
                              -------------------
        git sha              : $Format:%H$
        copyright            : (C) 2017 by D.J Paek
        email                : dj dot paek1 at gmail dot com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'D.J Paek'
__date__ = 'March 2019'
__copyright__ = '(C) 2019, D.J Paek'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from processing.gui.wrappers import WidgetWrapper, DIALOG_STANDARD
from processing.tools import dataobjects
from qgis.PyQt.QtCore import pyqtSignal, Qt
from qgis.PyQt.QtWidgets import QMenu, QListWidgetItem
import os
from qgis.PyQt import uic
from qgis.core import QgsProject, QgsProcessingUtils, QgsMessageLog
from qgis.core import (
    QgsProject,
    QgsProcessingUtils,
    QgsMessageLog,
    QgsMapLayer,
    QgsWkbTypes,
)

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'VariableWidget.ui'))


class VariableWidget(BASE, WIDGET):

    hasChanged = pyqtSignal()

    def __init__(self):
        super(VariableWidget, self).__init__(None)
        self.setupUi(self)
        self.attrs = []
        self.v_type = 'geom'
        self.normalized = False
        self.allow_attribute_selection = True
        self.buttonGroup.buttonClicked.connect(self.setVariables)
        self.fieldLW.itemChanged.connect(self.changeAttributes)
        self.checkBox.stateChanged.connect(self.is_normalized)
        self.fieldLW.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fieldLW.customContextMenuRequested.connect(self.showContextMenu)

    def setSource(self, source, layer=None):
        self.fieldLW.clear()
        self.attrs = []
        self.v_type = 'geom'
        if source is None:
            self.source = None
            self.allow_attribute_selection = True
            self.geom.setEnabled(True)
            self.geom.setChecked(True)
            self.attrsRB.setChecked(False)
            self.attrsRB.setEnabled(False)
            self.fieldLW.setEnabled(False)
            self.hasChanged.emit()
            return

        self.source = source
        has_geometry = self._has_geometry(source, layer)
        is_tabular = self._is_tabular_layer(layer, source)
        if not has_geometry and not is_tabular and layer is not None:
            try:
                is_tabular = layer.fields().count() > 0
            except Exception:
                is_tabular = False

        fields = self._collect_fields(source, layer)
        if not fields:
            self.allow_attribute_selection = False
            self.geom.setEnabled(has_geometry)
            if has_geometry:
                self.geom.setChecked(True)
                self.attrsRB.setChecked(False)
            else:
                self.geom.setChecked(False)
                self.attrsRB.setChecked(False)
            self.fieldLW.setEnabled(False)
            self.attrsRB.setEnabled(False)
            self.hasChanged.emit()
            return

        if not has_geometry and is_tabular:
            candidate_fields = [f.name() for f in fields]
        else:
            candidate_fields = [f.name() for f in fields if f.isNumeric()]
        for f in candidate_fields:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.fieldLW.addItem(item)
        if has_geometry:
            self.allow_attribute_selection = True
        else:
            self.allow_attribute_selection = self.fieldLW.count() > 0

        self.geom.setEnabled(has_geometry)
        if has_geometry:
            self.geom.setChecked(True)
            self.attrsRB.setChecked(False)
        else:
            self.geom.setChecked(False)
            self.attrsRB.setChecked(True)

        enable_fields = False
        if not has_geometry and self.allow_attribute_selection:
            enable_fields = True
        self.fieldLW.setEnabled(enable_fields)
        self.attrsRB.setEnabled(self.allow_attribute_selection and self.fieldLW.count() > 0)
        self.setVariables()

    def setVariables(self):
        if self.geom.isChecked():
            self.v_type = 'geom'
            self.attrs = []
            self.fieldLW.setEnabled(False)
        else:
            self.v_type = 'attrs'
            self.attrs = self.checkedItems()
            enable_fields = self.allow_attribute_selection and self.fieldLW.count() > 0
            self.fieldLW.setEnabled(enable_fields)
            if not enable_fields:
                self.attrsRB.setChecked(False)
        self.hasChanged.emit()

    def changeAttributes(self):
        self.attrs = self.checkedItems()
        self.hasChanged.emit()

    def showContextMenu(self, pos):
        menu = QMenu(self)
        menu.addAction(self.tr('Select All'), self.selectAll)
        menu.addAction(self.tr('Clear Selection'), self.clearSelection)
        menu.addAction(self.tr('Toggle Selection'), self.toggleSelection)
        menu.exec_(self.fieldLW.mapToGlobal(pos))

    def selectAll(self):
        for i in range(self.fieldLW.count()):
            item = self.fieldLW.item(i)
            item.setCheckState(Qt.Checked)
        self.changeAttributes()

    def clearSelection(self):
        for i in range(self.fieldLW.count()):
            item = self.fieldLW.item(i)
            item.setCheckState(Qt.Unchecked)
        self.changeAttributes()

    def toggleSelection(self):
        for i in range(self.fieldLW.count()):
            item = self.fieldLW.item(i)
            state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(state)
        self.changeAttributes()

    def is_normalized(self):
        self.normalized = True if self.checkBox.isChecked() else False
        self.hasChanged.emit()

    def setValue(self, value):
        return True

    def value(self):
        v = [self.v_type, self.attrs, self.normalized]
        return v

    def checkedItems(self):
        items = []
        for i in range(self.fieldLW.count()):
            item = self.fieldLW.item(i)
            if item.checkState() == Qt.Checked:
                items.append(item.text())
        return items

    def _collect_fields(self, source, layer=None):
        for obj in (source, layer):
            if obj is None:
                continue
            if hasattr(obj, 'fields'):
                try:
                    fields = obj.fields()
                except Exception:
                    continue
                if getattr(fields, 'count', None):
                    try:
                        if fields.count() > 0:
                            return fields
                    except Exception:
                        pass
                else:
                    # QgsFields behaves like iterable even without count()
                    try:
                        if len(fields) > 0:
                            return fields
                    except Exception:
                        pass
        return None

    def _has_geometry(self, source, layer=None):
        for obj in (layer, source):
            if obj is None:
                continue
            if hasattr(obj, 'hasGeometry'):
                try:
                    return bool(obj.hasGeometry())
                except TypeError:
                    pass
            if hasattr(obj, 'wkbType'):
                try:
                    wkb = obj.wkbType()
                except TypeError:
                    continue
                if wkb != QgsWkbTypes.Unknown and wkb != QgsWkbTypes.NoGeometry:
                    return True
                if wkb == QgsWkbTypes.NoGeometry:
                    return False
        return False

    def _is_tabular_layer(self, layer, source=None):
        provider = ''
        source_str = ''

        for obj in (layer, source):
            if obj is None:
                continue
            if not provider and hasattr(obj, 'providerType'):
                try:
                    provider = obj.providerType() or ''
                except Exception:
                    provider = ''
            if not source_str and hasattr(obj, 'source'):
                attr = getattr(obj, 'source')
                try:
                    source_str = attr() if callable(attr) else attr
                except Exception:
                    source_str = ''
            if not source_str and hasattr(obj, 'sourceName'):
                try:
                    source_str = obj.sourceName()
                except Exception:
                    source_str = ''

        if not isinstance(source_str, str):
            source_str = ''

        base_path = source_str.split('|')[0]
        base_path = base_path.split('?')[0]

        _, extension = os.path.splitext(base_path.lower())

        tabular_exts = {'.csv', '.txt', '.tsv', '.xlsx', '.xls'}
        tabular_providers = {'delimitedtext', 'spreadsheet'}

        if extension in tabular_exts:
            return True

        if provider in tabular_providers:
            return True
        if provider == 'ogr' and extension in tabular_exts:
            return True
        return False

class VariableWidgetWrapper(WidgetWrapper):

    def __init__(self, param, dialog, row=3, col=0, **kwargs):
        super().__init__(param, dialog, row, col, **kwargs)
        self.context = dataobjects.createContext()

    def createWidget(self):
        if self.dialogType == DIALOG_STANDARD:
            widget = VariableWidget()
            widget.hasChanged.connect(lambda: self.widgetValueHasChanged.emit(self))
            return widget

    def postInitialize(self, wrappers):
        if self.dialogType != DIALOG_STANDARD:
            return
        for wrapper in wrappers:
            if wrapper.parameterDefinition().name() == self.param.layer_param:
                self.setSource(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.layerChanged)

    def layerChanged(self, wrapper):
        self.setSource(wrapper.parameterValue())

    def setSource(self, source):
        layer = None
        if isinstance(source, QgsMapLayer):
            layer = source
        elif isinstance(source, str):
            try:
                layer = QgsProcessingUtils.mapLayerFromString(source, self.context)
            except Exception:
                layer = None
        else:
            source_str = ''
            if hasattr(source, 'source'):
                attr = getattr(source, 'source')
                if isinstance(attr, str):
                    source_str = attr
                elif callable(attr):
                    try:
                        source_str = attr()
                    except Exception:
                        source_str = ''
            if source_str:
                try:
                    layer = QgsProcessingUtils.mapLayerFromString(source_str, self.context)
                except Exception:
                    layer = None
        source = QgsProcessingUtils.variantToSource(source, self.context)
        self.widget.setSource(source, layer)

    def setValue(self, value):
        self.widget.setValue(value)
        
    def value(self):
        return self.widget.value()

