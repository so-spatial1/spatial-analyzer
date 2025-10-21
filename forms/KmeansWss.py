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
import os
import tempfile
import numpy as np
from numbers import Real
from scipy.cluster.vq import kmeans, kmeans2, vq
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.vq import kmeans2

import plotly
import plotly as plt
import plotly.graph_objs as go
from qgis.PyQt import uic
from qgis.core import Qgis, QgsMessageLog, QgsNetworkAccessManager, QgsProcessingUtils
from qgis.gui import QgsMessageBar, QgsCollapsibleGroupBox
from PyQt5.QtCore import QDate
from qgis.PyQt.QtWidgets import QVBoxLayout
from qgis.PyQt.QtWebKit import QWebSettings
from qgis.PyQt.QtWebKitWidgets import QWebView
from qgis.PyQt.QtCore import QUrl, Qt, QCoreApplication
from qgis.PyQt.QtGui import QDesktopServices
from PyQt5.QtWidgets import QMessageBox

from .sklearn_utils import has_sklearn, SKLEARN_INSTALL_MESSAGE

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'KmeansWss.ui'))


class WssWidget(BASE, WIDGET):

    def __init__(self):
        super(WssWidget, self).__init__(None)
        self.setupUi(self)
        
        # load the webview of the plot
        self.wss_webview_layout = QVBoxLayout()
        self.wss_webview_layout.setContentsMargins(0,0,0,0)
        self.wss_panel.setLayout(self.wss_webview_layout)
        self.wss_webview = QWebView()
        self.wss_webview.page().setNetworkAccessManager(QgsNetworkAccessManager.instance())
        wss_webview_settings = self.wss_webview.settings()
        wss_webview_settings.setAttribute(QWebSettings.WebGLEnabled, True)
        wss_webview_settings.setAttribute(QWebSettings.DeveloperExtrasEnabled, True)
        wss_webview_settings.setAttribute(QWebSettings.Accelerated2dCanvasEnabled, True)
        self.wss_webview_layout.addWidget(self.wss_webview)

        # Connect signals
        self.wssBtn.clicked.connect(self.plotView)
        self.browserBtn.clicked.connect(self.browserVeiw)
        self._table_style_injected = False
        self.backend_choice = 'scipy'
        self._sklearn_message_logged = False

    def setSource(self, source):
        if not source:
            return
        self.source = source

    def setOptions(self, options):
        self.options = options

    def setBackend(self, backend_value):
        backend = 'scipy'
        if isinstance(backend_value, str):
            backend = backend_value if backend_value in ('scipy', 'sklearn') else 'scipy'
        else:
            try:
                backend = 'sklearn' if int(backend_value) == 1 else 'scipy'
            except (TypeError, ValueError):
                backend = 'scipy'
        self.backend_choice = backend

    def wss_once(k):
        centroids, labels = kmeans2(features, k, minit='++', iter=100)
        diffs = features - centroids[labels]
        return float(np.sum(np.einsum('ij,ij->i', diffs, diffs)))

    def getWss(self):
        cLayer =  self.source
        to_cluster, variable_fields, normalized = self.options
        maxK = self.maxK.value()

        feat_list = list(cLayer.getFeatures())
        if not feat_list:
            return self.tr('Input layer contains no features.')

        cleaned_rows = []
        if to_cluster == 'geom':
            for feat in feat_list:
                geom = feat.geometry()
                if not geom or geom.isEmpty():
                    continue
                try:
                    point = geom.centroid().asPoint()
                except Exception:  # geometry cannot provide a point centroid
                    continue
                row = [point.x(), point.y()]
                if np.any(~np.isfinite(row)):
                    continue
                cleaned_rows.append(row)
        else:
            if not variable_fields:
                return self.tr('Select at least one field.')
            for feat in feat_list:
                row = []
                valid = True
                for field_name in variable_fields:
                    value = feat[field_name]
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        valid = False
                        break
                    if not np.isfinite(numeric):
                        valid = False
                        break
                    row.append(numeric)
                if valid:
                    cleaned_rows.append(row)

        if not cleaned_rows:
            return self.tr('No valid records available for clustering.')

        features = np.asarray(cleaned_rows, dtype=float)

        if features.shape[0] < 2:
            return self.tr('At least two valid records are required to compute WSS.')

        if maxK > features.shape[0]:
            return self.tr('Clusters should be less than or equal to the valid feature count.')

        if normalized:
            scale = np.std(features, axis=0)
            scale[scale == 0] = 1.0
            features = features / scale

        sklearn_available = has_sklearn()
        backend = self.backend_choice if getattr(self, 'backend_choice', None) else 'scipy'
        if backend == 'sklearn' and not sklearn_available:
            backend = 'scipy'
        if not sklearn_available:
            self._log_missing_sklearn()

        wss = np.asarray(self._compute_wss(features, maxK, backend), dtype=float)

        silhouette_scores = self._compute_silhouette_scores(features, maxK, backend)
        self.silhouette_scores = silhouette_scores
        self.silhouette_table_html = self._build_silhouette_table_html(
            silhouette_scores,
            maxK,
        )
        self._table_style_injected = False

        diff = np.diff(wss)
        diff = np.append(0, -diff)
        diff_ratio = diff / wss[0] * 100
        diff_ratio = [
            self.tr('{value:.1f}%').format(value=i)
            for i in diff_ratio
        ]

        #그래프
        x_range = [i for i in range(1, maxK+1)]
        x_axis = {
            'title': self.tr('Clusters (K)'),
            'dtick': 1,
            'side': 'bottom',
            'color': 'gray',
            'zeroline': True,
            'showline': True,
            'linecolor': '#666666',
            'linewidth': 1,
            'showgrid': False,
            'autorange': True,
            'rangemode': 'nonnegative',
            'visible': True
        }
        y_axis = {
            'type': '-',
            'ticks': '',
            'zeroline': False,
            'showline': False,
            'showgrid': False,
            'autorange': True,
            'rangemode': 'nonnegative',
            'visible': False,
            'mirror':False
        }
        wssHover = [
            self.tr('K = {cluster}<br>WSS = {wss}').format(
                cluster=k,
                wss='{0:,.0f}'.format(w)
            )
            for k, w in zip(x_range, wss)
        ]
        
        trace0 = go.Scatter(
            x=x_range[1:],
            y=wss[1:],
            text=wssHover[1:],
            mode="lines+markers",
            line=dict(color='orange'),
            marker=dict(color='white', size=10, line=dict(color='orange', width=2)),
            textposition='top center',
            hoverinfo='text',
            name=self.tr('WSS')
        )
        trace1 = go.Bar(
            x=x_range[1:],
            y=diff[1:],
            text=diff_ratio[1:],
            hoverinfo='none',
            marker=dict(color='white', line=dict(width=1, color='gray')),
            textposition='outside',
            name=self.tr('ΔWSS÷TSS')
        )

        data = [trace0, trace1]
        layout = {
            'plot_bgcolor': 'white',
            'hovermode': 'closest',
            'hoverlabel': dict(font=dict(color='black')),
            'legend': dict(x=0.8, y=1, borderwidth = 0, orientation='v', traceorder='normal', tracegroupgap = 5, font={'size':12}),
            'showlegend': True,
            'xaxis1': dict(x_axis, **dict(domain=[0.0, 1.0], anchor='y1')),
            'yaxis1': dict(y_axis, **dict(domain=[0.0, 1.0], anchor='x1')),
            'margin': dict(l=0, r=0, t=15, b=45)
        }
        self.fig = {'data' : data, 'layout' : layout}
        return None
        
    def plotView(self):
        msg = self.getWss()
        if msg:
            QMessageBox.information(self, self.tr('Input Error'), msg)
        else:
            plot_path = self._create_plot_file_with_table()
            if plot_path:
                widget_layout = self.wss_webview_layout
                webview =  self.wss_webview
                plot_url = QUrl.fromLocalFile(plot_path)
                webview.load(plot_url)
                widget_layout.addWidget(webview)

    def browserVeiw(self):
        msg = self.getWss()
        if msg:
            QMessageBox.information(self, self.tr('Input Error'), msg)
        else:
            plot_path = self._create_plot_file_with_table()
            if plot_path:
                QDesktopServices.openUrl(QUrl.fromLocalFile(plot_path))

    def setValue(self, value):
        return True

    def value(self):
        return 1

    def _compute_wss(self, features, max_k, backend):
        if backend == 'sklearn':
            return self._compute_wss_sklearn(features, max_k)
        return self._compute_wss_scipy(features, max_k)

    def _compute_wss_scipy(self, features, max_k):
        wss_scores = []
        tolerance = 0.005  # 0.5 % relative change threshold for early stopping
        min_runs = 2
        max_runs = 8

        for k in range(1, max_k + 1):
            if k == 1:
                centroid = np.mean(features, axis=0)
                diffs = features - centroid
                wss_scores.append(float(np.sum(diffs ** 2)))
                continue

            total = 0.0
            prev_mean = None
            runs = 0

            while runs < max_runs:
                runs += 1
                wss_value = self._scipy_single_run_wss(features, k)
                total += wss_value
                mean = total / runs

                if runs >= min_runs and prev_mean is not None:
                    denom = abs(prev_mean) if prev_mean else 1.0
                    relative_change = abs(mean - prev_mean) / denom
                    if relative_change < tolerance:
                        prev_mean = mean
                        break
                prev_mean = mean

            wss_scores.append(float(prev_mean if prev_mean is not None else total))

        return np.asarray(wss_scores, dtype=float)

    def _scipy_single_run_wss(self, features, k):
        codebook = None
        try:
            codebook, _ = kmeans2(features, k, iter=50, minit='++')
        except TypeError:  # pragma: no cover - fallback for older SciPy
            codebook, _ = kmeans2(features, k, iter=50, minit='points')
        except Exception:
            codebook = None

        if codebook is None:
            codebook = kmeans(features, k)[0]

        distortion = vq(features, codebook)[1]
        return float(np.sum(distortion ** 2))

    def _compute_wss_sklearn(self, features, max_k):
        try:
            from sklearn.cluster import KMeans  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            self._log_missing_sklearn()
            return self._compute_wss_scipy(features, max_k)

        scores = []
        use_elkan = features.shape[1] > 1 if features.ndim > 1 else False
        for k in range(1, max_k + 1):
            try:
                init_args = dict(
                    n_clusters=k,
                    init='k-means++',
                    random_state=0,
                    n_init=1,
                    max_iter=150,
                    tol=1e-4,
                )
                if use_elkan:
                    init_args['algorithm'] = 'elkan'
                try:
                    model = KMeans(**init_args)
                except TypeError:
                    init_args.pop('algorithm', None)
                    model = KMeans(**init_args)
                model.fit(features)
                scores.append(float(model.inertia_))
            except Exception:
                return self._compute_wss_scipy(features, max_k)
        return np.asarray(scores, dtype=float)

    def _compute_silhouette_scores(self, features, max_k, backend):
        if backend == 'sklearn':
            return self._compute_silhouette_scores_sklearn(features, max_k)
        return self._compute_silhouette_scores_scipy(features, max_k)

    def _compute_silhouette_scores_scipy(self, features, max_k):
        if max_k < 2 or features.shape[0] <= 1:
            return []
        pairwise = squareform(pdist(features, metric='euclidean'))
        silhouettes = []
        for k in range(2, max_k + 1):
            codebook = kmeans(features, k)[0]
            labels = vq(features, codebook)[0]
            silhouettes.append((k, self._silhouette_for_labels(labels, pairwise)))
        return silhouettes

    def _compute_silhouette_scores_sklearn(self, features, max_k):
        if max_k < 2 or features.shape[0] <= 1:
            return []
        try:
            from sklearn.cluster import KMeans  # type: ignore
            from sklearn.metrics import silhouette_score  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            self._log_missing_sklearn()
            return self._compute_silhouette_scores_scipy(features, max_k)

        silhouettes = []
        for k in range(2, max_k + 1):
            try:
                model = KMeans(n_clusters=k, n_init=20, init='k-means++')
                labels = model.fit_predict(features)
                score = float(silhouette_score(features, labels, metric='euclidean'))
            except Exception:
                score = None
            silhouettes.append((k, score))
        return silhouettes

    def _log_missing_sklearn(self):
        if not getattr(self, '_sklearn_message_logged', False):
            QgsMessageLog.logMessage(
                QCoreApplication.translate('sklearn_utils', SKLEARN_INSTALL_MESSAGE),
                'SpatialAnalysis',
                Qgis.Info
            )
            self._sklearn_message_logged = True

    def _silhouette_for_labels(self, labels, pairwise):
        unique_labels = np.unique(labels)
        n_samples = labels.shape[0]
        silhouette_vals = np.zeros(n_samples, dtype=float)
        for idx in range(n_samples):
            own_label = labels[idx]
            same_cluster = labels == own_label
            same_cluster_count = np.sum(same_cluster) - 1
            if same_cluster_count > 0:
                a = np.sum(pairwise[idx, same_cluster]) / same_cluster_count
            else:
                a = 0.0
            b = np.inf
            for other_label in unique_labels:
                if other_label == own_label:
                    continue
                other_cluster = labels == other_label
                if not np.any(other_cluster):
                    continue
                dist = np.mean(pairwise[idx, other_cluster])
                if dist < b:
                    b = dist
            if not np.isfinite(b):
                silhouette_vals[idx] = 0.0
                continue
            denom = max(a, b)
            silhouette_vals[idx] = 0.0 if denom == 0 else (b - a) / denom
        return float(np.mean(silhouette_vals))

    def _build_silhouette_table_html(self, silhouettes, max_k=None):
        if not silhouettes:
            return ""

        prepared = []
        for entry in silhouettes:
            k_value = None
            score_value = None
            if isinstance(entry, dict):
                k_value = entry.get('k') if 'k' in entry else entry.get('K')
                score_value = (
                    entry.get('score')
                    if 'score' in entry
                    else entry.get('silhouette')
                )
                if score_value is None:
                    score_value = entry.get('silhouette_score')
            else:
                try:
                    k_value, score_value = entry
                except (TypeError, ValueError):
                    continue

            coerced_k = self._coerce_cluster_count(k_value)
            coerced_score = self._coerce_silhouette_score(score_value)
            if coerced_k is None or coerced_k < 2:
                continue
            prepared.append((coerced_k, coerced_score))

        if not prepared:
            return ""

        collapsed = {}
        for k, score in prepared:
            if k not in collapsed or collapsed[k] is None:
                collapsed[k] = score
            elif score is not None:
                collapsed[k] = score

        if max_k is None:
            max_k = None
            try:
                if hasattr(self, 'maxK'):
                    max_widget_value = getattr(self.maxK, 'value', None)
                    if callable(max_widget_value):
                        max_k = max_widget_value()
            except Exception:
                max_k = None

        if max_k is not None:
            try:
                max_k = int(max_k)
            except (TypeError, ValueError):
                max_k = None

        if max_k is not None and max_k < 2:
            max_k = None

        if max_k is not None:
            k_sequence = list(range(2, max_k + 1))
        else:
            available_keys = sorted(k for k in collapsed if k >= 2)
            if not available_keys:
                available_keys = [2]
            if available_keys[0] != 2:
                available_keys = [2] + [k for k in available_keys if k != 2]
            k_sequence = available_keys

        column_entries = []
        for k in k_sequence:
            column_entries.append((k, collapsed.get(k)))

        valid_scores = [score for _, score in column_entries if score is not None]
        best_score = max(valid_scores) if valid_scores else None

        def _scores_match(candidate, best):
            if candidate is None or best is None:
                return False
            try:
                matches = np.isclose(candidate, best)
            except Exception:
                return candidate == best
            if isinstance(matches, np.ndarray):
                return bool(matches.item()) if matches.shape == () else bool(matches.any())
            return bool(matches)

        best_indices = {
            idx for idx, (_, score) in enumerate(column_entries)
            if _scores_match(score, best_score)
        }

        header_cells = []
        value_cells = []
        na_text = self.tr('N/A')
        k_header = self.tr('K')
        silhouette_header = self.tr('Silhouette')
        for idx, (k, score) in enumerate(column_entries):
            style = "font-weight:bold;" if idx in best_indices else ""
            header_cells.append(
                f"<th style='border:1px solid #ccc; padding:6px 8px; {style}'>" +
                f"{k}</th>"
            )
            if score is None:
                display = na_text
            else:
                display = f"{score:.3f}"
            value_cells.append(
                f"<td style='border:1px solid #ccc; padding:6px 8px; {style}'>" +
                f"{display}</td>"
            )
        table_rows = [
            (
                "<tr><th style='border:1px solid #ccc; padding:6px 8px;'>"
                + k_header
                + "</th>"
                + "".join(header_cells)
                + "</tr>"
            ),
            (
                "<tr><th style='border:1px solid #ccc; padding:6px 8px;'>"
                + silhouette_header
                + "</th>"
                + "".join(value_cells)
                + "</tr>"
            ),
        ]
        table_html = (
            "<table class='silhouette-table' "
            "style='width:100%; border-collapse:collapse; margin-bottom:24px; text-align:center; border:1px solid #ccc;'>"
            + "".join(table_rows)
            + "</table>"
        )
        return table_html

    def _coerce_silhouette_score(self, score):
        if score is None:
            return None
        if isinstance(score, Real):
            return float(score)
        if isinstance(score, np.ndarray):
            if score.size == 0:
                return None
            return self._coerce_silhouette_score(score.flat[0])
        if isinstance(score, (list, tuple)):
            for item in score:
                coerced = self._coerce_silhouette_score(item)
                if coerced is not None:
                    return coerced
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    def _coerce_cluster_count(self, value):
        if value is None:
            return None
        if isinstance(value, Real):
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return None
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return self._coerce_cluster_count(value.flat[0])
        if isinstance(value, (list, tuple)):
            for item in value:
                coerced = self._coerce_cluster_count(item)
                if coerced is not None:
                    return coerced
            return None
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return None

    def _create_plot_file_with_table(self):
        try:
            plot_path = os.path.join(tempfile.gettempdir(), 'wss' + '.html')
            plotly.offline.plot(self.fig, filename=plot_path, auto_open=False, include_plotlyjs=True)
            if getattr(self, 'silhouette_table_html', None):
                with open(plot_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                insertion = self._silhouette_table_html_with_style()
                html = html.replace('<body>', '<body>' + insertion, 1)
                with open(plot_path, 'w', encoding='utf-8') as f:
                    f.write(html)
            return plot_path
        except Exception:
            QgsMessageLog.logMessage(
                self.tr('Failed to create silhouette table view.'),
                'SpatialAnalysis',
                Qgis.Warning
            )
            return None

    def _silhouette_table_html_with_style(self):
        style_block = (
            "<style>"
            ".silhouette-table th {background-color:#f7f7f7;}"
            ".silhouette-table td, .silhouette-table th {font-size:14px;}"
            "</style>"
        )
        if not self._table_style_injected:
            self._table_style_injected = True
            return style_block + (self.silhouette_table_html or "")
        return self.silhouette_table_html or ""

class WssWidgetWrapper(WidgetWrapper):

    def __init__(self, param, dialog, row=3, col=0, **kwargs):
        super().__init__(param, dialog, row, col, **kwargs)
        self.context = dataobjects.createContext()

    def _panel(self):
        return WssWidget()

    def createWidget(self):
        if self.dialogType == DIALOG_STANDARD:
            return self._panel()

    def postInitialize(self, wrappers):
        if self.dialogType != DIALOG_STANDARD:
            return
        for wrapper in wrappers:
            if wrapper.parameterDefinition().name() == self.param.layer_param:
                self.setSource(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.layerChanged)
            elif wrapper.parameterDefinition().name() == self.param.variable_options:
                self.setOptions(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.optionsChanged)
            elif getattr(self.param, 'backend_param', None) and \
                    wrapper.parameterDefinition().name() == self.param.backend_param:
                self.setBackend(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.backendChanged)

    def layerChanged(self, wrapper):
        self.setSource(wrapper.parameterValue())

    def setSource(self, source):
        source = QgsProcessingUtils.variantToSource(source, self.context)
        self.widget.setSource(source)

    def backendChanged(self, wrapper):
        self.setBackend(wrapper.parameterValue())

    def setBackend(self, backend):
        self.widget.setBackend(backend)

    def optionsChanged(self, wrapper):
        self.setOptions(wrapper.parameterValue())

    def setOptions(self, options):
        self.widget.setOptions(options)

    def setValue(self, value):
        return self.widget.setValue(value)

    def value(self):
        return self.widget.value()

