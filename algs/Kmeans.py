# -*- coding: utf-8 -*-
"""
/***************************************************************************
                                 A QGIS plugin
SpatialAnalyzer
                              -------------------
        git sha              : $Format:%H$
        copyright            : (C) 2018 by D.J Paek
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

import os
import codecs

from qgis.PyQt.QtCore import QVariant, QUrl
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.PyQt.QtWebKitWidgets import QWebView

from qgis.core import (QgsWkbTypes,
                       QgsFeature,
                       QgsFeatureRequest,
                       QgsGeometry, QgsMessageLog,
                       QgsPointXY,
                       QgsField,
                       QgsFields,
                       QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingUtils,
                       QgsFeatureSink,
                       QgsSymbol,
                       QgsRendererCategory,
                       QgsCategorizedSymbolRenderer,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterFileDestination)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
from spatial_analysis.forms.KmeansWssParam import ParameterWss
from spatial_analysis.forms.VariableParam import ParameterVariable
from spatial_analysis.forms.sklearn_utils import has_sklearn, SKLEARN_INSTALL_MESSAGE
import numpy as np
from scipy.cluster.vq import kmeans2
import geopandas as gpd
from ..locale_utils import localized_menu_text

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Kmeans(QgisAlgorithm):

    INPUT = 'INPUT_LAYER'
    BACKEND = 'BACKEND'
    MINIT = 'MINIT'
    ITER = 'ITER'
    K = 'K'
    V_OPTIONS = 'V_OPTIONS'
    NORMALIZE = 'NORMALIZE'
    WSS = 'WSS'
    OUTPUT = 'OUTPUT'
    OUTPUT_CENTROID = 'OUTPUT_CENTROID'
    OUTPUT_REPORT = 'OUTPUT_REPORT'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'cluster.svg'))

    def group(self):
        return localized_menu_text('Clustering', self.tr('Clustering'))

    def groupId(self):
        return 'clustering'
    
    def name(self):
        return 'kmeans'

    def displayName(self):
        return localized_menu_text('K-Means', self.tr('K-Means'))

    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr(u'Input Layer'),
                                                              [QgsProcessing.TypeVector]))
        backend_param = QgsProcessingParameterEnum(
            self.BACKEND,
            self.tr('Backend'),
            ['SciPy', 'scikit-learn'],
            defaultValue=0
        )
        backend_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.KmeansBackendSelector.BackendSelectorWrapper'}})
        self.addParameter(backend_param)
        variable_param = ParameterVariable(self.V_OPTIONS, self.tr(u'Variable Fields'), layer_param=self.INPUT)
        variable_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.VariableWidget.VariableWidgetWrapper'}})
        self.addParameter(variable_param)
        self.minit_name=[['KMeans++', 'Random', 'Points'], ['++', 'random', 'points']]
        self.addParameter(QgsProcessingParameterEnum(self.MINIT,
                                                     self.tr(u'Initialization Method'),
                                                     defaultValue  = 0,
                                                     options = self.minit_name[0]))
        self.addParameter(QgsProcessingParameterNumber(self.ITER,
                                                       self.tr(u'Number of iterations'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       10, False, 1, 99999999))
        self.addParameter(QgsProcessingParameterNumber(self.K,
                                                       self.tr(u'Number of Clusters(K)'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       3, False, 2, 99999999))
        wss_param = ParameterWss(
            self.WSS,
            self.tr('<hr> '),
            layer_param=self.INPUT,
            variable_options=self.V_OPTIONS,
            backend_param=self.BACKEND)
        wss_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.KmeansWss.WssWidgetWrapper'}})
        self.addParameter(wss_param)
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr(u'Output Layer with K_Clusters'),
                                                            QgsProcessing.TypeVector))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_CENTROID, 
                                                            self.tr(u'Centroids of Clusters'),
                                                            QgsProcessing.TypeVectorPoint))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT_REPORT, self.tr('Output Report'),
                                                                self.tr('HTML files (*.html)'), None, True))

    def processAlgorithm(self, parameters, context, feedback):
        # input parameters
        cLayer = self.parameterAsSource(parameters, self.INPUT, context)
        to_cluster, variable_fields, normalized = self.parameterAsMatrix(parameters, self.V_OPTIONS, context)
        backend_idx = self.parameterAsEnum(parameters, self.BACKEND, context)
        backend_choice = 'sklearn' if backend_idx == 1 else 'scipy'
        sklearn_available = has_sklearn()
        if not sklearn_available:
            feedback.pushInfo(SKLEARN_INSTALL_MESSAGE)
        if backend_choice == 'sklearn' and not sklearn_available:
            backend_choice = 'scipy'
        minit_idx = self.parameterAsEnum(parameters, self.MINIT, context)
        iter = self.parameterAsInt(parameters, self.ITER, context)
        k = self.parameterAsInt(parameters, self.K, context)
        feat_count = len(cLayer)
        if k > feat_count:
            feedback.pushInfo(self.tr(u'The number of clusters is greater than the number of data.<br>The number of clusters was adjusted to the number of data.'))
            k = feat_count
        if to_cluster == 'attrs' and not variable_fields:
            raise QgsProcessingException(self.tr(u'No Fields Selected.'))
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))

        # input --> numpy array
        if to_cluster == 'geom':
            raw_features = [[f.geometry().centroid().asPoint().x(), f.geometry().centroid().asPoint().y()] for f in cLayer.getFeatures()]
            raw_features = np.stack(raw_features, axis=0)
        else:
            raw_features = [[f[fld] for f in cLayer.getFeatures()] for fld in variable_fields]
            raw_features = np.stack(raw_features, axis=1)
        raw_features = raw_features.astype(float)
        if normalized:
            scale = np.std(raw_features, axis=0)
            scale[scale == 0] = 1.0
            features = raw_features / scale
            if_normalized = "Yes"
        else:
            scale = np.ones(raw_features.shape[1], dtype=float)
            features = raw_features.copy()
            if_normalized = "No"
        total_ss = np.sum((features - np.mean(features, axis = 0))**2)
        indices = np.arange(feat_count, dtype=int)

        if backend_choice == 'scipy':
            centroids, label = kmeans2(features, k, iter, minit=self.minit_name[1][minit_idx])
        else:
            centroids, label = self._run_sklearn_kmeans(features, k, iter, self.minit_name[1][minit_idx])

        label = label.astype(int)
        distances = features - centroids[label]
        distance_squared = np.einsum('ij,ij->i', distances, distances)
        pts = np.column_stack((indices, label, distance_squared, features))
        pts = pts[np.argsort(pts[:, 0]), :]

        wss_by_cluster = np.bincount(label, weights=distance_squared, minlength=k)

        centroids_scaled = np.column_stack((np.arange(k), wss_by_cluster, centroids))
        centroids_original_vals = centroids * scale
        centroids_original = np.column_stack((np.arange(k), wss_by_cluster, centroids_original_vals))
        cluster = pts[:, 1]
        distance = pts[:, 2]
        
        feedback.pushInfo("End of Algorithm")
        feedback.pushInfo("Building Layers")
	
        # cluster layer
        cluster_colors = [QColor.fromHsv(int(360 * i / max(1, k)), 255, 200) for i in range(k)]
        fields = cLayer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('K_Cluster_ID', QVariant.Int))
        new_fields.append(QgsField('Within_Distance', QVariant.Double))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (cluster_sink, cluster_dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                          fields, cLayer.wkbType(), cLayer.sourceCrs())
        for i, feat in enumerate(cLayer.getFeatures()):
            outFeat = feat
            attrs = feat.attributes()
            attrs.extend([int(cluster[i]), float(distance[i])])
            outFeat.setAttributes(attrs)
            cluster_sink.addFeature(outFeat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / feat_count * 100))
        feedback.setProgress(0)
        feedback.pushInfo("Done with Cluster Layer")

        cluster_layer = QgsProcessingUtils.mapLayerFromString(cluster_dest_id, context)
        cluster_geom_type = cluster_layer.geometryType()
        if cluster_geom_type != QgsWkbTypes.NullGeometry:
            categories = []
            for idx, color in enumerate(cluster_colors):
                symbol = QgsSymbol.defaultSymbol(cluster_geom_type)
                if symbol is None:
                    continue
                symbol.setColor(color)
                categories.append(QgsRendererCategory(idx, symbol, str(idx)))
            if categories:
                renderer = QgsCategorizedSymbolRenderer('K_Cluster_ID', categories)
                cluster_layer.setRenderer(renderer)
                cluster_layer.triggerRepaint()
                style_path = os.path.join(QgsProcessingUtils.tempFolder(), 'kmeans_cluster.qml')
                cluster_layer.saveNamedStyle(style_path)
                if context.willLoadLayerOnCompletion(cluster_dest_id):
                    details = context.layersToLoadOnCompletion()[cluster_dest_id]
                    details.style = style_path

        # centroid layer
        if to_cluster == 'geom':
            xy_fields = QgsFields()
            xy_fields.append(QgsField('K_Cluster_ID', QVariant.Int))
            xy_fields.append(QgsField('WSS', QVariant.Double))
            (centroid_sink, centroid_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_CENTROID, context,
                                                xy_fields, QgsWkbTypes.Point, cLayer.sourceCrs())

            total = k
            for j, center in enumerate(centroids_original):
                centerFeat = QgsFeature()
                if to_cluster == 'geom':
                    centerGeom = QgsGeometry.fromPointXY(QgsPointXY(center[2],center[3]))
                    centerFeat.setGeometry(centerGeom)
                attrs = centerFeat.attributes()
                attrs.extend([int(center[0]), float(center[1])])
                centerFeat.setAttributes(attrs)
                centroid_sink.addFeature(centerFeat, QgsFeatureSink.FastInsert)
                feedback.setProgress(int(j / total * 100))
            feedback.pushInfo(self.tr("Done with Cluster <br> Centroid Layer"))

            centroid_layer = QgsProcessingUtils.mapLayerFromString(centroid_dest_id, context)
            categories = []
            for idx, color in enumerate(cluster_colors):
                symbol = QgsSymbol.defaultSymbol(centroid_layer.geometryType())
                symbol.setColor(color)
                categories.append(QgsRendererCategory(idx, symbol, str(idx)))
            renderer = QgsCategorizedSymbolRenderer('K_Cluster_ID', categories)
            centroid_layer.setRenderer(renderer)
            centroid_layer.triggerRepaint()
            centroid_style = os.path.join(QgsProcessingUtils.tempFolder(), 'kmeans_centroid.qml')
            centroid_layer.saveNamedStyle(centroid_style)
            if context.willLoadLayerOnCompletion(centroid_dest_id):
                details = context.layersToLoadOnCompletion()[centroid_dest_id]
                details.style = centroid_style

        else:
            centroid_sink = None
            centroid_dest_id = None

        # output report
        output_report = self.parameterAsFileOutput(parameters, self.OUTPUT_REPORT, context)

        total_wss = float(np.sum(wss_by_cluster))
        td_blue = '<td rowspan="1" \
                    colspan="1" \
                    bgcolor="rgb(0, 80, 141)" \
                    style="word-break: break-all; background-color: rgb(0, 80, 141); \
                    height: 24px; \
                    padding: 3px 4px 2px;" \
                    data-origin-bgcolor="rgb(0, 80, 141)">' + \
                    '<div style="text-align: center;">' + \
                    '<span style="color: rgb(255, 255, 255); font-weight: bold;">'
        td_white = '<td rowspan="1" \
                    colspan="1" \
                    bgcolor="#ffffff" \
                    style="word-break: break-all; background-color: rgb(255, 255, 255); \
                    height: 24px; \
                    padding: 3px 4px 2px;">' + \
                    '<div style="text-align: center;">' + \
                    '<span>'

        with codecs.open(output_report, 'w', encoding='utf-8') as f:
            f.write('<html><head>\n')
            f.write('<meta http-equiv="Content-Type" content="text/html; \
                    charset=utf-8" /></head><body>\n')
            f.write('<p> Number of clusters: ' + str(k) + '</p>\n')
            f.write('<p> Initialization Method: ' + self.minit_name[0][minit_idx] + '</p>\n')
            f.write('<p> Number of iterations: ' + str(iter) + '</p>\n')
            f.write('<p> Normalized: ' + if_normalized + '</p>\n')
            f.write('<p> The total sum of squares: ' + str(total_ss) + '</p>\n')
            f.write('<p> The within cluster sum of squares: ' + str(total_wss) + '</p>\n')
            f.write('<p> The between cluster sum of squares: ' + str(total_ss-total_wss) + '</p>\n')
            f.write('<p> The ratio of between to total sum of squares: ' + str((total_ss-total_wss) / total_ss * 100) + '%</p>\n')
            # start of table
            
            cols = ['X', 'Y'] if to_cluster == 'geom' else variable_fields

            def write_centroid_table(title, centroid_rows):
                f.write('<p><strong>' + title + '</strong></p>')
                f.write('<table cellpadding="0" cellspacing="1" bgcolor="#ffffff" style="background-color: rgb(204, 204, 204);">')
                f.write('<tbody>')
                f.write('<tr style="">')
                f.write(td_blue + 'Cluster Centers' + '</span></div></td>')
                f.write(td_blue + 'WSS' + '</span></div></td>')
                for c in cols:
                    f.write(td_blue + str(c) + '</span></div></td>')
                for centroid in centroid_rows:
                    f.write('<tr style="height: 24px;">')
                    for cent in centroid:
                        f.write(td_white + str(cent) + '</span></div></td>')
                    f.write('<tr>')
                f.write('</tbody></table>')

            if normalized:
                write_centroid_table('Cluster Centers (Standardized)', centroids_scaled)
                write_centroid_table('Cluster Centers (Original Units)', centroids_original)
            else:
                write_centroid_table('Cluster Centers', centroids_original)

            f.write('</body></html>\n')

        results = {}
        results[self.OUTPUT] = cluster_dest_id
        results[self.OUTPUT_CENTROID] = centroid_dest_id
        results[self.OUTPUT_REPORT] = output_report
        return results

    def _run_sklearn_kmeans(self, features, k, max_iter, init_key):
        try:
            from sklearn.cluster import KMeans  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise QgsProcessingException(
                self.tr('scikit-learn backend is unavailable: {}').format(exc)
            )

        if init_key == '++':
            init_param = 'k-means++'
            n_init = 20
        elif init_key == 'random':
            init_param = 'random'
            n_init = 20
        else:
            if features.shape[0] < k:
                init_points = features.copy()
            else:
                init_indices = np.random.choice(features.shape[0], k, replace=False)
                init_points = features[init_indices]
            init_param = init_points
            n_init = 1

        model = KMeans(
            n_clusters=k,
            init=init_param,
            n_init=n_init,
            max_iter=max_iter
        )
        labels = model.fit_predict(features)
        centroids = model.cluster_centers_
        return centroids, labels
