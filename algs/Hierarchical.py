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

import os
import codecs

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QIcon, QColor

from qgis.core import (QgsField,
                       QgsFields,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsSymbol,
                       QgsRendererCategory,
                       QgsCategorizedSymbolRenderer,
                       QgsWkbTypes,
                       QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingUtils,
                       QgsFeatureSink,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterFileDestination)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
import numpy as np
from ..locale_utils import localized_menu_text

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from spatial_analysis.forms.HarchiParam import ParameterHarchi
from spatial_analysis.forms.VariableParam import ParameterVariable

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Hierarchical(QgisAlgorithm):

    INPUT = 'INPUT_LAYER'
    V_OPTIONS = 'V_OPTIONS'
    LINKAGE = 'LINKAGE'
    HARCHIPARAM = 'HARCHIPARAM'
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
        return 'hierarchical'

    def displayName(self):
        return localized_menu_text('Hierarchical', self.tr('Hierarchical'))

    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr(u'Input Layer'),
                                                              [QgsProcessing.TypeVector]))
        variable_param = ParameterVariable(self.V_OPTIONS, self.tr(u'Variable Fields'), layer_param=self.INPUT)
        variable_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.VariableWidget.VariableWidgetWrapper'}})
        self.addParameter(variable_param)
        self.dMethod = ['centroid', 'ward', 'single', 'complete', 'average']
        self.addParameter(QgsProcessingParameterEnum(self.LINKAGE,
                                                        self.tr('Linkage(Distance) Method'),
                                                        defaultValue  = 0,
                                                        options = self.dMethod))
        harchi_param = ParameterHarchi(self.HARCHIPARAM, self.tr('<hr> '), layer_param=self.INPUT, variable_options=self.V_OPTIONS, linkage_param=self.LINKAGE)
        harchi_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.Harchi.HarchiWidgetWrapper'}})
        self.addParameter(harchi_param)
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT,
                                                            self.tr('Output Layer with H_Clusters'),
                                                            QgsProcessing.TypeVector))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_CENTROID,
                                                            self.tr('Centroids of Clusters'),
                                                            QgsProcessing.TypeVectorPoint))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT_REPORT,
                                                                self.tr('Output Report'),
                                                                self.tr('HTML files (*.html)'),
                                                                None, True))

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT, context)
        to_cluster, variable_fields, normalized = self.parameterAsMatrix(parameters, self.V_OPTIONS, context)
        dMethodIndex = self.parameterAsEnum(parameters, self.LINKAGE, context)
        harchi_param = self.parameterAsMatrix(parameters, self.HARCHIPARAM, context)
        if harchi_param[0] == 0:
            criterion = 'maxclust'
        else:
            criterion = 'distance'
        threshold = harchi_param[1]

        feat_list = list(cLayer.getFeatures())
        feature_count = len(feat_list)
        if feature_count == 0:
            raise QgsProcessingException(self.tr('Input layer contains no features.'))
        if to_cluster == 'attrs' and not variable_fields:
            raise QgsProcessingException(self.tr('No Fields Selected.'))
        has_geometry = QgsWkbTypes.geometryType(cLayer.wkbType()) != QgsWkbTypes.NullGeometry

        if to_cluster == 'geom':
            if not has_geometry:
                raise QgsProcessingException(self.tr('Input layer has no geometry for geometric clustering.'))
            raw_features = np.array([[f.geometry().centroid().asPoint().x(),
                                      f.geometry().centroid().asPoint().y()] for f in feat_list], dtype=float)
        else:
            raw_features = np.array([[f[fld] for fld in variable_fields] for f in feat_list], dtype=float)

        if normalized:
            scale = np.std(raw_features, axis=0)
            scale[scale == 0] = 1.0
            features = raw_features / scale
            if_normalized = "Yes"
        else:
            scale = np.ones(raw_features.shape[1], dtype=float)
            features = raw_features.copy()
            if_normalized = "No"
		
        ## perform hierarchical clustering and get centers of clusters
        Z = linkage(features, method=self.dMethod[dMethodIndex], metric='euclidean')
        cluster = fcluster(Z, t=threshold, criterion=criterion)
        feedback.pushInfo("End of Algorithm")
        feedback.pushInfo("Building Layers")
		
        ## cluster layer
        fields = cLayer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('H_Cluster', QVariant.Int))

        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (cluster_sink, cluster_dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                          fields, cLayer.wkbType(), cLayer.sourceCrs())
        for i, feat in enumerate(feat_list):
            outFeat = QgsFeature(feat)
            attrs = feat.attributes()
            attrs.extend([int(cluster[i])])
            outFeat.setAttributes(attrs)
            cluster_sink.addFeature(outFeat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / max(1, feature_count) * 100))
        feedback.setProgress(0)
        feedback.pushInfo("Done with Cluster Layer")
   

        # centroid layer
        centroid_sink = None
        centroid_dest_id = None
        geom_coords = None
        unique_clusters = np.unique(cluster)
        if has_geometry:
            geom_coords = np.array([[f.geometry().centroid().asPoint().x(),
                                     f.geometry().centroid().asPoint().y()] for f in feat_list], dtype=float)
            xy_fields = QgsFields()
            xy_fields.append(QgsField('H_Cluster', QVariant.Int))
            (centroid_sink, centroid_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_CENTROID, context,
                                                  xy_fields, QgsWkbTypes.Point, cLayer.sourceCrs())
            total = len(unique_clusters)
            for j, cid in enumerate(sorted(unique_clusters)):
                coords = geom_coords[cluster == cid]
                if coords.size == 0:
                    continue
                center_x, center_y = np.mean(coords, axis=0)
                centerFeat = QgsFeature()
                centerFeat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(center_x, center_y)))
                centerFeat.setAttributes([int(cid)])
                centroid_sink.addFeature(centerFeat, QgsFeatureSink.FastInsert)
                feedback.setProgress(int(j / max(1, total) * 100))
            feedback.pushInfo(self.tr("Done with Cluster Centroid Layer"))

        # build color mapping for categories
        color_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                      '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                      '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                      '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
        color_map = {cid: QColor(color_list[i % len(color_list)])
                     for i, cid in enumerate(sorted(unique_clusters))}

        result_layer = QgsProcessingUtils.mapLayerFromString(cluster_dest_id, context)
        if result_layer:
            categories = []
            geom_type = result_layer.geometryType()
            for cid in sorted(unique_clusters):
                symbol = QgsSymbol.defaultSymbol(geom_type)
                if symbol is None:
                    continue
                symbol.setColor(color_map[cid])
                category = QgsRendererCategory(int(cid), symbol, str(int(cid)))
                categories.append(category)
            if categories:
                renderer = QgsCategorizedSymbolRenderer('H_Cluster', categories)
                result_layer.setRenderer(renderer)
                result_layer.triggerRepaint()
                style_path = os.path.join(QgsProcessingUtils.tempFolder(), 'hierarchical_clusters.qml')
                result_layer.saveNamedStyle(style_path)
                if context.willLoadLayerOnCompletion(cluster_dest_id):
                    details = context.layersToLoadOnCompletion()[cluster_dest_id]
                    details.style = style_path

        centroid_layer = QgsProcessingUtils.mapLayerFromString(centroid_dest_id, context) if centroid_dest_id else None
        if centroid_layer:
            categories = []
            for cid in sorted(unique_clusters):
                symbol = QgsSymbol.defaultSymbol(centroid_layer.geometryType())
                if symbol is None:
                    continue
                symbol.setColor(color_map[cid])
                category = QgsRendererCategory(int(cid), symbol, str(int(cid)))
                categories.append(category)
            if categories:
                renderer = QgsCategorizedSymbolRenderer('H_Cluster', categories)
                centroid_layer.setRenderer(renderer)
                centroid_layer.triggerRepaint()
                style_path = os.path.join(QgsProcessingUtils.tempFolder(), 'hierarchical_cluster_centroids.qml')
                centroid_layer.saveNamedStyle(style_path)
                if context.willLoadLayerOnCompletion(centroid_dest_id):
                    details = context.layersToLoadOnCompletion()[centroid_dest_id]
                    details.style = style_path

        # build report
        output_report = self.parameterAsFileOutput(parameters, self.OUTPUT_REPORT, context)
        cols = ['X', 'Y'] if to_cluster == 'geom' else variable_fields
        unique_clusters = np.unique(cluster)
        centroid_rows_std = []
        centroid_rows_orig = []
        for cid in sorted(unique_clusters):
            mask = cluster == cid
            cluster_data_std = features[mask]
            if cluster_data_std.size == 0:
                centroid_std = [np.nan] * len(cols)
            else:
                centroid_std = np.mean(cluster_data_std, axis=0)
            centroid_orig = (np.array(centroid_std) * scale).tolist()
            centroid_rows_std.append([int(cid), int(np.sum(mask))] + list(centroid_std))
            centroid_rows_orig.append([int(cid), int(np.sum(mask))] + centroid_orig)

        td_blue = '<td rowspan="1" colspan="1" bgcolor="rgb(0, 80, 141)" ' \
                  'style="word-break: break-all; background-color: rgb(0, 80, 141); height: 24px; padding: 3px 4px 2px;" ' \
                  'data-origin-bgcolor="rgb(0, 80, 141)"><div style="text-align: center;">' \
                  '<span style="color: rgb(255, 255, 255); font-weight: bold;">'
        td_white = '<td rowspan="1" colspan="1" bgcolor="#ffffff" ' \
                   'style="word-break: break-all; background-color: rgb(255, 255, 255); height: 24px; padding: 3px 4px 2px;">' \
                   '<div style="text-align: center;"><span>'

        def write_centroid_table(file_handle, title, centroid_rows):
            file_handle.write('<p><strong>' + title + '</strong></p>')
            file_handle.write('<table cellpadding="0" cellspacing="1" bgcolor="#ffffff" style="background-color: rgb(204, 204, 204);">')
            file_handle.write('<tbody>')
            file_handle.write('<tr style="">')
            headers = ['Cluster', 'Count'] + [str(c) for c in cols]
            for header in headers:
                file_handle.write(td_blue + header + '</span></div></td>')
            for centroid in centroid_rows:
                file_handle.write('<tr style="height: 24px;">')
                for value in centroid:
                    file_handle.write(td_white + str(value) + '</span></div></td>')
                file_handle.write('<tr>')
            file_handle.write('</tbody></table>')

        with codecs.open(output_report, 'w', encoding='utf-8') as f:
            f.write('<html><head>\n')
            f.write('<meta http-equiv="Content-Type" content="text/html; charset=utf-8" /></head><body>\n')
            f.write('<p> Number of clusters: ' + str(len(unique_clusters)) + '</p>\n')
            f.write('<p> Linkage Method: ' + self.dMethod[dMethodIndex].title() + '</p>\n')
            f.write('<p> Threshold: ' + str(threshold) + '</p>\n')
            f.write('<p> Criterion: ' + criterion + '</p>\n')
            f.write('<p> Normalized: ' + if_normalized + '</p>\n')

            if normalized:
                write_centroid_table(f, 'Cluster Centers (Standardized)', centroid_rows_std)
                write_centroid_table(f, 'Cluster Centers (Original Units)', centroid_rows_orig)
            else:
                write_centroid_table(f, 'Cluster Centers', centroid_rows_orig)
            f.write('</body></html>\n')

        results = {}
        results[self.OUTPUT] = cluster_dest_id
        results[self.OUTPUT_CENTROID] = centroid_dest_id
        results[self.OUTPUT_REPORT] = output_report
        return results
        