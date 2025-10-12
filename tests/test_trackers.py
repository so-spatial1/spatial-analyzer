import sys
import os
import types
import pytest

# Stub minimal qgis modules required for utilities
qgis_mod = types.ModuleType("qgis")
core_mod = types.ModuleType("qgis.core")

class QgsFeature:
    def __init__(self):
        self._geometry = None
        self._attributes = []
    def geometry(self):
        return self._geometry
    def setGeometry(self, geom):
        self._geometry = geom
    def attributes(self):
        return self._attributes
    def setAttributes(self, attrs):
        self._attributes = attrs

class QgsPointXY(tuple):
    def __new__(cls, x, y):
        return tuple.__new__(cls, (x, y))

class QgsGeometry:
    @staticmethod
    def fromPointXY(pt):
        return pt

core_mod.QgsFeature = QgsFeature
core_mod.QgsPointXY = QgsPointXY
core_mod.QgsGeometry = QgsGeometry
qgis_mod.core = core_mod
sys.modules.setdefault("qgis", qgis_mod)
sys.modules.setdefault("qgis.core", core_mod)

# Stub numpy and scipy modules to satisfy imports without heavy dependencies
numpy_mod = types.ModuleType("numpy")
numpy_mod.asarray = lambda x, dtype=None: list(x)
numpy_mod.float32 = float
numpy_mod.sum = lambda arr: sum(arr)
numpy_mod.zeros = lambda shape: [0] * (shape[0] if isinstance(shape, (list, tuple)) else shape)
numpy_mod.stack = lambda arrays, axis=-1: list(zip(*arrays))
numpy_mod.argmin = lambda arr, axis=None: min(range(len(arr)), key=lambda i: arr[i])
sys.modules.setdefault("numpy", numpy_mod)

scipy_mod = types.ModuleType("scipy")
spatial_mod = types.ModuleType("scipy.spatial")
distance_mod = types.ModuleType("scipy.spatial.distance")
distance_mod.pdist = lambda coords, metric=None: []
distance_mod.squareform = lambda x: []
spatial_mod.distance = distance_mod
optimize_mod = types.ModuleType("scipy.optimize")
optimize_mod.minimize = lambda func, x0, method=None: types.SimpleNamespace(x=x0)
scipy_mod.spatial = spatial_mod
scipy_mod.optimize = optimize_mod
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.spatial", spatial_mod)
sys.modules.setdefault("scipy.spatial.distance", distance_mod)
sys.modules.setdefault("scipy.optimize", optimize_mod)

# ensure package importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from spatial_analysis.utilities import (
    getPointCoords,
    getMeanCenter,
    getMedianCenter,
    getCentralFeature,
)


class DummyGeometry:
    def __init__(self, point):
        self._pt = point
    def asPoint(self):
        if self._pt is None:
            raise AttributeError("Invalid point")
        return self._pt

class DummyFeature:
    def __init__(self, point=None, attrs=None):
        self._geom = DummyGeometry(point)
        self._attrs = attrs or []
    def geometry(self):
        return self._geom
    def attributes(self):
        return self._attrs


def run_mean_center(features):
    try:
        x, y, w = getPointCoords(features, -1)
    except ValueError:
        return None
    try:
        return getMeanCenter(x, y, w, 1)
    except ValueError:
        return None


def run_median_center(features):
    try:
        x, y, w = getPointCoords(features, -1)
    except ValueError:
        return None
    try:
        return getMedianCenter(x, y, w, 1)
    except ValueError:
        return None


def run_central_feature(features):
    try:
        x, y, w = getPointCoords(features, -1)
    except ValueError:
        return None
    try:
        return getCentralFeature(x, y, w, 1, 0)
    except ValueError:
        return None


def test_empty_feature_sets():
    feats = []
    assert run_mean_center(feats) is None
    assert run_median_center(feats) is None
    assert run_central_feature(feats) is None


def test_malformed_features():
    feats = [DummyFeature(point=None)]
    assert run_mean_center(feats) is None
    assert run_median_center(feats) is None
    assert run_central_feature(feats) is None

