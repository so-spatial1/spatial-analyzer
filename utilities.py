from qgis.core import QgsFeature, QgsPointXY, QgsGeometry
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

def getPointCoords(feat, weightFieldIndex):
    """Extract x, y coordinate lists and associated weights from features.

    Parameters
    ----------
    feat : iterable
        Iterable of QgsFeature like objects.
    weightFieldIndex : int
        Index of the field containing the weight value.  A negative value
        treats all weights as 1.

    Returns
    -------
    tuple
        Three element tuple of (x coordinates, y coordinates, weights).

    Raises
    ------
    ValueError
        If any feature is missing required geometry/attributes or if the
        iterator is empty.
    """
    try:
        pts = []
        weights = []
        for f in feat:
            pts.append(f.geometry().asPoint())
            if weightFieldIndex >= 0:
                weights.append(f.attributes()[weightFieldIndex])
            else:
                weights.append(1)
        if not pts:
            raise ValueError("No features provided")
        weights = np.asarray(weights, dtype=np.float32)
        x = [pt[0] for pt in pts]
        y = [pt[1] for pt in pts]
        return x, y, weights
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Invalid feature data") from exc

def getMeanCenter(x, y, weights, id):
    """Return a feature representing the weighted mean center."""
    try:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        if weights.size == 0 or np.sum(weights) == 0:
            raise ValueError("Weights must not be empty or all zeros")
        wx = x * weights
        wy = y * weights
        mx = np.sum(wx) / np.sum(weights)
        my = np.sum(wy) / np.sum(weights)

        meanCenter = QgsPointXY(mx, my)

        centerFeat = QgsFeature()
        centerGeom = QgsGeometry.fromPointXY(meanCenter)
        attrs = centerFeat.attributes()
        centerFeat.setGeometry(centerGeom)
        attrs.extend([id])
        centerFeat.setAttributes(attrs)
        return centerFeat
    except (ZeroDivisionError, TypeError, ValueError) as exc:
        raise ValueError("Failed to compute mean center") from exc

def getMedianCenter(x, y, weights, id):
    """Return a feature representing the weighted median center."""
    try:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        sumWeights = np.sum(weights)
        if weights.size == 0 or sumWeights == 0:
            raise ValueError("Weights must not be empty or all zeros")
        mx = np.sum(x) / len(x)
        my = np.sum(y) / len(y)

        # initial guesses
        cMedianCenter = np.zeros(2)
        cMedianCenter[0] = mx
        cMedianCenter[1] = my

        # define objective
        def objective(cMedianCenter):
            return np.sum(np.sqrt((cMedianCenter[0] - x) ** 2 +
                                  (cMedianCenter[1] - y) ** 2) * weights / sumWeights)

        solution = minimize(objective, cMedianCenter, method='Nelder-Mead')
        medianCenter = QgsPointXY(solution.x[0], solution.x[1])
		
        centerFeat = QgsFeature()
        centerGeom = QgsGeometry.fromPointXY(medianCenter)
        attrs = centerFeat.attributes()
        centerFeat.setGeometry(centerGeom)
        attrs.extend([id])
        centerFeat.setAttributes(attrs)
        return centerFeat
    except (ZeroDivisionError, TypeError, ValueError) as exc:
        raise ValueError("Failed to compute median center") from exc

def getCentralFeature(x, y, weights, id, dMetricIndex):
    """Return a feature representing the weighted central feature."""
    try:
        dMetric = ['euclidean', 'cityblock']
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        sumWeights = np.sum(weights)
        if weights.size == 0 or sumWeights == 0:
            raise ValueError("Weights must not be empty or all zeros")
        coords = np.stack([x, y], axis=-1)
        distanceVector = pdist(coords, metric=dMetric[dMetricIndex])
        distanceMatrix = squareform(distanceVector) / weights * sumWeights
        minDistanceIndex = np.argmin(np.sum(distanceMatrix, axis=1))
        centralFeat = QgsFeature()
        centralGeom = QgsGeometry.fromPointXY(QgsPointXY(x[minDistanceIndex], y[minDistanceIndex]))
        attrs = centralFeat.attributes()
        centralFeat.setGeometry(centralGeom)
        attrs.extend([id])
        centerFeat = centralFeat
        centerFeat.setAttributes(attrs)
        return centerFeat
    except (IndexError, ZeroDivisionError, TypeError, ValueError) as exc:
        raise ValueError("Failed to compute central feature") from exc


def compute_silhouette_coefficient(features, labels):
    """Compute the mean silhouette coefficient for clustering results."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)

    if features.ndim != 2 or features.shape[0] < 2:
        return np.nan

    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return np.nan

    distance_matrix = squareform(pdist(features, metric='euclidean'))
    silhouettes = np.zeros(features.shape[0], dtype=float)

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_size = np.sum(mask)
        if cluster_size == 0:
            continue
        if cluster_size == 1:
            intra_dist = np.zeros(1, dtype=float)
        else:
            intra_sum = np.sum(distance_matrix[np.ix_(mask, mask)], axis=1)
            intra_dist = intra_sum / (cluster_size - 1)

        inter_dist = np.full(cluster_size, np.inf, dtype=float)
        for other_id in unique_labels:
            if other_id == cluster_id:
                continue
            other_mask = labels == other_id
            if not np.any(other_mask):
                continue
            mean_dist = np.mean(distance_matrix[np.ix_(mask, other_mask)], axis=1)
            inter_dist = np.minimum(inter_dist, mean_dist)

        denom = np.maximum(intra_dist, inter_dist)
        silhouettes[mask] = np.where(denom > 0, (inter_dist - intra_dist) / denom, 0.0)

    return float(np.mean(silhouettes))