from qgis.PyQt.QtCore import QLocale
from qgis.core import QgsSettings

_KOREAN_TRANSLATIONS = {
    'SpatialAnalyzer - Spatial Analysis Toolbox': 'SpatialAnalyzer - 공간 분석 도구상자',
    'Sample Data': '샘플 데이터',
    'Spatial Central Tendency': '공간 중심경향성',
    'Spatial Dispersion': '공간분산',
    'Clustering': '군집분석',
    'Gravity Model': '중력모형',
    'Spatial Regression': '공간 회귀분석',
    'Dimension Reduction': '차원축소',
    'Spatial Autocorrelation': '공간자기상관',
    'Gi*': 'Gi*',
    'Centers(Mean Center, Median Center, Central Feature)': '중심경향성(평균좌표, 중앙좌표 , 중심피처)',
    'Central Feature Tracker': '누적 중심피처',
    'DBSCAN': '밀도기반 군집분석(DBSCAN)',
    'Geographically Weighted Regression': '지리 가중 회귀분석',
    'Gravity': '중력분석',
    'Hierarchical': '계층적 군집분석(Hierarchical)',
    'K-Means': 'K-평균 군집분석(K-Means)',
    'Mean Center Tracker': '누적 평균좌표',
    'Median Center Tracker': '누적 중앙좌표',
    'Principal Component Analysis': '주성분 분석(PCA)',
    "Local Moran's i": '로컬 모란지수',
    'Standard Deviation Ellipse': '표준편차 타원체',
    'Standard Distance': '표준거리',
    't-SNE': 't-SNE',
}


def _user_locale_name():
    try:
        locale = QgsSettings().value('locale/userLocale', '', type=str)
    except Exception:
        locale = ''
    if not locale:
        locale = QLocale().name()
    return locale or ''


def _is_korean_locale():
    return _user_locale_name().lower().startswith('ko')



def localized_menu_text(english_text, fallback_text=None):
    """Return a locale-aware menu label.

    Parameters
    ----------
    english_text: str
        The canonical English string used as the lookup key.
    fallback_text: Optional[str]
        The text to return when no translation is available. Defaults to
        ``english_text`` if omitted.
    """

    if fallback_text is None:
        fallback_text = english_text

    if _is_korean_locale():
        text = _KOREAN_TRANSLATIONS.get(english_text, fallback_text)
    else:
        text = fallback_text

    return text
