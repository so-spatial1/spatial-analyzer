"""Utility helpers for optional scikit-learn support."""
from functools import lru_cache

SKLEARN_INSTALL_MESSAGE = (
    "Installing scikit-learn provides faster and more stable results.\n"
    "Install scikit-learn: Open the OSGeo4W Shell installed with QGIS as Administrator and type:\n"
    "$ python -m pip install --upgrade pip\n"
    "$ python -m pip install scikit-learn -U"
)


@lru_cache(maxsize=1)
def has_sklearn():
    """Return True if scikit-learn is available."""
    try:
        import sklearn  # noqa: F401
    except Exception:  # pragma: no cover - environment specific dependency
        return False
    return True