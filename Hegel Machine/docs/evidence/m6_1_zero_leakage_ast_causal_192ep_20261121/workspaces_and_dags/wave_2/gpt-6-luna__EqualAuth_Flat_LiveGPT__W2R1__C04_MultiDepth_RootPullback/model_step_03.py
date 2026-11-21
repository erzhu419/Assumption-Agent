import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    indoor = float(row.get('u_indoor', 0.0))
    outdoor = float(row.get('u_outdoor', 0.0))
    weather = float(row.get('p_weather', 0.0))
    impact = float(row.get('impact_idx', 0.0))
    foot = float(row.get('a_foot', 1.0))
    # Saturating indoor exercise benefit; weather-weighted outdoor input;
    # impact-related loss and footprint adjustment.
    return np.array([1.0, indoor * indoor / (1.5 + indoor), outdoor * weather, impact, 1.0 / foot], dtype=float)

def fit(train_rows):
    global _PARAMS
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residuals(p):
        return X @ p - y
    result = least_squares(residuals, x0=np.zeros(X.shape[1]), max_nfev=5000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _PARAMS)