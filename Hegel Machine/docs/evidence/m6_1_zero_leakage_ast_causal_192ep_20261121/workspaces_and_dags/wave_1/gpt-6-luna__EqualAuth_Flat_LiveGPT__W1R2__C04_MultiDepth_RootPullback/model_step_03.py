import numpy as np
from scipy.optimize import least_squares

_COEF = None

def _features(row):
    indoor = float(row.get('u_indoor', 0.0))
    outdoor_weather = float(row.get('u_outdoor', 0.0)) * float(row.get('p_weather', 0.0))
    foot = float(row.get('a_foot', 0.0))
    impact = float(row.get('impact_idx', 0.0))
    return np.array([1.0, indoor, indoor * indoor, outdoor_weather, impact, impact * impact, foot, impact * foot], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # Mild regularization stabilizes correlated footprint/impact terms while retaining data fit.
    scale = np.maximum(np.linalg.norm(X, axis=0), 1.0)
    A = X / scale
    def residual(c):
        return np.concatenate([A @ c - y, 0.015 * c[1:]])
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=5000)
    _COEF = result.x / scale
    return _COEF

def predict(row):
    if _COEF is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _COEF)
