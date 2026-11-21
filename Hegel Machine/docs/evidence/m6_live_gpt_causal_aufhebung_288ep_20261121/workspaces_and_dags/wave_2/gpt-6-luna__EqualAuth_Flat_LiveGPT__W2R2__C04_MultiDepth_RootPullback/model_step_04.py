import numpy as np
from scipy.optimize import least_squares

_COEF = None

def _features(row):
    outdoor = float(row.get('u_outdoor', 0.0))
    indoor = float(row.get('u_indoor', 0.0))
    weather = float(row.get('p_weather', 0.0))
    foot = max(float(row.get('a_foot', 1.0)), 1e-8)
    impact = float(row.get('impact_idx', 0.0))
    dose = indoor / foot
    return np.array([1.0, outdoor * weather, dose, dose * dose, impact, impact * dose], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    # Fit a low-order response with a separate biomechanical loss term.
    result = least_squares(lambda b: X @ b - y, np.zeros(X.shape[1]), max_nfev=5000)
    _COEF = result.x
    return _COEF

def predict(row):
    if _COEF is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _COEF)