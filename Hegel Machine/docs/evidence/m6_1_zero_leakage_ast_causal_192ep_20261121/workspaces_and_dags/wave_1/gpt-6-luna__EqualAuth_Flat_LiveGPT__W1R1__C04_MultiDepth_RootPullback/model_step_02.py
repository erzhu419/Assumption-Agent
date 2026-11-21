import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_outdoor', 'u_indoor', 'a_foot', 'impact_idx')

def _features(row):
    o = float(row.get('u_outdoor', 0.0))
    i = float(row.get('u_indoor', 0.0))
    a = float(row.get('a_foot', 0.0))
    m = float(row.get('impact_idx', 0.0))
    p = float(row.get('p_weather', 0.0))
    # Compact mechanistic basis: rainy-weather outdoor dose, indoor dose
    # (including saturating response), and impact/space penalties.
    return np.array([1.0, o*p, i, i*i/(1.5+i), m, a, m*a, i*m], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # Mild ridge stabilization keeps the underdetermined small pilot from
    # assigning extreme coefficients to correlated terms.
    scale = np.maximum(np.linalg.norm(X, axis=0), 1.0)
    Z = X / scale
    def residual(b):
        return np.concatenate((Z @ b - y, 0.015 * b[1:]))
    result = least_squares(residual, np.zeros(X.shape[1]), max_nfev=5000)
    return result.x / scale

def predict(row):
    # fit() attaches the pilot-estimated coefficients as module state.
    return float(np.dot(_features(row), _COEF))

_COEF = np.zeros(8)
_original_fit = fit
def fit(train_rows):
    global _COEF
    _COEF = _original_fit(train_rows)
    return _COEF