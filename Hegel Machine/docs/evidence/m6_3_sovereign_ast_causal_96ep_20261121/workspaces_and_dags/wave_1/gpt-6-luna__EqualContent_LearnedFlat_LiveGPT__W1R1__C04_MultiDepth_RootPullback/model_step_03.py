import math
import numpy as np
from scipy.optimize import least_squares

_COEF = None

def _features(row):
    i = float(row.get('u_indoor', 0.0))
    o = float(row.get('u_outdoor', 0.0))
    w = float(row.get('p_weather', 0.0))
    a = float(row.get('a_foot', 0.0))
    q = float(row.get('impact_idx', 0.0))
    # Center area near the observed range; include nonlinear impact and
    # indoor-by-impact/area effects to capture space-limited indoor benefit.
    ac = a - 1.4
    return np.array([1.0, i, o*w, q, ac, i*q, q*ac, i*i, q*q], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Mild ridge stabilization keeps the small pilot from selecting extreme
    # coefficients while allowing the dominant mechanistic terms to fit.
    lam = 0.02
    A = np.vstack([X, np.sqrt(lam) * np.eye(X.shape[1])])
    b = np.concatenate([y, np.zeros(X.shape[1])])
    result = least_squares(lambda c: A @ c - b, np.zeros(X.shape[1]))
    _COEF = result.x
    return predict

def predict(row):
    if _COEF is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_features(row) @ _COEF)