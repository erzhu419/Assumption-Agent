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
    ac = a - 1.4
    # Quadratic response surface permits indoor capacity and impact costs to
    # depend on each other and on available floor area.
    return np.array([
        1.0, i, q, ac, o*w,
        i*i, q*q, ac*ac,
        i*q, i*ac, q*ac,
        i*q*ac, i*i*q, i*q*q
    ], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    # Fit the observed response surface with a small ridge penalty to limit
    # unstable coefficients in this small pilot.
    lam = 1e-4
    A = np.vstack([X, math.sqrt(lam) * np.eye(X.shape[1])])
    b = np.concatenate([y, np.zeros(X.shape[1])])
    result = least_squares(lambda c: A @ c - b, np.zeros(X.shape[1]))
    _COEF = result.x
    return predict

def predict(row):
    if _COEF is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_features(row) @ _COEF)