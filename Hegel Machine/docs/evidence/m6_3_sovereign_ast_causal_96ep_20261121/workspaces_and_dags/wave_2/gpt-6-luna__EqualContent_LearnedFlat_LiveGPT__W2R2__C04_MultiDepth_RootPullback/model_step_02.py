import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ['u_outdoor', 'u_indoor', 'p_weather', 'a_foot', 'impact_idx']
_COEF = None

def _features(row):
    o = float(row['u_outdoor']) * float(row['p_weather'])
    i = float(row['u_indoor'])
    f = float(row['a_foot'])
    m = float(row['impact_idx'])
    x = [o, i, f, m]
    return np.array([1.0] + x + [v*v for v in x] + [x[a]*x[b] for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))], dtype=float)

def fit(train_rows):
    global _COEF
    X = np.vstack([_features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Minimum-norm quadratic response, with a tiny penalty to stabilize the
    # underdetermined fit while preserving the observed training response.
    def residual(c):
        return np.concatenate([X @ c - y, 1e-7 * c[1:]])
    _COEF = least_squares(residual, np.zeros(X.shape[1])).x
    return predict

def predict(row):
    return float(_features(row) @ _COEF)
