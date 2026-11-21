import numpy as np
from scipy.optimize import least_squares
import math

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    x = float(row['x_now'])
    m = float(row['mem_conv']) if row.get('mem_conv') is not None else 0.0
    h = float(row['H'])
    up = float(row['u_past'])
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    # Include the archived sink-modulated pool and the observed history/transient variables.
    return np.array([base, -sink * base, math.sqrt(max(u, 0.0)), u, x, m, (u-up) * math.exp(-h), h], dtype=float)

def fit(train_rows):
    X = np.vstack([_features(r) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    # Numerically fit coefficients; regularize only through a small stabilizing residual.
    def residual(p):
        return np.r_[X @ p - y, 1e-7 * p]
    starts = [np.zeros(X.shape[1]), np.ones(X.shape[1]) * 0.1]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, max_nfev=5000)
        if best is None or np.linalg.norm(res.fun) < np.linalg.norm(best.fun):
            best = res
    return best.x

def predict(row):
    return float(_features(row) @ _PARAMS)
