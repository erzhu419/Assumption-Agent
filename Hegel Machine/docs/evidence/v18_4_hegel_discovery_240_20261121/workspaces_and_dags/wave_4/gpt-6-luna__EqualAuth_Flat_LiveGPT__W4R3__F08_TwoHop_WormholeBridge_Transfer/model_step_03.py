import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def fit(train_rows):
    rows = list(train_rows)
    X = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    u, s = X[:, 0], X[:, 1]
    def pred(p):
        a, b, c, d, e, f = p
        return a + b * (1.0 - np.exp(-np.maximum(c, 1e-6) * u)) + d*s + e*u*s + f*s*s
    p0 = np.array([1.0, 3.0, 0.5, 0.0, 0.0, 0.0])
    result = least_squares(lambda p: pred(p) - y, p0, max_nfev=20000)
    return result.x

def predict(row):
    p = fit([row])
    u, s = _features(row)
    a, b, c, d, e, f = p
    return float(a + b * (1.0 - np.exp(-max(c, 1e-6) * u)) + d*s + e*u*s + f*s*s)
