import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def _model(p, row):
    # Saturating recruitment/transfer modulated by the signed bridge gradient.
    a, k, c, d = p
    u, s = _features(row)
    sat = a * u / (k + max(u, 1e-12))
    return sat + c * (1.0 - s) * u / (1.0 + d * u)

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_model(p, r) for r in rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, 0.5, 0.2]),
                           bounds=([0.0, 1e-6, -20.0, 0.0], [50.0, 50.0, 20.0, 20.0]),
                           max_nfev=20000)
    return result.x

def predict(row):
    p = fit(train_rows)
    return float(_model(p, row))