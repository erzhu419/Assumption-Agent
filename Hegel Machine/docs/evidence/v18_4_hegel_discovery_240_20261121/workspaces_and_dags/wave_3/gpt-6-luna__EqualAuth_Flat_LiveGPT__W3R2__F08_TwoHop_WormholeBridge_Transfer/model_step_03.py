import math
import numpy as np
from scipy.optimize import least_squares


def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = float(row['H'])
    # Saturating two-hop transfer, modulated by the sink gradient.
    return u, s, H


def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    X = np.array([_features(r) for r in rows], dtype=float)

    def predict_params(p, z):
        a, b, c, d, e = p
        u, s, H = z
        transfer = 1.0 - math.exp(-max(c, 1e-8) * max(u, 0.0) * max(H, 0.0))
        return a + b * transfer + d * (s - 1.0) + e * (s - 1.0) * transfer

    def residual(p):
        return np.array([predict_params(p, z) for z in X]) - y

    p0 = np.array([0.0, 5.0, 1.0, -0.4, 0.0])
    result = least_squares(residual, p0, max_nfev=20000)
    fit.params_ = result.x
    return fit


def predict(row):
    p = fit.params_
    u, s, H = _features(row)
    a, b, c, d, e = p
    transfer = 1.0 - math.exp(-max(c, 1e-8) * max(u, 0.0) * max(H, 0.0))
    return float(a + b * transfer + d * (s - 1.0) + e * (s - 1.0) * transfer)
