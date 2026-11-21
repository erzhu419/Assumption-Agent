import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    return float(row['u_now']), float(row['s_grad'])

def fit(train_rows):
    u = np.array([_features(r)[0] for r in train_rows], dtype=float)
    s = np.array([_features(r)[1] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    # Saturating sink-modulated pool, with a smooth thresholded leak.
    def pred(p):
        vmax, rho, km, leak, threshold, power = p
        base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u)
        excess = np.maximum(s - threshold, 0.0)
        return base - leak * u * excess ** power
    def resid(p):
        return pred(p) - y
    best = None
    for p0 in ([5.0, 0.7, 1.2, 1.0, 1.2, 2.0], [4.0, 0.5, 1.0, 2.0, 1.0, 1.5], [6.0, 0.8, 1.5, 0.5, 1.5, 3.0]):
        res = least_squares(resid, p0, bounds=([0.01, 0.0, 0.01, 0.0, 0.0, 0.5], [20.0, 3.0, 10.0, 20.0, 4.0, 4.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    fit.params = best.x
    return fit

def predict(row):
    p = fit.params
    u, s = _features(row)
    vmax, rho, km, leak, threshold, power = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u)
    return float(base - leak * u * max(s - threshold, 0.0) ** power)
