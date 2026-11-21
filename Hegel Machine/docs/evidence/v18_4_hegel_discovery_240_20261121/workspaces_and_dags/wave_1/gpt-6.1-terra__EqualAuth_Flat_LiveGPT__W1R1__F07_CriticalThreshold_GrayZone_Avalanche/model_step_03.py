import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, s):
    vmax, rho, threshold, strength, power = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base - strength * np.maximum(s - threshold, 0.0) ** power

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.3], [100.0, 1.5, 3.58, 100.0, 5.0])
    best = None
    for threshold in (2.0, 2.3, 2.5, 2.7):
        for power in (1.0, 1.5, 2.0, 3.0):
            p0 = [8.0, 0.7, threshold, 4.0, power]
            result = least_squares(lambda p: _values(p, u, s) - y, p0, bounds=bounds, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['s_grad'])))
