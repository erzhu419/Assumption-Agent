import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, s):
    vmax, rho, c, amplitude, threshold, power = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(np.maximum(u, 0.0))
    return base - amplitude * np.maximum(s - threshold, 0.0) ** power

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    bounds = ([0.0, 0.0, -20.0, 0.0, 1.5, 1.0], [50.0, 1.5, 20.0, 100.0, 2.77, 5.0])
    best = None
    for threshold in (2.0, 2.3, 2.5, 2.65):
        for power in (1.5, 2.0, 3.0):
            initial = [5.0, 0.7, 1.0, 3.0, threshold, power]
            result = least_squares(lambda p: _values(p, u, s) - y, initial, bounds=bounds, max_nfev=12000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['s_grad'])))