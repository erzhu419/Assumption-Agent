import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, rho, offset, strength, threshold, exponent = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base + offset - strength * np.maximum(s - threshold, 0.0) ** exponent

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    lower = [0.0, 0.69, -10.0, 0.0, 0.0, 0.5]
    upper = [100.0, 0.72, 10.0, 100.0, 3.5, 5.0]
    best = None
    for threshold in (1.8, 2.2, 2.4, 2.6, 2.8):
        for exponent in (1.0, 2.0, 3.0):
            start = [8.0, 0.705, 0.0, 3.0, threshold, exponent]
            result = least_squares(lambda p: _model(p, u, s) - y, start, bounds=(lower, upper), max_nfev=5000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))
