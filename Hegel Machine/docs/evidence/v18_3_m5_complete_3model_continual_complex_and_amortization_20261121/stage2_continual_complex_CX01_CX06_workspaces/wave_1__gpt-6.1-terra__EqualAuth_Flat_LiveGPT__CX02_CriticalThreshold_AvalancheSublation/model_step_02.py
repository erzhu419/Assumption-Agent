import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, km, rho, amplitude, threshold, power = p
    base = vmax * u / (km + u) * (1.0 - rho * s / (1.0 + s))
    return base - amplitude * np.maximum(s - threshold, 0.0) ** power

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    lower = [0.01, 0.01, 0.0, 0.0, 0.5, 0.5]
    upper = [100.0, 20.0, 1.5, 100.0, 1.54, 4.0]
    best = None
    for threshold in (1.0, 1.2, 1.4):
        for power in (1.0, 2.0, 3.0):
            initial = [8.0, 1.2, 0.7, 3.0, threshold, power]
            result = least_squares(lambda p: _model(p, u, s) - y, initial,
                                   bounds=(lower, upper), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))
