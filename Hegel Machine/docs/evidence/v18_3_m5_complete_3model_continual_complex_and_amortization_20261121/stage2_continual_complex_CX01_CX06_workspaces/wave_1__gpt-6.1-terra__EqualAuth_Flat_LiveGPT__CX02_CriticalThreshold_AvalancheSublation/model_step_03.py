import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, rho, c, strength, threshold, power = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(np.maximum(u, 0.0)) - strength * np.maximum(s - threshold, 0.0) ** power

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    lower = [0.0, 0.69, -10.0, 0.0, 0.0, 0.5]
    upper = [50.0, 0.72, 10.0, 100.0, 1.54, 3.5]
    best = None
    for threshold in (0.8, 1.1, 1.3, 1.45):
        for power in (1.0, 1.5, 2.0):
            result = least_squares(lambda p: _model(p, u, s) - y, [7.5, 0.705, 0.2, 4.0, threshold, power], bounds=(lower, upper), max_nfev=10000)
            if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))
