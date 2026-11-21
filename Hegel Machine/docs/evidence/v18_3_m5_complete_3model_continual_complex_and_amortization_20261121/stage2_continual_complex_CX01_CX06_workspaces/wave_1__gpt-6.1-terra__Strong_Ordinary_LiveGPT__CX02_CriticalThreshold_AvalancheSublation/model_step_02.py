import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    v, vrho, c, amp, threshold, power = p
    base = (v - vrho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(np.maximum(u, 0.0))
    return base - amp * np.maximum(s - threshold, 0.0) ** power

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    lower = [0.0, 0.0, -20.0, 0.0, 0.0, 0.5]
    upper = [100.0, 100.0, 20.0, 100.0, 2.0, 5.0]
    best = None
    for threshold in (0.8, 1.0, 1.2, 1.4, 1.6):
        for power in (1.0, 2.0, 3.0):
            p0 = [8.0, 5.6, 1.0, 4.0, threshold, power]
            result = least_squares(lambda p: _model(p, u, s) - y, p0, bounds=(lower, upper), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))
