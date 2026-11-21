import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, row):
    vmax, rho, c, strength, threshold = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u)
    return base - strength * max(0.0, s - threshold) ** 2

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_evaluate(p, r) for r in rows]) - y
    best = None
    for threshold in (1.5, 2.0, 2.3, 2.5, 2.7):
        result = least_squares(residual, [6.0, 0.7, 0.5, 2.0, threshold], bounds=([0.0, 0.0, -20.0, 0.0, 0.0], [50.0, 2.0, 20.0, 100.0, 3.58]), max_nfev=5000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_evaluate(_params, row))
