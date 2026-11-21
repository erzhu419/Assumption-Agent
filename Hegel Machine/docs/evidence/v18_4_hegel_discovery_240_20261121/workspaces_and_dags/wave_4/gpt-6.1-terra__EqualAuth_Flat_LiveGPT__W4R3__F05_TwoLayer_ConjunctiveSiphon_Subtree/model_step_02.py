import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, km, rho, c = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * np.sqrt(np.maximum(u, 0.0)) - y
    best = None
    for km in (0.05, 0.3, 1.2, 3.0):
        result = least_squares(residual, [30.0, km, 0.5, 3.0], bounds=([0.0, 0.001, 0.0, -100.0], [10000.0, 100.0, 1.5, 100.0]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    vmax, km, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * np.sqrt(max(u, 0.0)))