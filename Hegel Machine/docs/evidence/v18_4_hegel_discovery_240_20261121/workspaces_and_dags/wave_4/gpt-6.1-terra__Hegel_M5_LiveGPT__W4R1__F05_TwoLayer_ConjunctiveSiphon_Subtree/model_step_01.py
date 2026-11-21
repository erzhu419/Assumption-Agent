import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 4.0 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u) - y
    result = least_squares(residual, [6.5, 0.7, 0.1], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    vmax, rho, c = _params
    u = row['u_now'] / 4.0
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u))