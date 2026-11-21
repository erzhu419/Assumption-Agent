import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(np.maximum(u, 0.0)) - y
    _params = least_squares(residual, [5.0, 0.5, 0.5], max_nfev=10000).x

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, rho, c = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(max(u, 0.0)))