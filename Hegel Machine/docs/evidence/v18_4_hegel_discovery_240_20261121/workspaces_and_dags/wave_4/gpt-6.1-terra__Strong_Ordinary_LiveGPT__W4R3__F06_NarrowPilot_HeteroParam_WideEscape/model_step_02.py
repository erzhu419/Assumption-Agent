import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    def residual(p):
        return p[0] * (1.0 - p[1] * sink) * base + p[2] * np.sqrt(u) - y
    result = least_squares(residual, [8.0, 0.7, 1.0], max_nfev=10000)
    _params = result.x
    return _params

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    v, rho, c = _params
    return float(v * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u))