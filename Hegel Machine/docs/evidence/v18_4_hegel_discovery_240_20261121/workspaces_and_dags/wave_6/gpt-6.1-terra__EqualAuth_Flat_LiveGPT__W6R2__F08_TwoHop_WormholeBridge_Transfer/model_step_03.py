import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        a, b, c = p
        return a * u / ((b + u) * (1.0 + c * s)) - y
    result = least_squares(residual, [5.5, 1.2, 0.7], bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x

def predict(row):
    a, b, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(a * u / ((b + u) * (1.0 + c * s)))