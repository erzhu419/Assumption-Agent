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
        return a * u / ((1.0 + b * u) * (1.0 + c * s)) - y
    result = least_squares(residual, [6.0, 1.0, 0.6], bounds=(0.0, np.inf), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    a, b, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(a * u / ((1.0 + b * u) * (1.0 + c * s)))