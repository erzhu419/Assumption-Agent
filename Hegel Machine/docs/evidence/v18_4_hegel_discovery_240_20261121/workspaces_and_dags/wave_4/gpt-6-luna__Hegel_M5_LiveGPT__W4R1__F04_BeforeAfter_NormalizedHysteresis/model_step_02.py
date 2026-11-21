import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    H = np.array([r['H'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    du = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def predict_params(p):
        theta, kappa, E0, tau = p
        return theta + (x - theta) * np.exp(-kappa * H) + E0 * du * np.exp(-H / tau)

    def residual(p):
        return predict_params(p) - y

    result = least_squares(residual, x0=np.array([0.0, 0.5, 2.0, 1.0]), bounds=([-20.0, 1e-6, -20.0, 1e-6], [20.0, 20.0, 20.0, 20.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    theta, kappa, E0, tau = _params
    H = float(row['H'])
    x = float(row['x_now'])
    du = float(row['u_now']) - float(row['u_past'])
    return float(theta + (x - theta) * math.exp(-kappa * H) + E0 * du * math.exp(-H / tau))