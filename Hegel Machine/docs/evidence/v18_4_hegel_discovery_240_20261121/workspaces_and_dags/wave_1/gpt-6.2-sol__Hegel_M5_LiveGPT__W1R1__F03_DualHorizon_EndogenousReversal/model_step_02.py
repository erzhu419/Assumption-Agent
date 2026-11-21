import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        a_fast, a_slow, tau_fast, gap = p
        tau_slow = tau_fast + gap
        return u * (-a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)) - y

    result = least_squares(residual, [1.5, 2.5, 0.5, 10.0], bounds=([0.0, 0.0, 1e-8, 1e-8], np.inf), x_scale='jac')
    _params = result.x

def predict(row) -> float:
    a_fast, a_slow, tau_fast, gap = _params
    h = float(row['H'])
    return float(row['u_now'] * (-a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / (tau_fast + gap))))