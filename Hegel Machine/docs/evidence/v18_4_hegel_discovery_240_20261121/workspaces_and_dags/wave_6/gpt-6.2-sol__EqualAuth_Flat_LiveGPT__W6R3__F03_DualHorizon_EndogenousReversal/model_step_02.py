import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, h, y = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T

    def residual(log_params):
        fast, slow, tau_fast, gap = np.exp(log_params)
        tau_slow = tau_fast + gap
        return u * (fast * (-np.expm1(-h / tau_fast)) - slow * (-np.expm1(-h / tau_slow))) - y

    result = least_squares(residual, np.log([1.0, 2.0, 0.5, 10.0]), bounds=(-12.0, 12.0), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = np.exp(result.x)

def predict(row) -> float:
    fast, slow, tau_fast, gap = _params
    h = float(row['H'])
    return float(row['u_now'] * (fast * (-math.expm1(-h / tau_fast)) - slow * (-math.expm1(-h / (tau_fast + gap)))))