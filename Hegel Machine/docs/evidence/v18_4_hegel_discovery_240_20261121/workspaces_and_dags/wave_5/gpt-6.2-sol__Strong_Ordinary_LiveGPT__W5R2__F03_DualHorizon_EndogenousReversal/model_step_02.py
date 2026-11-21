import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, a_slow, tau_fast, gap = p
        tau_slow = tau_fast + gap
        return u * (-a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)) - y

    _params = least_squares(residual, [1.5, 2.5, 0.5, 10.0], bounds=([0.0, 0.0, 1e-6, 1e-6], np.inf), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    a_fast, a_slow, tau_fast, gap = _params
    h = float(row['H'])
    return float(row['u_now'] * (-a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / (tau_fast + gap))))
