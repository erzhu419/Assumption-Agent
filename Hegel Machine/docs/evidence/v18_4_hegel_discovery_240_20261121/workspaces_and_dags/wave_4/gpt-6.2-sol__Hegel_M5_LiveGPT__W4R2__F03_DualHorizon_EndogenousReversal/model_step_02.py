import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, tau_fast, a_slow, tau_gap = p
        tau_slow = tau_fast + tau_gap
        return u * (-a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)) - y

    _params = least_squares(residual, [1.6, 0.4, 2.3, 10.0], bounds=([0.0, 1e-8, 0.0, 1e-8], np.inf)).x

def predict(row) -> float:
    a_fast, tau_fast, a_slow, tau_gap = _params
    h = row['H']
    tau_slow = tau_fast + tau_gap
    return float(row['u_now'] * (-a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / tau_slow)))