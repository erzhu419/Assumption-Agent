import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, a_slow, tau_fast, tau_gap = p
        tau_slow = tau_fast + tau_gap
        gain = -a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)
        return u * gain - y

    _params = least_squares(residual, [2.0, 3.0, 0.4, 10.0], bounds=(1e-8, np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    a_fast, a_slow, tau_fast, tau_gap = _params
    h = row['H']
    gain = -a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / (tau_fast + tau_gap))
    return float(row['u_now'] * gain)
