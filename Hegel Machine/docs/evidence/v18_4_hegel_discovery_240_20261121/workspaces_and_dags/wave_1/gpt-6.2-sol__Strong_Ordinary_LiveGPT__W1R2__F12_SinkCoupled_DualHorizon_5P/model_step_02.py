import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    sat = u / (1.2 + u)
    sink = s / (1.0 + s)

    def residual(p):
        vmax, rho, loss, tau_fast, tau_slow = p
        gain = vmax * sat * (1.0 - rho * sink) * (-np.expm1(-h / tau_fast))
        adaptation = loss * u * (-np.expm1(-h / tau_slow))
        return gain - adaptation - y

    _params = least_squares(residual, [6.0, 0.65, 2.0, 0.4, 10.0], bounds=([0.0, 0.0, 0.0, 1e-4, 1e-4], [np.inf, 1.0, np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row):
    vmax, rho, loss, tau_fast, tau_slow = _params
    u, s, h = row['u_now'], row['s_grad'], row['H']
    gain = vmax * (u / (1.2 + u)) * (1.0 - rho * s / (1.0 + s)) * (-math.expm1(-h / tau_fast))
    adaptation = loss * u * (-math.expm1(-h / tau_slow))
    return float(gain - adaptation)
