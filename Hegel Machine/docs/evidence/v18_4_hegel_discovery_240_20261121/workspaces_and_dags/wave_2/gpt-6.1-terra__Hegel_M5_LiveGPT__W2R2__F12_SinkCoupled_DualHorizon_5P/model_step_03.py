import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            * (-np.expm1(-h / tau_fast))
            - amp_adapt * u * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    h = np.asarray([r['H'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _model(p, u, s, h) - y,
                           x0=[6.5, 0.7, 0.5, 1.0, 10.0],
                           bounds=([0.0, 0.0, 0.001, 0.0, 0.01],
                                   [50.0, 1.0, 50.0, 20.0, 500.0]),
                           max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    return float(_model(_params, row['u_now'], row['s_grad'], row['H']))
