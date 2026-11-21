import numpy as np
from scipy.optimize import least_squares
import math

def _features(r):
    tau = 0.004
    un = r['u_now'] / tau
    up = r['u_past'] / tau
    H = r['H']
    return un, up, H

def fit(train_rows):
    rows = list(train_rows)
    def pred(p, r):
        gain, tau_relax, k_eq = p
        un, up, H = _features(r)
        return gain * (un - up) * math.exp(-H / tau_relax) + k_eq * un
    def residual(p):
        return np.array([pred(p, r) - r['y'] for r in rows])
    result = least_squares(residual, x0=[1.0, 0.01, 0.0], bounds=([-100.0, 1e-8, -100.0], [100.0, 10.0, 100.0]))
    return result.x

def predict(row):
    p = fit(_TRAIN_ROWS)
    un, up, H = _features(row)
    gain, tau_relax, k_eq = p
    return float(gain * (un - up) * math.exp(-H / tau_relax) + k_eq * un)

_TRAIN_ROWS = []
# fit(train_rows) saves fitted training rows for predict, as required by the evaluation interface.
_fit_impl = fit
def fit(train_rows):
    global _TRAIN_ROWS
    _TRAIN_ROWS = list(train_rows)
    return _fit_impl(_TRAIN_ROWS)
