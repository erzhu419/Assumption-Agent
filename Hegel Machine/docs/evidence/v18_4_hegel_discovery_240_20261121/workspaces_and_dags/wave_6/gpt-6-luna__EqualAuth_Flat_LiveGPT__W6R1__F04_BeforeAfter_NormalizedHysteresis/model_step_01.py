import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 5.0

def _features(row):
    up = float(row['u_past']) / TAU_SCALE
    un = float(row['u_now']) / TAU_SCALE
    h = float(row['H'])
    return un, up, h

def _value(row, p):
    gain, tau_relax, k_eq = p
    un, up, h = _features(row)
    return gain * (un - up) * math.exp(-h / tau_relax) + k_eq * un

def fit(train_rows):
    def residuals(p):
        return np.asarray([_value(r, p) - float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(residuals, x0=np.array([5.0, 5.0, 0.0]), bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row) -> float:
    p = fit(_TRAIN_ROWS)
    return float(_value(row, p))

_TRAIN_ROWS = []

# fit() records the dataset so predict(row) uses the parameters estimated from training.
_fit_impl = fit
def fit(train_rows):
    global _TRAIN_ROWS
    _TRAIN_ROWS = list(train_rows)
    return _fit_impl(_TRAIN_ROWS)
