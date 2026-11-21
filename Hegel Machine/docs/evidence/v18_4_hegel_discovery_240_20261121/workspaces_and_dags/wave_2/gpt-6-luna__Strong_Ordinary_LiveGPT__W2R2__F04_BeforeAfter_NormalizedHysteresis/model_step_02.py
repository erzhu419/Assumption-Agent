import math
import numpy as np
from scipy.optimize import least_squares

def _pred(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / 5.0
    u_p = float(row['u_past']) / 5.0
    h = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-h / tau_relax) + k_eq * u_n

def fit(train_rows):
    rows = list(train_rows)
    if not rows:
        raise ValueError('fit requires training rows')
    def residual(p):
        return np.array([_pred(r, p) - float(r['y']) for r in rows], dtype=float)
    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5]), bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    return result.x

def predict(row):
    if not hasattr(predict, '_params'):
        raise RuntimeError('Call fit(train_rows) and set predict parameters before predicting')
    return float(_pred(row, predict._params))

_original_fit = fit
def fit(train_rows):
    params = _original_fit(train_rows)
    predict._params = params
    return params