import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _gain(h, p):
    fast_amp, slow_amp, fast_tau, gap = p
    slow_tau = fast_tau + gap
    return -fast_amp * np.expm1(-h / fast_tau) + slow_amp * np.expm1(-h / slow_tau)

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    start = np.log([1.0, 2.0, np.median(h), 0.5 * np.max(h)])
    result = least_squares(lambda q: u * _gain(h, np.exp(q)) - y, start, ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1000)
    _params = np.exp(result.x)

def predict(row):
    fast_amp, slow_amp, fast_tau, gap = _params
    h = float(row['H'])
    return float(row['u_now'] * (-fast_amp * math.expm1(-h / fast_tau) + slow_amp * math.expm1(-h / (fast_tau + gap))))