import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    scale = 0.004
    un = float(row['u_now']) / scale
    up = float(row['u_past']) / scale
    h = float(row['H'])
    return un, up, h

def _value(p, row):
    gain, tau_relax, k_eq = p
    un, up, h = _features(row)
    return gain * (un - up) * math.exp(-h / tau_relax) + k_eq * un

def fit(train_rows):
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    best = None
    for p0 in ([1.0, 0.3, 0.1], [0.5, 1.0, 0.0], [2.0, 0.05, 0.2]):
        res = least_squares(residual, p0, bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x

def predict(row):
    # Refit from rows supplied to fit via returned parameter vector is not possible here;
    # predict accepts the fitted parameter vector attached by the evaluation harness.
    p = row.get('_fit_params')
    if p is None:
        raise ValueError('predict expects _fit_params returned by fit')
    return float(_value(p, row))