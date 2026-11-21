import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _model(row, p):
    a_fast, a_slow, tau_fast, tau_slow, c_sink = p
    h = float(row['H'])
    gain = a_fast * (1.0 - math.exp(-h / tau_fast)) - a_slow * (1.0 - math.exp(-h / tau_slow))
    return float(row['u_now']) * gain + c_sink * float(row['s_grad'])

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in rows]) - y
    p0 = np.array([3.0, 4.0, 1.0, 10.0, 0.0])
    lo = [0.0, 0.0, 1e-4, 0.5001, -10.0]
    hi = [20.0, 20.0, 0.5, 1000.0, 10.0]
    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=20000)
    _P = result.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return _model(row, _P)
