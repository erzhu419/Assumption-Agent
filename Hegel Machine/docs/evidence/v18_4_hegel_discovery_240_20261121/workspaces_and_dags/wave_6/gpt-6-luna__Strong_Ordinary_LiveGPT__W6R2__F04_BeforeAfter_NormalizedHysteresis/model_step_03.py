import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / 0.004
    u_p = float(row['u_past']) / 0.004
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([1.0, 0.5, 0.5]), bounds=([-100.0, 1e-5, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    _P = result.x
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _P))