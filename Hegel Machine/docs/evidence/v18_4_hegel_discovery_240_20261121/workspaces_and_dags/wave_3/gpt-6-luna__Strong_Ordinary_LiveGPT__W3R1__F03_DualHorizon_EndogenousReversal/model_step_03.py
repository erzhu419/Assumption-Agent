import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def fit(train_rows):
    global _P
    H = np.array([float(r['H']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    def model(p):
        amp_fast, tau_fast, amp_slow, tau_slow = p
        return u * (amp_fast * (-np.expm1(-H / tau_fast)) - amp_slow * (-np.expm1(-H / tau_slow)))

    starts = ([1.0, 1.0, 1.0, 20.0], [2.0, 0.5, 1.0, 30.0], [1.0, 2.0, 0.5, 15.0])
    fits = [least_squares(lambda p: model(p) - y, x0=s,
                          bounds=([0.0, 1e-4, 0.0, 1.0], [20.0, 10.0, 20.0, 500.0]),
                          max_nfev=20000) for s in starts]
    best = min(fits, key=lambda z: float(np.dot(z.fun, z.fun)))
    _P = tuple(float(v) for v in best.x)
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp_fast, tau_fast, amp_slow, tau_slow = _P
    h = float(row['H'])
    u = float(row['u_now'])
    return float(u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow))))