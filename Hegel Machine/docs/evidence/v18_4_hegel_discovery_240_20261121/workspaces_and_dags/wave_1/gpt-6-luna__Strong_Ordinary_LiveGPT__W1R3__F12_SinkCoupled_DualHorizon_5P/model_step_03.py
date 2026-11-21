import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _model(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _P
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows], dtype=float)
    # Log parameters keep amplitudes and timescales positive; tau_slow > tau_fast.
    def unpack(q):
        tf = math.exp(float(q[1]))
        return (math.exp(float(q[0])), tf, math.exp(float(q[2])), tf + math.exp(float(q[3])))
    def residual(q):
        p = unpack(q)
        return np.array([_model(r, p) - y for r, y in zip(rows, ys)], dtype=float)
    starts = [np.log([0.8, 1.0, 1.2, 7.0]), np.log([1.0, 0.5, 1.0, 10.0]), np.log([1.5, 2.0, 2.0, 12.0])]
    best = None
    for q0 in starts:
        res = least_squares(residual, q0, max_nfev=20000)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, unpack(res.x))
    _P = best[1]
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _P))