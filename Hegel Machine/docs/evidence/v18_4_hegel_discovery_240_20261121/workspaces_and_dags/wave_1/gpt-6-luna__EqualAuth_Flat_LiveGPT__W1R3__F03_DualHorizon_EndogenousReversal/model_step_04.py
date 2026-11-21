import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _curve(H, p):
    # Positive amplitudes and ordered positive time constants.
    af, ass = np.exp(p[0]), np.exp(p[1])
    tf = np.exp(p[2])
    ts = tf + np.exp(p[3])
    return af * (-np.expm1(-H / tf)) - ass * (-np.expm1(-H / ts))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    h = np.array([float(r['H']) for r in rows])
    u = np.array([float(r['u_now']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    # The mechanism predicts y proportional to u; fit normalized gains to
    # prevent the largest-u rows dominating the timescale estimate.
    target = y / np.maximum(np.abs(u), 1e-8)
    def residual(p):
        return _curve(h, p) - target
    best = None
    # Multistart addresses the strong correlation between amplitudes and
    # timescales over a broad horizon range.
    for tf0 in (0.05, 0.2, 0.8, 2.0):
        for ts0 in (4.0, 10.0, 25.0, 60.0):
            if ts0 <= tf0:
                continue
            p0 = np.log([1.2, 1.0, tf0, ts0 - tf0])
            result = least_squares(residual, p0, max_nfev=20000)
            score = float(np.dot(result.fun, result.fun))
            if best is None or score < best[0]:
                best = (score, result.x)
    _params = best[1]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(float(row['u_now']) * _curve(float(row['H']), _params))
