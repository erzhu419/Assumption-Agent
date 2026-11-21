import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = np.exp(p)
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (-math.expm1(-h / tau_fast)) - amp_slow * (-math.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    # Positive amplitudes and timescales; ordered timescales encode fast and slow components.
    def unpack(p):
        af, ass, tf, delta = np.exp(p)
        return af, ass, tf, tf + delta
    def residual(p):
        af, ass, tf, ts = unpack(p)
        pred = [float(r['u_now']) * (af * (-math.expm1(-float(r['H']) / tf)) - ass * (-math.expm1(-float(r['H']) / ts))) for r in rows]
        return np.asarray(pred) - y
    starts = [np.log([1.0, 1.0, 1.0, 10.0]), np.log([1.5, 1.5, 0.5, 15.0]), np.log([2.0, 1.0, 2.0, 20.0])]
    results = [least_squares(residual, s, max_nfev=20000) for s in starts]
    best = min(results, key=lambda z: float(np.dot(z.fun, z.fun)))
    af, ass, tf, ts = unpack(best.x)
    _params = np.log([af, ass, tf, ts - tf])
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _params))