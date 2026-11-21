import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _unpack(q):
    a_fast, a_slow, tau_fast, gap = np.exp(q)
    return a_fast, a_slow, tau_fast, tau_fast + gap

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(q):
        af, az, tf, tz = _unpack(q)
        return u * (-af * np.expm1(-h / tf) + az * np.expm1(-h / tz)) - y

    best = None
    for seed in [(1.5, 2.5, 0.45, 11.0), (2.5, 4.0, 0.9, 22.0)]:
        result = least_squares(residual, np.log(seed), bounds=(-12.0, 12.0), ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=400)
        if best is None or result.cost < best.cost:
            best = result
    _params = tuple(map(float, _unpack(best.x)))

def predict(row) -> float:
    af, az, tf, tz = _params
    h = float(row['H'])
    return float(row['u_now']) * (-af * math.expm1(-h / tf) + az * math.expm1(-h / tz))
