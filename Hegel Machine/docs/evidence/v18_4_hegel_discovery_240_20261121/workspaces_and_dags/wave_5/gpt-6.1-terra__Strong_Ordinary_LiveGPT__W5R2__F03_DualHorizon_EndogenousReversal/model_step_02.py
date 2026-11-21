import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        af, ts, a_s, gap = np.exp(p)
        tf = ts
        slow = tf + gap
        return u * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / slow))) - y

    best = None
    for tf in (0.3, 1.0, 3.0):
        for slow in (8.0, 20.0, 50.0):
            if slow <= tf:
                continue
            initial = np.log([2.0, tf, 3.0, slow - tf])
            result = least_squares(residual, initial, bounds=(-10, 10), max_nfev=3000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = np.exp(best.x)

def predict(row):
    af, tf, a_s, gap = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - a_s * (-np.expm1(-h / (tf + gap)))))
