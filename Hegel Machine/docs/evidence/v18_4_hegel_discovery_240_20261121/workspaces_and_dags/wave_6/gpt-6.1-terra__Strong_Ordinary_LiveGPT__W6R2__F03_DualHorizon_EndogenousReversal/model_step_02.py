import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        af, slow_amp, tf, delta = p
        ts = tf + delta
        return u * (af * (-np.expm1(-h / tf)) - slow_amp * (-np.expm1(-h / ts))) - y

    best = None
    for tf in (0.3, 1.0, 3.0):
        for ts in (5.0, 15.0, 40.0):
            if ts <= tf:
                continue
            result = least_squares(residual, [2.0, 3.0, tf, ts - tf], bounds=([0.0, 0.0, 0.01, 0.01], [100.0, 100.0, 100.0, 500.0]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    af, slow_amp, tf, delta = _params
    h = float(row['H'])
    return float(row['u_now'] * (af * (-np.expm1(-h / tf)) - slow_amp * (-np.expm1(-h / (tf + delta)))))