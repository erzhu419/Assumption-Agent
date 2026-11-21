import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, rows):
    v, rho, af, tf, ass, ts = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    base = v * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base + u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    best = None
    for tf in (0.3, 1.5, 4.0):
        for ts in (5.0, 15.0, 35.0):
            p0 = [3.0, 0.7, 2.0, tf, 3.0, ts]
            result = least_squares(lambda p: _values(p, rows) - y, p0,
                bounds=([0.0, 0.0, 0.0, 0.02, 0.0, 0.2],
                        [30.0, 2.0, 30.0, 30.0, 30.0, 200.0]),
                max_nfev=4000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_values(_params, [row])[0])