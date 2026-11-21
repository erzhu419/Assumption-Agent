import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    vmax, rho, c, af, tf, ass, ts = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(np.maximum(u, 0.0))
    dynamic = u * (af * (-np.expm1(-h / tf)) - ass * (-np.expm1(-h / ts)))
    return base + dynamic

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows] if False else np.array([r['y'] for r in rows], dtype=float)
    lower = [0.0, 0.0, -10.0, 0.0, 0.01, 0.0, 0.5]
    upper = [30.0, 2.0, 10.0, 20.0, 15.0, 20.0, 100.0]
    best = None
    for tf in (0.3, 1.5, 5.0):
        for ts in (5.0, 15.0, 35.0):
            start = [3.0, 0.7, 0.3, 1.0, tf, 1.5, ts]
            result = least_squares(lambda p: _values(p, rows) - y, start, bounds=(lower, upper), max_nfev=12000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_values(_params, [row])[0])