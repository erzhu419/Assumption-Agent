import numpy as np
from scipy.optimize import least_squares

_params = None
_coupled = False

def _evaluate(p, rows, coupled):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    a, b, c, fast, slow, tf, delta = p[:7]
    ts = tf + delta
    base = (a - b * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(u)
    adaptation = u * (fast * (-np.expm1(-h / tf)) - slow * (-np.expm1(-h / ts)))
    if coupled:
        adaptation -= p[7] * u * s / (1.0 + s) * (-np.expm1(-h / ts))
    return base + adaptation

def fit(train_rows):
    global _params, _coupled
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    best = None
    for coupled in (False, True):
        lower = [-20, -20, -20, -20, -20, 0.03, 0.03] + ([-20] if coupled else [])
        upper = [20, 20, 20, 20, 20, 50, 200] + ([20] if coupled else [])
        candidate = None
        for tf in (0.5, 2.0, 5.0):
            for delta in (3.0, 12.0, 35.0):
                p0 = [2.0, 1.4, 0.3, 1.0, 2.0, tf, delta] + ([0.0] if coupled else [])
                result = least_squares(lambda p: _evaluate(p, rows, coupled) - y, p0, bounds=(lower, upper), max_nfev=5000)
                error = float(np.mean(result.fun ** 2))
                if candidate is None or error < candidate[0]:
                    candidate = (error, result.x, coupled)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if not coupled and candidate[0] < 1e-8:
            break
    _params, _coupled = best[1], best[2]

def predict(row):
    return float(_evaluate(_params, [row], _coupled)[0])
