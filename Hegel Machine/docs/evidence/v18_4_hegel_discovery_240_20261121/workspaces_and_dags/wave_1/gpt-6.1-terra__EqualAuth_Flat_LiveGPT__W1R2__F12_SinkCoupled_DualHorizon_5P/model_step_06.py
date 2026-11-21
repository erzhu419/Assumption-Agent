import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _calc(p, rows, kind):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    h = np.array([r['H'] for r in rows], dtype=float)
    q = s / (1.0 + s)
    base = p[0] * (1.0 - p[1] * q) * u / (1.2 + u) + p[2] * np.sqrt(np.maximum(u, 0.0))
    tf, ts = np.exp(p[5]), np.exp(p[6])
    if kind == 0:
        fast = 1.0 - np.exp(-h / tf)
        slow = 1.0 - np.exp(-h * np.exp(p[7] * q) / ts)
        return base + u * (p[3] * fast - p[4] * slow)
    if kind == 1:
        fast = 1.0 - np.exp(-h / tf)
        slow = 1.0 - np.exp(-h / ts)
        return base + u * (p[3] * fast - p[4] * (1.0 + p[7] * q) * slow)
    fast = 1.0 - np.exp(-h * np.exp(p[7] * q) / tf)
    slow = 1.0 - np.exp(-h / ts)
    return base + u * (p[3] * fast - p[4] * slow)

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    best = None
    lower = [-20, -3, -10, -10, -10, np.log(0.02), np.log(0.3), -5]
    upper = [20, 3, 10, 10, 10, np.log(10), np.log(100), 5]
    for kind in range(3):
        for tf in (0.5, 2.0):
            for ts in (5.0, 20.0):
                p0 = [3.0, 0.7, 0.2, 1.0, 1.5, np.log(tf), np.log(ts), 0.0]
                result = least_squares(lambda p: _calc(p, rows, kind) - y, p0, bounds=(lower, upper), max_nfev=5000)
                score = np.sum(result.fun ** 2)
                if best is None or score < best[0]:
                    best = (score, kind, result.x)
    _model, _params = best[1], best[2]

def predict(row):
    return float(_calc(_params, [row], _model)[0])