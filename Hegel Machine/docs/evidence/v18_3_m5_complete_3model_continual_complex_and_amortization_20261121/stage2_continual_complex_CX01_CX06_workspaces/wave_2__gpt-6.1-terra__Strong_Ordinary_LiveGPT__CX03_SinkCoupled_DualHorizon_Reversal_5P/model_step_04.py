import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(row, p, mode):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    vmax, rho, loc, fast, slow, tf, ts = p
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + loc * math.sqrt(max(u, 0.0))
    f = -math.expm1(-h / tf)
    q = -math.expm1(-h / ts)
    if mode == 0:
        return base + u * (fast * f - slow * q)
    if mode == 1:
        return base * (1.0 + fast * f - slow * q)
    if mode == 2:
        return base + fast * f - slow * u * q
    return base + fast * u * f - slow * base * q

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    lower = [0.0, 0.0, -5.0, 0.0, 0.0, 0.02, 0.5]
    upper = [20.0, 2.0, 5.0, 20.0, 20.0, 10.0, 200.0]
    best = None
    for mode in range(4):
        for tf in (0.3, 1.5, 4.0):
            for ts in (8.0, 25.0, 60.0):
                p0 = [2.5, 0.7, 0.4, 0.7, 1.5, tf, ts]
                result = least_squares(lambda p: np.array([_value(r, p, mode) for r in rows]) - y, p0, bounds=(lower, upper), max_nfev=2500)
                score = np.mean(result.fun ** 2)
                if best is None or score < best[0]:
                    best = (score, mode, result.x)
    _model, _params = best[1], best[2]

def predict(row):
    return float(_value(row, _params, _model))
