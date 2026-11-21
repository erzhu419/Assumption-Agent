import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(row, p, kind):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    sink = 1.0 - 0.70 * s / (1.0 + s)
    base = p[0] * sink * u / (1.2 + u) + p[1] * math.sqrt(max(0.0, u))
    tf, ts = math.exp(p[4]), math.exp(p[5])
    fast = -math.expm1(-h / tf)
    slow = -math.expm1(-h / ts)
    if kind == 0:
        return base + u * (p[2] * fast - p[3] * slow)
    if kind == 1:
        return base + u * sink * (p[2] * fast - p[3] * slow)
    if kind == 2:
        return base + u * (p[2] * fast - p[3] * s / (1.0 + s) * slow)
    return base + u * (p[2] * sink * fast - p[3] * slow)

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    y = np.array [float(r['y']) for r in rows]
    best = None
    for kind in range(4):
        def residual(p):
            return np.array([_value(r, p, kind) for r in rows]) - y
        for tf, ts in [(0.3, 8.0), (1.0, 10.0), (3.0, 15.0), (0.1, 20.0), (5.0, 30.0)]:
            p0 = [2.0, 0.2, 1.0, 2.0, math.log(tf), math.log(ts)]
            result = least_squares(residual, p0, bounds=([-20, -20, -20, -20, -4.6, -2.3], [20, 20, 20, 20, 4.6, 5.3]), max_nfev=3000)
            error = float(np.dot(result.fun, result.fun))
            if best is None or error < best[0]:
                best = (error, kind, result.x.copy())
    _, _model, _params = best

def predict(row):
    return float(_value(row, _params, _model))