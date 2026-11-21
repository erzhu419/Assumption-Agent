import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_kind = None

def _value(row, p, kind):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    v, rho, c, af, tf, a_s, ts = p
    sink = 1.0 - rho * s / (1.0 + s)
    base = v * sink * u / (1.2 + u) + c * math.sqrt(max(0.0, u))
    fast = -math.expm1(-h / tf)
    slow = -math.expm1(-h / ts)
    if kind == 0:
        return base + u * (af * fast - a_s * slow)
    if kind == 1:
        return base + u * sink * (af * fast - a_s * slow)
    if kind == 2:
        return base * (1.0 + af * fast - a_s * slow)
    return base + u * (af * fast - a_s * slow * (1.0 + s / (1.0 + s)))

def fit(train_rows):
    global _params, _kind
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    lower = np.array([0.0, 0.0, -5.0, 0.0, 0.02, 0.0, 1.0])
    upper = np.array([20.0, 1.0, 5.0, 20.0, 8.0, 20.0, 100.0])
    starts = [np.array([3.0, 0.7, 0.2, 1.0, 1.0, 2.0, 12.0]),
              np.array([4.0, 0.7, -0.2, 2.0, 0.2, 3.0, 25.0])]
    def solve(indices, kind, guesses):
        def residual(p):
            return np.array([_value(rows[i], p, kind) - y[i] for i in indices])
        results = [least_squares(residual, x, bounds=(lower, upper), max_nfev=3000) for x in guesses]
        return min(results, key=lambda r: np.dot(r.fun, r.fun))
    best = None
    n = len(rows)
    for kind in range(4):
        full = solve(range(n), kind, starts)
        errors = []
        for j in range(n):
            result = solve([i for i in range(n) if i != j], kind, [full.x])
            errors.append(_value(rows[j], result.x, kind) - y[j])
        score = float(np.mean(np.square(errors)))
        if best is None or score < best[0]:
            best = (score, kind, full.x)
    _, _kind, _params = best

def predict(row):
    return float(_value(row, _params, _kind))