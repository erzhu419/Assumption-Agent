import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    t_fast = math.exp(p[6])
    t_slow = t_fast + math.exp(p[7])
    return p[0] + u * (p[1] + p[2] * math.exp(-h / t_fast) + p[3] * math.exp(-h / t_slow)) + s * (p[4] + p[5] * u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(p, r) for r in rows]) - y
    lower = [-20, -20, -20, -20, -20, -20, math.log(0.02), math.log(0.05)]
    upper = [20, 20, 20, 20, 20, 20, math.log(15), math.log(200)]
    best = None
    for fast in (0.2, 1.0, 4.0):
        for slow in (5.0, 15.0, 50.0):
            if slow <= fast:
                continue
            p0 = [0.2, -1.0, 1.0, 1.0, -0.3, 0.0, math.log(fast), math.log(slow - fast)]
            result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=5000)
            loss = float(np.dot(result.fun, result.fun))
            if best is None or loss < best[0]:
                best = (loss, result.x)
    _params = best[1]

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return float(_value(_params, row))