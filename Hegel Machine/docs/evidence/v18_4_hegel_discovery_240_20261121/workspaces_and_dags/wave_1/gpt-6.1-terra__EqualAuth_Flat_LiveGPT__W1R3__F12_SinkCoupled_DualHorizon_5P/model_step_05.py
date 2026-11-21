import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _value(row, p, mode):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    vmax, rho, local, af, slow, tf, ts = p
    sink = 1.0 - rho * s / (1.0 + s)
    base = vmax * sink * u / (1.2 + u) + local * math.sqrt(max(u, 0.0))
    fast = -math.expm1(-h / tf)
    delayed = -math.expm1(-h / ts)
    if mode == 0:
        adaptation = u * (af * fast - slow * delayed)
    elif mode == 1:
        adaptation = u * sink * (af * fast - slow * delayed)
    elif mode == 2:
        adaptation = u * (af * sink * fast - slow * delayed)
    elif mode == 3:
        adaptation = u * (af * fast - slow * sink * delayed)
    else:
        adaptation = u * (af * fast - slow * delayed) / (1.0 + rho * s)
    return base + adaptation

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    lower = [0.0, 0.0, -10.0, -10.0, -10.0, 0.02, 0.5]
    upper = [30.0, 2.0, 10.0, 10.0, 10.0, 8.0, 100.0]
    best = None
    for mode in range(5):
        for tf, ts in [(0.3, 8.0), (1.0, 15.0), (3.0, 25.0)]:
            p0 = [3.0, 0.7, 0.2, 1.0, 2.0, tf, ts]
            result = least_squares(lambda p: np.array([_value(r, p, mode) for r in rows]) - y, p0, bounds=(lower, upper), max_nfev=3000)
            error = float(np.sum(result.fun ** 2))
            if best is None or error < best[0]:
                best = (error, mode, result.x)
    _, _model, _params = best

def predict(row):
    return float(_value(row, _params, _model))
