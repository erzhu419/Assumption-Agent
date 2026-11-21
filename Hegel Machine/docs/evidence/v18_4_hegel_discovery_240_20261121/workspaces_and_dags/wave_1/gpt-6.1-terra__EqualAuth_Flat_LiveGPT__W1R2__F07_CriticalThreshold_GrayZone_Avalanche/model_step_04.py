import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c, amp, threshold, power = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u)
    return base - amp * max(0.0, s - threshold) ** power

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    bounds = ([0.0, 0.69, -10.0, 0.0, 0.0, 0.3], [30.0, 0.72, 10.0, 100.0, 2.779, 5.0])
    solutions = []
    for threshold in (1.8, 2.2, 2.5):
        for power in (1.0, 2.0, 3.0):
            p0 = [8.0, 0.705, 0.5, 2.0, threshold, power]
            result = least_squares(residual, p0, bounds=bounds, max_nfev=10000)
            solutions.append(result)
    _params = min(solutions, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return float(_value(row, _params))