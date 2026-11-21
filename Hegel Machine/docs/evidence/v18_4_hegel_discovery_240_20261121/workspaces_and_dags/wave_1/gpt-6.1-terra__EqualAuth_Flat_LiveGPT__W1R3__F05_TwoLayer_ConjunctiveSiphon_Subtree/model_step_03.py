import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    q = u / (4.8 + u)
    g = s / (1.0 + s)
    return q, g, u

def _value(row, p):
    q, g, u = _features(row)
    return q * (p[0] * (1.0 - p[1] * g) + p[2] * g*g + p[3] * g*g * u / (4.8 + u))

def fit(train_rows):
    global _params
    def residual(p):
        errors = [_value(r, p) - float(r['y']) for r in train_rows]
        return np.array(errors + [0.05 * (p[1] - 0.705), 0.01 * p[3]])
    result = least_squares(residual, [7.3, 0.705, 0.3, 0.5], bounds=([0.0, 0.0, -10.0, -10.0], [30.0, 1.5, 10.0, 10.0]), max_nfev=10000)
    _params = result.x

def predict(row):
    return float(_value(row, _params))
