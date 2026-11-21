import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        _PARAMS = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        return _PARAMS
    X = np.array([[float(r.get('u_outdoor', 0.0)), float(r.get('u_indoor', 0.0)), float(r.get('p_weather', 0.0)), float(r.get('a_foot', 1.0)), float(r.get('impact_idx', 0.0))] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def pred(p, x):
        b0, b_out, amp, rate, impact_pow = p
        outdoor = x[:, 0] * x[:, 2]
        dose = np.maximum(x[:, 1], 0.0) / np.maximum(x[:, 3], 1e-6) * np.maximum(1.0 - x[:, 4], 0.0) ** impact_pow
        return b0 + b_out * outdoor + amp * (1.0 - np.exp(-rate * dose))
    res = least_squares(lambda p: pred(p, X) - y, x0=np.array([0.0, 2.0, 6.0, 0.6, 1.0]), bounds=([-20.0, -50.0, -50.0, 0.001, 0.05], [20.0, 50.0, 50.0, 20.0, 5.0]), max_nfev=10000)
    _PARAMS = res.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    x = np.array([[float(row.get('u_outdoor', 0.0)), float(row.get('u_indoor', 0.0)), float(row.get('p_weather', 0.0)), float(row.get('a_foot', 1.0)), float(row.get('impact_idx', 0.0))]])
    b0, b_out, amp, rate, impact_pow = _PARAMS
    outdoor = x[0, 0] * x[0, 2]
    dose = max(x[0, 1], 0.0) / max(x[0, 3], 1e-6) * max(1.0 - x[0, 4], 0.0) ** impact_pow
    return float(b0 + b_out * outdoor + amp * (1.0 - math.exp(-rate * dose)))