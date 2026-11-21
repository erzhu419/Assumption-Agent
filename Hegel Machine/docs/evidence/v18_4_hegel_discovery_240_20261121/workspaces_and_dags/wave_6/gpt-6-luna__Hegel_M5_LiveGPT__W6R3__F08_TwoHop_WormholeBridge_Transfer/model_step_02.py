import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def _pred(p, row):
    a, b, c, d = p
    u, s = _features(row)
    return a + b * u / (c + u) + d * (1.0 - s)

def fit(train_rows):
    rows = list(train_rows)
    ys = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_pred(p, r) for r in rows]) - ys
    # Positive saturation scale; unconstrained offsets/amplitudes allow the data to determine signs.
    result = least_squares(residual, x0=np.array([0.0, 3.0, 1.0, 0.8]), bounds=([-20.0, -50.0, 0.01, -20.0], [20.0, 50.0, 50.0, 20.0]))
    return result.x

def predict(row):
    p = fit(row.get('train_rows', []))
    return float(_pred(p, row))