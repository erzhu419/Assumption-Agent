import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    dose = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor'])
    impact = float(row['impact_idx']) * float(row['a_foot'])
    return dose, impact

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    X = [_features(r) for r in train_rows]
    def residual(p):
        gain, k, loss = p
        return np.asarray([gain * d / (k + d) - loss * imp for d, imp in X]) - y
    result = least_squares(residual, x0=[6.0, 0.5, 3.0], bounds=([0.0, 1e-8, 0.0], [20.0, 20.0, 20.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    gain, k, loss = _PARAMS
    dose, impact = _features(row)
    return float(gain * dose / (k + dose) - loss * impact)
