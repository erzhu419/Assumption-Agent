import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_params(p, row):
    o = float(row['u_outdoor'])
    i = float(row['u_indoor'])
    w = float(row['p_weather'])
    a = float(row['a_foot'])
    m = float(row['impact_idx'])
    # Rainy-regime response: indoor and weather-adjusted outdoor activity,
    # with footprint and impact effects and their interactions.
    z = np.array([1.0, i, o*w, a, m, i*a, i*m, a*m, (o*w)*a, (o*w)*m, i*i, (o*w)**2])
    return float(np.dot(p, z))

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    p0 = np.zeros(12, dtype=float)
    p0[0] = 1.0
    p0[1] = 1.0
    result = least_squares(lambda p: np.array([_predict_params(p, r) for r in train_rows]) - y, p0, max_nfev=20000)
    _params = result.x
    return predict

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return _predict_params(_params, row)
