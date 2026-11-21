import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    dose = float(row['u_outdoor']) * float(row['p_weather']) + float(row['u_indoor'])
    burden = float(row['impact_idx']) * float(row['a_foot'])
    return dose, burden

def _predict_params(row, params):
    g, k, beta = params
    dose, burden = _features(row)
    return g * dose / (k + dose) - beta * burden

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([7.0, 0.8, 3.0]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_predict_params(row, _PARAMS))