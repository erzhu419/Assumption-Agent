import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = (1.0, 1.0)

def _predict_with(params, row):
    E, eta = params
    u = float(row['u_now'])
    H = float(row['H'])
    z = E * H / eta
    return (u / E) * (-math.expm1(-z))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows or any('y' not in r for r in rows):
        raise ValueError("fit requires labeled rows containing y")
    def residual(log_params):
        params = np.exp(log_params)
        return np.asarray([_predict_with(params, r) - float(r['y']) for r in rows])
    result = least_squares(residual, np.log([1.0, 1.0]), max_nfev=20000)
    _PARAMS = tuple(np.exp(result.x))
    return _PARAMS

def predict(row) -> float:
    return float(_predict_with(_PARAMS, row))