import numpy as np
from scipy.optimize import least_squares

def _predict_params(row, p):
    E, eta = p
    H = float(row['H'])
    u = float(row['u_now'])
    z = E * H / eta
    return (u / E) * (-np.expm1(-z))

def fit(train_rows):
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.5, 0.75]), bounds=([1e-8, 1e-8], [100.0, 1e6]))
    return result.x

def predict(row):
    return float(_predict_params(row, _FIT_PARAMS))

_FIT_PARAMS = None
_OLD_FIT = fit
def fit(train_rows):
    global _FIT_PARAMS
    _FIT_PARAMS = _OLD_FIT(train_rows)
    return _FIT_PARAMS
