import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 5.0

def _predict_params(row, p):
    gain, tau_relax, k_eq = p
    u_n = float(row['u_now']) / TAU_SCALE
    u_p = float(row['u_past']) / TAU_SCALE
    H = float(row['H'])
    return gain * (u_n - u_p) * math.exp(-H / tau_relax) + k_eq * u_n

def fit(train_rows):
    def residuals(p):
        return np.asarray([_predict_params(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residuals, x0=np.array([1.0, 1.0, 0.0]), bounds=([-100.0, 1e-6, -100.0], [100.0, 1000.0, 100.0]))
    return {'params': result.x.tolist()}

def predict(row):
    return float(_predict_params(row, _FIT_PARAMS))

_FIT_PARAMS = [1.0, 1.0, 0.0]
_original_fit = fit
def fit(train_rows):
    global _FIT_PARAMS
    model = _original_fit(train_rows)
    _FIT_PARAMS = model['params']
    return model
