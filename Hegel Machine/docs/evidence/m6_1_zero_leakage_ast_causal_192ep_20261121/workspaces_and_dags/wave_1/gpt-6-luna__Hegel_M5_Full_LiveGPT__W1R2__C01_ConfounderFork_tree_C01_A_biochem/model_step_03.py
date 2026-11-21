import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _gate(row):
    return 1.0 if float(row.get('z_1', 0.0)) > float(row.get('z_3', 0.0)) else 0.0

def _predict_params(row, p):
    vmax, k_half, c_ctrl, gate_effect = p
    x = float(row['x_ind']) / 1.5
    x2 = x * x
    hill = vmax * x2 / (k_half + x2)
    return hill - c_ctrl * float(row.get('u_ctrl', 0.0)) + gate_effect * _gate(row)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.7, 0.0]),
                           bounds=([0.0, 1e-8, -10.0, -20.0], [100.0, 100.0, 10.0, 20.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_predict_params(row, _PARAMS))