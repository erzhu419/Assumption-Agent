import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.004
_params = {'gain': 0.0, 'tau_relax': 0.01, 'k_eq': 0.0}

def _model(row, p):
    un = float(row['u_now']) / _TAU_SCALE
    up = float(row['u_past']) / _TAU_SCALE
    H = float(row['H'])
    return p[0] * (un - up) * math.exp(-H / p[1]) + p[2] * un

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.5, 0.1, 0.2]), bounds=([-100.0, 1e-6, -100.0], [100.0, 100.0, 100.0]), max_nfev=20000)
    _params = {'gain': float(result.x[0]), 'tau_relax': float(result.x[1]), 'k_eq': float(result.x[2])}
    return dict(_params)

def predict(row) -> float:
    p = [_params['gain'], _params['tau_relax'], _params['k_eq']]
    return float(_model(row, p))