import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    k_ss, a_eff = p
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    return k_ss * u + (x - k_ss * u) * np.exp(-a_eff * H)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([2.0, 0.7]), bounds=([0.0, 0.0], [np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _params))