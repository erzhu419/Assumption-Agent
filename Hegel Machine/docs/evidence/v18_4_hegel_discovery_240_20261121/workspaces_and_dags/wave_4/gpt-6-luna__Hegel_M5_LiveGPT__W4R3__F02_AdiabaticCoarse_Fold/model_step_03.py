import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    H = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def predict_params(p):
        E, eta = np.exp(p)
        z = np.clip(-E * H / eta, -700.0, 0.0)
        return (u / E) * (1.0 - np.exp(z))
    result = least_squares(lambda p: predict_params(p) - y,
                           x0=np.log([0.47, 0.25]), max_nfev=20000)
    _params = tuple(np.exp(result.x))
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    E, eta = _params
    H = float(row['H'])
    return (float(row['u_now']) / E) * (1.0 - math.exp(max(-700.0, -E * H / eta)))