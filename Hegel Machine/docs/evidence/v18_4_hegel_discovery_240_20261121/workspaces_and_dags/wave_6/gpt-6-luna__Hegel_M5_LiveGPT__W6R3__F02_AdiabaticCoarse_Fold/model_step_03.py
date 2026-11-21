import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    H = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(theta):
        E, eta = np.exp(theta)
        pred = (u / E) * (1.0 - np.exp(-E * H / eta))
        return pred - y

    result = least_squares(residual, np.log([0.5, 1.0]))
    _params = tuple(np.exp(result.x))
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    E, eta = _params
    u = float(row['u_now'])
    H = float(row['H'])
    return (u / E) * (1.0 - math.exp(-E * H / eta))