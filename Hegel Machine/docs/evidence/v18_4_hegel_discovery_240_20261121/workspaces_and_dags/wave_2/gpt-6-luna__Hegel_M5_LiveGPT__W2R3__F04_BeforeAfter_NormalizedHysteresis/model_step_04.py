import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    h = float(row['H'])
    du = float(row['u_now']) - float(row['u_past'])
    return float(row['x_now']) + du * np.array([1.0, h, h*h, h*h*h])

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_features(r) @ p - yy for r, yy in zip(train_rows, y)])
    result = least_squares(residual, np.zeros(4))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_features(row) @ _params)