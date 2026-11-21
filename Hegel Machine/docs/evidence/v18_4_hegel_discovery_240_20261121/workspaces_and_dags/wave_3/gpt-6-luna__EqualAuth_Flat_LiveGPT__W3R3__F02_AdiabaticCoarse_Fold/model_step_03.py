import numpy as np
from scipy.optimize import least_squares

_MODEL = {'k': 1.0}

def fit(train_rows):
    global _MODEL
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    result = least_squares(lambda p: p[0] * u - y, x0=np.array([1.0]))
    _MODEL = {'k': float(result.x[0])}
    return _MODEL

def predict(row) -> float:
    return float(_MODEL['k'] * float(row['u_now']))