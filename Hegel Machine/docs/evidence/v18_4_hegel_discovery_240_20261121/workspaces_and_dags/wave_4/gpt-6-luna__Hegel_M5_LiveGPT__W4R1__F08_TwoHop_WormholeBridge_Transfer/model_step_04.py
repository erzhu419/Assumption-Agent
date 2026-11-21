import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def pred(p):
        amp, half_sat, grad_rate, offset = p
        return offset + amp * u / (half_sat + u) * np.exp(-grad_rate * s)

    result = least_squares(lambda p: pred(p) - y,
                           x0=np.array([4.0, 1.0, 0.2, 0.0]),
                           bounds=([0.0, 1e-6, -5.0, -20.0], [50.0, 50.0, 5.0, 20.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp, half_sat, grad_rate, offset = _PARAMS
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(offset + amp * u / (half_sat + u) * math.exp(-grad_rate * s))