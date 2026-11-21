import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    f = u / (1.2 + u)
    g = f * s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        a, b, c, k, threshold = p
        return a*f - b*g + c*root - k*u*np.maximum(0.0, s-threshold) - y

    _params = least_squares(residual, [7.0, 5.0, 0.3, 1.0, 2.61],
                            bounds=([0.0, 0.0, 0.0, 0.0, 2.44],
                                    [np.inf, np.inf, np.inf, np.inf, 2.78])).x
    return _params

def predict(row) -> float:
    a, b, c, k, threshold = _params
    u, s = row['u_now'], row['s_grad']
    return float((a - b*s/(1.0+s))*u/(1.2+u) + c*math.sqrt(u)
                 - k*u*max(0.0, s-threshold))