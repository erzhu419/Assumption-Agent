import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([max(1e-8, float(r['u_now'])) for r in train_rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    # Concave increasing substrate response, multiplicatively attenuated by sink gradient.
    def pred(p):
        a, b, alpha, beta = p
        return (a + b * np.power(u, alpha)) * np.exp(-beta * s)
    res = least_squares(lambda p: pred(p) - y,
                        x0=np.array([0.1, 10.0, 0.5, 0.2]),
                        bounds=([0.0, 0.0, 0.05, 0.0], [20.0, 100.0, 1.0, 5.0]),
                        max_nfev=20000)
    _PARAMS = res.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    a, b, alpha, beta = _PARAMS
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return float((a + b * math.pow(u, alpha)) * math.exp(-beta * s))