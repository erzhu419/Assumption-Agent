import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    H = np.array([float(r['H']) for r in train_rows])
    u = np.array([float(r['u_now']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        E, eta = p
        return (u / E) * (1.0 - np.exp(-E * H / eta)) - y
    candidates = []
    for start in ([0.8, 0.1], [0.8, 0.5], [0.8, 2.0], [1.0, 0.2], [0.5, 0.1]):
        res = least_squares(residual, start, bounds=([1e-6, 1e-6], [100.0, 1e4]), max_nfev=20000)
        candidates.append(res)
    best = min(candidates, key=lambda r: float(np.dot(r.fun, r.fun)))
    _params = tuple(float(v) for v in best.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called first')
    E, eta = _params
    H = float(row['H'])
    u = float(row['u_now'])
    return (u / E) * (1.0 - math.exp(-E * H / eta))