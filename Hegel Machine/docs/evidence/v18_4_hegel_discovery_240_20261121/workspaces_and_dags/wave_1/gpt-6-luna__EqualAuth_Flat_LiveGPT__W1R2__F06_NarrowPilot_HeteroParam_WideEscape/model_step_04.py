import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([max(1e-6, float(r['u_now'])) for r in train_rows])
    s = np.array([max(0.0, float(r['s_grad'])) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def model(p, uu, ss):
        A, K, alpha, rho = p
        substrate = A * uu**alpha / (K**alpha + uu**alpha)
        gradient = 1.0 - rho * ss / (1.0 + ss)
        return substrate * gradient

    result = least_squares(
        lambda p: model(p, u, s) - y,
        x0=np.array([5.5, 1.0, 1.0, 0.7]),
        bounds=([0.01, 0.01, 0.1, 0.0], [100.0, 100.0, 5.0, 0.99]),
        max_nfev=20000,
    )
    _params = result.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    A, K, alpha, rho = _params
    u = max(1e-6, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return float(A * u**alpha / (K**alpha + u**alpha) * (1.0 - rho * s / (1.0 + s)))