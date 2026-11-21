import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return u, s

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def pred(p, r):
        b, vmax, rho, km = p
        u, s = _features(r)
        return b + vmax * (1.0 - rho * s / (1.0 + s)) * (u / (km + u))
    result = least_squares(lambda p: np.array([pred(p, r) for r in train_rows]) - y,
                           x0=[0.0, 4.0, 0.7, 1.2],
                           bounds=([-100.0, 0.0, 0.0, 1e-6], [100.0, 100.0, 1.0, 100.0]),
                           max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    b, vmax, rho, km = _params
    u, s = _features(row)
    return float(b + vmax * (1.0 - rho * s / (1.0 + s)) * (u / (km + u)))