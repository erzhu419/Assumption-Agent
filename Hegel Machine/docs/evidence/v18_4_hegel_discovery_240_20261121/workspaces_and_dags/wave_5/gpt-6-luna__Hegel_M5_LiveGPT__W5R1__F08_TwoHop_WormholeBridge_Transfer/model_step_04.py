import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([max(1e-8, float(r['u_now'])) for r in train_rows], dtype=float)
    s = np.array([float(r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    base = (1.0 - None if False else 1.0)  # placeholder
    def calc(p, sign):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + sign * c_loc * np.sqrt(u)
    best = None
    for sign in (-1.0, 1.0):
        res = least_squares(lambda p: calc(p, sign) - y, x0=[5.0, 0.7, 0.5], bounds=([0.0, 0.0, 0.0], [20.0, 2.0, 10.0]))
        err = float(np.mean((calc(res.x, sign) - y) ** 2))
        if best is None or err < best[0]:
            best = (err, sign, res.x)
    _PARAMS = (best[1], *[float(v) for v in best[2]])
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    sign, vmax, rho, c_loc = _PARAMS
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + sign * c_loc * math.sqrt(u))