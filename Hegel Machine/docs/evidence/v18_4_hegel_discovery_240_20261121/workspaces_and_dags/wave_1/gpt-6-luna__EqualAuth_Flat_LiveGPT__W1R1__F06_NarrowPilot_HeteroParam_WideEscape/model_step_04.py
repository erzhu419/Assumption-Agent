import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([max(1e-8, r['u_now']) for r in train_rows], dtype=float)
    s = np.array([max(1e-8, r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    # Saturating dependence on u and a fitted power response to the gradient.
    def pred(p):
        vmax, km, beta = p
        return vmax * u / (km + u) * np.power(s, beta)
    scale = max(1.0, float(np.max(y)))
    result = least_squares(lambda p: (pred(p) - y) / scale,
                           x0=[4.0, 0.8, -0.5],
                           bounds=([1e-6, 1e-6, -5.0], [100.0, 100.0, 5.0]),
                           max_nfev=20000)
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, km, beta = _PARAMS
    u = max(1e-8, float(row['u_now']))
    s = max(1e-8, float(row['s_grad']))
    return float(vmax * u / (km + u) * (s ** beta))