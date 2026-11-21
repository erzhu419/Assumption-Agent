import numpy as np
from scipy.optimize import least_squares

_THETA = None

def _features(rows):
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    x = np.array([r['x_now'] for r in rows], dtype=float)
    m = np.array([r['mem_conv'] for r in rows], dtype=float)
    return u, s, x, m

def fit(train_rows):
    global _THETA
    u, s, x, m = _features(train_rows)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    # Saturating hydraulic field, sink attenuation, and an additive branch
    # driven by the available state/memory features.
    def pred(p):
        vmax, km, rho, bx, bm, bp = p
        base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u)
        branch = bx * x + bm * m + bp
        return base + branch
    def residual(p):
        return pred(p) - y
    result = least_squares(residual, x0=np.array([5.0, 1.2, 0.70, 0.0, 0.0, 0.0]),
                           bounds=([0.01, 0.01, 0.0, -10, -10, -10],
                                   [20.0, 20.0, 0.99, 10, 10, 10]), max_nfev=20000)
    _THETA = result.x
    return _THETA

def predict(row):
    if _THETA is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, km, rho, bx, bm, bp = _THETA
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u)
    return float(base + bx * float(row['x_now']) + bm * float(row['mem_conv']) + bp)
