import math
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import expm

_P = None

def _predict_params(row, p):
    a, b, c, tau, vmax = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    H = max(0.0, float(row['H']))
    x0 = float(row['x_now'])
    z0 = float(row['mem_conv'] or 0.0)
    # Documented hydraulic-basin attenuation, rho fixed at its archived value.
    J = vmax * (1.0 - 0.70 * s / (1.0 + s)) * (u / (1.2 + u))
    M = np.array([[-a, b], [0.52 / tau, -1.0 / tau]], dtype=float)
    q = np.array([c * J, 0.0], dtype=float)
    # Augmented matrix gives exact propagation of the constant-forcing linear ODE.
    A = np.zeros((3, 3), dtype=float)
    A[:2, :2] = M
    A[:2, 2] = q
    v = expm(A * H) @ np.array([x0, z0, 1.0], dtype=float)
    return float(v[0])

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    best = None
    # A few bounded starts mitigate the ODE parameter correlations without a grid search.
    for start in ([0.15, 0.08, 1.0, 2.0, 2.0], [0.5, 0.2, 1.0, 5.0, 3.0], [0.05, 0.02, 1.0, 10.0, 1.0]):
        result = least_squares(residual, np.asarray(start, dtype=float), bounds=([0.001, 0.0, 0.01, 0.05, 0.01], [3.0, 3.0, 10.0, 100.0, 20.0]), max_nfev=500)
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _P = best[1]
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return _predict_params(row, _P)