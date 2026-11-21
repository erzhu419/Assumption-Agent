import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _readout(H, params):
    a, b, c, tau = params
    A = np.array([[-a, b, c], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    n = max(1, math.ceil(H / 0.004))
    M = (H / n) * A
    M2 = M @ M
    M3 = M2 @ M
    step = np.eye(3) + M + M2 / 2 + M3 / 6 + (M3 @ M) / 24
    return np.linalg.matrix_power(step, n)[0]

def fit(train_rows):
    global _params
    rows = list(train_rows)
    H = np.array([r['H'] for r in rows], dtype=float)
    X = np.array([[r['x_now'], r['mem_conv'], r['u_now']] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    groups = [(h, H == h) for h in np.unique(H)]
    start = np.array([2.0, 1.5, 1.2, 0.35])

    def residual(params):
        err = np.empty_like(y)
        for h, mask in groups:
            err[mask] = X[mask] @ _readout(h, params) - y[mask]
        if len(groups) == 1:
            # Select a representative tau when only three readout coefficients are identified.
            return np.r_[err, 1e-5 * math.log(params[3] / start[3])]
        return err

    _params = least_squares(residual, start, bounds=([0.0, 0.0, 0.0, 0.01], [20.0, 20.0, 20.0, 10.0]), ftol=1e-11, xtol=1e-11, gtol=1e-11).x

def predict(row):
    state = [row['x_now'], row['mem_conv'], row['u_now']]
    return float(_readout(float(row['H']), _params) @ state)
