import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _weights(H, p):
    a, b, c, tau = p
    n = max(1, math.ceil(H / 0.004))
    dt = H / n
    # Advance the linear system on (x, z, u), with du/dt = 0.
    D = dt * np.array([[-a, b, c], [0.52 / tau, -1.0 / tau, 0.0], [0.0, 0.0, 0.0]])
    D2 = D @ D
    step = np.eye(3) + D + D2 / 2.0 + (D2 @ D) / 6.0 + (D2 @ D2) / 24.0
    return np.linalg.matrix_power(step, n)[0]

def _features(row):
    return [row['x_now'], row['mem_conv'], row['u_now']]

def fit(train_rows):
    global _params
    rows = list(train_rows)
    X = np.array([_features(r) for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    horizons, group = np.unique([r['H'] for r in rows], return_inverse=True)
    start = np.log([1.8, 2.2, 1.2, 0.18])
    def residual(log_p):
        p = np.exp(log_p)
        W = np.array([_weights(float(H), p) for H in horizons])
        error = np.sum(X * W[group], axis=1) - y
        # At one horizon the rates are not separately identifiable.
        if len(horizons) == 1:
            return np.append(error, 0.01 * (log_p[3] - start[3]))
        return error
    result = least_squares(residual, start, bounds=(np.log([0.001, 0.001, 0.001, 0.015]), np.log([20.0, 30.0, 20.0, 1.0])), ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = np.exp(result.x)

def predict(row):
    return float(_weights(float(row['H']), _params) @ _features(row))