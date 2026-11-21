import math
import numpy as np
from scipy.optimize import least_squares

_fitted = None

def fit(train_rows):
    global _fitted
    rows = list(train_rows)
    d = np.array([float(r['u_now']) - float(r['u_past']) for r in rows], dtype=float)
    h = np.array([float(r['H']) for r in rows], dtype=float)
    x = np.array([float(r['x_now']) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)

    def residual(p):
        amplitude, log_tau = p
        tau = math.exp(float(log_tau))
        return x + amplitude * d * np.exp(-h / tau) - y

    result = least_squares(residual, np.array([2.0, math.log(1.5)]))
    _fitted = (float(result.x[0]), float(math.exp(float(result.x[1]))))
    return _fitted

def predict(row):
    if _fitted is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amplitude, tau = _fitted
    delta = float(row['u_now']) - float(row['u_past'])
    return float(row['x_now']) + amplitude * delta * math.exp(-float(row['H']) / tau)
