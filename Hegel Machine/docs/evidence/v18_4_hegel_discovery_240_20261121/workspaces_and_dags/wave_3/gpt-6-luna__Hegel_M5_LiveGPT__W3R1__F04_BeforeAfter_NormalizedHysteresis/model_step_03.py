import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    du = np.array([float(r['u_now']) - float(r['u_past']) for r in rows], dtype=float)
    H = np.array([float(r['H']) for r in rows], dtype=float)
    x = np.array([float(r['x_now']) for r in rows], dtype=float)
    y = np.array([float(r['y']) for r in rows], dtype=float)

    def residual(p):
        amp, tau = p
        return x - amp * du * np.exp(-H / tau) - y

    result = least_squares(residual, x0=np.array([100.0, 0.5]), bounds=([0.0, 1e-6], [10000.0, 100.0]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    amp, tau = _params
    du = float(row['u_now']) - float(row['u_past'])
    return float(row['x_now']) - amp * du * math.exp(-float(row['H']) / tau)
