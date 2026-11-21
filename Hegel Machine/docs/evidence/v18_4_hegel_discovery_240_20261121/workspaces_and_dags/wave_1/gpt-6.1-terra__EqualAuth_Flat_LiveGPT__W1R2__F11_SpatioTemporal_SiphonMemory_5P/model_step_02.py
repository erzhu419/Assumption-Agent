import math
import numpy as np
from scipy.optimize import least_squares

_model = None
_params = None

def _forecast(row, p, kind):
    u = float(row['u_now'])
    x = float(row['x_now'])
    m = float(row['mem_conv'])
    s = float(row['s_grad'])
    H = float(row['H'])
    n = max(1, int(math.ceil(abs(H) / 0.025)))
    dt = H / n
    def deriv(z):
        x, m = z
        if kind == 0:
            return np.array([p[0] + p[1]*u - p[2]*x - p[3]*s + p[4]*m, 0.0])
        if kind == 1:
            return np.array([p[0] + p[1]*u - p[2]*x - p[3]*s*x + p[4]*m, 0.0])
        if kind == 2:
            return np.array([p[0]*u - p[1]*x - p[2]*s + p[3]*m, p[4]*(u-m)])
        return np.array([p[0]*u - p[1]*x - p[2]*s*x + p[3]*m, p[4]*(u-m)])
    z = np.array([x, m], dtype=float)
    for _ in range(n):
        k1 = deriv(z)
        k2 = deriv(z + dt*k1/2)
        k3 = deriv(z + dt*k2/2)
        k4 = deriv(z + dt*k3)
        z += dt*(k1 + 2*k2 + 2*k3 + k4)/6
    return float(z[0])

def _fit_kind(rows, kind):
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_forecast(r, p, kind) for r in rows]) - y
    if kind < 2:
        bounds = ([-5, -5, 0, -5, -5], [5, 5, 8, 5, 5])
        start = [0.5, 0.5, 1.0, 0.3, 0.2]
    else:
        bounds = ([-5, 0, -5, -5, 0], [5, 8, 5, 5, 8])
        start = [0.5, 1.0, 0.3, 0.2, 1.0]
    return least_squares(residual, start, bounds=bounds, max_nfev=1200).x

def fit(train_rows):
    global _model, _params
    rows = list(train_rows)
    best = float('inf')
    for kind in range(4):
        errors = []
        try:
            for i, row in enumerate(rows):
                p = _fit_kind(rows[:i] + rows[i+1:], kind)
                errors.append(_forecast(row, p, kind) - float(row['y']))
            score = float(np.mean(np.square(errors)))
            p = _fit_kind(rows, kind)
            if np.isfinite(score) and score < best:
                best, _model, _params = score, kind, p
        except (ValueError, FloatingPointError, OverflowError):
            continue
    if _params is None:
        raise RuntimeError('Could not fit a transient model')

def predict(row):
    return _forecast(row, _params, _model)
