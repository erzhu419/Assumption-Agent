import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = row['u_now'] / 0.05
    s = row['s_grad']
    j_siph = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    H = row['H']
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    n = max(100, int(math.ceil(H / 0.001)))
    dt = H / n
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + j_siph, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = deriv(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x += (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        z += (dt / 6.0) * (k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_predict_with(r, p) for r in train_rows]) - y
    starts = [
        [0.8, 0.2, 6.5, 0.70, 0.05],
        [0.4, 0.8, 5.0, 0.65, 0.10],
        [1.5, -0.3, 8.0, 0.75, 0.03],
    ]
    lo = [0.001, -5.0, 0.01, 0.0, 0.001]
    hi = [10.0, 5.0, 30.0, 0.999, 2.0]
    results = [least_squares(residual, x0, bounds=(lo, hi), max_nfev=5000) for x0 in starts]
    best = min(results, key=lambda q: np.dot(q.fun, q.fun))
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _params))