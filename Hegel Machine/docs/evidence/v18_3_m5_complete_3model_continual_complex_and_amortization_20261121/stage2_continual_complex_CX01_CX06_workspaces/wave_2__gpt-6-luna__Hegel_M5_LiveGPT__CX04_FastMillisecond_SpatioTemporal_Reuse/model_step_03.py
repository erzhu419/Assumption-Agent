import math
import numpy as np
from scipy.optimize import least_squares

_TAU_SCALE = 0.005
_params = None

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = row['u_now'] / _TAU_SCALE
    s = row['s_grad']
    influx = vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n))
    H = row['H']
    # Resolve the fast memory state while keeping a fixed, reproducible RK4 scheme.
    n = max(200, int(math.ceil(H / max(tau_fast / 12.0, 1e-5))))
    dt = H / n
    x, z = float(row['x_now']), float(row['mem_conv'])
    def deriv(xv, zv):
        return -a_slow * xv + b_couple * zv + influx, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = deriv(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x += dt * (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        z += dt * (k1z + 2*k2z + 2*k3z + k4z) / 6.0
    return x

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_simulate(r, p) for r in train_rows]) - y
    lower = np.array([0.0, -2.0, 0.01, 0.0, 1e-4])
    upper = np.array([20.0, 2.0, 30.0, 0.999, 2.0])
    starts = [np.array([1.0, 0.2, 6.0, 0.7, 0.02]), np.array([2.0, -0.2, 8.0, 0.5, 0.05])]
    results = [least_squares(residual, x0=np.minimum(np.maximum(s, lower + 1e-8), upper - 1e-8), bounds=(lower, upper), max_nfev=3000) for s in starts]
    _params = min(results, key=lambda q: float(np.dot(q.fun, q.fun))).x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _params))