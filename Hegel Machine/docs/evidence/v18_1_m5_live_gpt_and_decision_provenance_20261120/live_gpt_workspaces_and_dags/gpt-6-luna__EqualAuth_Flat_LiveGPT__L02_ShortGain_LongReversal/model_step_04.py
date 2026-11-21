import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _integrate(row, p):
    a_slow, b_couple, c_drive, tau_fast = p
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    u = float(row['u_now'])
    H = max(0.0, float(row['H']))
    n = max(1, int(math.ceil(H / 0.004)))
    dt = H / n
    def rhs(xv, zv):
        return (-a_slow * xv + b_couple * zv + c_drive * u,
                (0.52 * xv - zv) / tau_fast)
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = rhs(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = rhs(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    ys = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(logp):
        p = np.exp(np.clip(logp, -12.0, 8.0))
        return np.asarray([_integrate(r, p) for r in train_rows]) - ys
    starts = [np.log([0.1, 0.1, 0.1, 1.0]), np.log([0.5, 0.5, 0.5, 2.0]), np.log([0.02, 0.02, 0.5, 0.2])]
    results = [least_squares(residual, s, max_nfev=250) for s in starts]
    best = min(results, key=lambda q: float(np.dot(q.fun, q.fun)))
    _PARAMS = np.exp(np.clip(best.x, -12.0, 8.0))
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_integrate(row, _PARAMS))