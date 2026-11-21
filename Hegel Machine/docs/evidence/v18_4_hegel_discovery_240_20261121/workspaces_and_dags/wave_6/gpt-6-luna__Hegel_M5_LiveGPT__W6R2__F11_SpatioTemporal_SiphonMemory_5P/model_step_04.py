import math
import numpy as np
from scipy.optimize import least_squares

_P = None
_TAU_SCALE = 0.005

def _simulate(rows, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    x = np.asarray([r['x_now'] for r in rows], dtype=float)
    z = np.asarray([r['mem_conv'] for r in rows], dtype=float)
    H = np.asarray([r['H'] for r in rows], dtype=float)
    u_n = np.asarray([r['u_now'] / _TAU_SCALE for r in rows], dtype=float)
    sg = np.asarray([r['s_grad'] for r in rows], dtype=float)
    drive = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    n = max(1, int(math.ceil(float(np.max(H)) / 0.00075)))
    dt = H / n
    def rhs(xx, zz):
        return -a_slow * xx + b_couple * zz + drive, (0.52 * xx - zz) / tau_fast
    for _ in range(n):
        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = rhs(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = rhs(x + dt*k3x, z + dt*k3z)
        x = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        z = z + (dt/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
    return x

def fit(train_rows):
    global _P
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    lo = np.array([0.0, -2.0, 1.0, 0.05, 0.0002])
    hi = np.array([8.0, 2.0, 15.0, 0.95, 0.3])
    starts = [np.array([1.0, 0.5, 6.0, 0.7, 0.02]),
              np.array([0.5, 1.0, 6.0, 0.7, 0.01])]
    best = None
    for x0 in starts:
        res = least_squares(lambda q: _simulate(train_rows, q) - y,
                            x0=np.minimum(np.maximum(x0, lo+1e-8), hi-1e-8),
                            bounds=(lo, hi), max_nfev=90, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        score = float(np.dot(res.fun, res.fun))
        if best is None or score < best[0]:
            best = (score, res.x)
    _P = best[1]
    return _P

def predict(row):
    if _P is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_simulate([row], _P)[0])