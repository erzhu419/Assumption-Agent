import math
import numpy as np
from scipy.optimize import least_squares


def _simulate(rows, p):
    a_slow, b_couple, c_drive, tau_fast = p
    hs = np.asarray([float(r['H']) for r in rows])
    x = np.asarray([float(r['x_now']) for r in rows], dtype=float).copy()
    z = np.asarray([0.0 if r.get('mem_conv') is None else float(r['mem_conv']) for r in rows], dtype=float).copy()
    u = np.asarray([float(r['u_now']) for r in rows], dtype=float)
    n = max(1, int(math.ceil(float(np.max(hs)) / 0.04)))
    dt = hs / n
    def deriv(xx, zz):
        return -a_slow * xx + b_couple * zz + c_drive * u, (0.52 * xx - zz) / tau_fast
    for _ in range(n):
        k1x, k1z = deriv(x, z)
        k2x, k2z = deriv(x + 0.5 * dt * k1x, z + 0.5 * dt * k1z)
        k3x, k3z = deriv(x + 0.5 * dt * k2x, z + 0.5 * dt * k2z)
        k4x, k4z = deriv(x + dt * k3x, z + dt * k3z)
        x += (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        z += (dt / 6.0) * (k1z + 2*k2z + 2*k3z + k4z)
    return x


def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return _simulate(train_rows, p) - y
    starts = [[0.15, 0.08, 0.5, 0.8], [0.5, 0.2, 1.0, 0.3], [0.03, 0.03, 0.7, 2.0]]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([1e-5, 0.0, 0.0, 0.01], [5.0, 5.0, 10.0, 20.0]), max_nfev=400, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    return best.x


def predict(row):
    return float(_simulate([row], fit(_TRAIN_ROWS))[0])