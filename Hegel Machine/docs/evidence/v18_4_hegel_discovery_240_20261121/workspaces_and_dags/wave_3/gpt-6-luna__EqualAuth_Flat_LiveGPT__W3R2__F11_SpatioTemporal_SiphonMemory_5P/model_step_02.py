import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
TAU_SCALE = 0.05

def _simulate(row, p):
    a_slow, b_couple, vmax, rho, tau_fast = p
    u_n = float(row['u_now']) / TAU_SCALE
    sg = float(row['s_grad'])
    j = vmax * (1.0 - rho * sg / (1.0 + sg)) * (u_n / (1.2 + u_n))
    x = float(row['x_now'])
    z = float(row['mem_conv'])
    H = float(row['H'])
    # Fixed-step RK4; fine resolution relative to the fitted fast timescale.
    n = max(1, int(math.ceil(H / min(0.002, max(tau_fast / 12.0, 1e-5)))))
    dt = H / n
    def f(xv, zv):
        return -a_slow * xv + b_couple * zv + j, (0.52 * xv - zv) / tau_fast
    for _ in range(n):
        k1x, k1z = f(x, z)
        k2x, k2z = f(x + 0.5*dt*k1x, z + 0.5*dt*k1z)
        k3x, k3z = f(x + 0.5*dt*k2x, z + 0.5*dt*k2z)
        k4x, k4z = f(x + dt*k3x, z + dt*k3z)
        x += dt*(k1x + 2*k2x + 2*k3x + k4x)/6.0
        z += dt*(k1z + 2*k2z + 2*k3z + k4z)/6.0
    return x

def fit(train_rows):
    global _PARAMS
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_simulate(r, p) for r in train_rows]) - ys
    starts = [np.array([0.5, 0.5, 1.0, 0.3, 0.05]), np.array([1.0, 1.0, 2.0, 0.5, 0.02]), np.array([0.2, 0.2, 1.5, 0.2, 0.1])]
    lo = np.array([0.001, 0.001, 0.001, 0.0, 0.001])
    hi = np.array([10.0, 10.0, 20.0, 0.999, 1.0])
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=(lo, hi), max_nfev=3000)
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_simulate(row, _PARAMS))